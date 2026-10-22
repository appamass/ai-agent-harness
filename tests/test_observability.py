"""OpenTelemetry listener tests (ADR 0022).

Verifies translation of harness metadata events to OpenTelemetry spans,
parent-child hierarchies, GenAI semantic conventions, and zero PII/secret leakage.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import BaseModel

from ai_agent_harness import (
    Autonomy,
    BaseTool,
    Event,
    EventType,
    Risk,
    RunState,
    RunStatus,
    ToolError,
    start,
    tool,
)
from ai_agent_harness.core.errors import ConfigError
from ai_agent_harness.core.types import Usage
from ai_agent_harness.observability import OpenTelemetryListener, log_listener
from ai_agent_harness.testing import text_response, tool_response

from .conftest import make_agent, run_sync


@pytest.fixture
def otel_setup() -> tuple[TracerProvider, InMemorySpanExporter]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_missing_dependency_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import ai_agent_harness.observability as obs

    monkeypatch.setattr(obs, "_otel_available", False)
    with pytest.raises(ConfigError, match="pip install 'ai-agent-harness\\[opentelemetry\\]'"):
        OpenTelemetryListener()


def test_full_agent_run_span_hierarchy_and_gen_ai_attributes(
    otel_setup: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    provider, exporter = otel_setup
    tracer = provider.get_tracer("test_harness")
    listener = OpenTelemetryListener(tracer)

    @tool(risk=Risk.READ)
    def fetch_weather(city: str) -> str:
        """Fetches weather for a city."""
        return f"Weather in {city}: 22C, sunny"

    resp1 = tool_response(("fetch_weather", {"city": "Berlin"})).model_copy(
        update={
            "model": "test-model",
            "usage": Usage(
                input_tokens=100, output_tokens=25, cache_read_tokens=15, cache_write_tokens=7, reasoning_tokens=5
            ),
        }
    )
    resp2 = text_response("The weather in Berlin is 22C and sunny.").model_copy(
        update={"model": "test-model", "usage": Usage(input_tokens=150, output_tokens=30)}
    )

    agent, _, _ = make_agent([resp1, resp2], [fetch_weather], autonomy=Autonomy.READ_ONLY)
    agent.events.subscribe(listener)

    final_state = run_sync(start(agent, "What is the weather in Berlin?"))
    assert final_state.status is RunStatus.DONE

    spans = exporter.get_finished_spans()
    # Expect 4 spans:
    # 1. tool span: execute_tool fetch_weather
    # 2. model span 1: chat test-model
    # 3. model span 2: chat test-model
    # 4. root span: run agent
    assert len(spans) == 4

    # Group spans by name
    run_spans = [s for s in spans if s.name.startswith("run ")]
    model_spans = [s for s in spans if s.name.startswith("chat ")]
    tool_spans = [s for s in spans if s.name.startswith("execute_tool ")]

    assert len(run_spans) == 1
    assert len(model_spans) == 2
    assert len(tool_spans) == 1

    root = run_spans[0]
    assert root.name == "run test"
    assert root.attributes is not None
    assert "gen_ai.system" not in root.attributes
    assert root.attributes["gen_ai.workflow.name"] == "test"
    assert root.attributes["gen_ai.agent.name"] == "test"
    assert root.attributes["harness.run.status"] == "done"
    assert root.status.status_code is StatusCode.OK

    # Verify hierarchy: child spans must have parent.span_id == root.context.span_id
    root_context = root.context
    assert root_context is not None
    root_span_id = root_context.span_id
    for child in (*model_spans, *tool_spans):
        parent_ctx = child.parent
        assert parent_ctx is not None
        assert parent_ctx.span_id == root_span_id

    # Verify model span attributes
    first_model = model_spans[0]
    assert first_model.attributes is not None
    assert first_model.attributes["gen_ai.operation.name"] == "chat"
    assert first_model.attributes["gen_ai.response.model"] == "test-model"
    assert first_model.attributes["gen_ai.usage.input_tokens"] == 100
    assert first_model.attributes["gen_ai.usage.output_tokens"] == 25
    assert first_model.attributes["gen_ai.usage.cache_read_input_tokens"] == 15
    assert first_model.attributes["gen_ai.usage.cache_creation_input_tokens"] == 7
    assert first_model.attributes["harness.usage.reasoning_tokens"] == 5
    assert first_model.attributes["gen_ai.response.finish_reasons"] == ("tool_use",)
    assert first_model.status.status_code is StatusCode.OK

    # Verify tool span attributes
    t_span = tool_spans[0]
    assert t_span.name == "execute_tool fetch_weather"
    assert t_span.attributes is not None
    assert t_span.attributes["gen_ai.tool.name"] == "fetch_weather"
    assert t_span.attributes["harness.tool.is_error"] is False
    assert t_span.status.status_code is StatusCode.OK


def test_tool_failure_marks_span_as_error(
    otel_setup: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    provider, exporter = otel_setup
    tracer = provider.get_tracer("test_harness")
    listener = OpenTelemetryListener(tracer)

    @tool(risk=Risk.READ)
    def fail_tool() -> str:
        """Always fails."""
        raise ToolError("Database connection refused")

    resp1 = tool_response(("fail_tool", {}))
    resp2 = text_response("Tool failed, stopping.")
    agent, _, _ = make_agent([resp1, resp2], [fail_tool])
    agent.events.subscribe(listener)

    run_sync(start(agent, "run fail tool"))

    spans = exporter.get_finished_spans()
    tool_spans = [s for s in spans if s.name.startswith("execute_tool ")]
    assert len(tool_spans) == 1
    t_span = tool_spans[0]
    assert t_span.attributes is not None
    assert t_span.attributes["harness.tool.is_error"] is True
    assert t_span.status.status_code is StatusCode.ERROR


def test_tool_blocked_creates_blocked_span(
    otel_setup: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    provider, exporter = otel_setup
    tracer = provider.get_tracer("test_harness")
    listener = OpenTelemetryListener(tracer)

    @tool(risk=Risk.IRREVERSIBLE)
    def drop_table() -> str:
        """Destructive action."""
        return "table dropped"

    resp1 = tool_response(("drop_table", {}))
    resp2 = text_response("Blocked, stopping.")
    agent, _, _ = make_agent([resp1, resp2], [drop_table], autonomy=Autonomy.READ_ONLY)
    agent.events.subscribe(listener)

    run_sync(start(agent, "drop everything"))

    spans = exporter.get_finished_spans()
    tool_spans = [s for s in spans if s.name.startswith("execute_tool ")]
    assert len(tool_spans) == 1
    t_span = tool_spans[0]
    assert t_span.attributes is not None
    assert t_span.attributes["harness.tool.blocked"] is True
    assert t_span.attributes["harness.tool.block_source"] == "policy"
    assert t_span.status.status_code is StatusCode.ERROR


def test_provider_retry_and_failover_recorded_as_span_events(
    otel_setup: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    provider, exporter = otel_setup
    tracer = provider.get_tracer("test_harness")
    listener = OpenTelemetryListener(tracer)

    ev_req = Event(
        type=EventType.MODEL_REQUEST,
        run_id="r1",
        tenant_id="t1",
        step=1,
        data={"messages": 3, "tools": 2},
    )
    ev_retry = Event(
        type=EventType.PROVIDER_RETRY,
        run_id="r1",
        tenant_id="t1",
        step=1,
        data={"provider": "anthropic", "attempt": 1, "delay_s": 0.5},
    )
    ev_failover = Event(
        type=EventType.PROVIDER_FAILOVER,
        run_id="r1",
        tenant_id="t1",
        step=1,
        data={"to": "openai", "retryable": True},
    )
    ev_resp = Event(
        type=EventType.MODEL_RESPONSE,
        run_id="r1",
        tenant_id="t1",
        step=1,
        data={"provider": "openai", "model": "gpt-4o", "stop_reason": "end", "usage": {"input_tokens": 50}},
    )

    run_sync(listener.handle(ev_req))
    run_sync(listener.handle(ev_retry))
    run_sync(listener.handle(ev_failover))
    run_sync(listener.handle(ev_resp))

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    model_span = spans[0]
    event_names = [e.name for e in model_span.events]
    assert "gen_ai.retry" in event_names
    assert "gen_ai.failover" in event_names

    retry_event = next(e for e in model_span.events if e.name == "gen_ai.retry")
    assert retry_event.attributes is not None
    assert retry_event.attributes["provider"] == "anthropic"
    assert retry_event.attributes["attempt"] == 1
    assert retry_event.attributes["delay_s"] == 0.5

    failover_event = next(e for e in model_span.events if e.name == "gen_ai.failover")
    assert failover_event.attributes is not None
    assert failover_event.attributes["to"] == "openai"
    assert failover_event.attributes["retryable"] is True


def test_approval_and_checkpoint_events_recorded_on_run_span(
    otel_setup: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    provider, exporter = otel_setup
    tracer = provider.get_tracer("test_harness")
    listener = OpenTelemetryListener(tracer)

    ev_start = Event(type=EventType.RUN_STARTED, run_id="r2", tenant_id="t1", step=0, data={"agent": "tester"})
    ev_appr = Event(
        type=EventType.APPROVAL_REQUESTED,
        run_id="r2",
        tenant_id="t1",
        step=1,
        data={"tool": "deploy_prod", "risk": "irreversible"},
    )
    ev_ckpt = Event(type=EventType.CHECKPOINT_SAVED, run_id="r2", tenant_id="t1", step=1, data={"revision": 4})
    ev_finish = Event(
        type=EventType.RUN_FINISHED, run_id="r2", tenant_id="t1", step=1, data={"status": "waiting_for_approval"}
    )

    run_sync(listener.handle(ev_start))
    run_sync(listener.handle(ev_appr))
    run_sync(listener.handle(ev_ckpt))
    run_sync(listener.handle(ev_finish))

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    root = spans[0]
    event_names = [e.name for e in root.events]
    assert "harness.approval_requested" in event_names
    assert "harness.checkpoint_saved" in event_names


def test_zero_sensitive_data_leakage_in_spans(
    otel_setup: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    """Verifies invariant: never leak raw prompts, tool arguments, or secrets in OTel spans (ADR 0011)."""
    provider, exporter = otel_setup
    tracer = provider.get_tracer("test_harness")
    listener = OpenTelemetryListener(tracer)

    secret_prompt = "MY_SUPER_SECRET_PROMPT_98765"
    secret_arg = "TOP_SECRET_CREDENTIAL_XYZ"
    secret_output = "INTERNAL_DATABASE_PASSWORD_456"

    class SecretInput(BaseModel):
        token: str

    class SecretTool(BaseTool[SecretInput]):
        name = "secret_action"
        description = "processes sensitive data"
        input_model = SecretInput
        risk = Risk.READ

        async def run(self, args: SecretInput, state: RunState) -> str:
            return f"Processed {args.token} with {secret_output}"

    resp1 = tool_response(("secret_action", {"token": secret_arg}))
    resp2 = text_response("Finished safely.")
    agent, _, _ = make_agent([resp1, resp2], [SecretTool()])
    agent.events.subscribe(listener)

    run_sync(start(agent, secret_prompt))

    all_spans = exporter.get_finished_spans()
    assert len(all_spans) >= 3

    for span in all_spans:
        # Check span name
        assert secret_prompt not in span.name
        assert secret_arg not in span.name
        assert secret_output not in span.name

        # Check attributes
        if span.attributes:
            for key, val in span.attributes.items():
                val_str = str(val)
                assert secret_prompt not in key and secret_prompt not in val_str
                assert secret_arg not in key and secret_arg not in val_str
                assert secret_output not in key and secret_output not in val_str

        # Check span events
        for event in span.events:
            assert secret_prompt not in event.name
            assert secret_arg not in event.name
            assert secret_output not in event.name
            if event.attributes:
                for ek, ev in event.attributes.items():
                    ev_str = str(ev)
                    assert secret_prompt not in ek and secret_prompt not in ev_str
                    assert secret_arg not in ek and secret_arg not in ev_str
                    assert secret_output not in ek and secret_output not in ev_str


def test_log_listener_logs_the_event_with_its_ids(caplog: pytest.LogCaptureFixture) -> None:
    ev = Event(type=EventType.RUN_STARTED, run_id="run-99", tenant_id="tenant-1", step=0, data={"agent": "assistant"})
    caplog.set_level(logging.INFO, logger="ai_agent_harness.run")

    run_sync(log_listener(ev))

    records = [r for r in caplog.records if getattr(r, "harness_event", {}).get("run_id") == "run-99"]
    assert len(records) == 1 and records[0].getMessage().startswith("run_started")


# ------------------------------------------------------------------ Metrics, attribute names, bounded memory


def test_genai_metrics_are_recorded(otel_setup: tuple[TracerProvider, InMemorySpanExporter]) -> None:
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    provider, _ = otel_setup
    reader = InMemoryMetricReader()
    meter = MeterProvider(metric_readers=[reader]).get_meter("test_harness")
    listener = OpenTelemetryListener(provider.get_tracer("test_harness"), meter=meter)

    resp = text_response("hello").model_copy(
        update={"model": "test-model", "usage": Usage(input_tokens=100, output_tokens=25)}
    )
    agent, _, _ = make_agent([resp], [])
    agent.events.subscribe(listener)
    run_sync(start(agent, "hi"))

    data = reader.get_metrics_data()
    assert data is not None
    metrics = {m.name: m for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics}
    assert set(metrics) == {"gen_ai.client.token.usage", "gen_ai.client.operation.duration"}

    usage_points = {
        dict(point.attributes or {})["gen_ai.token.type"]: point.sum
        for point in cast("Any", metrics["gen_ai.client.token.usage"].data).data_points
    }
    assert usage_points == {"input": 100, "output": 25}
    duration_points = list(cast("Any", metrics["gen_ai.client.operation.duration"].data).data_points)
    assert len(duration_points) == 1
    assert duration_points[0].count == 1


def test_non_standard_attributes_use_the_harness_prefix(
    otel_setup: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    provider, exporter = otel_setup
    listener = OpenTelemetryListener(provider.get_tracer("test_harness"))
    agent, _, _ = make_agent([text_response("hello")], [])
    agent.events.subscribe(listener)
    run_sync(start(agent, "hi"))

    model_span = next(s for s in exporter.get_finished_spans() if s.name.startswith("chat "))
    assert model_span.attributes is not None
    names = set(model_span.attributes)
    assert {"harness.request.messages", "gen_ai.provider.name"} <= names
    assert not {"gen_ai.request.messages_count", "gen_ai.request.tools_count", "gen_ai.system"} & names


def test_open_runs_are_bounded(otel_setup: tuple[TracerProvider, InMemorySpanExporter]) -> None:
    provider, exporter = otel_setup
    listener = OpenTelemetryListener(provider.get_tracer("test_harness"), max_open_runs=2)

    async def started(run_id: str) -> None:
        await listener(Event(type=EventType.RUN_STARTED, run_id=run_id, tenant_id="t", step=0, data={"agent": "a"}))

    for run_id in ("r1", "r2", "r3"):
        run_sync(started(run_id))

    ended = [s for s in exporter.get_finished_spans() if s.name == "run a"]
    assert len(ended) == 1  # the oldest run was ended as abandoned instead of leaking
    assert ended[0].status.status_code is StatusCode.ERROR
    assert ended[0].attributes is not None
    assert ended[0].attributes["harness.run_id"] == "r1"


def test_invalid_max_open_runs_rejected() -> None:
    with pytest.raises(ConfigError, match="max_open_runs"):
        OpenTelemetryListener(max_open_runs=0)


# ------------------------------------------------------------------ Run outcomes and resumed runs


def _event(type_: EventType, run: str = "r", **data: object) -> Event:
    return Event(type=type_, run_id=run, tenant_id="t", step=1, data=data)


def test_a_failed_run_is_an_error_span_and_a_waiting_run_records_what_it_waits_for(
    otel_setup: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    provider, exporter = otel_setup
    listener = OpenTelemetryListener(provider.get_tracer("test_harness"))
    events = (
        _event(EventType.RUN_STARTED, "failed-run", agent="x"),
        _event(EventType.RUN_FINISHED, "failed-run", status="failed"),
        _event(EventType.RUN_STARTED, "waiting-run", agent="x"),
        _event(
            EventType.RUN_FINISHED, "waiting-run", status="waiting_for_approval", waiting_for="approval", tool="deploy"
        ),
    )
    for event in events:
        run_sync(listener.handle(event))

    spans = {str((s.attributes or {})["harness.run_id"]): s for s in exporter.get_finished_spans()}
    assert spans["failed-run"].status.status_code is StatusCode.ERROR
    waiting = spans["waiting-run"]
    assert waiting.status.status_code is StatusCode.OK
    assert (waiting.attributes or {})["harness.run.waiting_for"] == "approval"
    assert (waiting.attributes or {})["harness.run.waiting_tool"] == "deploy"


def test_a_resume_without_an_open_span_creates_one_and_open_children_end_on_finish(
    otel_setup: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    provider, exporter = otel_setup
    listener = OpenTelemetryListener(provider.get_tracer("test_harness"))
    events = (
        _event(EventType.RUN_RESUMED, **{"from": "waiting_for_approval"}),
        _event(EventType.MODEL_REQUEST, messages=1, tools=0),
        _event(EventType.TOOL_STARTED, tool="x"),
        _event(EventType.INPUT_REQUESTED, tool="ask_user"),
        _event(EventType.RUN_FINISHED, status="cancelled"),
    )
    for event in events:
        run_sync(listener.handle(event))

    spans = exporter.get_finished_spans()
    assert sorted(s.name for s in spans) == ["execute_tool x", "model_request", "resume r"]
    root = next(s for s in spans if s.name == "resume r")
    assert (root.attributes or {})["harness.resumed_from"] == "waiting_for_approval"
    assert "harness.input_requested" in [e.name for e in root.events]
    children = [s for s in spans if s.name != "resume r"]
    assert children and all(s.status.status_code is StatusCode.ERROR for s in children)  # abandoned, not "ok"


def test_zero_valued_optional_usage_is_not_recorded(
    otel_setup: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    provider, exporter = otel_setup
    listener = OpenTelemetryListener(provider.get_tracer("test_harness"))
    usage = {"input_tokens": 1, "output_tokens": 2, "cache_read_tokens": 0, "reasoning_tokens": 0}
    events = (
        _event(EventType.RUN_STARTED, agent="x"),
        _event(EventType.MODEL_REQUEST, messages=1, tools=0),
        _event(EventType.MODEL_RESPONSE, provider="p", model="m", stop_reason="end", usage=usage),
        _event(EventType.RUN_FINISHED, status="done"),
    )
    for event in events:
        run_sync(listener.handle(event))

    chat = next(s for s in exporter.get_finished_spans() if s.name.startswith("chat"))
    attributes = chat.attributes or {}
    assert attributes["gen_ai.usage.input_tokens"] == 1
    assert "gen_ai.usage.cache_read_input_tokens" not in attributes
    assert "harness.usage.reasoning_tokens" not in attributes
