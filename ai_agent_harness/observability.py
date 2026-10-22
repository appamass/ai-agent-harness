"""Observability: listeners for the event bus.

`log_listener` writes every event as structured log output.
`OpenTelemetryListener` translates harness metadata events into OpenTelemetry spans and the two
GenAI client metrics (`gen_ai.client.token.usage`, `gen_ai.client.operation.duration`). Attributes outside the
GenAI semantic conventions are prefixed `harness.` (ADR 0022).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, cast

from ._version import __version__
from .core.errors import ConfigError
from .core.events import Event, EventType

try:
    from opentelemetry import metrics, trace
    from opentelemetry.metrics import Histogram, Meter
    from opentelemetry.trace import Span, SpanKind, StatusCode, Tracer

    _otel_available = True
except ImportError:
    _otel_available = False
    if TYPE_CHECKING:
        from opentelemetry import metrics, trace
        from opentelemetry.metrics import Histogram, Meter
        from opentelemetry.trace import Span, SpanKind, StatusCode, Tracer

log = logging.getLogger("ai_agent_harness.run")

_USAGE_ATTRIBUTES = (
    # (usage key, span attribute, only recorded when > 0)
    ("input_tokens", "gen_ai.usage.input_tokens", False),
    ("output_tokens", "gen_ai.usage.output_tokens", False),
    ("cache_read_tokens", "gen_ai.usage.cache_read_input_tokens", True),
    ("cache_write_tokens", "gen_ai.usage.cache_creation_input_tokens", True),
    ("reasoning_tokens", "harness.usage.reasoning_tokens", True),
)


async def log_listener(event: Event) -> None:
    if not log.isEnabledFor(logging.INFO):
        return
    log.info(
        "%s run=%s tenant=%s step=%d %s",
        event.type,
        event.run_id,
        event.tenant_id,
        event.step,
        event.data,
        extra={"harness_event": event.model_dump(mode="json")},
    )


class OpenTelemetryListener:
    """Event listener that emits OpenTelemetry spans following GenAI Semantic Conventions.

    Adheres strictly to ADR 0011 & ADR 0022: events emit metadata only (token counts,
    model names, tool names, execution times, status flags). No prompts, user messages,
    tool arguments, or decrypted secrets are ever recorded in spans or attributes.
    """

    def __init__(
        self,
        tracer: Tracer | None = None,
        *,
        tracer_name: str = "ai_agent_harness",
        tracer_version: str = __version__,
        meter: Meter | None = None,
        max_open_runs: int = 1000,
    ) -> None:
        if not _otel_available:
            raise ConfigError("The OpenTelemetry listener needs the SDK: pip install 'ai-agent-harness[opentelemetry]'")
        if max_open_runs < 1:
            raise ConfigError("OpenTelemetryListener: max_open_runs must be >= 1")
        self._tracer: Tracer = tracer if tracer is not None else trace.get_tracer(tracer_name, tracer_version)
        self._max_open_runs = max_open_runs
        meter = meter if meter is not None else metrics.get_meter(tracer_name, tracer_version)
        self._token_usage: Histogram = meter.create_histogram(
            "gen_ai.client.token.usage", unit="{token}", description="Number of input and output tokens used"
        )
        self._operation_duration: Histogram = meter.create_histogram(
            "gen_ai.client.operation.duration", unit="s", description="GenAI operation duration"
        )
        self._run_spans: dict[str, Span] = {}
        self._model_spans: dict[str, tuple[Span, float]] = {}  # run_id -> (span, monotonic start)
        self._tool_spans: dict[tuple[str, str], Span] = {}

    async def __call__(self, event: Event) -> None:
        await self.handle(event)

    async def handle(self, event: Event) -> None:
        match event.type:
            case EventType.RUN_STARTED:
                self._on_run_started(event)
            case EventType.RUN_RESUMED:
                self._on_run_resumed(event)
            case EventType.MODEL_REQUEST:
                self._on_model_request(event)
            case EventType.MODEL_RESPONSE:
                self._on_model_response(event)
            case EventType.PROVIDER_RETRY:
                self._on_provider_retry(event)
            case EventType.PROVIDER_FAILOVER:
                self._on_provider_failover(event)
            case EventType.TOOL_STARTED:
                self._on_tool_started(event)
            case EventType.TOOL_FINISHED:
                self._on_tool_finished(event)
            case EventType.TOOL_BLOCKED:
                self._on_tool_blocked(event)
            case EventType.APPROVAL_REQUESTED:
                self._on_approval_requested(event)
            case EventType.INPUT_REQUESTED:
                self._on_input_requested(event)
            case EventType.CHECKPOINT_SAVED:
                self._on_checkpoint_saved(event)
            case EventType.RUN_FINISHED:
                self._on_run_finished(event)

    def _start(self, name: str, kind: SpanKind, event: Event, *, child: bool = True) -> Span:
        """Starts a span with the attributes every harness span carries; `child` attaches it to the run span."""
        parent = self._run_spans.get(event.run_id) if child else None
        context = trace.set_span_in_context(parent) if parent is not None else None
        span = self._tracer.start_span(name, context=context, kind=kind)
        span.set_attribute("harness.run_id", event.run_id)
        span.set_attribute("harness.tenant_id", event.tenant_id)
        span.set_attribute("harness.step", event.step)
        return span

    def _open_run_span(self, span: Span, event: Event) -> None:
        self._run_spans[event.run_id] = span
        # Bounds memory when runs never report `RUN_FINISHED` (killed process, dropped events)
        while len(self._run_spans) > self._max_open_runs:
            self._abandon_run(next(iter(self._run_spans)), "Abandoned: too many open runs")

    def _on_run_started(self, event: Event) -> None:
        agent_name = str(event.data.get("agent", "agent"))
        span = self._start(f"run {agent_name}", SpanKind.INTERNAL, event, child=False)
        span.set_attribute("gen_ai.workflow.name", agent_name)
        span.set_attribute("gen_ai.agent.name", agent_name)
        self._open_run_span(span, event)

    def _on_run_resumed(self, event: Event) -> None:
        previous = str(event.data.get("from"))
        existing = self._run_spans.get(event.run_id)
        if existing is not None:
            existing.add_event("harness.run_resumed", {"from": previous})
            return
        span = self._start(f"resume {event.run_id}", SpanKind.INTERNAL, event, child=False)
        span.set_attribute("harness.resumed_from", previous)
        self._open_run_span(span, event)

    def _on_model_request(self, event: Event) -> None:
        span = self._start("model_request", SpanKind.CLIENT, event)
        span.set_attribute("gen_ai.operation.name", "chat")
        for key, attribute in (("messages", "harness.request.messages"), ("tools", "harness.request.tools")):
            value = event.data.get(key)
            if isinstance(value, int):
                span.set_attribute(attribute, value)
        self._model_spans[event.run_id] = (span, time.monotonic())

    def _on_model_response(self, event: Event) -> None:
        entry = self._model_spans.pop(event.run_id, None)
        if entry is None:
            return
        span, started = entry
        provider = str(event.data.get("provider", "unknown"))
        model = str(event.data.get("model", "unknown"))
        span.update_name(f"chat {model}")
        span.set_attribute("gen_ai.provider.name", provider)
        span.set_attribute("gen_ai.response.model", model)
        stop_reason = event.data.get("stop_reason")
        if stop_reason is not None:
            span.set_attribute("gen_ai.response.finish_reasons", [str(stop_reason)])

        usage_raw = event.data.get("usage")
        if isinstance(usage_raw, dict):
            usage = cast("dict[str, object]", usage_raw)
            for key, attribute, positive_only in _USAGE_ATTRIBUTES:
                value = usage.get(key)
                if isinstance(value, int) and (value > 0 or not positive_only):
                    span.set_attribute(attribute, value)
            self._record_token_usage(usage, provider, model)

        self._operation_duration.record(
            time.monotonic() - started,
            {"gen_ai.operation.name": "chat", "gen_ai.provider.name": provider, "gen_ai.response.model": model},
        )
        span.set_status(StatusCode.OK)
        span.end()

    def _record_token_usage(self, usage: dict[str, object], provider: str, model: str) -> None:
        for token_type, key in (("input", "input_tokens"), ("output", "output_tokens")):
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                self._token_usage.record(
                    value,
                    {
                        "gen_ai.operation.name": "chat",
                        "gen_ai.provider.name": provider,
                        "gen_ai.response.model": model,
                        "gen_ai.token.type": token_type,
                    },
                )

    def _on_provider_retry(self, event: Event) -> None:
        self._add_model_event(event, "gen_ai.retry", ("provider", str), ("attempt", int), ("delay_s", float))

    def _on_provider_failover(self, event: Event) -> None:
        self._add_model_event(event, "gen_ai.failover", ("to", str), ("retryable", bool))

    def _add_model_event(self, event: Event, name: str, *fields: tuple[str, type]) -> None:
        """Adds a span event with the listed metadata fields to the active model span."""
        entry = self._model_spans.get(event.run_id)
        if entry is None:
            return
        attrs: dict[str, str | int | float | bool] = {}
        for key, kind in fields:
            value = event.data.get(key)
            if kind is str and value is not None:
                attrs[key] = str(value)
            elif kind is float and isinstance(value, int | float) and not isinstance(value, bool):
                attrs[key] = float(value)
            elif (kind is int and isinstance(value, int) and not isinstance(value, bool)) or (
                kind is bool and isinstance(value, bool)
            ):
                attrs[key] = value
        entry[0].add_event(name, attrs)

    def _on_tool_started(self, event: Event) -> None:
        tool_name = str(event.data.get("tool", "tool"))
        self._tool_spans[(event.run_id, tool_name)] = self._start_tool_span(event, tool_name)

    def _start_tool_span(self, event: Event, tool_name: str) -> Span:
        span = self._start(f"execute_tool {tool_name}", SpanKind.INTERNAL, event)
        span.set_attribute("gen_ai.tool.name", tool_name)
        return span

    def _on_tool_finished(self, event: Event) -> None:
        span = self._tool_spans.pop((event.run_id, str(event.data.get("tool", "tool"))), None)
        if span is None:
            return
        is_error = bool(event.data.get("is_error", False))
        span.set_attribute("harness.tool.is_error", is_error)
        if is_error:
            span.set_status(StatusCode.ERROR, description="Tool execution reported error")
        else:
            span.set_status(StatusCode.OK)
        span.end()

    def _on_tool_blocked(self, event: Event) -> None:
        tool_name = str(event.data.get("tool", "tool"))
        source = str(event.data.get("source", ""))
        span = self._tool_spans.pop((event.run_id, tool_name), None) or self._start_tool_span(event, tool_name)
        span.set_attribute("harness.tool.blocked", True)
        span.set_attribute("harness.tool.block_source", source)
        span.set_status(StatusCode.ERROR, description=f"Tool blocked by {source}")
        span.end()

    def _on_approval_requested(self, event: Event) -> None:
        span = self._run_spans.get(event.run_id)
        if span is not None:
            span.add_event(
                "harness.approval_requested",
                {"tool": str(event.data.get("tool", "")), "risk": str(event.data.get("risk", ""))},
            )

    def _on_input_requested(self, event: Event) -> None:
        span = self._run_spans.get(event.run_id)
        if span is not None:
            span.add_event("harness.input_requested", {"tool": str(event.data.get("tool", ""))})

    def _on_checkpoint_saved(self, event: Event) -> None:
        raw_rev = event.data.get("revision")
        span = self._run_spans.get(event.run_id)
        if span is not None and isinstance(raw_rev, int):
            span.add_event("harness.checkpoint_saved", {"revision": raw_rev})

    def _end_children(self, run_id: str, description: str) -> None:
        """Ends the tool and model spans still open for a run, marked as abandoned."""
        children = [self._tool_spans.pop(key) for key in [k for k in self._tool_spans if k[0] == run_id]]
        model_entry = self._model_spans.pop(run_id, None)
        if model_entry is not None:
            children.append(model_entry[0])
        for child in children:
            child.set_status(StatusCode.ERROR, description=description)
            child.end()

    def _abandon_run(self, run_id: str, description: str) -> None:
        self._end_children(run_id, description)
        run_span = self._run_spans.pop(run_id, None)
        if run_span is not None:
            run_span.set_status(StatusCode.ERROR, description=description)
            run_span.end()

    def _on_run_finished(self, event: Event) -> None:
        self._end_children(event.run_id, "Abandoned on run finish")
        span = self._run_spans.pop(event.run_id, None)
        if span is None:
            return
        status = str(event.data.get("status", "unknown"))
        span.set_attribute("harness.run.status", status)
        if event.data.get("waiting_for"):
            span.set_attribute("harness.run.waiting_for", str(event.data["waiting_for"]))
        if event.data.get("tool"):
            span.set_attribute("harness.run.waiting_tool", str(event.data["tool"]))

        if status == "failed":
            span.set_status(StatusCode.ERROR, description="Run failed")
        else:
            span.set_status(StatusCode.OK)
        span.end()
