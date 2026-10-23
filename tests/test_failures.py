"""Robustness: every failure path ends with a well-defined status and a consistent history."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import ClassVar

import pytest

from ai_agent_harness import (
    ContextBuilder,
    Decision,
    Event,
    EventType,
    Hook,
    HookPipeline,
    InMemoryCheckpointStore,
    ModelRequest,
    RunState,
    RunStatus,
    start,
)
from ai_agent_harness.core import ProviderError, ToolInvocation
from ai_agent_harness.testing import text_response, tool_response

from .conftest import Echo, assert_consistent, make_agent, no_sleep, results_of, run_sync


class EndRecorder(Hook):
    def __init__(self) -> None:
        self.ended: list[RunStatus] = []

    async def on_run_end(self, state: RunState) -> None:
        self.ended.append(state.status)


def test_provider_failure_ends_with_failed_status_and_full_cleanup() -> None:
    """Regression K1: previously crashed without setting status, emitting event, running hook, or saving checkpoint."""
    recorder = EndRecorder()
    store = InMemoryCheckpointStore()
    agent, _, events = make_agent(
        [ProviderError("API down", retryable=False)], hooks=HookPipeline([recorder]), checkpoints=store
    )
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.FAILED
    assert "API down" in (state.status_reason or "")
    assert recorder.ended == [RunStatus.FAILED]
    assert events[-1].type is EventType.RUN_FINISHED
    saved = run_sync(store.load(state.tenant_id, state.run_id))
    assert saved is not None and saved.status is RunStatus.FAILED


def test_unexpected_adapter_exception_does_not_crash() -> None:
    agent, _, _ = make_agent([RuntimeError("adapter bug")])
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.FAILED
    assert "adapter bug" not in (state.status_reason or "")  # no internal details leaked


def test_crashing_hook_fails_closed() -> None:
    """Regression K6: a broken guardrail must never let anything pass."""

    class Broken(Hook):
        async def before_model(self, state: RunState, request: ModelRequest) -> Decision:
            raise ValueError("hook bug")

    agent, provider, _ = make_agent([text_response("x")], hooks=HookPipeline([Broken()]))
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.BLOCKED
    assert provider.requests == []


def test_crashing_tool_hook_blocks_only_that_call() -> None:
    class Broken(Hook):
        async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
            raise ValueError("bug")

    echo = Echo()
    agent, _, _ = make_agent([tool_response(("echo", {})), text_response("ok")], [echo], hooks=HookPipeline([Broken()]))
    state = run_sync(start(agent, "go"))
    assert echo.calls == 0
    assert results_of(state)[0].is_error
    assert state.status is RunStatus.DONE


def test_invalid_decision_from_hook_fails_closed() -> None:
    class Wrong(Hook):
        async def before_model(self, state: RunState, request: ModelRequest) -> Decision:
            return Decision.retry("retry is not allowed here")

    agent, _, _ = make_agent([text_response("x")], hooks=HookPipeline([Wrong()]))
    assert run_sync(start(agent, "go")).status is RunStatus.BLOCKED


def test_later_block_wins_over_earlier_approval() -> None:
    class Approve(Hook):
        async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
            return Decision.require_approval("review")

    class Block(Hook):
        async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
            return Decision.block("forbidden")

    agent, _, _ = make_agent(
        [tool_response(("echo", {})), text_response("ok")], [Echo()], hooks=HookPipeline([Approve(), Block()])
    )
    state = run_sync(start(agent, "go"))
    assert "forbidden" in results_of(state)[0].content


def test_broken_event_listener_is_logged_not_swallowed_silently(caplog: pytest.LogCaptureFixture) -> None:
    agent, _, _ = make_agent([text_response("ok")])

    async def broken(event: Event) -> None:
        raise RuntimeError("listener broken")

    agent.events.subscribe(broken)
    with caplog.at_level(logging.ERROR, logger="ai_agent_harness.events"):
        state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.DONE
    assert any("Event listener" in r.message for r in caplog.records)


def test_crashing_on_run_end_hook_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    class Broken(Hook):
        async def on_run_end(self, state: RunState) -> None:
            raise RuntimeError("x")

    agent, _, _ = make_agent([text_response("ok")], hooks=HookPipeline([Broken()]))
    with caplog.at_level(logging.ERROR, logger="ai_agent_harness.hooks"):
        assert run_sync(start(agent, "go")).status is RunStatus.DONE
    assert caplog.records


def test_failing_context_source_fails_run_cleanly() -> None:
    class BrokenSource:
        async def render(self, state: RunState) -> str | None:
            raise OSError("RAG unreachable")

    agent, _, _ = make_agent([text_response("x")])
    agent = dataclasses.replace(agent, context=ContextBuilder("s", sources=(BrokenSource(),)))
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.FAILED
    assert state.status_reason == "Internal error (OSError)"


def test_cancellation_ends_with_cancelled_and_consistent_history() -> None:

    from pydantic import BaseModel

    from ai_agent_harness import BaseTool, Risk

    class Empty(BaseModel):
        pass

    class Slow(BaseTool[Empty]):
        name: ClassVar[str] = "slow"
        description: ClassVar[str] = "slow"
        input_model: ClassVar[type[BaseModel]] = Empty
        risk: ClassVar[Risk] = Risk.READ

        async def run(self, args: Empty, state: RunState) -> str:
            await asyncio.sleep(10)
            return "never"

    store = InMemoryCheckpointStore()
    agent, _, _ = make_agent([tool_response(("echo", {}), ("slow", {}))], [Echo(), Slow()], checkpoints=store)

    async def scenario() -> None:
        task = asyncio.create_task(start(agent, "go", run_id="cancelme"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run_sync(scenario())
    state = run_sync(store.load("default", "cancelme"))
    assert state is not None and state.status is RunStatus.CANCELLED
    results = results_of(state)
    assert results[0].content == "echo" and not results[0].is_error  # completed work is preserved
    assert results[1].is_error
    assert_consistent(state)


def test_concurrent_resume_is_detected() -> None:
    from ai_agent_harness import resume
    from ai_agent_harness.core import ConcurrencyError

    store = InMemoryCheckpointStore()
    agent, _, _ = make_agent([text_response("1"), text_response("2"), text_response("3")], checkpoints=store)
    state = run_sync(start(agent, "go"))
    copy_a = run_sync(store.load(state.tenant_id, state.run_id))
    copy_b = run_sync(store.load(state.tenant_id, state.run_id))
    assert copy_a is not None and copy_b is not None
    run_sync(resume(agent, copy_a, user_input="a"))
    with pytest.raises(ConcurrencyError):
        run_sync(resume(agent, copy_b, user_input="b"))


def test_events_never_contain_content() -> None:
    """Approval previews, hook reasons, tool arguments, and model text must never end up in events."""
    from pydantic import BaseModel

    from ai_agent_harness import Autonomy, BaseTool, Risk

    secret = "AT611904300234573201"

    class MailInput(BaseModel):
        to: str
        body: str

    class Mail(BaseTool[MailInput]):
        name: ClassVar[str] = "send_email"
        description: ClassVar[str] = "sends email"
        input_model: ClassVar[type[BaseModel]] = MailInput
        risk: ClassVar[Risk] = Risk.IRREVERSIBLE

        async def run(self, args: MailInput, state: RunState) -> str:
            return "ok"

    class LeakyReason(Hook):
        async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
            body = getattr(invocation.args, "body", "")
            return Decision.block(f"contains {body}") if "block_me" in body else Decision()

    agent, _, events = make_agent(
        [
            tool_response(("send_email", {"to": "x@y.at", "body": f"block_me {secret}"})),
            tool_response(("send_email", {"to": "x@y.at", "body": f"Salary {secret}"})),
        ],
        [Mail()],
        autonomy=Autonomy.AUTONOMOUS,
        hooks=HookPipeline([LeakyReason()]),
    )
    state = run_sync(start(agent, f"please use {secret}"))
    assert state.status is RunStatus.WAITING_FOR_APPROVAL
    assert secret in (state.status_reason or "")  # preview stays in RunState ...
    dumped = "".join(e.model_dump_json() for e in events)
    assert secret not in dumped  # ... but never in events
    assert events[-1].data == {"status": "waiting_for_approval", "waiting_for": "approval", "tool": "send_email"}


def test_event_data_is_restricted_to_simple_values() -> None:
    from pydantic import ValidationError

    long_text = Event(type=EventType.RUN_STARTED, run_id="r", tenant_id="t", step=0, data={"x": "a" * 1000})
    assert len(long_text.data["x"]) <= 201
    with pytest.raises(ValidationError):
        Event(type=EventType.RUN_STARTED, run_id="r", tenant_id="t", step=0, data={"x": ["list", "with", "content"]})


def test_router_events_contain_no_error_text() -> None:
    """Provider error messages often echo the input — they must not appear in retry/failover events."""
    from ai_agent_harness import ModelRouter, RetryPolicy
    from ai_agent_harness.testing import FakeProvider

    secret = "AT611904300234573201"
    primary = FakeProvider([ProviderError(f"429 too many requests: {secret}", retryable=True)] * 2, id="primary")
    fallback = FakeProvider([text_response("ok")], id="backup")
    agent, _, events = make_agent([])
    agent = dataclasses.replace(
        agent,
        router=ModelRouter(
            primary, [fallback], RetryPolicy(max_attempts=2, base_delay_s=0.0, max_delay_s=0.0), sleep=no_sleep
        ),
    )
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.DONE
    types = [e.type for e in events]
    assert EventType.PROVIDER_RETRY in types
    assert EventType.PROVIDER_FAILOVER in types
    assert secret not in "".join(e.model_dump_json() for e in events)


def test_invalid_event_data_never_breaks_a_run(caplog: pytest.LogCaptureFixture) -> None:
    """Events are for observability: an invalid value (e.g. an exotic adapter ID) is dropped and logged."""
    from typing import Any

    from ai_agent_harness import ModelRouter
    from ai_agent_harness.testing import FakeProvider

    odd_id: Any = ("not", "a_string")  # hashable, but not an allowed event value
    agent, _, _ = make_agent([])
    agent = dataclasses.replace(agent, router=ModelRouter(FakeProvider([text_response("ok")], id=odd_id)))
    with caplog.at_level(logging.ERROR):
        state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.DONE
    assert "Event" in caplog.text


def test_dropped_event_data_is_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Dropped event data is the exact data that might contain sensitive content — it must not be logged."""
    from typing import Any

    from ai_agent_harness import ModelRouter
    from ai_agent_harness.testing import FakeProvider

    secret = "secret42"  # short so Pydantic's truncated repr would not accidentally hide it
    odd_id: Any = (secret,)
    agent, _, _ = make_agent([])
    agent = dataclasses.replace(agent, router=ModelRouter(FakeProvider([text_response("ok")], id=odd_id)))
    with caplog.at_level(logging.DEBUG):
        assert run_sync(start(agent, "go")).status is RunStatus.DONE
    assert "dropped" in caplog.text
    assert secret not in caplog.text
