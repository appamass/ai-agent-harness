"""`cancel` (ADR 0031): ends a waiting or crashed run as cancelled without running the model or any tool."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import pytest
from pydantic import BaseModel

from ai_agent_harness import (
    AskUser,
    Autonomy,
    BaseTool,
    Decision,
    Hook,
    HookPipeline,
    InMemoryCheckpointStore,
    Risk,
    RunState,
    RunStatus,
    ToolInvocation,
    cancel,
    resume,
    start,
)
from ai_agent_harness.core import ConcurrencyError, InvalidStateError
from ai_agent_harness.testing import text_response, tool_response

from .conftest import Echo, EmptyInput, WriteFile, make_agent, results_of, run_sync

NOT_EXECUTED = "Not executed: run was cancelled."


def test_cancelling_a_run_that_waits_for_approval_keeps_finished_results_and_runs_nothing() -> None:
    store = InMemoryCheckpointStore()
    echo, write = Echo(), WriteFile()
    agent, provider, events = make_agent(
        [tool_response(("echo", {}), ("write_file", {"path": "a.txt"}), ("echo", {}))],
        [echo, write],
        autonomy=Autonomy.SUGGEST,
        checkpoints=store,
    )
    waiting = run_sync(start(agent, "Go", run_id="r1"))
    assert waiting.status is RunStatus.WAITING_FOR_APPROVAL

    state = run_sync(cancel(agent, waiting, reason="Nobody is there to approve"))

    assert state.status is RunStatus.CANCELLED and state.status_reason == "Nobody is there to approve"
    assert [(r.content, r.is_error) for r in results_of(state)] == [
        ("echo", False),
        (NOT_EXECUTED, True),
        (NOT_EXECUTED, True),
    ]
    state.check_protocol()
    assert state.pending is None and state.pending_id is None
    assert (echo.calls, write.written, len(provider.requests)) == (1, [], 1)
    stored = run_sync(store.load("default", "r1"))
    assert stored is not None and stored.status is RunStatus.CANCELLED
    assert events[-1].data == {"status": "cancelled", "waiting_for": None, "tool": None}


def test_cancelling_a_run_that_waits_for_input() -> None:
    agent, _, _ = make_agent([tool_response(("ask_user", {"question": "Which file?"}))], [AskUser()])
    waiting = run_sync(start(agent, "Go"))
    assert waiting.status is RunStatus.WAITING_FOR_INPUT

    state = run_sync(cancel(agent, waiting))

    assert state.status is RunStatus.CANCELLED and state.status_reason == "Run was cancelled"
    assert [r.content for r in results_of(state)] == [NOT_EXECUTED]


def test_a_crashed_run_needs_recover_and_its_interrupted_call_is_reported_as_unknown() -> None:
    class CrashingSend(BaseTool[EmptyInput]):
        name: ClassVar[str] = "send"
        description: ClassVar[str] = "Side effect, then the process dies."
        input_model: ClassVar[type[BaseModel]] = EmptyInput
        risk: ClassVar[Risk] = Risk.WRITE

        async def run(self, args: EmptyInput, state: RunState) -> str:
            raise SystemExit

    store = InMemoryCheckpointStore()
    agent, _, _ = make_agent(
        [tool_response(("echo", {}), ("send", {}))],
        [Echo(), CrashingSend()],
        autonomy=Autonomy.AUTONOMOUS,
        checkpoints=store,
    )
    with pytest.raises(SystemExit):
        run_sync(start(agent, "Go", run_id="crashed"))
    crashed = run_sync(store.load("default", "crashed"))
    assert crashed is not None and crashed.status is RunStatus.RUNNING

    with pytest.raises(InvalidStateError, match="recover=True"):
        run_sync(cancel(agent, crashed))
    state = run_sync(cancel(agent, crashed, recover=True))

    assert state.status is RunStatus.CANCELLED
    echo_result, send_result = results_of(state)
    assert echo_result.content == "echo" and not echo_result.is_error
    assert send_result.is_error and "unknown" in send_result.content
    state.check_protocol()


def test_finished_runs_cannot_be_cancelled_and_recover_needs_a_running_run() -> None:
    agent, _, _ = make_agent([text_response("done"), tool_response(("ask_user", {"question": "?"}))], [AskUser()])
    done = run_sync(start(agent, "Go"))
    with pytest.raises(InvalidStateError, match="done"):
        run_sync(cancel(agent, done))

    waiting = run_sync(start(agent, "Ask"))
    with pytest.raises(InvalidStateError, match="recover=True"):
        run_sync(cancel(agent, waiting, recover=True))


def test_a_run_of_another_agent_is_rejected() -> None:
    agent, _, _ = make_agent([tool_response(("ask_user", {"question": "?"}))], [AskUser()])
    waiting = run_sync(start(agent, "Go"))
    other = dataclasses.replace(agent, name="other")
    with pytest.raises(InvalidStateError, match="belongs to agent"):
        run_sync(cancel(other, waiting))


def test_two_workers_cannot_both_cancel_or_resume_the_same_waiting_run() -> None:
    store = InMemoryCheckpointStore()
    agent, _, _ = make_agent(
        [tool_response(("write_file", {"path": "a.txt"}))], [WriteFile()], autonomy=Autonomy.SUGGEST, checkpoints=store
    )
    run_sync(start(agent, "Go", run_id="r1"))
    first = run_sync(store.load("default", "r1"))
    second = run_sync(store.load("default", "r1"))
    assert first is not None and second is not None

    run_sync(cancel(agent, first))
    with pytest.raises(ConcurrencyError):
        run_sync(resume(agent, second, approved=True, pending_id=second.pending_id))
    stored = run_sync(store.load("default", "r1"))
    assert stored is not None and stored.status is RunStatus.CANCELLED


def test_on_run_end_runs_and_a_cancelled_run_can_continue_with_new_input() -> None:
    ended: list[RunStatus] = []

    class Record(Hook):
        async def on_run_end(self, state: RunState) -> None:
            ended.append(state.status)

    agent, _, _ = make_agent(
        [tool_response(("ask_user", {"question": "?"})), text_response("Starting over.")],
        [AskUser()],
        hooks=HookPipeline([Record()]),
    )
    cancelled = run_sync(cancel(agent, run_sync(start(agent, "Go"))))
    assert ended == [RunStatus.CANCELLED]

    state = run_sync(resume(agent, cancelled, user_input="Let us start over"))
    assert state.status is RunStatus.DONE and state.last_text == "Starting over."


def test_cancel_events_carry_metadata_only() -> None:
    secret = "AT611904300234573201"
    agent, _, events = make_agent([tool_response(("ask_user", {"question": f"IBAN {secret}?"}))], [AskUser()])
    run_sync(cancel(agent, run_sync(start(agent, "Go")), reason=f"Gave up on {secret}"))

    assert not [e for e in events if secret in e.model_dump_json()]


def _parallel_read(tool_name: str) -> BaseTool[EmptyInput]:
    class Read(BaseTool[EmptyInput]):
        name: ClassVar[str] = tool_name
        description: ClassVar[str] = "A parallel-safe read."
        input_model: ClassVar[type[BaseModel]] = EmptyInput
        risk: ClassVar[Risk] = Risk.READ
        parallel_safe: ClassVar[bool] = True

        async def run(self, args: EmptyInput, state: RunState) -> str:
            return f"{tool_name} done"

    return Read()


def test_cancelling_a_paused_parallel_group_keeps_every_finished_result() -> None:
    class ApproveB(Hook):
        async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
            return Decision.require_approval("b needs a human") if invocation.tool_name == "b" else Decision()

    agent, provider, _ = make_agent(
        [tool_response(("a", {}), ("b", {}), ("c", {}))],
        [_parallel_read("a"), _parallel_read("b"), _parallel_read("c")],
        hooks=HookPipeline([ApproveB()]),
        max_parallel_tools=3,
    )
    waiting = run_sync(start(agent, "Go"))
    assert waiting.status is RunStatus.WAITING_FOR_APPROVAL
    done_before = {r.call_id: r.content for r in waiting.pending.done} if waiting.pending else {}
    assert sorted(done_before.values()) == ["a done", "c done"]  # the group ran around the waiting call

    state = run_sync(cancel(agent, waiting))

    by_call = {c.id: c.name for m in state.messages for c in m.tool_calls}
    results = {by_call[r.call_id]: r.content for r in results_of(state)}
    assert results["b"] == NOT_EXECUTED
    assert all(results[by_call[call_id]] == content for call_id, content in done_before.items())
    assert len(provider.requests) == 1
    state.check_protocol()


def test_cancelling_a_run_with_an_output_model_leaves_no_output_and_keeps_the_binding() -> None:
    class Answer(BaseModel):
        text: str

    agent, _, _ = make_agent([tool_response(("ask_user", {"question": "?"}))], [AskUser()])
    waiting = run_sync(start(agent, "Go", output_model=Answer))
    state = run_sync(cancel(agent, waiting))

    assert state.status is RunStatus.CANCELLED and state.output is None
    with pytest.raises(InvalidStateError, match="output_model"):
        run_sync(resume(agent, state, user_input="again"))


def test_a_cancel_after_another_worker_claimed_the_run_fails_and_changes_nothing() -> None:
    store = InMemoryCheckpointStore()
    write = WriteFile()
    agent, _, _ = make_agent(
        [tool_response(("write_file", {"path": "a.txt"})), text_response("written")],
        [write],
        autonomy=Autonomy.SUGGEST,
        checkpoints=store,
    )
    run_sync(start(agent, "Go", run_id="r1"))
    first = run_sync(store.load("default", "r1"))
    second = run_sync(store.load("default", "r1"))
    assert first is not None and second is not None

    run_sync(resume(agent, first, approved=True, pending_id=first.pending_id))
    with pytest.raises(ConcurrencyError):
        run_sync(cancel(agent, second))
    stored = run_sync(store.load("default", "r1"))
    assert stored is not None and stored.status is RunStatus.DONE and write.written == ["a.txt"]


@pytest.mark.parametrize("reason", ["", "   ", "x" * 50_001])
def test_the_reason_must_be_short_text(reason: str) -> None:
    agent, _, _ = make_agent([tool_response(("ask_user", {"question": "?"}))], [AskUser()])
    waiting = run_sync(start(agent, "Go"))
    with pytest.raises(ValueError, match="reason"):
        run_sync(cancel(agent, waiting, reason=reason))
    assert waiting.status is RunStatus.WAITING_FOR_INPUT


def test_a_run_that_crashed_before_any_tool_is_cancelled_without_open_calls() -> None:
    store = InMemoryCheckpointStore()
    agent, _, _ = make_agent([SystemExit()], checkpoints=store)
    with pytest.raises(SystemExit):
        run_sync(start(agent, "Go", run_id="early"))
    crashed = run_sync(store.load("default", "early"))
    assert crashed is not None and crashed.status is RunStatus.RUNNING and not crashed.open_tool_calls()

    state = run_sync(cancel(agent, crashed, recover=True))
    assert state.status is RunStatus.CANCELLED and results_of(state) == []
    state.check_protocol()
