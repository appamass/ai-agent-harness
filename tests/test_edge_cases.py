"""Rare but production-critical paths: mid-run checkpoints, crash recovery, chained pauses."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel

from ai_agent_harness import (
    AskUser,
    Autonomy,
    BaseTool,
    Decision,
    EventType,
    Hook,
    HookPipeline,
    InMemoryCheckpointStore,
    InputRequest,
    JsonFileCheckpointStore,
    ModelRequest,
    Risk,
    RunState,
    RunStatus,
    resume,
    start,
)
from ai_agent_harness.core import InvalidStateError
from ai_agent_harness.testing import text_response, tool_response

from .conftest import Crash, Echo, EmptyInput, WriteFile, assert_consistent, make_agent, results_of, run_sync


def test_checkpoint_is_saved_after_every_step() -> None:
    store = InMemoryCheckpointStore()
    agent, _, events = make_agent(
        [tool_response(("echo", {})), tool_response(("echo", {})), text_response("ok")], [Echo()], checkpoints=store
    )
    state = run_sync(start(agent, "go"))
    saved = [e for e in events if e.type is EventType.CHECKPOINT_SAVED]
    # Two checkpoints per tool step: before execution (write-ahead) and after.
    assert [e.data["revision"] for e in saved] == [2, 3, 4, 5]
    assert state.revision == 6  # start + 2 × (before, after) + finish


def test_recover_from_file_checkpoint_written_mid_run(tmp_path: Path) -> None:
    """Crash after an intermediate checkpoint: load, recover, finish — with a consistent history."""
    store = JsonFileCheckpointStore(tmp_path)

    class CrashAfterCheckpoint(Hook):
        async def before_model(self, state: RunState, request: ModelRequest) -> Decision:
            if state.step == 1:
                raise SystemExit  # simulates hard process termination (bypasses normal exception handlers)
            return Decision()

    agent, _, _ = make_agent(
        [tool_response(("echo", {}))], [Echo()], checkpoints=store, hooks=HookPipeline([CrashAfterCheckpoint()])
    )
    with pytest.raises(SystemExit):
        run_sync(start(agent, "go", run_id="crash1"))

    crashed = run_sync(store.load("default", "crash1"))
    assert crashed is not None and crashed.status is RunStatus.RUNNING
    survivor, _, _ = make_agent([text_response("done")], [Echo()], checkpoints=store)
    state = run_sync(resume(survivor, crashed, recover=True))
    assert state.status is RunStatus.DONE
    assert_consistent(state)


def test_approval_then_tool_asks_for_input() -> None:
    class Confirming(BaseTool[EmptyInput]):
        name: ClassVar[str] = "confirming"
        description: ClassVar[str] = "writes after asking a clarifying question"
        input_model: ClassVar[type[BaseModel]] = EmptyInput
        risk: ClassVar[Risk] = Risk.WRITE

        async def run(self, args: EmptyInput, state: RunState) -> str | InputRequest:
            return InputRequest(question="Really overwrite?")

    agent, _, _ = make_agent(
        [tool_response(("confirming", {}), ("echo", {})), text_response("ok")],
        [Confirming(), Echo()],
        autonomy=Autonomy.SUGGEST,
    )
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.WAITING_FOR_APPROVAL
    state = run_sync(resume(agent, state, approved=True, pending_id=state.pending_id))
    assert state.status is RunStatus.WAITING_FOR_INPUT
    assert state.status_reason == "Really overwrite?"
    assert_consistent(state)
    state = run_sync(resume(agent, state, answer="yes", pending_id=state.pending_id))
    assert state.status is RunStatus.DONE
    assert [r.content for r in results_of(state)] == ["yes", "echo"]


def test_second_pause_in_remaining_batch() -> None:
    agent, _, _ = make_agent(
        [tool_response(("ask_user", {"question": "1?"}), ("ask_user", {"question": "2?"})), text_response("ok")],
        [AskUser()],
    )
    state = run_sync(start(agent, "go"))
    state = run_sync(resume(agent, state, answer="one", pending_id=state.pending_id))
    assert state.status is RunStatus.WAITING_FOR_INPUT and state.status_reason == "2?"
    state = run_sync(resume(agent, state, answer="two", pending_id=state.pending_id))
    assert [r.content for r in results_of(state)] == ["one", "two"]
    assert state.status is RunStatus.DONE


class BlockWord(Hook):
    async def on_user_input(self, state: RunState, text: str) -> Decision:
        return Decision.block("forbidden") if "evil" in text else Decision()


def test_blocked_answer_ends_run_consistently() -> None:
    agent, _, _ = make_agent(
        [tool_response(("ask_user", {"question": "?"}))], [AskUser()], hooks=HookPipeline([BlockWord()])
    )
    state = run_sync(start(agent, "go"))
    state = run_sync(resume(agent, state, answer="evil answer", pending_id=state.pending_id))
    assert state.status is RunStatus.BLOCKED
    assert "evil" not in "".join(r.content for r in results_of(state))
    assert_consistent(state)


def test_blocked_follow_up_input() -> None:
    agent, provider, _ = make_agent([text_response("ok")], hooks=HookPipeline([BlockWord()]))
    state = run_sync(start(agent, "go"))
    state = run_sync(resume(agent, state, user_input="evil"))
    assert state.status is RunStatus.BLOCKED
    assert len(provider.requests) == 1


def test_denial_comment_is_also_checked_by_input_guardrails() -> None:
    agent, _, _ = make_agent(
        [tool_response(("write_file", {"path": "a"})), text_response("ok")],
        [WriteFile()],
        autonomy=Autonomy.SUGGEST,
        hooks=HookPipeline([BlockWord()]),
    )
    state = run_sync(start(agent, "go"))
    state = run_sync(resume(agent, state, approved=False, answer="evil", pending_id=state.pending_id))
    assert results_of(state)[0].content == "Approval was denied."


def test_failing_checkpoint_store_is_reported_loudly() -> None:
    class BrokenStore(InMemoryCheckpointStore):
        async def save(self, state: RunState) -> None:
            raise OSError("disk full")

    agent, _, _ = make_agent([text_response("ok")], checkpoints=BrokenStore())
    with pytest.raises(OSError, match="disk full"):
        run_sync(start(agent, "go"))


def test_max_tokens_with_calls_respects_retry_limit() -> None:
    from ai_agent_harness import Limits
    from ai_agent_harness.core import StopReason

    truncated = tool_response(("echo", {}), stop=StopReason.MAX_TOKENS)
    agent, _, _ = make_agent([truncated] * 5, [Echo()], limits=Limits(max_consecutive_retries=1))
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.FAILED
    assert_consistent(state)


def test_before_model_block_mid_run() -> None:
    class StopAfterFirst(Hook):
        async def before_model(self, state: RunState, request: ModelRequest) -> Decision:
            return Decision.block("budget hook") if state.step >= 1 else Decision()

    agent, _, _ = make_agent([tool_response(("echo", {}))], [Echo()], hooks=HookPipeline([StopAfterFirst()]))
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.BLOCKED
    assert_consistent(state)


@pytest.mark.parametrize(
    ("status", "kwargs", "match"),
    [
        (RunStatus.RUNNING, {"recover": True, "user_input": "x"}, "does not allow additional arguments"),
        (RunStatus.DONE, {"recover": True}, "only allowed when status is 'running'"),
        (RunStatus.WAITING_FOR_INPUT, {"answer": "x", "approved": True}, "pass only 'answer'"),
        (RunStatus.WAITING_FOR_APPROVAL, {}, "approved=True/False"),
        (RunStatus.WAITING_FOR_APPROVAL, {"approved": True, "user_input": "x"}, "do not pass 'user_input'"),
    ],
)
def test_resume_argument_validation(status: RunStatus, kwargs: dict[str, object], match: str) -> None:
    if status is RunStatus.WAITING_FOR_APPROVAL:
        script = [tool_response(("write_file", {"path": "a"}))]
        agent, _, _ = make_agent(script, [WriteFile()], autonomy=Autonomy.SUGGEST)
    elif status is RunStatus.WAITING_FOR_INPUT:
        agent, _, _ = make_agent([tool_response(("ask_user", {"question": "?"}))], [AskUser()])
    else:
        agent, _, _ = make_agent([text_response("ok")])
    state = run_sync(start(agent, "go"))
    if status is RunStatus.RUNNING:
        state.set_status(RunStatus.RUNNING)
    assert state.status is status
    kwargs = {**kwargs, "pending_id": state.pending_id}  # correct id, so each case reaches its own check
    with pytest.raises(InvalidStateError, match=match):
        run_sync(resume(agent, state, **kwargs))  # type: ignore[arg-type]


def test_concurrent_resume_of_approval_never_executes_twice() -> None:
    from ai_agent_harness.core import ConcurrencyError

    store = InMemoryCheckpointStore()
    write = WriteFile()
    agent, _, _ = make_agent(
        [tool_response(("write_file", {"path": "a"})), text_response("ok"), text_response("ok")],
        [write],
        autonomy=Autonomy.SUGGEST,
        checkpoints=store,
    )
    state = run_sync(start(agent, "go"))
    worker_a = run_sync(store.load(state.tenant_id, state.run_id))
    worker_b = run_sync(store.load(state.tenant_id, state.run_id))
    assert worker_a is not None
    assert worker_b is not None
    run_sync(resume(agent, worker_a, approved=True, pending_id=worker_a.pending_id))
    with pytest.raises(ConcurrencyError):
        run_sync(resume(agent, worker_b, approved=True, pending_id=worker_b.pending_id))
    assert write.written == ["a"]  # executed exactly once


def test_crash_during_tool_execution_is_reported_as_unknown_after_recover(tmp_path: Path) -> None:
    """Write-ahead: the tool call is persisted before execution so a mid-tool crash is
    reported as 'outcome unknown' on recovery — instead of being silently re-executed."""

    class CrashingSend(BaseTool[EmptyInput]):
        name: ClassVar[str] = "send"
        description: ClassVar[str] = "side effect, then process crashes"
        input_model: ClassVar[type[BaseModel]] = EmptyInput
        risk: ClassVar[Risk] = Risk.WRITE
        executed = 0

        async def run(self, args: EmptyInput, state: RunState) -> str:
            CrashingSend.executed += 1
            raise SystemExit  # process dies after the side effect

    store = JsonFileCheckpointStore(tmp_path)
    agent, _, _ = make_agent(
        [tool_response(("send", {}))], [CrashingSend()], autonomy=Autonomy.AUTONOMOUS, checkpoints=store
    )
    with pytest.raises(SystemExit):
        run_sync(start(agent, "go", run_id="wal1"))

    crashed = run_sync(store.load("default", "wal1"))
    assert crashed is not None
    assert [c.name for c in crashed.open_tool_calls()] == ["send"]  # call was saved before execution

    survivor, provider, _ = make_agent([text_response("checked")], [CrashingSend()], checkpoints=store)
    state = run_sync(resume(survivor, crashed, recover=True))
    assert state.status is RunStatus.DONE
    assert CrashingSend.executed == 1  # not automatically re-executed
    assert "unknown" in provider.requests[0].messages[-1].tool_results[0].content


class CrashAfterSideEffect(BaseTool[EmptyInput]):
    name: ClassVar[str] = "crash"
    description: ClassVar[str] = "side effect, then process dies"
    input_model: ClassVar[type[BaseModel]] = EmptyInput
    risk: ClassVar[Risk] = Risk.WRITE

    async def run(self, args: EmptyInput, state: RunState) -> str:
        raise Crash


@pytest.mark.parametrize("autonomy", [Autonomy.AUTONOMOUS, Autonomy.SUGGEST], ids=["batch", "after-approval"])
def test_finished_results_survive_a_crash_later_in_the_batch(tmp_path: Path, autonomy: Autonomy) -> None:
    """If tool 2 in a batch dies, tool 1 keeps its real result — even when tool 2 runs only after an
    approval. Otherwise the model would think a completed action had an unknown outcome and might repeat it."""
    store = JsonFileCheckpointStore(tmp_path)
    agent, _, _ = make_agent(
        [tool_response(("echo", {}), ("crash", {}))],
        [Echo(), CrashAfterSideEffect()],
        autonomy=autonomy,
        checkpoints=store,
    )

    async def run_until_crash() -> None:
        state = await start(agent, "go", run_id="r")
        assert state.status is RunStatus.WAITING_FOR_APPROVAL  # only on suggest; autonomous crashes earlier
        assert state.pending is not None
        assert [r.content for r in state.pending.done] == ["echo"]
        await resume(agent, state, approved=True, pending_id=state.pending_id)

    with pytest.raises(Crash):
        run_sync(run_until_crash())

    crashed = run_sync(store.load("default", "r"))
    assert crashed is not None
    assert [r.content for r in crashed.batch_done] == ["echo"]  # preserved in the checkpoint

    survivor, provider, _ = make_agent([text_response("checked")], [Echo(), CrashAfterSideEffect()], checkpoints=store)
    done = run_sync(resume(survivor, crashed, recover=True))
    assert done.status is RunStatus.DONE
    first, second = provider.requests[0].messages[-1].tool_results
    assert first.content == "echo"  # real result
    assert "unknown" in second.content  # only the genuinely interrupted tool
    assert_consistent(done)


def test_cancel_during_tool_run_marks_started_call_as_unknown_and_later_calls_as_not_executed() -> None:
    """If CancelledError hits mid-`tool.run()`, the already started tool must not be marked 'Not executed'
    (its side effect may have already happened) — whereas subsequent calls in the batch must be."""
    import asyncio

    entered = asyncio.Event()

    class SlowWrite(BaseTool[EmptyInput]):
        name: ClassVar[str] = "slow_write"
        description: ClassVar[str] = "starts side effect and waits"
        input_model: ClassVar[type[BaseModel]] = EmptyInput
        risk: ClassVar[Risk] = Risk.WRITE
        side_effects = 0

        async def run(self, args: EmptyInput, state: RunState) -> str:
            SlowWrite.side_effects += 1
            entered.set()
            await asyncio.Future()  # waits until task.cancel() arrives
            return "never"

    store = InMemoryCheckpointStore()
    agent, _, _ = make_agent(
        [tool_response(("echo", {}), ("slow_write", {}), ("echo", {}))],
        [Echo(), SlowWrite()],
        autonomy=Autonomy.AUTONOMOUS,
        checkpoints=store,
    )

    async def main() -> None:
        task = asyncio.create_task(start(agent, "go", run_id="cancel-mid-tool"))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run_sync(main())
    saved = run_sync(store.load("default", "cancel-mid-tool"))
    assert saved is not None and saved.status is RunStatus.CANCELLED
    assert SlowWrite.side_effects == 1
    first, second, third = results_of(saved)
    assert first.content == "echo" and not first.is_error
    assert second.is_error and "unknown" in second.content and "Not executed" not in second.content
    assert third.is_error and "Not executed" in third.content
    assert_consistent(saved)


def test_cancel_during_tool_finished_event_preserves_completed_tool_result() -> None:
    """Once tool.run() and after_tool finish, cancelling inside an event listener must keep the result."""
    import asyncio

    from ai_agent_harness import Event

    store = InMemoryCheckpointStore()
    agent, _, _ = make_agent(
        [tool_response(("echo", {}), ("echo", {}))],
        [Echo()],
        checkpoints=store,
    )
    task_ref: dict[str, asyncio.Task[RunState]] = {}

    async def cancel_on_finish(event: Event) -> None:
        if event.type is EventType.TOOL_FINISHED:
            task_ref["task"].cancel()
            await asyncio.sleep(0)  # allow cancellation to take effect

    agent.events.subscribe(cancel_on_finish)

    async def main() -> None:
        task_ref["task"] = asyncio.create_task(start(agent, "go", run_id="cancel-after-tool"))
        with pytest.raises(asyncio.CancelledError):
            await task_ref["task"]

    run_sync(main())
    saved = run_sync(store.load("default", "cancel-after-tool"))
    assert saved is not None and saved.status is RunStatus.CANCELLED
    first, second = results_of(saved)
    assert first.content == "echo" and not first.is_error
    assert second.is_error and "Not executed" in second.content
    assert_consistent(saved)


def test_cancel_during_after_tool_hook_withholds_unverified_result_fail_closed() -> None:
    """If `after_tool` is cancelled, neither the unverified result nor 'Not executed' may be recorded."""
    import asyncio

    from ai_agent_harness.core import ToolInvocation, ToolResult

    in_hook = asyncio.Event()

    class SecretTool(BaseTool[EmptyInput]):
        name: ClassVar[str] = "secret_tool"
        description: ClassVar[str] = "returns a sensitive result"
        input_model: ClassVar[type[BaseModel]] = EmptyInput
        risk: ClassVar[Risk] = Risk.READ

        async def run(self, args: EmptyInput, state: RunState) -> str:
            return "secret-token-42"

    class SlowAfterToolGuard(Hook):
        async def after_tool(self, state: RunState, invocation: ToolInvocation, result: ToolResult) -> Decision:
            in_hook.set()
            await asyncio.Future()
            return Decision.block("contains secret")

    store = InMemoryCheckpointStore()
    agent, _, _ = make_agent(
        [tool_response(("secret_tool", {}))],
        [SecretTool()],
        hooks=HookPipeline([SlowAfterToolGuard()]),
        checkpoints=store,
    )

    async def main() -> None:
        task = asyncio.create_task(start(agent, "go", run_id="cancel-in-after-tool"))
        await in_hook.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run_sync(main())
    saved = run_sync(store.load("default", "cancel-in-after-tool"))
    assert saved is not None and saved.status is RunStatus.CANCELLED
    (result,) = results_of(saved)
    assert result.is_error
    assert "secret-token-42" not in result.content
    assert "Not executed" not in result.content
    assert "withheld" in result.content
    assert_consistent(saved)
