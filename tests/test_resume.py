"""Pausing and resuming: clarifying questions, approvals, follow-up questions, crash recovery."""

from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import BaseModel

from ai_agent_harness import (
    AskUser,
    Autonomy,
    BaseTool,
    InMemoryCheckpointStore,
    InputRequest,
    Limits,
    Risk,
    RunState,
    RunStatus,
    restore_state,
    resume,
    start,
)
from ai_agent_harness.core import InvalidStateError
from ai_agent_harness.core.types import Message, ToolCall
from ai_agent_harness.testing import text_response, tool_response
from ai_agent_harness.tools.base import ToolOutput

from .conftest import (
    FIXTURES_DIR,
    Echo,
    PathInput,
    SendMail,
    WriteFile,
    assert_consistent,
    make_agent,
    results_of,
    run_sync,
)


def test_ask_user_pauses_and_answer_becomes_tool_result() -> None:
    agent, provider, _ = make_agent(
        [tool_response(("ask_user", {"question": "Which file?"})), text_response("thanks")], [AskUser()]
    )
    state = run_sync(start(agent, "read the file"))
    assert state.status is RunStatus.WAITING_FOR_INPUT
    assert state.status_reason == "Which file?"
    assert_consistent(state)

    state = run_sync(resume(agent, state, answer="a.txt", pending_id=state.pending_id))
    assert state.status is RunStatus.DONE
    answer = results_of(state)[0]
    assert answer.content == "a.txt" and not answer.is_error
    roles = [m.role for m in provider.requests[1].messages]
    assert roles == ["user", "assistant", "user"]  # no two consecutive user messages
    assert_consistent(state)


def test_after_answer_other_tools_do_not_stop_the_run() -> None:
    """Regression K3: previously the old question stayed pending and halted every subsequent tool call."""
    agent, _, _ = make_agent(
        [
            tool_response(("ask_user", {"question": "Which one?"})),
            tool_response(("echo", {})),
            text_response("done"),
        ],
        [AskUser(), Echo()],
    )
    state = run_sync(start(agent, "go"))
    state = run_sync(resume(agent, state, answer="the first one", pending_id=state.pending_id))
    assert state.status is RunStatus.DONE
    assert state.last_text == "done"


def test_pause_in_the_middle_of_a_batch_keeps_order_and_results() -> None:
    echo = Echo()
    agent, _, _ = make_agent(
        [tool_response(("echo", {}), ("ask_user", {"question": "?"}), ("echo", {})), text_response("ok")],
        [AskUser(), echo],
    )
    state = run_sync(start(agent, "go"))
    assert echo.calls == 1  # only the call before the question
    state = run_sync(resume(agent, state, answer="yes", pending_id=state.pending_id))
    assert echo.calls == 2
    assert [r.content for r in results_of(state)] == ["echo", "yes", "echo"]
    assert_consistent(state)


def test_approval_granted_executes_tool() -> None:
    write = WriteFile()
    agent, _, _ = make_agent(
        [tool_response(("write_file", {"path": "a.py"})), text_response("ok")], [write], autonomy=Autonomy.SUGGEST
    )
    state = run_sync(start(agent, "write"))
    assert state.status is RunStatus.WAITING_FOR_APPROVAL
    assert "write_file" in (state.status_reason or "") and "a.py" in (state.status_reason or "")
    assert write.written == []

    state = run_sync(resume(agent, state, approved=True, pending_id=state.pending_id))
    assert write.written == ["a.py"]
    assert state.status is RunStatus.DONE
    assert_consistent(state)


def test_approval_denied_reports_to_model_with_comment() -> None:
    mail = SendMail()
    agent, _, _ = make_agent(
        [tool_response(("send_mail", {})), text_response("understood")], [mail], autonomy=Autonomy.ACT_WITH_APPROVAL
    )
    state = run_sync(start(agent, "send"))
    state = run_sync(resume(agent, state, approved=False, answer="not to the customer", pending_id=state.pending_id))
    assert mail.sent == 0
    result = results_of(state)[0]
    assert result.is_error and "not to the customer" in result.content
    assert state.status is RunStatus.DONE


def test_follow_up_question_after_done() -> None:
    agent, provider, _ = make_agent([text_response("one"), text_response("two")])
    state = run_sync(start(agent, "question 1"))
    state = run_sync(resume(agent, state, user_input="question 2"))
    assert state.status is RunStatus.DONE
    assert [m.role for m in provider.requests[1].messages] == ["user", "assistant", "user"]


@pytest.mark.parametrize(
    ("kwargs", "status", "match"),
    [
        ({}, RunStatus.WAITING_FOR_INPUT, "'answer' must be a non-empty string"),
        ({"answer": "x", "approved": True}, RunStatus.WAITING_FOR_INPUT, "pass only 'answer'"),
        ({"answer": "x", "user_input": "x"}, RunStatus.WAITING_FOR_INPUT, "pass only 'answer'"),
        ({"answer": "x"}, RunStatus.DONE, "not waiting"),
        ({}, RunStatus.DONE, "Nothing to resume"),
    ],
)
def test_resume_rejects_wrong_arguments(kwargs: dict[str, object], status: RunStatus, match: str) -> None:
    script = [tool_response(("ask_user", {"question": "?"}))] if status.is_waiting else [text_response("ok")]
    agent, _, _ = make_agent(script, [AskUser()])
    state = run_sync(start(agent, "go"))
    assert state.status is status
    kwargs = {**kwargs, "pending_id": state.pending_id}  # correct id, so each case reaches its own check
    with pytest.raises((InvalidStateError, ValueError), match=match):
        run_sync(resume(agent, state, **kwargs))  # type: ignore[arg-type]


def test_running_state_requires_recover_and_reports_unknown_outcome() -> None:
    agent, _, _ = make_agent([text_response("continued")], [Echo()])
    crashed = run_sync(start(make_agent([tool_response(("echo", {}))], [Echo()])[0], "go"))
    # Simulate a crash mid-tool execution: open calls, status running
    crashed.messages = crashed.messages[:2]
    crashed.set_status(RunStatus.RUNNING)
    crashed = crashed.model_copy(update={"agent_name": "test"})

    with pytest.raises(InvalidStateError):
        run_sync(resume(agent, crashed))
    state = run_sync(resume(agent, crashed, recover=True))
    assert state.status is RunStatus.DONE
    assert "unknown" in results_of(state)[0].content
    assert_consistent(state)


def test_failed_run_can_be_resumed_without_new_input() -> None:
    from ai_agent_harness.core import ProviderError

    agent, _, _ = make_agent([ProviderError("down", retryable=False), text_response("back online")])
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.FAILED
    state = run_sync(resume(agent, state))
    assert state.status is RunStatus.DONE


def test_limits_apply_across_resumes() -> None:
    agent, _, _ = make_agent(
        [tool_response(("ask_user", {"question": "?"})), tool_response(("echo", {})), text_response("x")],
        [AskUser(), Echo()],
        limits=Limits(max_steps=2),
    )
    state = run_sync(start(agent, "go"))
    state = run_sync(resume(agent, state, answer="yes", pending_id=state.pending_id))
    assert state.status is RunStatus.LIMIT_REACHED
    assert state.step == 2


def test_resume_with_wrong_agent_is_rejected() -> None:
    agent, _, _ = make_agent([text_response("ok")])
    state = run_sync(start(agent, "go"))
    other = state.model_copy(update={"agent_name": "other"})
    with pytest.raises(InvalidStateError):
        run_sync(resume(agent, other, user_input="x"))


def test_checkpoint_roundtrip_of_waiting_run() -> None:
    store = InMemoryCheckpointStore()
    agent, _, _ = make_agent(
        [tool_response(("ask_user", {"question": "?"})), text_response("ok")], [AskUser()], checkpoints=store
    )
    state = run_sync(start(agent, "go"))
    loaded = run_sync(store.load(state.tenant_id, state.run_id))
    assert loaded is not None and loaded.status is RunStatus.WAITING_FOR_INPUT
    done = run_sync(resume(agent, loaded, answer="yes", pending_id=loaded.pending_id))
    assert done.status is RunStatus.DONE


def test_corrupted_history_is_rejected_on_resume() -> None:
    agent, _, _ = make_agent([text_response("ok")])
    state = run_sync(start(agent, "go"))
    state.messages.append(Message(role="assistant", content=(ToolCall(id="x", name="echo"),)))
    with pytest.raises(Exception, match="role"):
        run_sync(resume(agent, state, user_input="next"))


# ------------------------------------------------------------ Binding resume to the pending call (ADR 0014)

PINNED_V1_PENDING_ID = "b30f1cb883e0238e"  # tests/fixtures/checkpoint_v1_waiting_for_approval.json


def test_stale_approval_never_approves_the_next_action() -> None:
    """Regression: an approval given for call A must not execute call B that paused the run afterward."""
    write = WriteFile()
    store = InMemoryCheckpointStore()
    agent, _, _ = make_agent(
        [
            tool_response(("write_file", {"path": "a.py"})),
            tool_response(("write_file", {"path": "b.py"})),
            text_response("ok"),
        ],
        [write],
        autonomy=Autonomy.SUGGEST,
        checkpoints=store,
    )
    state = run_sync(start(agent, "write", tenant_id="t"))
    shown_to_both = state.pending_id  # both approvers see the dialog for a.py
    assert shown_to_both is not None

    async def click_approve(run_id: str, pending_id: str) -> None:  # UI handler: load + resume
        loaded = await store.load("t", run_id)
        assert loaded is not None
        await resume(agent, loaded, approved=True, pending_id=pending_id)

    run_sync(click_approve(state.run_id, shown_to_both))
    assert write.written == ["a.py"]
    with pytest.raises(InvalidStateError, match="pending_id"):
        run_sync(click_approve(state.run_id, shown_to_both))  # second approver, stale dialog
    assert write.written == ["a.py"]  # b.py was never shown, so it is never executed
    current = run_sync(store.load("t", state.run_id))
    assert current is not None and current.status is RunStatus.WAITING_FOR_APPROVAL


@pytest.mark.parametrize("pending_id", [None, "0000000000000000"])
def test_approval_requires_the_current_pending_id(pending_id: str | None) -> None:
    write = WriteFile()
    agent, _, _ = make_agent([tool_response(("write_file", {"path": "a.py"}))], [write], autonomy=Autonomy.SUGGEST)
    state = run_sync(start(agent, "write"))
    before = state.model_dump()
    with pytest.raises(InvalidStateError, match="pending_id"):
        run_sync(resume(agent, state, approved=True, pending_id=pending_id))
    assert write.written == []
    assert state.model_dump() == before  # rejected before any state is modified


@pytest.mark.parametrize("pending_id", [None, "0000000000000000"])
def test_answer_requires_the_current_pending_id(pending_id: str | None) -> None:
    agent, _, _ = make_agent([tool_response(("ask_user", {"question": "Delete everything?"}))], [AskUser()])
    state = run_sync(start(agent, "go"))
    before = state.model_dump()
    with pytest.raises(InvalidStateError, match="pending_id"):
        run_sync(resume(agent, state, answer="yes", pending_id=pending_id))
    assert state.model_dump() == before


def test_pending_id_is_rejected_when_the_run_is_not_waiting() -> None:
    agent, _, _ = make_agent([text_response("done")])
    state = run_sync(start(agent, "go"))
    assert state.pending_id is None
    with pytest.raises(InvalidStateError, match="pending_id"):
        run_sync(resume(agent, state, user_input="more", pending_id="0000000000000000"))


def test_approval_and_following_question_of_the_same_call_have_different_ids() -> None:
    class AskAfterApproval(BaseTool[PathInput]):
        name: ClassVar[str] = "write_file"
        description: ClassVar[str] = "Asks a question after approval (test stub)."
        input_model: ClassVar[type[BaseModel]] = PathInput
        risk: ClassVar[Risk] = Risk.WRITE

        async def run(self, args: PathInput, state: RunState) -> ToolOutput:
            return InputRequest(question="Which branch?")

    agent, _, _ = make_agent(
        [tool_response(("write_file", {"path": "a.py"})), text_response("ok")],
        [AskAfterApproval()],
        autonomy=Autonomy.SUGGEST,
    )
    state = run_sync(start(agent, "write"))
    approval_id = state.pending_id
    state = run_sync(resume(agent, state, approved=True, pending_id=approval_id))
    assert state.status is RunStatus.WAITING_FOR_INPUT
    assert state.pending_id is not None and state.pending_id != approval_id
    state = run_sync(resume(agent, state, answer="main", pending_id=state.pending_id))
    assert state.status is RunStatus.DONE


def test_stale_answer_never_answers_the_next_question() -> None:
    store = InMemoryCheckpointStore()
    agent, _, _ = make_agent(
        [
            tool_response(("ask_user", {"question": "Which file?"})),
            tool_response(("ask_user", {"question": "Delete it?"})),
            text_response("ok"),
        ],
        [AskUser()],
        checkpoints=store,
    )
    state = run_sync(start(agent, "go"))
    shown = state.pending_id
    state = run_sync(resume(agent, state, answer="a.txt", pending_id=shown))
    assert state.status is RunStatus.WAITING_FOR_INPUT and state.status_reason == "Delete it?"
    loaded = run_sync(store.load(state.tenant_id, state.run_id))
    assert loaded is not None
    with pytest.raises(InvalidStateError, match="pending_id"):
        run_sync(resume(agent, loaded, answer="yes", pending_id=shown))  # "yes" was meant for "Which file?"


def test_reused_run_id_and_other_tenant_get_different_pending_ids() -> None:
    def paused(tenant_id: str) -> RunState:
        agent, _, _ = make_agent([tool_response(("ask_user", {"question": "?"}))], [AskUser()])
        return run_sync(start(agent, "go", tenant_id=tenant_id, run_id="inc-1"))

    first, again, other_tenant = paused("t"), paused("t"), paused("u")
    assert len({first.pending_id, again.pending_id, other_tenant.pending_id}) == 3


def test_pending_id_formula_is_stable_for_persisted_states() -> None:
    """The formula is part of the contract (ADR 0014): changing it invalidates every open dialog."""
    state = restore_state((FIXTURES_DIR / "checkpoint_v1_waiting_for_approval.json").read_text(encoding="utf-8"))
    assert state.pending_id == PINNED_V1_PENDING_ID


def test_pending_id_survives_a_checkpoint_round_trip() -> None:
    agent, _, _ = make_agent([tool_response(("ask_user", {"question": "?"}))], [AskUser()])
    state = run_sync(start(agent, "go"))
    restored = restore_state(state.model_dump_json(), tenant_id=state.tenant_id, run_id=state.run_id)
    assert restored.pending_id == state.pending_id
    assert "pending_id" not in state.model_dump()  # derived, never persisted
