from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_agent_harness import Decision, Message, ModelResponse, RunState, RunStatus, StopReason
from ai_agent_harness.core import PendingBatch, PendingCall, ProtocolError
from ai_agent_harness.core.types import TextBlock, ThinkingBlock, ToolCall, ToolResult, Usage


def assistant_with_call(call_id: str = "c1") -> Message:
    return Message(role="assistant", content=(ToolCall(id=call_id, name="echo"),))


class TestMessage:
    def test_tool_call_not_allowed_in_user_message(self) -> None:
        with pytest.raises(ValidationError):
            Message(role="user", content=(ToolCall(id="c1", name="x"),))

    def test_tool_result_not_allowed_in_assistant_message(self) -> None:
        with pytest.raises(ValidationError):
            Message(role="assistant", content=(ToolResult(call_id="c1", content="x"),))

    def test_thinking_only_in_assistant(self) -> None:
        with pytest.raises(ValidationError):
            Message(role="user", content=(ThinkingBlock(text="hm"),))

    def test_duplicate_call_ids_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Message(role="assistant", content=(ToolCall(id="c1", name="a"), ToolCall(id="c1", name="b")))

    def test_empty_content_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Message(role="user", content=())

    def test_unknown_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TextBlock.model_validate({"type": "text", "text": "x", "typo": 1})

    def test_response_must_be_assistant(self) -> None:
        with pytest.raises(ValidationError):
            ModelResponse(message=Message.user("x"), stop_reason=StopReason.END)

    def test_thinking_provider_data_roundtrip(self) -> None:
        msg = Message(role="assistant", content=(ThinkingBlock(text="t", provider_data={"signature": "abc"}),))
        assert Message.model_validate_json(msg.model_dump_json()) == msg

    def test_usage_addition(self) -> None:
        assert (Usage(input_tokens=1, output_tokens=2) + Usage(input_tokens=3, output_tokens=4)).total == 10


class TestDecision:
    def test_non_continue_requires_reason(self) -> None:
        with pytest.raises(ValidationError):
            Decision.model_validate({"action": "block"})
        with pytest.raises(ValidationError):
            Decision.block("  ")


class TestRunStateProtocol:
    def test_assistant_must_follow_user(self) -> None:
        state = RunState(agent_name="a")
        with pytest.raises(ProtocolError):
            state.add_assistant(Message(role="assistant", content=(TextBlock(text="x"),)))

    def test_user_text_with_open_calls_rejected(self) -> None:
        state = RunState(agent_name="a")
        state.add_user_text("hi")
        state.add_assistant(assistant_with_call())
        with pytest.raises(ProtocolError):
            state.add_user_text("one more thing")

    def test_results_must_match_open_calls_exactly(self) -> None:
        state = RunState(agent_name="a")
        state.add_user_text("hi")
        state.add_assistant(assistant_with_call("c1"))
        with pytest.raises(ProtocolError):
            state.add_tool_results([ToolResult(call_id="other", content="x")])
        with pytest.raises(ProtocolError):
            state.add_tool_results([])

    def test_consecutive_user_text_is_merged(self) -> None:
        state = RunState(agent_name="a")
        state.add_user_text("one")
        state.add_user_text("two")
        assert len(state.messages) == 1
        assert state.messages[0].text == "one\ntwo"  # never glued together without separator (Finding 2.2)

    def test_close_open_calls_keeps_known_results(self) -> None:
        state = RunState(agent_name="a")
        state.add_user_text("hi")
        state.add_assistant(Message(role="assistant", content=(ToolCall(id="a", name="x"), ToolCall(id="b", name="y"))))
        state.close_open_tool_calls("cancelled", known=[ToolResult(call_id="a", content="ok")])
        results = state.messages[-1].tool_results
        assert [(r.call_id, r.is_error) for r in results] == [("a", False), ("b", True)]

    def test_check_protocol_detects_unanswered_calls_in_finished_run(self) -> None:
        state = RunState(agent_name="a")
        state.add_user_text("hi")
        state.add_assistant(assistant_with_call())
        state.set_status(RunStatus.DONE)
        with pytest.raises(ProtocolError):
            state.check_protocol()

    def test_check_protocol_detects_wrong_role_order(self) -> None:
        state = RunState(agent_name="a", messages=[Message.user("a"), Message.user("b")])
        with pytest.raises(ProtocolError):
            state.check_protocol()

    def test_check_protocol_detects_pending_mismatch(self) -> None:
        state = RunState(agent_name="a")
        state.add_user_text("hi")
        state.add_assistant(assistant_with_call("c1"))
        state.pending = PendingBatch(waiting=PendingCall(kind="input", call=ToolCall(id="zz", name="x"), prompt="?"))
        state.set_status(RunStatus.WAITING_FOR_INPUT)
        with pytest.raises(ProtocolError):
            state.check_protocol()

    def test_ids_are_validated(self) -> None:
        with pytest.raises(ValidationError):
            RunState(agent_name="a", run_id="../escape")
        with pytest.raises(ValidationError):
            RunState(agent_name="a", tenant_id="a/b")


def _state_with_open_call() -> RunState:
    state = RunState(agent_name="t")
    state.add_user_text("go")
    call = ToolCall(id="c1", name="echo")
    state.add_assistant(Message(role="assistant", content=(call,)))
    return state


def test_batch_done_for_unknown_call_is_a_protocol_error() -> None:
    state = _state_with_open_call()
    state.batch_done = (ToolResult(call_id="foreign", content="x"),)
    with pytest.raises(ProtocolError):
        state.check_protocol()


def test_duplicate_batch_done_is_a_protocol_error() -> None:
    state = _state_with_open_call()
    state.batch_done = (ToolResult(call_id="c1", content="x"), ToolResult(call_id="c1", content="y"))
    with pytest.raises(ProtocolError):
        state.check_protocol()


def test_batch_done_while_waiting_is_a_protocol_error() -> None:
    state = _state_with_open_call()
    state.pending = PendingBatch(waiting=PendingCall(kind="input", call=ToolCall(id="c1", name="echo"), prompt="?"))
    state.status = RunStatus.WAITING_FOR_INPUT
    state.batch_done = (ToolResult(call_id="c1", content="x"),)
    with pytest.raises(ProtocolError):
        state.check_protocol()


def test_long_arguments_error_is_truncated_not_rejected() -> None:
    """A parser's error message must never itself cause response validation to fail."""
    call = ToolCall(id="c1", name="echo", arguments_error="x" * 5000)
    assert call.arguments_error is not None
    assert len(call.arguments_error) <= 500
