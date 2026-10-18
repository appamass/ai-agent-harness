"""Agent loop: normal flows, stop reasons, retries, and limits."""

from __future__ import annotations

from ai_agent_harness import (
    Autonomy,
    Decision,
    EventType,
    Hook,
    HookPipeline,
    Limits,
    ModelResponse,
    RunState,
    RunStatus,
    start,
)
from ai_agent_harness.core.types import Message, StopReason, TextBlock
from ai_agent_harness.testing import text_response, tool_response

from .conftest import Echo, SendMail, WriteFile, assert_consistent, calls_of, make_agent, results_of, run_sync


def test_tool_call_then_answer() -> None:
    echo = Echo()
    agent, provider, events = make_agent([tool_response(("echo", {})), text_response("done")], [echo])
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.DONE
    assert state.last_text == "done"
    assert echo.calls == 1
    assert results_of(state)[0].content == "echo"
    assert [t.name for t in provider.requests[0].tools] == ["echo"]
    assert events[-1].type is EventType.RUN_FINISHED
    assert_consistent(state)


def test_invalid_arguments_go_back_to_model_without_values() -> None:
    agent, _, _ = make_agent(
        [tool_response(("write_file", {"path": 123, "secret": "pw"})), text_response("ok")],
        [WriteFile()],
        autonomy=Autonomy.AUTONOMOUS,
    )
    state = run_sync(start(agent, "go"))
    result = results_of(state)[0]
    assert result.is_error and "Invalid arguments" in result.content
    assert "pw" not in result.content
    assert_consistent(state)


def test_unknown_tool_lists_only_visible_tools() -> None:
    agent, _, _ = make_agent([tool_response(("nope", {})), text_response("ok")], [Echo(), WriteFile()])
    state = run_sync(start(agent, "go"))
    content = results_of(state)[0].content
    assert "echo" in content and "write_file" not in content


def test_denied_tools_are_not_offered_and_not_executed() -> None:
    write = WriteFile()
    agent, provider, _ = make_agent([tool_response(("write_file", {"path": "a"})), text_response("ok")], [write])
    state = run_sync(start(agent, "go"))
    assert provider.requests[0].tools == ()
    assert write.written == []
    assert "not allowed" in results_of(state)[0].content


def test_irreversible_tool_never_runs_without_approval_even_when_autonomous() -> None:
    mail = SendMail()
    agent, _, _ = make_agent([tool_response(("send_mail", {}))], [mail], autonomy=Autonomy.AUTONOMOUS)
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.WAITING_FOR_APPROVAL
    assert mail.sent == 0
    assert_consistent(state)


def test_refusal_ends_run() -> None:
    agent, _, _ = make_agent([text_response("No.", stop=StopReason.REFUSAL)])
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.REFUSED
    assert state.status_reason == "No."


def test_max_tokens_without_calls_asks_to_continue() -> None:
    agent, provider, _ = make_agent([text_response("Part 1", stop=StopReason.MAX_TOKENS), text_response("Part 2")])
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.DONE
    assert "truncated" in provider.requests[1].messages[-1].text
    assert_consistent(state)


def test_last_text_joins_continued_answer() -> None:
    """Finding 2.3: after a max-tokens continuation, `last_text` returns the full answer, not just Part 2."""
    agent, _, _ = make_agent(
        [
            text_response("Part 1 ", stop=StopReason.MAX_TOKENS),
            text_response("Part 2 ", stop=StopReason.MAX_TOKENS),
            text_response("Part 3"),
        ]
    )
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.DONE
    assert state.last_text == "Part 1 Part 2 Part 3"


def test_last_text_does_not_join_separate_turns() -> None:
    from ai_agent_harness import resume

    agent, _, _ = make_agent([text_response("first answer"), text_response("second answer")])
    first = run_sync(start(agent, "question 1"))
    second = run_sync(resume(agent, first, user_input="question 2"))
    assert second.last_text == "second answer"


def test_max_tokens_with_tool_calls_does_not_execute_and_keeps_protocol() -> None:
    echo = Echo()
    agent, _, _ = make_agent([tool_response(("echo", {}), stop=StopReason.MAX_TOKENS), text_response("ok")], [echo])
    state = run_sync(start(agent, "go"))
    assert echo.calls == 0
    assert results_of(state)[0].is_error
    assert state.status is RunStatus.DONE
    assert_consistent(state)


def test_end_with_tool_calls_executes_them() -> None:
    echo = Echo()
    agent, _, _ = make_agent([tool_response(("echo", {}), stop=StopReason.END), text_response("ok")], [echo])
    state = run_sync(start(agent, "go"))
    assert echo.calls == 1
    assert_consistent(state)


def test_tool_use_without_calls_is_rejected_and_bounded() -> None:
    empty_tool_use = text_response("hm", stop=StopReason.TOOL_USE)
    agent, provider, _ = make_agent([empty_tool_use] * 10, limits=Limits(max_consecutive_retries=2))
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.FAILED
    assert provider.remaining == 7  # stops after 3 attempts, does not spin up to max_steps
    assert_consistent(state)


class RetryOnce(Hook):
    def __init__(self) -> None:
        self.seen = 0

    async def after_model(self, state: RunState, response: ModelResponse) -> Decision:
        self.seen += 1
        return Decision.retry("wrong format") if self.seen == 1 else Decision()


def test_retry_after_tool_call_keeps_protocol_and_skips_execution() -> None:
    echo = Echo()
    agent, provider, _ = make_agent(
        [tool_response(("echo", {})), text_response("ok")], [echo], hooks=HookPipeline([RetryOnce()])
    )
    state = run_sync(start(agent, "go"))
    assert echo.calls == 0
    assert state.status is RunStatus.DONE
    feedback = provider.requests[1].messages[-1]
    assert feedback.tool_results[0].is_error and "wrong format" in feedback.text
    assert_consistent(state)


def test_after_model_block_closes_open_calls() -> None:
    class BlockAll(Hook):
        async def after_model(self, state: RunState, response: ModelResponse) -> Decision:
            return Decision.block("unwanted")

    agent, _, _ = make_agent([tool_response(("echo", {}))], [Echo()], hooks=HookPipeline([BlockAll()]))
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.BLOCKED
    assert_consistent(state)


def test_step_limit() -> None:
    agent, _, _ = make_agent([tool_response(("echo", {}))] * 5, [Echo()], limits=Limits(max_steps=2))
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.LIMIT_REACHED
    assert state.step == 2
    assert_consistent(state)


def test_token_limit() -> None:
    agent, _, _ = make_agent([tool_response(("echo", {}))] * 5, [Echo()], limits=Limits(max_total_tokens=30))
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.LIMIT_REACHED
    assert "Token" in (state.status_reason or "")


def test_blocked_user_input_is_not_stored() -> None:
    class NoSecrets(Hook):
        async def on_user_input(self, state: RunState, text: str) -> Decision:
            return Decision.block("contains password") if "password" in text else Decision()

    agent, provider, _ = make_agent([], hooks=HookPipeline([NoSecrets()]))
    state = run_sync(start(agent, "my password is 123"))
    assert state.status is RunStatus.BLOCKED
    assert state.messages == []
    assert provider.requests == []


def test_thinking_blocks_are_passed_back_unchanged() -> None:
    from ai_agent_harness.core.types import ThinkingBlock

    thinking = ThinkingBlock(text="thinking", provider_data={"signature": "sig"})
    first = tool_response(("echo", {}))
    first = first.model_copy(update={"message": Message(role="assistant", content=(thinking, *first.message.content))})
    agent, provider, _ = make_agent([first, text_response("ok")], [Echo()])
    run_sync(start(agent, "go"))
    passed_back = provider.requests[1].messages[1].content[0]
    # Content and provider_data unchanged; plus the origin provider_id (schema v2, for failover)
    assert passed_back == thinking.model_copy(update={"provider_id": "fake"})


def test_events_cover_the_run() -> None:
    agent, _, events = make_agent([tool_response(("echo", {})), text_response("ok")], [Echo()])
    run_sync(start(agent, "go"))
    types = [e.type for e in events]
    assert types[0] is EventType.RUN_STARTED
    assert EventType.TOOL_STARTED in types and EventType.TOOL_FINISHED in types
    assert types[-1] is EventType.RUN_FINISHED


def test_message_history_never_has_unanswered_calls_after_tools() -> None:
    agent, _, _ = make_agent([tool_response(("echo", {}), ("echo", {})), text_response("ok")], [Echo()])
    state = run_sync(start(agent, "go"))
    assert len(calls_of(state)) == len(results_of(state)) == 2
    assert isinstance(state.messages[0].content[0], TextBlock)
