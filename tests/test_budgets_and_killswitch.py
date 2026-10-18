"""Context budget, input length limit, and kill switch (ADR 0011)."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from pydantic import BaseModel

from ai_agent_harness import (
    AskUser,
    Autonomy,
    BaseTool,
    HookPipeline,
    KillSwitch,
    Limits,
    Message,
    ModelRequest,
    Risk,
    RunState,
    RunStatus,
    ToolCall,
    ToolResult,
    resume,
    start,
)
from ai_agent_harness.core import ConfigError, ImageBlock, TextBlock, ThinkingBlock, ToolSpec
from ai_agent_harness.testing import text_response, tool_response

from .conftest import Echo, EmptyInput, WriteFile, assert_consistent, make_agent, run_sync


class Big(BaseTool[EmptyInput]):
    name: ClassVar[str] = "big"
    description: ClassVar[str] = "returns a large text"
    input_model: ClassVar[type[BaseModel]] = EmptyInput
    risk: ClassVar[Risk] = Risk.READ

    async def run(self, args: EmptyInput, state: RunState) -> str:
        return "x" * 5_000


# ------------------------------------------------------------------ Estimation


def test_estimated_chars_counts_every_text_part_but_no_image_data() -> None:
    call = ToolCall(id="c1", name="echo", arguments={"a": "bb"})
    spec = ToolSpec(name="echo", description="d", input_schema={})
    request = ModelRequest(
        system="sys",  # 3
        messages=(
            Message(role="user", content=(TextBlock(text="hello"),)),  # 5
            Message(role="assistant", content=(ThinkingBlock(text="think", provider_data={"sig": "xyz"}), call)),
            Message(
                role="user",
                content=(
                    ToolResult(call_id="c1", content="12345678"),
                    ImageBlock(media_type="image/png", data="A" * 999),
                ),
            ),  # 8, image 0
        ),
        tools=(spec,),
    )
    thinking = 5 + len('{"sig": "xyz"}')  # text + provider_data (encrypted reasoning can be large)
    expected = 3 + 5 + thinking + len(call.model_dump_json()) + 8 + len(spec.model_dump_json())
    assert request.estimated_chars == expected


def test_estimate_never_fails_on_odd_provider_data() -> None:
    """Provider data is opaque; estimation must never fail on unusual values."""
    odd = ThinkingBlock(text="t", provider_data={"raw": b"\xff\xfe", "n": float("nan")})
    request = ModelRequest(
        system="s",
        messages=(Message(role="user", content=(TextBlock(text="u"),)), Message(role="assistant", content=(odd,))),
    )
    assert request.estimated_chars > 3


# ------------------------------------------------------------------ Context budget


def test_context_over_budget_ends_as_limit_reached_before_the_provider_is_called() -> None:
    agent, provider, _ = make_agent([text_response("never")], limits=Limits(max_context_chars=10))
    state = run_sync(start(agent, "This input is too large for the budget"))
    assert state.status is RunStatus.LIMIT_REACHED
    assert state.status_reason is not None
    assert "Context too large" in state.status_reason
    assert provider.requests == []
    assert_consistent(state)


def test_context_budget_boundary_is_inclusive() -> None:
    """At the exact limit the call still runs; one character over does not."""
    text = "How big is this?"
    size = len("You are a test agent.") + len(text)  # system + user text, no tools
    ok, _, _ = make_agent([text_response("ok")], limits=Limits(max_context_chars=size))
    assert run_sync(start(ok, text)).status is RunStatus.DONE
    over, provider, _ = make_agent([text_response("never")], limits=Limits(max_context_chars=size - 1))
    assert run_sync(start(over, text)).status is RunStatus.LIMIT_REACHED
    assert provider.requests == []


def test_growing_context_stops_cleanly_after_a_big_tool_result() -> None:
    agent, provider, _ = make_agent(
        [tool_response(("big", {})), text_response("never")], [Big()], limits=Limits(max_context_chars=3_000)
    )
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.LIMIT_REACHED
    assert len(provider.requests) == 1  # second call would have exceeded the budget
    assert state.messages[-1].tool_results[0].content.startswith("xxx")
    assert_consistent(state)


# ------------------------------------------------------------------ Input limit


def test_too_long_start_input_is_rejected() -> None:
    agent, provider, _ = make_agent([], limits=Limits(max_input_chars=10))
    with pytest.raises(ValueError, match="too long"):
        run_sync(start(agent, "x" * 11))
    assert provider.requests == []


def test_input_at_the_limit_is_accepted() -> None:
    agent, _, _ = make_agent([text_response("ok")], limits=Limits(max_input_chars=10))
    assert run_sync(start(agent, "x" * 10)).status is RunStatus.DONE


def test_too_long_answer_and_follow_up_are_rejected_without_touching_state() -> None:
    agent, _, _ = make_agent(
        [tool_response(("ask_user", {"question": "Who?"})), text_response("done")],
        [AskUser()],
        limits=Limits(max_input_chars=10),
    )
    waiting = run_sync(start(agent, "ask"))
    assert waiting.status is RunStatus.WAITING_FOR_INPUT
    before = waiting.model_dump_json()
    with pytest.raises(ValueError, match="'answer' is too long"):
        run_sync(resume(agent, waiting, answer="y" * 11, pending_id=waiting.pending_id))
    assert waiting.model_dump_json() == before
    done = run_sync(resume(agent, waiting, answer="Anna", pending_id=waiting.pending_id))
    assert done.status is RunStatus.DONE
    with pytest.raises(ValueError, match="'user_input' is too long"):
        run_sync(resume(agent, done, user_input="z" * 11))


def test_too_long_approval_comment_is_rejected() -> None:
    agent, _, _ = make_agent(
        [tool_response(("write_file", {"path": "a.py"}))],
        [WriteFile()],
        autonomy=Autonomy.SUGGEST,
        limits=Limits(max_input_chars=10),
    )
    waiting = run_sync(start(agent, "write"))
    assert waiting.status is RunStatus.WAITING_FOR_APPROVAL
    with pytest.raises(ValueError, match="too long"):
        run_sync(resume(agent, waiting, approved=False, answer="k" * 11, pending_id=waiting.pending_id))
    assert waiting.status is RunStatus.WAITING_FOR_APPROVAL


@pytest.mark.parametrize("field", ["max_context_chars", "max_input_chars"])
def test_budgets_must_be_positive(field: str) -> None:
    with pytest.raises(ConfigError):
        Limits(**{field: 0})


# ------------------------------------------------------------------ Kill switch


def test_active_kill_switch_stops_before_the_first_model_call() -> None:
    agent, provider, _ = make_agent([text_response("never")], hooks=HookPipeline([KillSwitch(lambda _: True)]))
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.BLOCKED
    assert state.status_reason == "Kill switch is active"
    assert provider.requests == []
    assert_consistent(state)


def test_kill_switch_flipped_mid_run_prevents_the_tool() -> None:
    write = WriteFile()

    async def check(state: RunState) -> bool:  # async like a feature-flag service; flips after the 1st model call
        return state.step >= 1

    agent, provider, _ = make_agent(
        [tool_response(("write_file", {"path": "a.py"})), text_response("never")],
        [write],
        hooks=HookPipeline([KillSwitch(check)]),
        autonomy=Autonomy.AUTONOMOUS,
    )
    state = run_sync(start(agent, "write"))
    assert write.written == []
    assert len(provider.requests) == 1
    assert state.status is RunStatus.BLOCKED
    assert_consistent(state)


def test_kill_switch_blocks_an_already_approved_action() -> None:
    flag = {"on": False}
    write = WriteFile()
    agent, _, _ = make_agent(
        [tool_response(("write_file", {"path": "a.py"})), text_response("done")],
        [write],
        hooks=HookPipeline([KillSwitch(lambda _: flag["on"])]),
        autonomy=Autonomy.SUGGEST,
    )
    waiting = run_sync(start(agent, "write"))
    assert waiting.status is RunStatus.WAITING_FOR_APPROVAL
    flag["on"] = True
    state = run_sync(resume(agent, waiting, approved=True, pending_id=waiting.pending_id))
    assert write.written == []
    assert state.status is RunStatus.BLOCKED
    assert_consistent(state)


def test_kill_switch_also_blocks_user_input() -> None:
    from ai_agent_harness import Decision, KillSwitch
    from ai_agent_harness.core.hooks import Action

    decision: Decision = run_sync(KillSwitch(lambda _state: True).on_user_input(RunState(agent_name="a"), "hello"))
    assert decision.action is Action.BLOCK
    allowed: Decision = run_sync(KillSwitch(lambda _state: False).on_user_input(RunState(agent_name="a"), "hello"))
    assert allowed.action is Action.CONTINUE


def test_kill_switch_is_fail_closed() -> None:
    def broken(_: RunState) -> bool:
        raise ConnectionError("flag service unreachable")

    def not_a_bool(_: RunState) -> Any:
        return "yes"

    for check in (broken, not_a_bool):
        agent, provider, _ = make_agent([text_response("never")], [Echo()], hooks=HookPipeline([KillSwitch(check)]))
        state = run_sync(start(agent, "go"))
        assert state.status is RunStatus.BLOCKED
        assert provider.requests == []


def test_kill_switch_needs_a_callable() -> None:
    not_callable: Any = True
    with pytest.raises(TypeError):
        KillSwitch(not_callable)
