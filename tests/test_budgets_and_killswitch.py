"""Context budget, input length limit, and kill switch (ADR 0011)."""

from __future__ import annotations

import math
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel

from ai_agent_harness import (
    Agent,
    AskUser,
    Autonomy,
    BaseTool,
    ContextBuilder,
    Event,
    HookPipeline,
    KillSwitch,
    Limits,
    Message,
    ModelRequest,
    ModelResponse,
    ModelRouter,
    RetryPolicy,
    Risk,
    RunState,
    RunStatus,
    ToolCall,
    ToolRegistry,
    ToolResult,
    resume,
    start,
)
from ai_agent_harness import context as context_module
from ai_agent_harness.context import ContextMeter, estimate_tokens
from ai_agent_harness.core import ConfigError, ImageBlock, ProviderError, TextBlock, ThinkingBlock, ToolSpec
from ai_agent_harness.core.events import EventType
from ai_agent_harness.core.types import CHARS_PER_TOKEN, Usage, message_chars
from ai_agent_harness.testing import FakeProvider, text_response, tool_response

from .conftest import Echo, EmptyInput, WriteFile, assert_consistent, make_agent, no_sleep, run_sync


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


# ------------------------------------------------------------------ Token estimate (ADR 0026)


def _with_usage(response: ModelResponse, **usage: int) -> ModelResponse:
    return response.model_copy(update={"usage": Usage(**usage)})


def _requests(events: list[Event]) -> list[Event]:
    return [e for e in events if e.type is EventType.MODEL_REQUEST]


def test_limits_without_the_token_limit_work_as_before() -> None:
    """The new field comes last and defaults to no limit: existing `Limits(...)` calls keep working."""
    assert Limits().max_context_tokens is None
    assert Limits(max_steps=5).max_context_tokens is None
    assert Limits(5, 1000, 10.0, 1, 2000, 300).max_input_chars == 300
    assert Limits(max_context_tokens=1000).max_context_tokens == 1000


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "10", float("nan")])
def test_the_token_limit_must_be_a_positive_integer_or_none(value: Any) -> None:
    with pytest.raises(ConfigError, match="max_context_tokens"):
        Limits(max_context_tokens=value)


def test_other_limits_still_reject_none() -> None:
    none: Any = None
    with pytest.raises(ConfigError, match="max_steps"):
        Limits(max_steps=none)


def test_prompt_tokens_are_all_input_tokens_of_one_call() -> None:
    usage = Usage(input_tokens=10, cache_read_tokens=200, cache_write_tokens=3, output_tokens=99)
    assert usage.prompt_tokens == 213
    assert usage.total == 213 + 99
    assert Usage().prompt_tokens == 0


def test_estimate_tokens_uses_the_measurement_and_converts_only_new_characters() -> None:
    assert estimate_tokens(300) == 300 // CHARS_PER_TOKEN
    assert estimate_tokens(301) == 101  # a started token counts: the estimate errs on the high side
    assert estimate_tokens(1_300, 1_000, 500) == 500 + 300 // CHARS_PER_TOKEN
    assert estimate_tokens(700, 1_000, 500) == 500 - 100  # the view got smaller since the measurement
    assert estimate_tokens(0, 1_000, 50) == 0  # never negative


def test_context_meter_matches_the_estimate_and_measures_each_message_once(monkeypatch: pytest.MonkeyPatch) -> None:
    measured: list[Message] = []

    def counting(message: Message) -> int:
        measured.append(message)
        return message_chars(message)

    monkeypatch.setattr(context_module, "message_chars", counting)
    meter = ContextMeter()
    spec = ToolSpec(name="echo", description="d", input_schema={"type": "object"})
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    first = Message.user("hello")
    reply = Message(role="assistant", content=(TextBlock(text="hi there"),))
    third = Message.user("more")
    histories: list[tuple[Message, ...]] = [(first,), (first, reply), (first, reply, third)]
    for messages in histories:
        request = ModelRequest(system="sys", messages=messages, tools=(spec,), output_schema=schema)
        size = meter.measure(request)
        assert size.total == request.estimated_chars
        assert len(size.messages) == len(messages)
    assert measured == [first, reply, third]  # each message once, although it was in three requests
    # Only the last message can be replaced (merged user text): that one is measured again, the others are not.
    merged = Message(role="user", content=(*third.content, TextBlock(text=" and more")))
    request = ModelRequest(system="sys", messages=(first, reply, merged), tools=(spec,), output_schema=schema)
    assert meter.measure(request).total == request.estimated_chars
    assert measured[-1] is merged
    assert len(measured) == 4
    # Tools and schema that disappear are not counted any more.
    bare = ModelRequest(system="sys", messages=(first,))
    assert meter.measure(bare).total == bare.estimated_chars


def test_the_request_event_carries_the_size_and_the_token_estimate() -> None:
    agent, _, events = make_agent([text_response("ok")])
    run_sync(start(agent, "How big is this?"))
    (event,) = _requests(events)
    chars = len("You are a test agent.") + len("How big is this?")
    assert event.data["context_chars"] == chars
    assert event.data["context_tokens_est"] == math.ceil(chars / CHARS_PER_TOKEN)


def test_the_estimate_starts_from_the_prompt_tokens_the_provider_reported() -> None:
    reported = _with_usage(tool_response(("echo", {})), input_tokens=1_000, cache_read_tokens=400, output_tokens=5)
    agent, _, events = make_agent([reported, text_response("done")], [Echo()])
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.DONE
    first, second = _requests(events)
    added = state.messages[1:3]  # the assistant message with the call, then the result message
    new_chars = sum(message_chars(m) for m in added)
    assert second.data["context_chars"] == first.data["context_chars"] + new_chars
    assert second.data["context_tokens_est"] == 1_400 + math.ceil(new_chars / CHARS_PER_TOKEN)


def test_without_reported_tokens_the_estimate_is_based_on_the_size_alone() -> None:
    silent = _with_usage(tool_response(("echo", {})), output_tokens=5)  # no input tokens reported
    agent, _, events = make_agent([silent, text_response("done")], [Echo()])
    run_sync(start(agent, "go"))
    second = _requests(events)[1]
    assert second.data["context_tokens_est"] == math.ceil(second.data["context_chars"] / CHARS_PER_TOKEN)


def test_a_switch_of_provider_drops_the_measurement() -> None:
    primary = FakeProvider(
        [
            _with_usage(tool_response(("echo", {})), input_tokens=5_000),
            ProviderError("down", retryable=False),
            ProviderError("still down", retryable=False),
        ],
        id="primary",
    )
    backup = FakeProvider(
        [_with_usage(tool_response(("echo", {})), input_tokens=7_000), text_response("done")], id="backup"
    )
    router = ModelRouter(
        primary, [backup], retry=RetryPolicy(max_attempts=1, base_delay_s=0.0, max_delay_s=0.0), sleep=no_sleep
    )
    agent = Agent(
        name="test",
        router=router,
        context=ContextBuilder("You are a test agent."),
        tools=ToolRegistry([Echo()]),
    )
    events: list[Event] = []

    async def collect(event: Event) -> None:
        events.append(event)

    agent.events.subscribe(collect)
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.DONE
    first, second, third = _requests(events)
    assert second.data["context_tokens_est"] > 5_000  # step 2 still uses what `primary` reported
    # Step 2 was answered by `backup`: its numbers do not describe what `primary` will see in step 3.
    assert third.data["context_tokens_est"] == math.ceil(third.data["context_chars"] / CHARS_PER_TOKEN)
    assert first.data["context_tokens_est"] < 5_000


def test_the_token_limit_stops_the_run_before_the_provider_is_called() -> None:
    text = "x" * 300
    chars = len("You are a test agent.") + len(text)
    needed = math.ceil(chars / CHARS_PER_TOKEN)
    ok, _, _ = make_agent([text_response("ok")], limits=Limits(max_context_tokens=needed))
    assert run_sync(start(ok, text)).status is RunStatus.DONE
    over, provider, _ = make_agent([text_response("never")], limits=Limits(max_context_tokens=needed - 1))
    state = run_sync(start(over, text))
    assert state.status is RunStatus.LIMIT_REACHED
    assert state.status_reason is not None
    assert "Context too large" in state.status_reason
    assert "tokens" in state.status_reason
    assert provider.requests == []
    assert_consistent(state)


def test_the_token_limit_uses_what_the_provider_reported() -> None:
    """The size in characters alone would pass: only the reported tokens show that the context is too large."""
    reported = _with_usage(tool_response(("echo", {})), input_tokens=9_000)
    agent, provider, _ = make_agent(
        [reported, text_response("never")], [Echo()], limits=Limits(max_context_tokens=5_000)
    )
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.LIMIT_REACHED
    assert len(provider.requests) == 1
    assert_consistent(state)


def test_both_context_limits_apply() -> None:
    agent, provider, _ = make_agent(
        [text_response("never")], limits=Limits(max_context_chars=10, max_context_tokens=1_000_000)
    )
    state = run_sync(start(agent, "This input is too large for the budget"))
    assert state.status is RunStatus.LIMIT_REACHED
    assert state.status_reason is not None
    assert "chars" in state.status_reason
    assert provider.requests == []


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
