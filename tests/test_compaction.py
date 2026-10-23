"""Context compaction: old tool results are cleared in the model's view, never in `RunState.messages` (ADR 0026)."""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, ValidationError

from ai_agent_harness import (
    Agent,
    Autonomy,
    BaseTool,
    ContextBuilder,
    Event,
    InMemoryCheckpointStore,
    Limits,
    Message,
    ModelRequest,
    ModelResponse,
    Risk,
    RunState,
    RunStatus,
    ToolCall,
    ToolResult,
    restore_state,
    resume,
    start,
)
from ai_agent_harness.context import (
    COMPACT_TARGET,
    COMPACT_TRIGGER,
    KEEP_RECENT_MESSAGES,
    MIN_CLEARED_CHARS,
    MessageSize,
    RequestSize,
    char_budget,
    choose_floor,
    cleared_placeholder,
    compact_view,
)
from ai_agent_harness.core import ConfigError, ProtocolError, TextBlock
from ai_agent_harness.core.events import EventType
from ai_agent_harness.core.state import MIGRATIONS, SCHEMA_VERSION
from ai_agent_harness.core.types import Usage
from ai_agent_harness.runtime import runner as runner_module
from ai_agent_harness.testing import FakeProvider, text_response, tool_response

from .conftest import FIXTURES_DIR, EmptyInput, WriteFile, assert_consistent, make_agent, run_sync

FIXTURE_V5 = FIXTURES_DIR / "checkpoint_v5_waiting_with_floor.json"
FIXTURE_V4 = FIXTURES_DIR / "checkpoint_v4_done_with_output.json"
PLACEHOLDER = re.compile(r"^\[Result removed to save space \(\d+ chars\)\. Call the tool again if you need it\.\]$")


class Blob(BaseTool[EmptyInput]):
    """Returns a result of `size` characters that starts with a number: no two results are equal."""

    name: ClassVar[str] = "blob"
    description: ClassVar[str] = "Returns a long text."
    input_model: ClassVar[type[BaseModel]] = EmptyInput
    risk: ClassVar[Risk] = Risk.READ

    def __init__(self, size: int = 1_000) -> None:
        self.size = size
        self.calls = 0

    async def run(self, args: EmptyInput, state: RunState) -> str:
        self.calls += 1
        return f"R{self.calls:04d}" + "x" * (self.size - 5)


def _rounds(count: int) -> list[ModelResponse]:
    """Tool calls from a provider that reports no input tokens: the token estimate is based on the size alone."""
    silent = Usage(output_tokens=5)
    return [tool_response(("blob", {})).model_copy(update={"usage": silent}) for _ in range(count)]


def _agent(
    script: list[ModelResponse],
    *,
    tools: list[BaseTool[Any]] | None = None,
    compact: bool = True,
    limits: Limits | None = None,
    **kwargs: Any,
) -> tuple[Agent, FakeProvider, list[Event]]:
    return make_agent(
        script,
        tools if tools is not None else [Blob()],
        compact_context=compact,
        limits=limits or Limits(max_steps=200, max_context_chars=30_000),
        **kwargs,
    )


def _compactions(events: list[Event]) -> list[Event]:
    return [e for e in events if e.type is EventType.CONTEXT_COMPACTED]


def _valid_view(request: ModelRequest) -> None:
    """The model's view obeys the same protocol as the stored history."""
    RunState(agent_name="view", status=RunStatus.DONE, messages=list(request.messages)).check_protocol()


def _results(messages: list[Message] | tuple[Message, ...]) -> list[ToolResult]:
    return [result for message in messages for result in message.tool_results]


def _long_run(compact: bool = True) -> tuple[RunState, FakeProvider, list[Event]]:
    agent, provider, events = _agent([*_rounds(60), text_response("done")], compact=compact)
    return run_sync(start(agent, "go")), provider, events


# ------------------------------------------------------------------ Pure functions


def test_the_placeholder_names_the_size_and_how_to_get_the_data_back_but_no_content() -> None:
    assert (
        cleared_placeholder(1234) == "[Result removed to save space (1234 chars). Call the tool again if you need it.]"
    )


def _result_message(*results: ToolResult) -> Message:
    return Message(role="user", content=results)


def _call_message(*ids: str) -> Message:
    return Message(role="assistant", content=tuple(ToolCall(id=i, name="blob") for i in ids))


def test_compact_view_changes_only_long_results_before_the_floor() -> None:
    long_error = ToolResult(call_id="c2", content="e" * 500, is_error=True)
    messages = [
        Message.user("go"),
        _call_message("c1"),
        _result_message(ToolResult(call_id="c1", content="a" * 500)),
        _call_message("c2", "c3"),
        _result_message(long_error, ToolResult(call_id="c3", content="File not found")),
        _call_message("c4"),
        _result_message(ToolResult(call_id="c4", content="d" * 500)),
    ]
    view = compact_view(messages, 5)
    assert [m.role for m in view] == [m.role for m in messages]
    first, second = _results(view[2:5])[0], _results(view[2:5])[1:]
    assert PLACEHOLDER.match(first.content) and first.call_id == "c1"
    assert PLACEHOLDER.match(second[0].content) and second[0].is_error and second[0].call_id == "c2"  # still an error
    assert second[1].content == "File not found"  # too short to be worth a placeholder
    assert view[5:] == tuple(messages[5:])  # at and after the floor: untouched
    assert view[0] is messages[0] and view[1] is messages[1]
    assert messages[2].tool_results[0].content == "a" * 500  # the input is not changed
    assert compact_view(messages, 0) == tuple(messages)
    assert compact_view(messages, 99)[2:5] == view[2:5]  # a floor beyond the end clears everything before it


def test_a_result_at_the_minimum_length_is_kept_and_one_character_more_is_cleared() -> None:
    keep = _result_message(ToolResult(call_id="c1", content="k" * MIN_CLEARED_CHARS))
    clear = _result_message(ToolResult(call_id="c1", content="k" * (MIN_CLEARED_CHARS + 1)))
    assert compact_view([keep], 1)[0] is keep
    assert PLACEHOLDER.match(_results(compact_view([clear], 1))[0].content)


def test_the_placeholder_is_always_shorter_than_what_it_replaces() -> None:
    placeholder = len(cleared_placeholder(10**9))
    assert placeholder < MIN_CLEARED_CHARS


@pytest.mark.parametrize("floor", [0, 2, 3, 5, 7])
def test_the_size_of_the_view_is_exact_also_when_the_digits_of_the_size_change(floor: int) -> None:
    from ai_agent_harness.context import ContextMeter

    sizes = [201, 999, 1000, 12_345]
    messages: list[Message] = [Message.user("go")]
    for index, size in enumerate(sizes, 1):
        messages.append(_call_message(f"c{index}"))
        messages.append(_result_message(ToolResult(call_id=f"c{index}", content="z" * size)))
    request = ModelRequest(system="sys", messages=tuple(messages))
    measured = ContextMeter().measure(request)
    view = ModelRequest(system="sys", messages=compact_view(messages, floor))
    assert measured.total == request.estimated_chars
    assert measured.view(floor) == view.estimated_chars


def _size(*pairs: tuple[int, int], fixed: int = 0) -> RequestSize:
    """A request of messages given as `(chars, saved)`."""
    return RequestSize(fixed=fixed, messages=tuple(MessageSize(c, s, 1 if s else 0) for c, s in pairs))


def test_choose_floor_does_nothing_below_the_trigger() -> None:
    size = _size(*[(100, 90)] * 20)  # 2000 chars
    assert choose_floor(size, int(2000 / COMPACT_TRIGGER) + 1, 0) == 0
    assert choose_floor(size, int(2000 / COMPACT_TRIGGER) - 50, 0) > 0


def test_choose_floor_jumps_in_one_step_down_to_the_target() -> None:
    size = _size(*[(100, 90)] * 40)  # 4000 chars, each message could save 90
    budget = 4200  # the view is above 80%: 3360
    floor = choose_floor(size, budget, 0)
    assert size.view(floor) <= budget * COMPACT_TARGET
    assert size.view(floor - 1) > budget * COMPACT_TARGET  # no further than needed
    assert floor <= 40 - KEEP_RECENT_MESSAGES


def test_choose_floor_never_touches_the_last_messages() -> None:
    size = _size(*[(100, 90)] * 12)
    floor = choose_floor(size, 100, 0)  # impossible target: clear as far as allowed
    assert floor == 12 - KEEP_RECENT_MESSAGES
    short = _size(*[(100, 90)] * KEEP_RECENT_MESSAGES)
    assert choose_floor(short, 100, 0) == 0


def test_choose_floor_counts_what_the_current_floor_already_saved() -> None:
    size = _size(*[(100, 90)] * 40)
    budget = 4200
    first = choose_floor(size, budget, 0)
    assert choose_floor(size, budget, first) == first  # now below the trigger: nothing to do
    bigger = _size(*[(100, 90)] * 80)
    assert choose_floor(bigger, budget, first) > first


def test_choose_floor_returns_no_new_floor_when_nothing_can_be_cleared() -> None:
    assert choose_floor(_size(*[(100, 0)] * 40), 100, 0) == 0
    assert choose_floor(_size(*[(100, 0)] * 40), 100, 5) == 5


def test_char_budget_is_the_tighter_of_both_limits() -> None:
    assert char_budget(10_000, None, 0, 0) == 10_000
    assert char_budget(10_000, 1_000, 0, 0) == 3_000  # 1000 tokens * 3 chars
    assert char_budget(10_000, 1_000, 6_000, 1_500) == 6_000 - 1_500  # the last call is 500 tokens under the limit
    assert char_budget(10_000, 100_000, 0, 0) == 10_000
    assert char_budget(10_000, 100, 9_000, 5_000) == 1  # already far over the token limit: never below 1


# ------------------------------------------------------------------ Runs


def test_without_compaction_a_long_run_hits_the_limit_and_with_it_the_run_finishes() -> None:
    plain, _, _ = _long_run(compact=False)
    assert plain.status is RunStatus.LIMIT_REACHED
    state, provider, events = _long_run()
    assert state.status is RunStatus.DONE
    assert len(provider.requests) == 61
    assert _compactions(events)
    assert_consistent(state)


def test_the_stored_history_keeps_every_result_in_full() -> None:
    state, _, _ = _long_run()
    results = _results(state.messages)
    assert len(results) == 60
    assert all(len(r.content) == 1_000 and r.content.startswith("R") for r in results)
    assert [r.content[:5] for r in results] == [f"R{n:04d}" for n in range(1, 61)]
    assert state.context_floor > 0


def test_the_model_sees_placeholders_before_the_floor_and_the_recent_rounds_in_full() -> None:
    state, provider, _ = _long_run()
    request = provider.requests[-1]
    floor = state.context_floor
    assert len(request.messages) == len(state.messages) - 1  # the final answer is not in the last request
    cleared = _results(request.messages[:floor])
    assert cleared and all(PLACEHOLDER.match(r.content) for r in cleared)
    assert all("R00" not in r.content for r in cleared)  # nothing of the content is left
    recent = _results(request.messages[floor:])
    assert recent and all(len(r.content) == 1_000 for r in recent)
    assert floor <= len(request.messages) - KEEP_RECENT_MESSAGES
    last_four = _results(request.messages[-KEEP_RECENT_MESSAGES:])
    assert len(last_four) == 4 and all(len(r.content) == 1_000 for r in last_four)


def test_every_request_keeps_the_tool_call_protocol() -> None:
    _, provider, _ = _long_run()
    for request in provider.requests:
        _valid_view(request)


def test_the_floor_rises_in_big_steps_not_every_step() -> None:
    state, provider, events = _long_run()
    floors = [e.data["floor"] for e in _compactions(events)]
    assert floors == sorted(set(floors))  # only upwards
    assert floors[-1] == state.context_floor
    assert 3 <= len(floors) <= 10 < len(provider.requests) // 3
    gaps = [b - a for a, b in zip([0, *floors], floors, strict=False)]
    assert all(gap >= 6 for gap in gaps)  # a jump, not a creeping change that would break the prompt cache each step


def test_the_view_before_the_floor_stays_the_same_between_compactions() -> None:
    state, provider, events = _long_run()
    first = _compactions(events)[0].data["floor"]
    index = next(i for i, e in enumerate(events) if e.type is EventType.CONTEXT_COMPACTED)
    start = next(e for e in events[index:] if e.type is EventType.MODEL_REQUEST).data["messages"]
    after = [r for r in provider.requests if len(r.messages) >= start]  # every request from the first compaction on
    heads = {tuple(m.model_dump_json() for m in r.messages[:first]) for r in after}
    assert len(after) > 20
    assert len(heads) == 1  # this part is a stable prefix for the prompt cache
    assert state.context_floor >= first


def test_compaction_events_carry_counters_only() -> None:
    _, _, events = _long_run()
    compactions = _compactions(events)
    for event in compactions:
        assert set(event.data) == {"floor", "cleared_results", "chars_saved"}
        assert all(isinstance(v, int) and v > 0 for v in event.data.values())
    assert sum(e.data["cleared_results"] for e in compactions) > 10
    dumped = json.dumps([e.model_dump(mode="json") for e in compactions])
    assert "R00" not in dumped and "Result removed" not in dumped


def test_a_checkpoint_is_saved_before_the_model_call_only_when_the_floor_rose() -> None:
    script = [*_rounds(60), text_response("done")]
    agent, _, events = _agent(script, checkpoints=InMemoryCheckpointStore())
    run_sync(start(agent, "go"))
    big, _, quiet = _agent(
        [*_rounds(60), text_response("done")],
        checkpoints=InMemoryCheckpointStore(),
        limits=Limits(max_steps=200, max_context_chars=1_000_000),
    )
    run_sync(start(big, "go"))
    saved = [e for e in events if e.type is EventType.CHECKPOINT_SAVED]
    saved_quiet = [e for e in quiet if e.type is EventType.CHECKPOINT_SAVED]
    assert _compactions(quiet) == []
    assert len(saved) - len(saved_quiet) == len(_compactions(events))
    order = [e.type for e in events]
    for index, kind in enumerate(order):
        if kind is EventType.CONTEXT_COMPACTED:
            assert order[index + 1] is EventType.CHECKPOINT_SAVED
            assert order[index + 2] is EventType.MODEL_REQUEST


def test_the_compacted_head_is_built_again_only_when_the_floor_rises(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[int] = []

    def counting(messages: Any, floor: int) -> tuple[Message, ...]:
        built.append(floor)
        return compact_view(messages, floor)

    monkeypatch.setattr(runner_module, "compact_view", counting)
    state, provider, events = _long_run()
    assert built == [e.data["floor"] for e in _compactions(events)]
    assert len(built) < len(provider.requests) // 3
    assert state.status is RunStatus.DONE


def test_context_sources_are_rendered_once_per_step() -> None:
    rendered: list[int] = []

    class Counting:
        async def render(self, state: RunState) -> str | None:
            rendered.append(state.step)
            return None

    agent, provider, _ = _agent([*_rounds(60), text_response("done")])
    agent = dataclasses.replace(agent, context=ContextBuilder("You are a test agent.", sources=[Counting()]))
    run_sync(start(agent, "go"))
    assert len(rendered) == len(provider.requests) == 61


def test_the_view_is_the_same_after_a_pause_and_after_loading_the_checkpoint() -> None:
    store = InMemoryCheckpointStore()
    write = WriteFile()
    script = [*_rounds(40), tool_response(("write_file", {"path": "a.py"})), text_response("done")]
    agent, provider, _ = _agent(script, tools=[Blob(), write], autonomy=Autonomy.SUGGEST, checkpoints=store)
    waiting = run_sync(start(agent, "go"))
    assert waiting.status is RunStatus.WAITING_FOR_APPROVAL
    floor = waiting.context_floor
    assert floor > 0
    before = provider.requests[-1]

    loaded = run_sync(store.load(waiting.tenant_id, waiting.run_id))
    assert loaded is not None
    assert loaded.context_floor == floor
    assert loaded.measured_chars == waiting.measured_chars
    assert loaded.measured_prompt_tokens == waiting.measured_prompt_tokens
    done = run_sync(resume(agent, loaded, approved=True, pending_id=loaded.pending_id))
    assert done.status is RunStatus.DONE and write.written == ["a.py"]
    after = provider.requests[-1]
    assert after.messages[:floor] == before.messages[:floor]  # the same cleared head as before the pause
    assert done.context_floor >= floor
    _valid_view(after)


def test_the_flag_is_off_by_default_and_nothing_is_cleared() -> None:
    agent, provider, events = _agent([*_rounds(20), text_response("done")], compact=False)
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.DONE and state.context_floor == 0
    assert _compactions(events) == []
    assert all(len(r.content) == 1_000 for r in _results(provider.requests[-1].messages))
    assert Agent.create(FakeProvider([text_response("x")])).compact_context is False


def test_short_results_are_never_cleared_and_the_hard_limit_still_applies() -> None:
    agent, provider, events = _agent(
        [*_rounds(60), text_response("never")], tools=[Blob(100)], limits=Limits(max_steps=200, max_context_chars=2_000)
    )
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.LIMIT_REACHED
    assert _compactions(events) == []
    assert state.context_floor == 0
    assert all(len(r.content) == 100 for r in _results(provider.requests[-1].messages))
    assert_consistent(state)


def test_the_token_limit_drives_compaction_too() -> None:
    limits = Limits(max_steps=200, max_context_tokens=10_000)  # 30,000 characters without a measurement
    plain, _, _ = _agent([*_rounds(60), text_response("never")], compact=False, limits=limits)
    assert run_sync(start(plain, "go")).status is RunStatus.LIMIT_REACHED
    agent, _, events = _agent([*_rounds(60), text_response("done")], limits=limits)
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.DONE
    assert _compactions(events)


def test_the_token_measurement_is_dropped_when_the_floor_rises() -> None:
    state, _, events = _long_run()
    assert _compactions(events)
    requests = [e for e in events if e.type is EventType.MODEL_REQUEST]
    # right after a compaction the estimate is based on the size alone: it is the size divided by 3, rounded up
    order = [e.type for e in events]
    index = order.index(EventType.CONTEXT_COMPACTED)
    request = next(e for e in events[index:] if e.type is EventType.MODEL_REQUEST)
    assert request.data["context_tokens_est"] == -(-request.data["context_chars"] // 3)
    assert len(requests) == 61 and state.measured_provider_id == "fake"


def test_a_resumed_run_ignores_the_floor_when_the_agent_does_not_compact() -> None:
    state = restore_state(FIXTURE_V5.read_text(encoding="utf-8"))
    assert state.context_floor == 3
    provider = FakeProvider([text_response("done")])
    plain = dataclasses.replace(make_agent([], [WriteFile()], autonomy=Autonomy.SUGGEST)[0], router=_router(provider))
    run_sync(resume(plain, state, approved=True, pending_id=state.pending_id))
    assert all(len(r.content) == 400 for r in _results(provider.requests[-1].messages)[:2])  # nothing cleared


def _router(provider: FakeProvider) -> Any:
    from ai_agent_harness import ModelRouter

    return ModelRouter(provider)


# ------------------------------------------------------------------ Configuration


def test_compact_context_must_be_a_bool() -> None:
    base, _, _ = make_agent([])
    wrong: Any = "yes"
    with pytest.raises(ConfigError, match="compact_context"):
        dataclasses.replace(base, compact_context=wrong)
    assert dataclasses.replace(base, compact_context=True).compact_context is True


def test_agent_create_passes_compact_context_on() -> None:
    agent = Agent.create(FakeProvider([text_response("x")]), compact_context=True)
    assert agent.compact_context is True


# ------------------------------------------------------------------ State and schema v5


def _state_with_rounds(rounds: int) -> RunState:
    state = RunState(agent_name="a")
    state.add_user_text("go")
    for index in range(1, rounds + 1):
        call = ToolCall(id=f"c{index}", name="blob")
        state.add_assistant(Message(role="assistant", content=(call,)))
        state.add_tool_results([ToolResult(call_id=call.id, content="y" * 300)])
    return state


def test_the_floor_can_only_rise_and_never_pass_the_messages() -> None:
    state = _state_with_rounds(3)  # 7 messages
    state.raise_context_floor(3)
    assert state.context_floor == 3
    for bad in (3, 2, 8):
        with pytest.raises(ProtocolError, match="context floor"):
            state.raise_context_floor(bad)
    assert state.context_floor == 3
    state.raise_context_floor(7)
    assert state.context_floor == 7


def test_raising_the_floor_drops_the_token_measurement() -> None:
    state = _state_with_rounds(3)
    state.record_measurement("p", 1_000, 400)
    assert (state.measured_provider_id, state.measured_chars, state.measured_prompt_tokens) == ("p", 1_000, 400)
    state.raise_context_floor(3)
    assert (state.measured_provider_id, state.measured_chars, state.measured_prompt_tokens) == (None, 0, 0)


def test_a_measurement_of_another_provider_or_without_tokens_starts_over() -> None:
    state = RunState(agent_name="a")
    state.record_measurement("a", 1_000, 400)
    state.record_measurement("a", 2_000, 900)
    assert (state.measured_chars, state.measured_prompt_tokens) == (2_000, 900)
    state.record_measurement("b", 3_000, 1_200)
    assert (state.measured_provider_id, state.measured_chars, state.measured_prompt_tokens) == ("b", 0, 0)
    state.record_measurement("b", 3_000, 1_200)
    assert (state.measured_chars, state.measured_prompt_tokens) == (3_000, 1_200)
    state.record_measurement("b", 4_000, 0)  # the provider reported no input tokens
    assert (state.measured_provider_id, state.measured_chars, state.measured_prompt_tokens) == ("b", 0, 0)


def test_a_saved_floor_beyond_the_messages_is_rejected_when_loading() -> None:
    raw = json.loads(FIXTURE_V5.read_text(encoding="utf-8"))
    messages = len(raw["messages"])
    raw["context_floor"] = messages
    assert restore_state(json.dumps(raw)).context_floor == messages  # any value in range is a valid protocol
    raw["context_floor"] = messages + 1
    with pytest.raises(ProtocolError, match="Context floor"):
        restore_state(json.dumps(raw))
    raw["context_floor"] = -1
    with pytest.raises(ValidationError):
        restore_state(json.dumps(raw))


def test_the_v5_fixture_loads_and_resumes_with_the_same_head() -> None:
    state = restore_state(FIXTURE_V5.read_text(encoding="utf-8"))
    assert state.schema_version == SCHEMA_VERSION == 5
    assert state.status is RunStatus.WAITING_FOR_APPROVAL
    assert state.context_floor == 3
    assert (state.measured_provider_id, state.measured_chars, state.measured_prompt_tokens) == ("fake", 1_500, 700)
    provider = FakeProvider([text_response("done")])
    agent, _, _ = _agent([], tools=[Blob(), WriteFile()], autonomy=Autonomy.SUGGEST)
    agent = dataclasses.replace(agent, router=_router(provider))
    done = run_sync(resume(agent, state, approved=True, pending_id=state.pending_id))
    assert done.status is RunStatus.DONE
    view = provider.requests[-1].messages
    first, second, _ = _results(view)  # the third is the result of the approved write
    assert PLACEHOLDER.match(first.content)  # before the floor
    assert second.content == "q" * 400  # after the floor
    assert _results(done.messages)[0].content == "p" * 400  # the stored history is whole
    assert_consistent(done)


def test_a_v4_checkpoint_loads_with_no_floor_and_no_measurement() -> None:
    state = restore_state(FIXTURE_V4.read_text(encoding="utf-8"))
    assert state.schema_version == SCHEMA_VERSION
    assert state.context_floor == 0
    assert (state.measured_provider_id, state.measured_chars, state.measured_prompt_tokens) == (None, 0, 0)
    assert 4 in MIGRATIONS  # the chain from v4 to v5 exists, even though it only adds defaults


def test_the_text_of_a_user_message_is_not_a_result_and_is_never_cleared() -> None:
    long_text = "t" * 500
    message = Message(role="user", content=(TextBlock(text=long_text),))
    assert compact_view([message], 1)[0] is message
