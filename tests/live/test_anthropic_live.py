"""Live tests against the real Anthropic API. They cost a few cents and are never part of a default run.

    export ANTHROPIC_API_KEY=...        # in your own shell, never in a file
    pytest -m live -v                   # model: HARNESS_LIVE_MODEL (default claude-haiku-5-5)

Set `HARNESS_RECORD_DIR=/some/dir` to also save every successful API response as JSON. After a review (no
personal data), copy them over the hand-written files in `tests/fixtures/anthropic/`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, ConfigDict

from ai_agent_harness import (
    Agent,
    Autonomy,
    BaseTool,
    ContextBuilder,
    Event,
    InMemoryCheckpointStore,
    Limits,
    ModelRouter,
    ReadFile,
    RetryPolicy,
    Risk,
    RunState,
    RunStatus,
    resume,
    start,
)
from ai_agent_harness.core import InvalidStateError, ProviderError
from ai_agent_harness.core.types import Message, ModelRequest, ThinkingBlock
from ai_agent_harness.providers.anthropic import AnthropicProvider

from ..conftest import WriteFile
from .helpers import MODEL, live_provider, run_live, track

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY is not set"),
]

SMALL_LIMITS = Limits(max_steps=8, max_total_tokens=60_000, max_seconds=120.0)  # a bug can never run up a bill


def live_agent(
    tools: Sequence[BaseTool[Any]] = (),
    *,
    autonomy: Autonomy = Autonomy.READ_ONLY,
    max_tokens: int = 2048,
    **kwargs: Any,
) -> tuple[Agent, list[Event]]:
    events: list[Event] = []
    agent = Agent.create(
        ModelRouter(live_provider(), retry=RetryPolicy(max_attempts=2, base_delay_s=2.0, max_delay_s=10.0)),
        ContextBuilder("You are a precise assistant in an automated test. Be brief.", max_tokens=max_tokens),
        name="live",
        tools=tools,
        autonomy=autonomy,
        limits=SMALL_LIMITS,
        **kwargs,
    )

    async def collect(event: Event) -> None:
        events.append(event)

    agent.events.subscribe(collect)
    return agent, events


class AddOneInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: int


class AddOne(BaseTool[AddOneInput]):
    name: ClassVar[str] = "add_one"
    description: ClassVar[str] = "Returns value + 1. Use it whenever you need to add one to a number."
    input_model: ClassVar[type[BaseModel]] = AddOneInput
    risk: ClassVar[Risk] = Risk.READ

    async def run(self, args: AddOneInput, state: RunState) -> str:
        return str(args.value + 1)


def test_plain_answer() -> None:
    agent, events = live_agent()
    state = run_live(start(agent, "Reply with exactly one word: pong"))
    assert state.status is RunStatus.DONE, state.status_reason
    assert "pong" in state.last_text.lower()
    assert state.step == 1 and state.usage.total > 0 and state.usage.output_tokens > 0
    state.check_protocol()
    assert all("pong" not in str(e.data).lower() for e in events)  # events carry metadata only


def test_tool_call_with_read_file(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("The secret code is 4711.", encoding="utf-8")
    agent, events = live_agent([ReadFile(tmp_path)])
    state = run_live(start(agent, "Read the file note.txt and tell me the secret code."))
    assert state.status is RunStatus.DONE, state.status_reason
    assert "4711" in state.last_text
    assert state.step >= 2
    state.check_protocol()
    assert all("4711" not in str(e.data) for e in events)


def test_sequential_tool_calls_keep_thinking_signatures_valid() -> None:
    """Several model turns in a row: the API verifies the thinking signatures we send back."""
    agent, _ = live_agent([AddOne()])
    prompt = (
        "Start with 0. Call add_one with 0, then call add_one again with the result, then once more with that "
        "result. Make exactly one call per step and wait for each result. Finally report the last number."
    )
    state = run_live(start(agent, prompt))
    assert state.status is RunStatus.DONE, state.status_reason
    assert state.step >= 4 and "3" in state.last_text
    state.check_protocol()


def test_high_effort_with_tool_calls_round_trips_thinking_if_the_model_thinks() -> None:
    """Best effort: at high effort the model may think before a tool call; if it does, the signature is sent back."""
    provider = live_provider(effort="high")
    agent = Agent.create(
        ModelRouter(provider, retry=RetryPolicy(max_attempts=2, base_delay_s=2.0)),
        ContextBuilder("You solve puzzles step by step and use tools for arithmetic.", max_tokens=4096),
        name="live-effort",
        tools=[AddOne()],
        limits=SMALL_LIMITS,
    )
    prompt = (
        "Anna has 3 boxes. Each box gets one more item than the box before, starting with the number of boxes. "
        "Use add_one to find the number of items in the 2nd box and then in the 3rd box (one call at a time, using the "
        "previous result) and tell me the total for all three boxes."
    )
    state = run_live(start(agent, prompt))
    assert state.status is RunStatus.DONE, state.status_reason
    assert state.step >= 3 and "12" in state.last_text
    thinking = [b for m in state.messages for b in m.content if isinstance(b, ThinkingBlock)]
    for block in thinking:  # empty if the model did not think: then there is nothing to round-trip
        assert block.provider_id == "anthropic" and block.provider_data.get("kind") in ("thinking", "redacted_thinking")
    state.check_protocol()


def test_write_tool_waits_for_approval_and_a_stale_approval_is_rejected() -> None:
    write = WriteFile()
    store = InMemoryCheckpointStore()
    agent, _ = live_agent([write], autonomy=Autonomy.SUGGEST, checkpoints=store)

    async def scenario() -> RunState:  # one event loop for the whole test: the provider is closed when it ends
        state = await start(agent, "Use the write_file tool to create the file report.txt, then say done.")
        assert state.status is RunStatus.WAITING_FOR_APPROVAL, state.status_reason
        assert write.written == []  # nothing happened before the human decided
        shown = state.pending_id

        with pytest.raises(InvalidStateError, match="pending_id"):
            await resume(agent, state, approved=True, pending_id="0000000000000000")
        assert write.written == []

        return await resume(agent, state, approved=True, pending_id=shown)

    state = run_live(scenario())
    assert state.status is RunStatus.DONE, state.status_reason
    assert write.written == ["report.txt"]
    state.check_protocol()


def test_truncated_answers_end_in_a_defined_status() -> None:
    agent, _ = live_agent(max_tokens=40)
    state = run_live(start(agent, "Write the numbers from 1 to 200 separated by spaces."))
    assert state.status in (RunStatus.DONE, RunStatus.LIMIT_REACHED, RunStatus.FAILED), state.status
    assert state.step >= 1
    state.check_protocol()


def test_unknown_model_is_not_retryable() -> None:
    provider = track(AnthropicProvider("claude-does-not-exist"))
    with pytest.raises(ProviderError) as caught:
        run_live(provider.complete(ModelRequest(system="x", messages=(Message.user("hi"),), max_tokens=16)))
    assert caught.value.retryable is False and "404" in str(caught.value)


def test_wrong_api_key_is_rejected_without_leaking_the_key() -> None:
    key = "sk-ant-this-key-is-wrong-0123456789"
    provider = track(AnthropicProvider(MODEL, api_key=key))
    with pytest.raises(ProviderError) as caught:
        run_live(provider.complete(ModelRequest(system="x", messages=(Message.user("hi"),), max_tokens=16)))
    assert caught.value.retryable is False and "401" in str(caught.value)
    assert key not in str(caught.value)
