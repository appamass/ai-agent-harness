"""Persisted format v2 (ADR 0012): cache/reasoning tokens, thinking block origin,
invalid tool-call JSON, and completed results in an in-progress tool batch."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_agent_harness import (
    Autonomy,
    Message,
    ModelResponse,
    ModelRouter,
    RunStatus,
    StopReason,
    ToolCall,
    resume,
    start,
    without_foreign_thinking,
)
from ai_agent_harness.core import MIGRATIONS, SCHEMA_VERSION, TextBlock, ThinkingBlock, Usage
from ai_agent_harness.testing import FakeProvider, text_response

from .conftest import FIXTURES_DIR, Echo, assert_consistent, make_agent, run_sync

# ------------------------------------------------------------------ Usage


def test_usage_counts_cache_and_reasoning_tokens() -> None:
    a = Usage(input_tokens=10, output_tokens=20, cache_read_tokens=100, cache_write_tokens=5, reasoning_tokens=15)
    b = Usage(input_tokens=1, output_tokens=2, cache_read_tokens=3, cache_write_tokens=4, reasoning_tokens=1)
    total = a + b
    assert total == Usage(
        input_tokens=11, output_tokens=22, cache_read_tokens=103, cache_write_tokens=9, reasoning_tokens=16
    )
    assert a.total == 10 + 20 + 100 + 5  # everything processed by the provider; reasoning is included in output


def test_reasoning_tokens_are_part_of_output_tokens() -> None:
    """Adapter contract: output_tokens includes reasoning. An adapter that double-counts fails immediately."""
    with pytest.raises(ValidationError):
        Usage(output_tokens=5, reasoning_tokens=6)


# ------------------------------------------------------------------ Thinking origin


def _thinking_response(text: str = "done") -> ModelResponse:
    return ModelResponse(
        message=Message(
            role="assistant",
            content=(ThinkingBlock(text="hm", provider_data={"signature": "s"}), TextBlock(text=text)),
        ),
        stop_reason=StopReason.END,
    )


def test_runner_stamps_the_answering_provider_on_thinking_blocks() -> None:
    agent, _, _ = make_agent([])
    agent = dataclasses.replace(agent, router=ModelRouter(FakeProvider([_thinking_response()], id="provider-a")))
    state = run_sync(start(agent, "go"))
    blocks = [b for b in state.messages[-1].content if isinstance(b, ThinkingBlock)]
    assert [b.provider_id for b in blocks] == ["provider-a"]


def test_failover_partner_never_sees_foreign_thinking() -> None:
    own = ThinkingBlock(text="mine", provider_id="b")
    foreign = ThinkingBlock(text="foreign", provider_id="a", provider_data={"signature": "only-for-a"})
    legacy = ThinkingBlock(text="old")  # saved before v2, origin unknown
    messages = (
        Message.user("go"),
        Message(role="assistant", content=(foreign, TextBlock(text="x"))),
        Message.user("next"),
        Message(role="assistant", content=(foreign,)),
        Message.user("and"),
        Message(role="assistant", content=(own, legacy, TextBlock(text="y"))),
    )
    cleaned = without_foreign_thinking(messages, "b")
    assert len(cleaned) == len(messages)
    assert cleaned[1].content == (TextBlock(text="x"),)
    assert cleaned[3].text  # never an empty message
    assert all(not isinstance(b, ThinkingBlock) for b in cleaned[3].content)
    assert cleaned[5].content == (own, legacy, TextBlock(text="y"))


# ------------------------------------------------------------------ Invalid JSON


def test_unparsable_tool_arguments_go_back_to_the_model_instead_of_failing_the_run() -> None:
    echo = Echo()
    broken = ModelResponse(
        message=Message(
            role="assistant",
            content=(ToolCall(id="c1", name="echo", arguments_error="Expecting ',' delimiter: line 1 column 9"),),
        ),
        stop_reason=StopReason.TOOL_USE,
    )
    agent, provider, _ = make_agent([broken, text_response("fixed")], [echo])
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.DONE
    assert echo.calls == 0
    result = provider.requests[1].messages[-1].tool_results[0]
    assert result.is_error
    assert "not valid JSON" in result.content
    assert_consistent(state)


def test_arguments_error_excludes_arguments() -> None:
    with pytest.raises(ValidationError):
        ToolCall(id="c1", name="echo", arguments={"a": 1}, arguments_error="broken")


# ------------------------------------------------------------------ Schema v2


def test_schema_is_v3_with_migrations() -> None:
    assert SCHEMA_VERSION == 3
    assert 1 in MIGRATIONS and 2 in MIGRATIONS


def test_v2_fixture_crashed_mid_batch_recovers_with_the_finished_result(tmp_path: Path) -> None:
    """Real v2 checkpoint: tool 1 finished, crash in tool 2. Recovery preserves result 1."""
    from ai_agent_harness import JsonFileCheckpointStore

    from .test_edge_cases import CrashAfterSideEffect

    # "customer-1" is the tenant_id recorded inside the immutable v2 fixture file
    (tmp_path / "customer-1").mkdir()
    (tmp_path / "customer-1" / "fixture-v2.json").write_bytes(
        (FIXTURES_DIR / "checkpoint_v2_crashed_mid_batch.json").read_bytes()
    )
    store = JsonFileCheckpointStore(tmp_path)
    crashed = run_sync(store.load("customer-1", "fixture-v2"))
    assert crashed is not None
    assert crashed.status is RunStatus.RUNNING
    assert [r.content for r in crashed.batch_done] == ["echo"]
    assert crashed.usage.cache_read_tokens > 0

    agent, provider, _ = make_agent(
        [text_response("checked")], [Echo(), CrashAfterSideEffect()], checkpoints=store, autonomy=Autonomy.AUTONOMOUS
    )
    agent = dataclasses.replace(agent, name="fixture")  # fixture belongs to agent "fixture"
    done = run_sync(resume(agent, crashed, recover=True))
    assert done.status is RunStatus.DONE
    first, second = provider.requests[0].messages[-1].tool_results
    assert first.content == "echo"
    assert "unknown" in second.content
