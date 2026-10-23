"""Configuration validation, context sources, logging listener, and checkpoint write failures."""

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
from typing import Any

import pytest

from ai_agent_harness import ContextBuilder, Limits, Risk, RunState, restore_state, start, stream_start, tool
from ai_agent_harness.core import ConfigError, Event, EventType, SchemaVersionError
from ai_agent_harness.observability import log_listener
from ai_agent_harness.persistence import JsonFileCheckpointStore
from ai_agent_harness.testing import text_response

from .conftest import make_agent, run_sync


@pytest.mark.parametrize(
    "kwargs",
    [{"max_steps": 0}, {"max_total_tokens": 0}, {"max_seconds": 0}, {"max_consecutive_retries": -1}],
)
def test_invalid_limits(kwargs: dict[str, int]) -> None:
    with pytest.raises(ConfigError):
        Limits(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_seconds": float("nan")},
        {"max_seconds": float("inf")},
        {"max_steps": float("nan")},
        {"max_consecutive_retries": float("nan")},
        {"max_steps": True},
        {"max_consecutive_retries": False},
        {"max_seconds": "600"},
    ],
)
def test_limits_reject_values_that_are_not_finite_numbers(kwargs: dict[str, Any]) -> None:
    """NaN compares false with everything, so `max_seconds=nan` would silently disable the time limit."""
    with pytest.raises(ConfigError):
        Limits(**kwargs)


def test_limits_accept_huge_integers() -> None:
    """`math.isfinite(10**400)` raises OverflowError; integers are always finite and must not crash validation."""
    assert Limits(max_steps=10**400, max_seconds=10**400).max_steps == 10**400


def test_invalid_context_builder() -> None:
    with pytest.raises(ConfigError):
        ContextBuilder("  ")
    with pytest.raises(ConfigError):
        ContextBuilder("s", max_tokens=0)


def test_invalid_agent_settings() -> None:
    with pytest.raises(ConfigError):
        make_agent([], max_result_chars=10)


def test_context_sources_are_appended_in_order() -> None:
    class Memory:
        async def render(self, state: RunState) -> str | None:
            return "User prefers concise answers."

    class Empty:
        async def render(self, state: RunState) -> str | None:
            return None

    agent, provider, _ = make_agent([text_response("ok")])
    agent = dataclasses.replace(agent, context=ContextBuilder("System.", sources=(Memory(), Empty())))
    run_sync(start(agent, "hi"))
    assert provider.requests[0].system == "System.\n\nUser prefers concise answers."


def test_log_listener_writes_structured_record(caplog: pytest.LogCaptureFixture) -> None:
    event = Event(type=EventType.RUN_STARTED, run_id="r1", tenant_id="t1", step=0, data={"agent": "a"})
    with caplog.at_level(logging.INFO, logger="ai_agent_harness.run"):
        run_sync(log_listener(event))
    record = caplog.records[0]
    assert "run_started" in record.getMessage()
    assert record.harness_event["tenant_id"] == "t1"  # type: ignore[attr-defined]


def test_failed_atomic_write_leaves_old_checkpoint_and_no_temp_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonFileCheckpointStore(tmp_path)
    state = RunState(agent_name="a", run_id="r1")
    state.add_user_text("first revision")
    run_sync(store.save(state))

    def broken_replace(src: str, dst: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken_replace)
    state.add_user_text(" - second revision")
    with pytest.raises(OSError, match="disk full"):
        run_sync(store.save(state))
    monkeypatch.undo()

    assert sorted(p.name for p in (tmp_path / "default").iterdir()) == [".lock", "r1.json"]
    loaded = run_sync(store.load("default", "r1"))
    assert loaded is not None
    assert loaded.messages[0].text == "first revision"
    assert state.revision == 1  # failed save does not increment revision


def test_unusable_checkpoint_directory_is_config_error(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(ConfigError):
        JsonFileCheckpointStore(blocker / "subdir")


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), 0, -1, True])
def test_a_tool_timeout_must_be_a_finite_positive_number(timeout: float) -> None:
    with pytest.raises(ConfigError, match="timeout_s"):
        tool(risk=Risk.READ, timeout_s=timeout)(_noop)


@pytest.mark.parametrize("idle_timeout", [float("nan"), float("inf"), 0])
def test_a_stream_idle_timeout_must_be_a_finite_positive_number(idle_timeout: float) -> None:
    agent, _, _ = make_agent([text_response("hi")])
    with pytest.raises(ConfigError, match="idle_timeout_s"):
        stream_start(agent, "hi", idle_timeout_s=idle_timeout)


def _noop() -> str:
    """Does nothing."""
    return "ok"


def test_huge_integers_are_valid_positive_numbers() -> None:
    assert tool(risk=Risk.READ, timeout_s=10**400, max_result_chars=10**400)(_noop).timeout_s == 10**400


def test_a_huge_schema_version_is_a_schema_version_error() -> None:
    with pytest.raises(SchemaVersionError):
        restore_state('{"schema_version": ' + "9" * 400 + "}")
