"""Schema versioning of persisted states (ADR 0010).

The fixture `checkpoint_v1_waiting_for_approval.json` is a real run persisted under schema
version 1 while waiting for approval. It must never be modified: every future template version
must be able to load and resume it — otherwise runs waiting in production would be lost on upgrade.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from ai_agent_harness import (
    Agent,
    Autonomy,
    ContextBuilder,
    JsonFileCheckpointStore,
    ModelRouter,
    RunState,
    RunStatus,
    ToolRegistry,
    restore_state,
    resume,
)
from ai_agent_harness.core import InvalidStateError, SchemaVersionError
from ai_agent_harness.core import state as state_module
from ai_agent_harness.core.types import canonical_digest
from ai_agent_harness.testing import FakeProvider, text_response

from .conftest import FIXTURES_DIR, Echo, WriteFile, assert_consistent, run_sync

FIXTURE = FIXTURES_DIR / "checkpoint_v1_waiting_for_approval.json"
FIXTURE_V2 = FIXTURES_DIR / "checkpoint_v2_waiting_for_approval.json"
FIXTURE_V3 = FIXTURES_DIR / "checkpoint_v3_done_with_output.json"
FIXTURE_V4 = FIXTURES_DIR / "checkpoint_v4_done_with_output.json"


def load_fixture(path: Path) -> RunState:
    return restore_state(path.read_text(encoding="utf-8"))


def _agent(write: WriteFile, store: JsonFileCheckpointStore | None = None) -> Agent:
    return Agent(
        name="fixture",
        router=ModelRouter(FakeProvider([text_response("completed")])),
        context=ContextBuilder("s"),
        tools=ToolRegistry([Echo(), write]),
        autonomy=Autonomy.SUGGEST,
        checkpoints=store,
    )


def test_v1_fixture_still_loads_and_resumes(tmp_path: Path) -> None:
    """Just like in production: old file sits in the store, a new worker loads, resumes, and saves it."""
    # "customer-1" is the tenant_id recorded inside the immutable v1 fixture file
    (tmp_path / "customer-1").mkdir()
    shutil.copy(FIXTURE, tmp_path / "customer-1" / "fixture-v1.json")
    store = JsonFileCheckpointStore(tmp_path)
    state = run_sync(store.load("customer-1", "fixture-v1"))
    assert state is not None
    assert state.schema_version == state_module.SCHEMA_VERSION
    assert state.status is RunStatus.WAITING_FOR_APPROVAL

    write = WriteFile()
    done = run_sync(resume(_agent(write, store), state, approved=True, pending_id=state.pending_id))
    assert done.status is RunStatus.DONE
    assert write.written == ["a.py"]
    assert_consistent(done)
    reloaded = run_sync(store.load("customer-1", "fixture-v1"))
    assert reloaded is not None
    assert reloaded.model_dump() == done.model_dump()


def test_v2_fixture_still_loads_and_resumes(tmp_path: Path) -> None:
    """Verifies that real v2 checkpoints load, migrate to the current version, and resume cleanly."""
    (tmp_path / "customer-1").mkdir()
    shutil.copy(FIXTURE_V2, tmp_path / "customer-1" / "fixture-v2-waiting.json")
    store = JsonFileCheckpointStore(tmp_path)
    state = run_sync(store.load("customer-1", "fixture-v2-waiting"))
    assert state is not None
    assert state.schema_version == state_module.SCHEMA_VERSION
    assert state.status is RunStatus.WAITING_FOR_APPROVAL

    write = WriteFile()
    done = run_sync(resume(_agent(write, store), state, approved=True, pending_id=state.pending_id))
    assert done.status is RunStatus.DONE
    assert write.written == ["a.py"]
    assert_consistent(done)
    reloaded = run_sync(store.load("customer-1", "fixture-v2-waiting"))
    assert reloaded is not None
    assert reloaded.model_dump() == done.model_dump()


@pytest.mark.parametrize("fixture", [FIXTURE_V3, FIXTURE_V4], ids=["v3", "v4"])
def test_a_fixture_keeps_its_structured_output_and_demands_the_same_model_on_resume(fixture: Path) -> None:
    """The v3 format added `output_schema`, `output_model_name`, and `output`; v4 keeps only the schema's digest."""
    from pydantic import BaseModel

    class Weather(BaseModel):
        city: str
        celsius: float

    class Other(BaseModel):
        value: int

    state = load_fixture(fixture)
    assert state.schema_version == state_module.SCHEMA_VERSION and state.status is RunStatus.DONE
    assert state.output_model_name == "Weather"
    assert state.output_as(Weather) == Weather(city="customer-city", celsius=3.5)
    assert state.output_schema_digest == canonical_digest(Weather.model_json_schema())

    write = WriteFile()
    with pytest.raises(InvalidStateError, match="output_model"):
        run_sync(resume(_agent(write), state, user_input="again"))
    with pytest.raises(InvalidStateError, match="does not match"):
        run_sync(resume(_agent(write), state, user_input="again", output_model=Other))


def test_resume_refuses_a_state_from_another_schema_version() -> None:
    """States loaded bypassing `restore_state` (e.g. a broken custom store) must not be allowed to run."""
    state = load_fixture(FIXTURE)
    state.schema_version = state_module.SCHEMA_VERSION + 1
    write = WriteFile()
    with pytest.raises(InvalidStateError, match="schema version"):
        run_sync(resume(_agent(write), state, approved=True, pending_id=state.pending_id))
    assert write.written == []


def test_migration_returning_garbage_is_a_schema_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(data: dict[str, Any]) -> Any:
        return None

    monkeypatch.setattr(state_module, "SCHEMA_VERSION", 2)
    monkeypatch.setattr(state_module, "MIGRATIONS", {1: broken})
    with pytest.raises(SchemaVersionError, match="Migration"):
        load_fixture(FIXTURE)


def test_state_without_version_is_treated_as_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the field, the migration chain starts at version 1."""
    started_at: list[int] = []

    def spy(data: dict[str, Any]) -> dict[str, Any]:
        started_at.append(data.get("schema_version", 1))
        return data

    monkeypatch.setattr(state_module, "MIGRATIONS", {**state_module.MIGRATIONS, 1: spy})
    raw: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw.pop("schema_version")
    assert restore_state(json.dumps(raw)).schema_version == state_module.SCHEMA_VERSION
    assert started_at == [1]


def test_newer_version_is_rejected_with_clear_error() -> None:
    raw: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["schema_version"] = state_module.SCHEMA_VERSION + 1
    with pytest.raises(SchemaVersionError, match="supports up to"):
        restore_state(json.dumps(raw))


@pytest.mark.parametrize("version", [0, -1, "1", True, None])
def test_invalid_version_is_rejected(version: object) -> None:
    raw: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["schema_version"] = version
    with pytest.raises(SchemaVersionError):
        restore_state(json.dumps(raw))


def test_migrations_are_applied_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulates a future upgrade to version 3 with two sequential migrations."""
    calls: list[int] = []

    def v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
        calls.append(1)
        data["status_reason"] = "migrated"
        return data

    def v2_to_v3(data: dict[str, Any]) -> dict[str, Any]:
        calls.append(2)
        return data

    monkeypatch.setattr(state_module, "SCHEMA_VERSION", 3)
    monkeypatch.setattr(state_module, "MIGRATIONS", {1: v1_to_v2, 2: v2_to_v3})
    state = load_fixture(FIXTURE)
    assert calls == [1, 2]
    assert state.schema_version == 3
    assert state.status_reason == "migrated"


def test_missing_migration_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(state_module, "SCHEMA_VERSION", 2)
    monkeypatch.setattr(state_module, "MIGRATIONS", {})
    with pytest.raises(SchemaVersionError, match="No migration"):
        load_fixture(FIXTURE)


def test_non_object_is_rejected() -> None:
    with pytest.raises(SchemaVersionError):
        restore_state("[1, 2, 3]")


def test_migrated_state_resumes_after_a_version_bump(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulates an upgrade to version 2: the migrated state must resume cleanly."""

    def v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
        return data

    monkeypatch.setattr(state_module, "SCHEMA_VERSION", 2)
    monkeypatch.setattr(state_module, "MIGRATIONS", {1: v1_to_v2})
    state = load_fixture(FIXTURE)
    assert state.schema_version == 2
    write = WriteFile()
    done = run_sync(resume(_agent(write), state, approved=True, pending_id=state.pending_id))
    assert done.status is RunStatus.DONE
    assert write.written == ["a.py"]
