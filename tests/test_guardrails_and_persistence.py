from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from pydantic import BaseModel

from ai_agent_harness import (
    Autonomy,
    HookPipeline,
    JsonFileCheckpointStore,
    ProtectedPaths,
    Risk,
    RunState,
    start,
)
from ai_agent_harness.core import (
    ConcurrencyError,
    ConfigError,
    InvalidStateError,
    ProtocolError,
    ToolCall,
    ToolInvocation,
)
from ai_agent_harness.core.hooks import Action
from ai_agent_harness.persistence import CheckpointStore
from ai_agent_harness.testing import text_response, tool_response

from .conftest import Echo, PathInput, WriteFile, make_agent, results_of, run_sync


class NestedInput(BaseModel):
    files: list[str]
    content: str = ""


class FilesByPath(BaseModel):
    files: dict[str, str]


def test_paths_used_as_dictionary_keys_are_checked() -> None:
    """Common tool pattern `write_files(files={path: content})` — the path is the dict key."""
    guard = ProtectedPaths()
    assert decide(guard, FilesByPath(files={"tests/test_x.py": "def test(): assert True"})) is Action.BLOCK
    assert decide(guard, FilesByPath(files={"conftest.py": "x"})) is Action.BLOCK
    assert decide(guard, FilesByPath(files={"src/a.py": "print(1)"})) is Action.CONTINUE


class Options(BaseModel):
    tests: bool = True


class JobInput(BaseModel):
    options: Options
    jobs: dict[str, str]


def test_ordinary_keys_and_nested_field_names_are_not_paths() -> None:
    """Only keys that look like paths (`/`, `\\`, `.`) count as paths — not `tests` as a plain key name."""
    guard = ProtectedPaths()
    assert decide(guard, JobInput(options=Options(), jobs={"tests": "pytest -q"})) is Action.CONTINUE
    assert decide(guard, JobInput(options=Options(), jobs={"tests/x.py": "pytest -q"})) is Action.BLOCK


def decide(guard: ProtectedPaths, args: BaseModel, risk: Risk = Risk.WRITE) -> Action:
    invocation = ToolInvocation(call=ToolCall(id="1", name="w"), tool_name="w", risk=risk, args=args)
    return run_sync(guard.before_tool(RunState(agent_name="t"), invocation)).action


class TestProtectedPaths:
    """Regression K5: case-insensitivity, ./, .., absolute paths, and alternative argument names."""

    @pytest.mark.parametrize(
        "path",
        [
            "tests/test_a.py",
            "Tests/test_a.py",
            "TESTS/test_a.py",
            "./tests/test_a.py",
            "src/../tests/test_a.py",
            "/repo/project/tests/test_a.py",
            "tests\\test_a.py",
            "tests",
            ".github/workflows/ci.yml",
            "./.GitHub/x",
            "pkg/evals/data.json",
            "config/THRESHOLDS.yaml",
            "pyproject.toml",
            "sub/conftest.py",
        ],
    )
    def test_protected(self, path: str) -> None:
        assert decide(ProtectedPaths(), PathInput(path=path)) is Action.BLOCK

    @pytest.mark.parametrize("path", ["src/app.py", "contests/x.py", "docs/testing.md", "latest/a.py"])
    def test_allowed(self, path: str) -> None:
        assert decide(ProtectedPaths(), PathInput(path=path)) is Action.CONTINUE

    def test_lists_and_other_key_names_are_checked(self) -> None:
        assert decide(ProtectedPaths(), NestedInput(files=["src/a.py", "tests/b.py"])) is Action.BLOCK

    def test_content_mentioning_tests_is_not_a_path(self) -> None:
        assert decide(ProtectedPaths(), PathInput(path="src/a.py", content="see tests/x.py")) is Action.CONTINUE

    def test_read_tools_are_ignored(self) -> None:
        assert decide(ProtectedPaths(), PathInput(path="tests/a.py"), risk=Risk.READ) is Action.CONTINUE

    def test_integrated_in_agent(self) -> None:
        write = WriteFile()
        agent, _, _ = make_agent(
            [tool_response(("write_file", {"path": "Tests/test_x.py"})), text_response("ok")],
            [write],
            autonomy=Autonomy.AUTONOMOUS,
            hooks=HookPipeline([ProtectedPaths()]),
        )
        state = run_sync(start(agent, "make the tests pass"))
        assert write.written == []
        assert "protected" in results_of(state)[0].content


class TestCheckpoints:
    def test_roundtrip_and_revision(self, any_store: CheckpointStore) -> None:
        state = RunState(agent_name="a", tenant_id="tenant1")
        state.add_user_text("hi")
        run_sync(any_store.save(state))
        run_sync(any_store.save(state))
        loaded = run_sync(any_store.load("tenant1", state.run_id))
        assert loaded is not None and loaded.revision == 2 == state.revision
        assert loaded.messages == state.messages

    def test_stale_copy_cannot_overwrite(self, any_store: CheckpointStore) -> None:
        state = RunState(agent_name="a")
        run_sync(any_store.save(state))
        stale = state.model_copy(deep=True)
        run_sync(any_store.save(state))
        with pytest.raises(ConcurrencyError):
            run_sync(any_store.save(stale))

    def test_tenants_are_separated(self, any_store: CheckpointStore) -> None:
        state = RunState(agent_name="a", tenant_id="tenant1", run_id="r1")
        run_sync(any_store.save(state))
        assert run_sync(any_store.load("tenant2", "r1")) is None

    @pytest.mark.parametrize(("tenant", "run"), [("../x", "r"), ("t", "../../escape"), ("t", "a/b")])
    def test_invalid_ids_are_rejected(self, any_store: CheckpointStore, tenant: str, run: str) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            run_sync(any_store.load(tenant, run))

    def test_file_store_writes_atomically_without_leftovers(self, tmp_path: Path) -> None:
        file_store = JsonFileCheckpointStore(tmp_path)
        state = RunState(agent_name="a", run_id="r1")
        run_sync(file_store.save(state))
        names = sorted(p.name for p in (tmp_path / "default").iterdir())
        assert names == [".lock", "r1.json"]

    def test_file_store_refuses_platforms_without_file_locks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without `fcntl` (e.g. Windows) there is no cross-process lock, so two workers could run one action twice.

        The store must refuse to start instead of silently running without locks.
        """
        import ai_agent_harness.persistence

        monkeypatch.setattr(ai_agent_harness.persistence, "fcntl", None)
        with pytest.raises(ConfigError, match="fcntl"):
            JsonFileCheckpointStore(tmp_path)

    def test_file_store_keeps_one_lock_per_tenant_not_per_run(self, tmp_path: Path) -> None:
        """Otherwise a lock file per run would remain on disk forever (Finding 2.4)."""
        file_store = JsonFileCheckpointStore(tmp_path)
        for i in range(30):
            run_sync(file_store.save(RunState(agent_name="a", run_id=f"r{i}")))
        locks = [p.name for p in (tmp_path / "default").iterdir() if p.suffix == ".lock" or p.name == ".lock"]
        assert locks == [".lock"]

    def test_checkpoint_copied_to_another_tenant_is_rejected(self, tmp_path: Path) -> None:
        """Copying files across tenant directories must not allow loading a foreign run."""
        file_store = JsonFileCheckpointStore(tmp_path)
        state = RunState(agent_name="a", tenant_id="tenant1", run_id="r1")
        run_sync(file_store.save(state))
        (tmp_path / "tenant2").mkdir()
        shutil.copy(tmp_path / "tenant1" / "r1.json", tmp_path / "tenant2" / "r1.json")
        shutil.copy(tmp_path / "tenant1" / "r1.json", tmp_path / "tenant1" / "r2.json")
        with pytest.raises(InvalidStateError, match="belongs"):
            run_sync(file_store.load("tenant2", "r1"))
        with pytest.raises(InvalidStateError, match="belongs"):
            run_sync(file_store.load("tenant1", "r2"))

    def test_save_never_overwrites_another_runs_file(self, tmp_path: Path) -> None:
        """On macOS/Windows `Customer`/`customer` share the same file — saving must not overwrite a foreign run."""
        file_store = JsonFileCheckpointStore(tmp_path)
        other = RunState(agent_name="a", tenant_id="Customer", run_id="r1")
        run_sync(file_store.save(other))
        (tmp_path / "customer").mkdir(exist_ok=True)
        target = tmp_path / "customer" / "r1.json"
        if not target.exists():  # case-sensitive filesystem: simulate collision
            shutil.copy(tmp_path / "Customer" / "r1.json", target)
        mine = RunState(agent_name="a", tenant_id="customer", run_id="r1", revision=other.revision)
        with pytest.raises(InvalidStateError, match="belongs"):
            run_sync(file_store.save(mine))
        assert json.loads(target.read_text(encoding="utf-8"))["tenant_id"] == "Customer"

    def test_tampered_checkpoint_is_rejected(self, tmp_path: Path) -> None:
        file_store = JsonFileCheckpointStore(tmp_path)
        state = RunState(agent_name="a", run_id="r1")
        state.add_user_text("hi")
        run_sync(file_store.save(state))
        path = tmp_path / "default" / "r1.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["messages"].append(data["messages"][0])  # two consecutive user messages
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ProtocolError):
            run_sync(file_store.load("default", "r1"))


class TestProtectedPathsUnknownKeys:
    """Finding 2.6: paths under arbitrary field names (location, save_to, uri, ...) must not slip through."""

    class Loc(BaseModel):
        location: str
        note: str = ""

    @pytest.mark.parametrize(
        "value", ["tests/test_x.py", "pyproject.toml", "./.github/workflows/ci.yml", "file:///repo/tests/a.py"]
    )
    def test_path_like_values_under_any_key_are_checked(self, value: str) -> None:
        assert decide(ProtectedPaths(), self.Loc(location=value)) is Action.BLOCK

    def test_free_text_with_spaces_stays_allowed(self) -> None:
        assert decide(ProtectedPaths(), self.Loc(location="src/a.py", note="see tests/x.py later")) is Action.CONTINUE

    def test_extra_keys_cover_paths_with_spaces(self) -> None:
        guard = ProtectedPaths(extra_keys=["location"])
        assert decide(guard, self.Loc(location="my folder/tests/a.py")) is Action.BLOCK


@pytest.mark.parametrize("scenario", ["once", "twice", "foreign_writer"])
def test_cancel_during_checkpoint_write_stays_a_clean_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    """Cancellation while the writer thread is running: always CancelledError, never ConcurrencyError.

    - once / twice: disk ends up with 'cancelled' and matching revision.
    - foreign_writer: another worker took over the run in the meantime — do not overwrite.
    """
    import asyncio
    import threading
    from collections.abc import Callable
    from typing import Any

    from ai_agent_harness import RunStatus, start

    original: Callable[..., Any] = JsonFileCheckpointStore.__dict__["_save_sync"]
    written, release = threading.Event(), threading.Event()
    count = {"n": 0}

    def slow_save(self: JsonFileCheckpointStore, tenant_id: str, run_id: str, expected: int, payload: str) -> None:
        original(self, tenant_id, run_id, expected, payload)
        count["n"] += 1
        if count["n"] == 2:  # write-ahead before tool: file written, revision not yet updated in memory
            if scenario == "foreign_writer":
                path = tmp_path / tenant_id / f"{run_id}.json"
                data = json.loads(path.read_text(encoding="utf-8"))
                data["revision"] += 1
                path.write_text(json.dumps(data), encoding="utf-8")
            written.set()
            release.wait(5)

    monkeypatch.setattr(JsonFileCheckpointStore, "_save_sync", slow_save)
    store = JsonFileCheckpointStore(tmp_path)
    agent, _, _ = make_agent([tool_response(("echo", {})), text_response("never")], [Echo()], checkpoints=store)

    async def main() -> None:
        task = asyncio.create_task(start(agent, "go", run_id="c1"))
        await asyncio.to_thread(written.wait, 5)
        task.cancel()
        if scenario == "twice":
            await asyncio.sleep(0.02)
            task.cancel()
        await asyncio.sleep(0.02)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    loaded = run_sync(store.load("default", "c1"))
    assert loaded is not None
    expected = RunStatus.RUNNING if scenario == "foreign_writer" else RunStatus.CANCELLED
    assert loaded.status is expected
