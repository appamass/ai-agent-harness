"""Checkpoints: persist state to resume after a crash, deployment, or user prompt.

Guarantees of both implementations:
- Tenant isolation: the key is always `(tenant_id, run_id)`.
- Optimistic locking: `save` fails with `ConcurrencyError` if another process has saved
  the run in the meantime. Two workers cannot silently resume the same run twice.
- Loaded states are verified for protocol consistency.

`JsonFileCheckpointStore` writes atomically (temporary file + `os.replace` + fsync),
does not block the event loop, and locks across processes via `fcntl`. It needs a POSIX system
(Linux, macOS): without `fcntl` (e.g. Windows) it raises `ConfigError` instead of running without locks.
`SqliteCheckpointStore` (one host, stdlib) and `PostgresCheckpointStore` (several workers) use one conditional
upsert (`INSERT ... ON CONFLICT DO UPDATE ... WHERE revision = :expected RETURNING revision`).
Custom stores must always load states via
`restore_state` (schema version, migration, protocol check) — never directly via `model_validate`.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Protocol

from .core.errors import ConcurrencyError, ConfigError, InvalidStateError
from .core.state import RunState, RunStatus, restore_state
from .core.types import ID_PATTERN

_ID = re.compile(ID_PATTERN)
_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]{0,62}")

DEFAULT_TABLE: Final = "ai_agent_harness_checkpoints"
MAX_LIST_LIMIT: Final = 10_000

try:  # POSIX file locks; without fcntl (Windows), JsonFileCheckpointStore refuses to start
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


class CheckpointStore(Protocol):
    async def save(self, state: RunState) -> None:
        """Saves and increments `state.revision`. Raises `ConcurrencyError` on conflict."""
        ...

    async def load(self, tenant_id: str, run_id: str) -> RunState | None: ...


@dataclass(frozen=True)
class RunInfo:
    """Metadata of a stored run, for approval queues and recovery jobs. Content stays in the state itself."""

    run_id: str
    status: RunStatus
    revision: int
    updated_at: datetime


@dataclass(frozen=True)
class WriteRecord:
    """One checkpoint write for the SQL stores: the new row values plus the revision the row must still have."""

    tenant_id: str
    run_id: str
    expected: int
    new_revision: int
    payload: str
    status: str
    schema_version: int

    @classmethod
    def from_state(cls, state: RunState) -> WriteRecord:
        new_revision = state.revision + 1
        return cls(
            tenant_id=state.tenant_id,
            run_id=state.run_id,
            expected=state.revision,
            new_revision=new_revision,
            payload=state.model_copy(update={"revision": new_revision}).model_dump_json(),
            status=str(state.status),
            schema_version=state.schema_version,
        )

    @property
    def params(self) -> tuple[str, str, int, str, int, str, int]:
        """Values in the order of the stores' upsert statement; the last one is the expected revision."""
        return (
            self.tenant_id,
            self.run_id,
            self.new_revision,
            self.status,
            self.schema_version,
            self.payload,
            self.expected,
        )


def sql_identifier(value: str, what: str) -> str:
    """Table and schema names go into DDL text, so only plain lower-case identifiers are accepted."""
    if not _IDENTIFIER.fullmatch(value):
        raise ConfigError(f"Invalid {what} identifier {value!r}: use 1-63 lower-case letters, digits, or underscores")
    return value


def status_index_name(table: str) -> str:
    """Names are cut at 63 characters: a short hash keeps long tables with a shared prefix from sharing an index."""
    return f"{table[:30]}_{hashlib.sha256(table.encode()).hexdigest()[:8]}_status_idx"


def check_tenant_id(tenant_id: str) -> None:
    if not _ID.fullmatch(tenant_id):
        raise ValueError(f"Invalid tenant_id: {tenant_id!r}")


def check_list_limit(limit: int) -> None:
    if not 1 <= limit <= MAX_LIST_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIST_LIMIT}")


def check_ids(tenant_id: str, run_id: str) -> None:
    for label, value in (("tenant_id", tenant_id), ("run_id", run_id)):
        if not _ID.fullmatch(value):
            raise ValueError(f"Invalid {label}: {value!r}")


async def settle_write(write: asyncio.Future[None], state: RunState, new_revision: int) -> None:
    """Waits for a checkpoint write to finish, even across cancellation, and syncs the in-memory revision.

    A write that was already started may still complete after the caller is cancelled. In-memory and stored
    revisions must stay in sync afterward, otherwise saving the status 'cancelled' would fail against our own lock
    (ADR 0006). So the write is awaited through repeated cancellations before the cancellation is re-raised.
    Used by every store whose write runs outside the event loop's control (threads, database round trips).
    """
    cancelled = False
    while not write.done():
        try:
            await asyncio.wait({write})  # does not cancel the write or raise its exception yet
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        if write.exception() is None:
            state.revision = new_revision
        raise asyncio.CancelledError
    write.result()  # raises e.g. ConcurrencyError
    state.revision = new_revision


class InMemoryCheckpointStore:
    """For tests and single-process runs that do not require disk persistence."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], tuple[int, str]] = {}
        self._lock = asyncio.Lock()

    async def save(self, state: RunState) -> None:
        key = (state.tenant_id, state.run_id)
        async with self._lock:
            stored = self._data.get(key)
            if stored is not None and stored[0] != state.revision:
                raise ConcurrencyError(f"Run {state.run_id} was modified concurrently")
            new_revision = state.revision + 1
            self._data[key] = (new_revision, state.model_copy(update={"revision": new_revision}).model_dump_json())
            state.revision = new_revision

    async def load(self, tenant_id: str, run_id: str) -> RunState | None:
        check_ids(tenant_id, run_id)
        stored = self._data.get((tenant_id, run_id))
        return restore_state(stored[1], tenant_id=tenant_id, run_id=run_id) if stored else None


class JsonFileCheckpointStore:
    def __init__(self, directory: str | Path) -> None:
        if not _file_locks_available():  # without a cross-process lock, two workers could repeat one action
            raise ConfigError(
                "JsonFileCheckpointStore needs POSIX file locks (fcntl), which this platform lacks. "
                "Use a database-backed CheckpointStore instead."
            )
        self._root = Path(directory).resolve()
        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigError(f"Checkpoint directory is not usable: {self._root}") from exc

    async def save(self, state: RunState) -> None:
        new_revision = state.revision + 1
        payload = state.model_copy(update={"revision": new_revision}).model_dump_json(indent=2)
        write = asyncio.ensure_future(
            asyncio.to_thread(self._save_sync, state.tenant_id, state.run_id, state.revision, payload)
        )
        await settle_write(write, state, new_revision)

    async def load(self, tenant_id: str, run_id: str) -> RunState | None:
        check_ids(tenant_id, run_id)
        path = self._path(tenant_id, run_id)
        raw = await asyncio.to_thread(lambda: path.read_text(encoding="utf-8") if path.exists() else None)
        return restore_state(raw, tenant_id=tenant_id, run_id=run_id) if raw is not None else None

    def _path(self, tenant_id: str, run_id: str) -> Path:
        path = (self._root / tenant_id / f"{run_id}.json").resolve()
        if not path.is_relative_to(self._root):  # defense in depth; IDs are already validated
            raise ValueError("Checkpoint path resolves outside the storage directory")
        return path

    def _save_sync(self, tenant_id: str, run_id: str, expected_revision: int, payload: str) -> None:
        check_ids(tenant_id, run_id)
        path = self._path(tenant_id, run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _file_lock(path.parent / ".lock"):  # one lock per tenant: avoids leaving per-run lock files behind
            if path.exists():
                stored = json.loads(path.read_text(encoding="utf-8"))
                if (stored.get("tenant_id"), stored.get("run_id")) != (tenant_id, run_id):
                    # e.g. `Customer`/`customer` on macOS/Windows: same file, different run
                    raise InvalidStateError(
                        f"Checkpoint file for {tenant_id}/{run_id} belongs to a different run "
                        "(case-insensitive filesystem?)"
                    )
                if stored.get("revision") != expected_revision:
                    raise ConcurrencyError(f"Run {run_id} was modified concurrently")
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{run_id}.", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, path)
            except BaseException:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(tmp)
                raise
            _fsync_dir(path.parent)


def _file_locks_available() -> bool:
    return fcntl is not None


@contextlib.contextmanager
def _file_lock(lock_path: Path) -> Generator[None]:
    assert fcntl is not None, "checked in JsonFileCheckpointStore.__init__"
    with open(lock_path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fsync_dir(directory: Path) -> None:
    if os.name != "posix":  # pragma: no cover
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
