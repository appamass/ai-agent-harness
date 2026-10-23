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
import json
import os
import tempfile
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

from .core.errors import ConcurrencyError, ConfigError, InvalidStateError
from .core.state import RunState, RunStatus, restore_state
from .core.types import ID_RE
from .persistence_sql import WriteRecord

if TYPE_CHECKING:
    from .encryption import StateCodec

MAX_LIST_LIMIT: Final = 10_000
ENCRYPTED_PREFIX: Final = "enc:v1:"

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


def check_tenant_id(tenant_id: str) -> None:
    if not ID_RE.fullmatch(tenant_id):
        raise ValueError(f"Invalid tenant_id: {tenant_id!r}")


def check_list_limit(limit: int) -> None:
    if not 1 <= limit <= MAX_LIST_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIST_LIMIT}")


def check_ids(tenant_id: str, run_id: str) -> None:
    for label, value in (("tenant_id", tenant_id), ("run_id", run_id)):
        if not ID_RE.fullmatch(value):
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


def load_state(
    raw: str,
    *,
    tenant_id: str,
    run_id: str,
    codec: StateCodec | None = None,
) -> RunState:
    """Decodes and restores a RunState from raw storage text."""
    check_ids(tenant_id, run_id)
    if codec is not None:
        if not raw.startswith(ENCRYPTED_PREFIX):
            raise InvalidStateError("Unencrypted checkpoint rejected: store is configured with encryption")
        plaintext = codec.decode(raw, tenant_id=tenant_id, run_id=run_id)
        return restore_state(plaintext, tenant_id=tenant_id, run_id=run_id)
    if raw.startswith(ENCRYPTED_PREFIX):
        raise InvalidStateError("Encrypted checkpoint cannot be loaded without a codec")
    return restore_state(raw, tenant_id=tenant_id, run_id=run_id)


def _extract_head(raw: str) -> tuple[str, str, int]:
    if raw.startswith(ENCRYPTED_PREFIX):
        header_json, _, _ = raw[len(ENCRYPTED_PREFIX) :].partition("\n")
        try:
            data = json.loads(header_json)
            return str(data["tenant_id"]), str(data["run_id"]), int(data["revision"])
        except Exception as exc:
            raise InvalidStateError("Corrupted encrypted checkpoint header") from exc
    try:
        data = json.loads(raw)
        return str(data.get("tenant_id")), str(data.get("run_id")), int(data.get("revision", 0))
    except Exception as exc:
        raise InvalidStateError("Corrupted checkpoint file") from exc


class InMemoryCheckpointStore:
    """For tests and single-process runs that do not require disk persistence."""

    def __init__(self, *, codec: StateCodec | None = None) -> None:
        self._data: dict[tuple[str, str], tuple[int, str]] = {}
        self._lock = asyncio.Lock()
        self._codec = codec

    async def save(self, state: RunState) -> None:
        key = (state.tenant_id, state.run_id)
        async with self._lock:
            stored = self._data.get(key)
            if stored is not None and stored[0] != state.revision:
                raise ConcurrencyError(f"Run {state.run_id} was modified concurrently")
            record = WriteRecord.from_state(state, self._codec)
            self._data[key] = (record.new_revision, record.payload)
            state.revision = record.new_revision

    async def load(self, tenant_id: str, run_id: str) -> RunState | None:
        check_ids(tenant_id, run_id)
        stored = self._data.get((tenant_id, run_id))
        return load_state(stored[1], tenant_id=tenant_id, run_id=run_id, codec=self._codec) if stored else None


class JsonFileCheckpointStore:
    def __init__(self, directory: str | Path, *, codec: StateCodec | None = None) -> None:
        if not _file_locks_available():  # without a cross-process lock, two workers could repeat one action
            raise ConfigError(
                "JsonFileCheckpointStore needs POSIX file locks (fcntl), which this platform lacks. "
                "Use a database-backed CheckpointStore instead."
            )
        self._root = Path(directory).resolve()
        self._codec = codec
        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigError(f"Checkpoint directory is not usable: {self._root}") from exc

    async def save(self, state: RunState) -> None:
        check_ids(state.tenant_id, state.run_id)
        new_revision = state.revision + 1
        write = asyncio.ensure_future(
            asyncio.to_thread(self._save_sync, state.tenant_id, state.run_id, state.revision, state)
        )
        await settle_write(write, state, new_revision)

    async def load(self, tenant_id: str, run_id: str) -> RunState | None:
        check_ids(tenant_id, run_id)
        path = self._path(tenant_id, run_id)
        return await asyncio.to_thread(self._load_sync, path, tenant_id, run_id)

    def _load_sync(self, path: Path, tenant_id: str, run_id: str) -> RunState | None:
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8")
        return load_state(raw, tenant_id=tenant_id, run_id=run_id, codec=self._codec)

    def _path(self, tenant_id: str, run_id: str) -> Path:
        path = (self._root / tenant_id / f"{run_id}.json").resolve()
        if not path.is_relative_to(self._root):  # defense in depth; IDs are already validated
            raise ValueError("Checkpoint path resolves outside the storage directory")
        return path

    def _save_sync(self, tenant_id: str, run_id: str, expected_revision: int, payload: str | RunState) -> None:
        check_ids(tenant_id, run_id)
        path = self._path(tenant_id, run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _file_lock(path.parent / ".lock"):  # one lock per tenant: avoids leaving per-run lock files behind
            if path.exists():
                raw_stored = path.read_text(encoding="utf-8")
                stored_tenant_id, stored_run_id, stored_revision = _extract_head(raw_stored)
                if (stored_tenant_id, stored_run_id) != (tenant_id, run_id):
                    # e.g. `Customer`/`customer` on macOS/Windows: same file, different run
                    raise InvalidStateError(
                        f"Checkpoint file for {tenant_id}/{run_id} belongs to a different run "
                        "(case-insensitive filesystem?)"
                    )
                if stored_revision != expected_revision:
                    raise ConcurrencyError(f"Run {run_id} was modified concurrently")
            if isinstance(payload, RunState):
                record = WriteRecord.from_state(payload, self._codec)
                payload_str = record.payload
            else:
                payload_str = payload
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{run_id}.", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload_str)
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
