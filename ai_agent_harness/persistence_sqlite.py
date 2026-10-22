"""SQLite checkpoint store for local, embedded, and single-host runs.

Part of standard library (zero extra dependencies). Meant for Linux, macOS, and Windows; CI only runs Linux (ADR 0020).

    store = SqliteCheckpointStore("checkpoints.sqlite")
    await store.setup()                       # CREATE TABLE IF NOT EXISTS (or run `schema_sql()`)
    agent = Agent(..., checkpoints=store)

How it stays safe:
- **One atomic statement per save.** `INSERT ... ON CONFLICT DO UPDATE ... WHERE revision = expected RETURNING revision`
  either stores the new state or changes nothing; no row returned means someone else saved first (`ConcurrencyError`).
- **Cancellation cannot desynchronize revisions** (shared logic in `persistence.settle_write`).
- **Loads go through `restore_state`**: schema migration, ownership check, and protocol check.
- **WAL mode enabled by default** (`PRAGMA journal_mode = WAL`, `PRAGMA busy_timeout = 5000`) allowing concurrent
  reads and handling concurrent writes cleanly across processes.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from .core.errors import ConcurrencyError, ConfigError
from .core.state import RunState, RunStatus, restore_state
from .persistence import RunInfo, check_ids, check_list_limit, check_tenant_id, settle_write
from .persistence_sql import DEFAULT_TABLE, WriteRecord, sql_identifier, statements, status_index_name

_NOW = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
_SQL = statements("?", _NOW)


def _create_statements(table: str) -> list[str]:
    tbl = sql_identifier(table, "table")
    idx = status_index_name(tbl)
    return [
        f"CREATE TABLE IF NOT EXISTS {tbl} ("
        "tenant_id TEXT NOT NULL, "
        "run_id TEXT NOT NULL, "
        "revision INTEGER NOT NULL, "
        "status TEXT NOT NULL, "
        "schema_version INTEGER NOT NULL, "
        "state TEXT NOT NULL, "
        f"updated_at TEXT NOT NULL DEFAULT ({_NOW}), "
        "PRIMARY KEY (tenant_id, run_id))",
        f"CREATE INDEX IF NOT EXISTS {idx} ON {tbl} (tenant_id, status)",
    ]


def schema_sql(table: str = DEFAULT_TABLE) -> str:
    """The DDL that `setup()` runs, for teams that manage their schema with a migration tool."""
    return ";\n".join(_create_statements(table)) + ";\n"


class SqliteCheckpointStore:
    """`CheckpointStore` backed by SQLite. Pass a file path or an open `sqlite3.Connection`."""

    def __init__(
        self,
        database: str | Path | None = None,
        *,
        connection: sqlite3.Connection | None = None,
        table: str = DEFAULT_TABLE,
        busy_timeout_ms: int = 5000,
    ) -> None:
        self._lock = threading.Lock()
        self._closed = False
        self._owns_connection = connection is None

        if database is not None and connection is not None:
            raise ConfigError("SqliteCheckpointStore needs at most one of database or connection")

        self._table = sql_identifier(table, "table")
        # Own connections apply these on first use in a worker thread (`journal_mode = WAL` can wait for another
        # process's write lock). A borrowed connection gets them now, while the caller has no transaction open.
        self._pragmas: tuple[str, ...] = (
            f"PRAGMA busy_timeout = {int(busy_timeout_ms)}",
            "PRAGMA journal_mode = WAL",
            "PRAGMA synchronous = NORMAL",
        )

        if connection is not None:
            self._connection = connection
            self._apply_pragmas()
        else:
            target = str(database if database is not None else "checkpoints.sqlite")
            if target != ":memory:" and not target.startswith("file:"):
                db_path = Path(target).resolve()
                try:
                    db_path.parent.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    raise ConfigError(f"Database directory is not usable: {db_path.parent}") from exc
                target = str(db_path)
            self._connection = sqlite3.connect(target, check_same_thread=False)

    def _apply_pragmas(self) -> None:
        """Runs the settings once; after a failure (for example a locked database) the next call tries again."""
        for pragma in self._pragmas:
            self._connection.execute(pragma)
        self._pragmas = ()

    def _sql(self, template: str) -> str:
        return template.format(table=self._table)  # validated by `sql_identifier`

    def _run(self, query: str, params: tuple[Any, ...] = (), *, commit: bool = False) -> tuple[list[Any], int]:
        """Runs one statement in the calling worker thread. Returns the rows and the row count."""
        if self._closed:
            raise ConfigError("SqliteCheckpointStore is closed")
        with self._lock:
            self._apply_pragmas()
            cursor = self._connection.execute(query, params)
            rows = cursor.fetchall()
            if commit:
                self._connection.commit()
            return rows, cursor.rowcount

    def close(self) -> None:
        """Closes the underlying SQLite connection if created by this store."""
        if not hasattr(self, "_lock"):
            return
        with self._lock:
            if not self._closed:
                self._closed = True
                if self._owns_connection and hasattr(self, "_connection"):
                    self._connection.close()

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> SqliteCheckpointStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    async def __aenter__(self) -> SqliteCheckpointStore:
        return self

    async def __aexit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------ Setup

    async def setup(self) -> None:
        """Creates the table and index if they do not exist (safe to call on every start)."""
        await asyncio.to_thread(self._setup_sync)

    def _setup_sync(self) -> None:
        for statement in _create_statements(self._table):
            self._run(statement, commit=True)

    # ------------------------------------------------------------------ CheckpointStore protocol

    async def save(self, state: RunState) -> None:
        """Saves and increments `state.revision`. Raises `ConcurrencyError` if another worker saved first."""
        check_ids(state.tenant_id, state.run_id)
        record = WriteRecord.from_state(state)
        write = asyncio.ensure_future(asyncio.to_thread(self._write_sync, record))
        await settle_write(write, state, record.new_revision)

    def _write_sync(self, record: WriteRecord) -> None:
        rows, _ = self._run(self._sql(_SQL.upsert), record.params, commit=True)
        if not rows:
            raise ConcurrencyError(f"Run {record.run_id} was modified concurrently")

    async def load(self, tenant_id: str, run_id: str) -> RunState | None:
        check_ids(tenant_id, run_id)
        rows, _ = await asyncio.to_thread(self._run, self._sql(_SQL.load), (tenant_id, run_id))
        return restore_state(str(rows[0][0]), tenant_id=tenant_id, run_id=run_id) if rows else None

    # ------------------------------------------------------------------ Queries and retention

    async def list_runs(self, tenant_id: str, *, status: RunStatus | None = None, limit: int = 100) -> list[RunInfo]:
        """Runs of one tenant, newest first."""
        check_tenant_id(tenant_id)
        check_list_limit(limit)
        params: tuple[Any, ...] = (tenant_id, limit) if status is None else (tenant_id, str(status), limit)
        query = self._sql(_SQL.list_all) if status is None else self._sql(_SQL.list_status)
        rows, _ = await asyncio.to_thread(self._run, query, params)
        return [
            RunInfo(
                run_id=str(r[0]),
                status=RunStatus(str(r[1])),
                revision=int(r[2]),
                updated_at=datetime.fromisoformat(str(r[3]).replace("Z", "+00:00")),
            )
            for r in rows
        ]

    async def delete_run(self, tenant_id: str, run_id: str) -> bool:
        """Deletes one run (retention policy, right to erasure). Returns whether a run existed."""
        check_ids(tenant_id, run_id)
        _, count = await asyncio.to_thread(self._run, self._sql(_SQL.delete_run), (tenant_id, run_id), commit=True)
        return count > 0

    async def delete_tenant(self, tenant_id: str) -> int:
        """Deletes all runs of a tenant. Returns how many were deleted."""
        check_tenant_id(tenant_id)
        _, count = await asyncio.to_thread(self._run, self._sql(_SQL.delete_tenant), (tenant_id,), commit=True)
        return count
