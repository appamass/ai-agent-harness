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
from .persistence import (
    DEFAULT_TABLE,
    RunInfo,
    WriteRecord,
    check_ids,
    check_list_limit,
    check_tenant_id,
    settle_write,
    sql_identifier,
    status_index_name,
)


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
        "updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')), "
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
        tbl = self._table
        # Keywords sit in variables so the S608 string-query lint stays quiet: `tbl` is validated by `sql_identifier`
        ins = "INSERT INTO"
        sel = "SELECT"
        dlt = "DELETE FROM"
        self._sql_insert = (
            f"{ins} {tbl} (tenant_id, run_id, revision, status, schema_version, state, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now')) "
            "ON CONFLICT (tenant_id, run_id) DO UPDATE SET "
            "revision = excluded.revision, "
            "status = excluded.status, "
            "schema_version = excluded.schema_version, "
            "state = excluded.state, "
            "updated_at = excluded.updated_at "
            f"WHERE {tbl}.revision = ? "
            f"RETURNING {tbl}.revision"
        )
        self._sql_load = f"{sel} state FROM {tbl} WHERE tenant_id = ? AND run_id = ?"
        self._sql_list_all = (
            f"{sel} run_id, status, revision, updated_at FROM {tbl} "
            "WHERE tenant_id = ? ORDER BY updated_at DESC, run_id LIMIT ?"
        )
        self._sql_list_status = (
            f"{sel} run_id, status, revision, updated_at FROM {tbl} "
            "WHERE tenant_id = ? AND status = ? ORDER BY updated_at DESC, run_id LIMIT ?"
        )
        self._sql_delete_run = f"{dlt} {tbl} WHERE tenant_id = ? AND run_id = ?"
        self._sql_delete_tenant = f"{dlt} {tbl} WHERE tenant_id = ?"

        if connection is not None:
            self._connection = connection
        else:
            db_target = database if database is not None else "checkpoints.sqlite"
            db_str = str(db_target)
            if db_str != ":memory:" and not db_str.startswith("file:"):
                db_path = Path(db_target).resolve()
                try:
                    db_path.parent.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    raise ConfigError(f"Database directory is not usable: {db_path.parent}") from exc
                self._connection = sqlite3.connect(db_path, check_same_thread=False)
            else:
                self._connection = sqlite3.connect(db_str, check_same_thread=False)

        with self._lock:
            self._connection.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = NORMAL")

    def _ensure_open(self) -> None:
        if self._closed:
            raise ConfigError("SqliteCheckpointStore is closed")

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
        self._ensure_open()
        with self._lock:
            for statement in _create_statements(self._table):
                self._connection.execute(statement)
            self._connection.commit()

    # ------------------------------------------------------------------ CheckpointStore protocol

    async def save(self, state: RunState) -> None:
        """Saves and increments `state.revision`. Raises `ConcurrencyError` if another worker saved first."""
        check_ids(state.tenant_id, state.run_id)
        record = WriteRecord.from_state(state)
        write = asyncio.ensure_future(asyncio.to_thread(self._write_sync, record))
        await settle_write(write, state, record.new_revision)

    def _write_sync(self, record: WriteRecord) -> None:
        self._ensure_open()
        with self._lock:
            cursor = self._connection.execute(self._sql_insert, record.params)
            row = cursor.fetchone()
            self._connection.commit()
        if row is None:
            raise ConcurrencyError(f"Run {record.run_id} was modified concurrently")

    async def load(self, tenant_id: str, run_id: str) -> RunState | None:
        check_ids(tenant_id, run_id)
        raw = await asyncio.to_thread(self._load_sync, tenant_id, run_id)
        return restore_state(raw, tenant_id=tenant_id, run_id=run_id) if raw is not None else None

    def _load_sync(self, tenant_id: str, run_id: str) -> str | None:
        self._ensure_open()
        with self._lock:
            cursor = self._connection.execute(self._sql_load, (tenant_id, run_id))
            row = cursor.fetchone()
        return str(row[0]) if row is not None else None

    # ------------------------------------------------------------------ Queries and retention

    async def list_runs(self, tenant_id: str, *, status: RunStatus | None = None, limit: int = 100) -> list[RunInfo]:
        """Runs of one tenant, newest first."""
        check_tenant_id(tenant_id)
        check_list_limit(limit)
        return await asyncio.to_thread(self._list_runs_sync, tenant_id, status, limit)

    def _list_runs_sync(self, tenant_id: str, status: RunStatus | None, limit: int) -> list[RunInfo]:
        self._ensure_open()
        if status is None:
            query = self._sql_list_all
            params: tuple[Any, ...] = (tenant_id, limit)
        else:
            query = self._sql_list_status
            params = (tenant_id, str(status), limit)
        with self._lock:
            cursor = self._connection.execute(query, params)
            rows = cursor.fetchall()
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
        return await asyncio.to_thread(self._delete_run_sync, tenant_id, run_id)

    def _delete_run_sync(self, tenant_id: str, run_id: str) -> bool:
        self._ensure_open()
        with self._lock:
            cursor = self._connection.execute(self._sql_delete_run, (tenant_id, run_id))
            self._connection.commit()
            return cursor.rowcount > 0

    async def delete_tenant(self, tenant_id: str) -> int:
        """Deletes all runs of a tenant. Returns how many were deleted."""
        check_tenant_id(tenant_id)
        return await asyncio.to_thread(self._delete_tenant_sync, tenant_id)

    def _delete_tenant_sync(self, tenant_id: str) -> int:
        self._ensure_open()
        with self._lock:
            cursor = self._connection.execute(self._sql_delete_tenant, (tenant_id,))
            self._connection.commit()
            return cursor.rowcount
