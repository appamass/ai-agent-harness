"""PostgreSQL checkpoint store for several workers. Optional extra: `pip install "ai-agent-harness[postgres]"`.

It implements the same `CheckpointStore` protocol and the same optimistic locking as the other stores, so a run can be
started in one process, wait days for an approval, and be resumed in another process or pod (ADR 0006, ADR 0018).

    store = PostgresCheckpointStore("postgresql://app@db/agents?sslmode=require")
    await store.setup()                       # CREATE TABLE IF NOT EXISTS (or run `schema_sql()` in a migration)
    agent = Agent(..., checkpoints=store)

How it stays safe:

- **One atomic statement per save.** `INSERT ... ON CONFLICT DO UPDATE ... WHERE revision = expected` either stores the
  new state or changes nothing; no row back means someone else saved first (`ConcurrencyError`). Two workers can never
  both resume the same state.
- **Cancellation cannot desynchronize revisions** (same shared logic as the file store, `persistence.settle_write`).
- **Loads go through `restore_state`**: schema migration, ownership check (a row moved to another tenant or run is
  rejected) and protocol check. The state is stored as the exact JSON text.
- **Parameterized SQL only.** Schema and table names are validated and quoted. IDs are validated before any SQL.
  Our own messages never contain the connection string or state content; the driver's connection errors do not
  contain the password either. Errors and logs of the database itself (a constraint you added, `log_statement`) may
  quote row data or parameters: treat them like checkpoints.
- **A failed commit is visible.** If the connection drops while a commit is in flight, the row may be stored while
  the in-memory revision was not advanced: the next save raises `ConcurrencyError`. Reload the state, or resume with
  `recover=True`.

What it does not do: encrypt. Checkpoints contain full conversations. Use TLS (`sslmode=require`), a least-privilege
database role, encryption at rest, or wrap the store like Use Case 01 (`EncryptedCheckpointStore`).
By default every call opens a short-lived connection (simple, independent of the event loop). Pass `pool=` (an
`AsyncConnectionPool` opened in your event loop) for a busy service.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from .core.errors import ConcurrencyError, ConfigError
from .core.state import RunState, RunStatus, restore_state
from .persistence import check_ids, settle_write

try:
    import psycopg
    from psycopg import sql
    from psycopg_pool import AsyncConnectionPool
except ImportError as exc:
    raise ConfigError("The Postgres store needs the driver: pip install 'ai-agent-harness[postgres]'") from exc

DEFAULT_TABLE: Final = "ai_agent_harness_checkpoints"
MAX_LIST_LIMIT: Final = 10_000
_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]{0,62}")
_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


@dataclass(frozen=True)
class RunInfo:
    """Metadata of a stored run, for approval queues and recovery jobs. Content stays in the state itself."""

    run_id: str
    status: RunStatus
    revision: int
    updated_at: datetime


@dataclass(frozen=True)
class _Record:
    tenant_id: str
    run_id: str
    expected: int
    new_revision: int
    payload: str
    status: str
    schema_version: int


def _identifier(value: str, what: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ConfigError(f"Invalid {what} identifier {value!r}: use 1-63 lower-case letters, digits, or underscores")
    return value


def _create_statements(schema: str, table: str) -> list[sql.Composed]:
    qualified = sql.Identifier(_identifier(schema, "schema"), _identifier(table, "table"))
    # Names are cut at 63 characters: a short hash keeps long tables with a shared prefix from sharing an index name
    index = sql.Identifier(f"{table[:30]}_{hashlib.sha256(table.encode()).hexdigest()[:8]}_status_idx")
    return [
        sql.SQL(
            "CREATE TABLE IF NOT EXISTS {table} ("
            "tenant_id text NOT NULL, run_id text NOT NULL, revision integer NOT NULL, status text NOT NULL, "
            "schema_version integer NOT NULL, state text NOT NULL, updated_at timestamptz NOT NULL DEFAULT now(), "
            "PRIMARY KEY (tenant_id, run_id))"
        ).format(table=qualified),
        sql.SQL("CREATE INDEX IF NOT EXISTS {index} ON {table} (tenant_id, status)").format(
            index=index, table=qualified
        ),
    ]


def schema_sql(schema: str = "public", table: str = DEFAULT_TABLE) -> str:
    """The DDL that `setup()` runs, for teams that manage their schema with a migration tool."""
    return ";\n".join(statement.as_string() for statement in _create_statements(schema, table)) + ";\n"


class PostgresCheckpointStore:
    """`CheckpointStore` backed by PostgreSQL. Pass exactly one of `dsn` or `pool`."""

    def __init__(
        self,
        dsn: str | None = None,
        *,
        pool: AsyncConnectionPool[Any] | None = None,
        schema: str = "public",
        table: str = DEFAULT_TABLE,
    ) -> None:
        if (dsn is None) == (pool is None):
            raise ConfigError("PostgresCheckpointStore needs exactly one of dsn or pool")
        self._dsn = dsn
        self._pool = pool
        self._schema = _identifier(schema, "schema")
        self._table = _identifier(table, "table")
        self._qualified = sql.Identifier(self._schema, self._table)

    # ------------------------------------------------------------------ Setup

    async def setup(self) -> None:
        """Creates the table and index if they do not exist (safe to call on every start)."""
        async with self._connection() as connection:
            for statement in _create_statements(self._schema, self._table):
                await connection.execute(statement)

    # ------------------------------------------------------------------ CheckpointStore protocol

    async def save(self, state: RunState) -> None:
        """Saves and increments `state.revision`. Raises `ConcurrencyError` if another worker saved first."""
        check_ids(state.tenant_id, state.run_id)
        new_revision = state.revision + 1
        payload = state.model_copy(update={"revision": new_revision}).model_dump_json()
        record = _Record(
            state.tenant_id,
            state.run_id,
            state.revision,
            new_revision,
            payload,
            str(state.status),
            state.schema_version,
        )
        write = asyncio.ensure_future(self._write(record))
        await settle_write(write, state, new_revision)

    async def load(self, tenant_id: str, run_id: str) -> RunState | None:
        check_ids(tenant_id, run_id)
        query = sql.SQL("SELECT state FROM {table} WHERE tenant_id = %s AND run_id = %s").format(table=self._qualified)
        async with self._connection() as connection:
            row = await (await connection.execute(query, (tenant_id, run_id))).fetchone()
        return restore_state(str(row[0]), tenant_id=tenant_id, run_id=run_id) if row is not None else None

    async def _write(self, record: _Record) -> None:
        query = sql.SQL(
            "INSERT INTO {table} AS c (tenant_id, run_id, revision, status, schema_version, state, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, now()) "
            "ON CONFLICT (tenant_id, run_id) DO UPDATE SET revision = EXCLUDED.revision, status = EXCLUDED.status, "
            "schema_version = EXCLUDED.schema_version, state = EXCLUDED.state, updated_at = now() "
            "WHERE c.revision = %s RETURNING c.revision"
        ).format(table=self._qualified)
        params = (
            record.tenant_id,
            record.run_id,
            record.new_revision,
            record.status,
            record.schema_version,
            record.payload,
            record.expected,
        )
        async with self._connection() as connection:
            row = await (await connection.execute(query, params)).fetchone()
        if row is None:  # the row exists with another revision: someone saved in the meantime
            raise ConcurrencyError(f"Run {record.run_id} was modified concurrently")

    # ------------------------------------------------------------------ Queries and retention

    async def list_runs(self, tenant_id: str, *, status: RunStatus | None = None, limit: int = 100) -> list[RunInfo]:
        """Runs of one tenant, newest first. For example `status=RunStatus.WAITING_FOR_APPROVAL` for an approval queue.
        For a recovery job use `RunStatus.RUNNING` and look at `updated_at`: a run that is `running` but has not been
        saved for longer than your longest step has probably lost its process (resume it with `recover=True`)."""
        if not _ID.fullmatch(tenant_id):
            raise ValueError(f"Invalid tenant_id: {tenant_id!r}")
        if not 1 <= limit <= MAX_LIST_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_LIST_LIMIT}")
        if status is None:
            query = sql.SQL(
                "SELECT run_id, status, revision, updated_at FROM {table} WHERE tenant_id = %s "
                "ORDER BY updated_at DESC, run_id LIMIT %s"
            ).format(table=self._qualified)
            params: tuple[Any, ...] = (tenant_id, limit)
        else:
            query = sql.SQL(
                "SELECT run_id, status, revision, updated_at FROM {table} WHERE tenant_id = %s AND status = %s "
                "ORDER BY updated_at DESC, run_id LIMIT %s"
            ).format(table=self._qualified)
            params = (tenant_id, str(status), limit)
        async with self._connection() as connection:
            records = await (await connection.execute(query, params)).fetchall()
        return [RunInfo(str(r[0]), RunStatus(str(r[1])), int(r[2]), r[3]) for r in records]

    async def delete_run(self, tenant_id: str, run_id: str) -> bool:
        """Deletes one run (retention policy, right to erasure). Returns whether a run existed."""
        check_ids(tenant_id, run_id)
        query = sql.SQL("DELETE FROM {table} WHERE tenant_id = %s AND run_id = %s").format(table=self._qualified)
        async with self._connection() as connection:
            cursor = await connection.execute(query, (tenant_id, run_id))
            return cursor.rowcount > 0

    async def delete_tenant(self, tenant_id: str) -> int:
        """Deletes all runs of a tenant (for example when an account is closed). Returns how many were deleted."""
        if not _ID.fullmatch(tenant_id):
            raise ValueError(f"Invalid tenant_id: {tenant_id!r}")
        query = sql.SQL("DELETE FROM {table} WHERE tenant_id = %s").format(table=self._qualified)
        async with self._connection() as connection:
            cursor = await connection.execute(query, (tenant_id,))
            return cursor.rowcount

    # ------------------------------------------------------------------ Connections

    @asynccontextmanager
    async def _connection(self) -> AsyncGenerator[psycopg.AsyncConnection[Any]]:
        if self._pool is not None:
            async with self._pool.connection() as pooled:
                yield pooled
        else:
            assert self._dsn is not None  # the constructor guarantees one of dsn or pool
            async with await psycopg.AsyncConnection.connect(self._dsn) as connection:
                yield connection
