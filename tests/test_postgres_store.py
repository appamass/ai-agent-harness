"""PostgresCheckpointStore. Database tests need `HARNESS_TEST_POSTGRES_DSN` (AGENTS.md); the rest always runs."""

from __future__ import annotations

import asyncio
import importlib
import sys
from collections.abc import Iterator
from typing import Any, LiteralString, cast

import psycopg
import pytest
from psycopg import sql

from ai_agent_harness.core import ConfigError
from ai_agent_harness.persistence_postgres import PostgresCheckpointStore, schema_sql

from .conftest import fresh_postgres_schema, run_sync, waiting_state

TABLE = "ai_agent_harness_checkpoints"


@pytest.fixture
def database() -> Iterator[tuple[str, str]]:
    with fresh_postgres_schema() as dsn_and_schema:
        yield dsn_and_schema


@pytest.fixture
def store(database: tuple[str, str]) -> PostgresCheckpointStore:
    dsn, schema = database
    created = PostgresCheckpointStore(dsn, schema=schema)
    run_sync(created.setup())
    return created


def rows(database: tuple[str, str], query: LiteralString, *params: object) -> list[tuple[Any, ...]]:
    dsn, schema = database
    with psycopg.connect(dsn, autocommit=True) as connection:
        statement = sql.SQL(query).format(table=sql.Identifier(schema, TABLE))
        cursor = connection.execute(statement, params)
        return cursor.fetchall() if cursor.description else []


# ------------------------------------------------------------------ Without a database


def test_a_missing_driver_gives_an_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "psycopg", None)  # `import psycopg` fails like an uninstalled extra
    monkeypatch.delitem(sys.modules, "ai_agent_harness.persistence_postgres")
    with pytest.raises(ConfigError, match=r"ai-agent-harness\[postgres\]"):
        importlib.import_module("ai_agent_harness.persistence_postgres")


def test_exactly_one_of_dsn_or_pool_is_required() -> None:
    with pytest.raises(ConfigError, match="dsn or pool"):
        PostgresCheckpointStore()
    with pytest.raises(ConfigError, match="dsn or pool"):
        PostgresCheckpointStore("postgresql://x/y", pool=cast("Any", object()))


@pytest.mark.parametrize("name", ["x; drop table y", "Bad", "", "a" * 64, "1abc", 'a"b', "a b", "ä"])
def test_schema_and_table_names_are_validated(name: str) -> None:
    with pytest.raises(ConfigError, match="identifier"):
        PostgresCheckpointStore("postgresql://x/y", schema=name)
    with pytest.raises(ConfigError, match="identifier"):
        PostgresCheckpointStore("postgresql://x/y", table=name)
    with pytest.raises(ConfigError, match="identifier"):
        schema_sql(name, "t")


def test_ids_are_validated_before_any_sql() -> None:
    unreachable = PostgresCheckpointStore("postgresql://nobody@127.0.0.1:1/none")  # nothing listens here
    with pytest.raises(ValueError, match="Invalid"):
        run_sync(unreachable.load("../x", "r"))
    with pytest.raises(ValueError, match="Invalid"):
        run_sync(unreachable.delete_run("t", "a/b"))
    with pytest.raises(ValueError, match="Invalid"):
        run_sync(unreachable.list_runs("bad tenant"))


def test_connection_errors_do_not_leak_the_password() -> None:
    unreachable = PostgresCheckpointStore("postgresql://user:top-secret-pw@127.0.0.1:1/db")
    with pytest.raises(psycopg.OperationalError) as caught:
        run_sync(unreachable.save(waiting_state()))
    assert "top-secret-pw" not in str(caught.value)


def test_schema_sql_describes_the_table_and_its_lock_column() -> None:
    text = schema_sql("app", "runs")
    assert '"app"."runs"' in text and "PRIMARY KEY (tenant_id, run_id)" in text and "revision" in text
    assert "CREATE TABLE IF NOT EXISTS" in text and "(tenant_id, status)" in text


def test_limit_must_be_sensible() -> None:
    store = PostgresCheckpointStore("postgresql://nobody@127.0.0.1:1/none")
    for bad in (0, -1, 10_001):
        with pytest.raises(ValueError, match="limit"):
            run_sync(store.list_runs("t", limit=bad))


# ------------------------------------------------------------------ With a database


def test_long_table_names_with_a_shared_prefix_each_get_their_own_index(database: tuple[str, str]) -> None:
    dsn, schema = database
    first, second = "a" * 45 + "x", "a" * 45 + "y"
    for table in (first, second):
        run_sync(PostgresCheckpointStore(dsn, schema=schema, table=table).setup())
    with psycopg.connect(dsn, autocommit=True) as connection:
        found = connection.execute(
            "SELECT tablename, count(*) FROM pg_indexes WHERE schemaname = %s GROUP BY tablename", (schema,)
        ).fetchall()
    assert sorted(found) == [(first, 2), (second, 2)]  # primary key + the (tenant_id, status) index, for both tables


def test_a_missing_row_is_created_even_with_a_nonzero_revision(store: PostgresCheckpointStore) -> None:
    state = waiting_state()
    state.revision = 5  # the other stores create too: a missing row is not a conflict
    run_sync(store.save(state))
    loaded = run_sync(store.load("t1", "r1"))
    assert loaded is not None and loaded.revision == 6 == state.revision


def test_a_corrupt_state_is_not_silently_accepted(database: tuple[str, str], store: PostgresCheckpointStore) -> None:
    run_sync(store.save(waiting_state()))
    rows(database, "UPDATE {table} SET state = %s", '{"schema_version": 99}')
    with pytest.raises(Exception, match="schema version"):
        run_sync(store.load("t1", "r1"))


@pytest.mark.parametrize("scenario", ["once", "twice"])
def test_cancellation_during_a_write_keeps_the_revisions_in_sync(
    store: PostgresCheckpointStore, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    original = PostgresCheckpointStore.__dict__["_write"]
    written, release = asyncio.Event(), asyncio.Event()

    async def slow_write(self: PostgresCheckpointStore, *args: Any) -> None:
        await original(self, *args)
        written.set()
        await release.wait()  # the row is stored, the in-memory revision is not updated yet

    monkeypatch.setattr(PostgresCheckpointStore, "_write", slow_write)
    state = waiting_state()

    async def main() -> None:
        task = asyncio.create_task(store.save(state))
        await written.wait()
        task.cancel()
        if scenario == "twice":
            await asyncio.sleep(0.02)
            task.cancel()
        await asyncio.sleep(0.02)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    run_sync(main())
    stored = run_sync(store.load("t1", "r1"))
    assert stored is not None and stored.revision == 1 == state.revision  # saving 'cancelled' next works


def test_a_connection_pool_works(database: tuple[str, str]) -> None:
    from psycopg_pool import AsyncConnectionPool

    dsn, schema = database

    async def main() -> int:
        async with AsyncConnectionPool(dsn, min_size=1, max_size=3, open=False) as pool:
            pooled = PostgresCheckpointStore(pool=pool, schema=schema)
            await pooled.setup()
            state = waiting_state()
            await pooled.save(state)
            await pooled.save(state)
            loaded = await pooled.load("t1", "r1")
            assert loaded is not None
            return loaded.revision

    assert run_sync(main()) == 2
