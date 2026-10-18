"""PostgresCheckpointStore. Database tests need `HARNESS_TEST_POSTGRES_DSN` (CONTRIBUTING.md); the rest always runs."""

from __future__ import annotations

import asyncio
import importlib
import sys
from collections.abc import Iterator
from typing import Any, LiteralString, cast

import psycopg
import pytest
from psycopg import sql

from ai_agent_harness import Autonomy, RunState, RunStatus, resume, start
from ai_agent_harness.core import ConcurrencyError, ConfigError, InvalidStateError
from ai_agent_harness.persistence_postgres import PostgresCheckpointStore, schema_sql
from ai_agent_harness.testing import text_response, tool_response

from .conftest import WriteFile, fresh_postgres_schema, make_agent, run_sync

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


def waiting_state(tenant: str = "t1", run: str = "r1") -> RunState:
    state = RunState(agent_name="a", tenant_id=tenant, run_id=run)
    state.add_user_text("hi")
    return state


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


def test_setup_is_idempotent_and_the_row_mirrors_the_state(
    database: tuple[str, str], store: PostgresCheckpointStore
) -> None:
    run_sync(store.setup())  # second call: nothing happens
    state = waiting_state()
    run_sync(store.save(state))
    ((tenant, run, revision, status, version),) = rows(
        database, "SELECT tenant_id, run_id, revision, status, schema_version FROM {table}"
    )
    assert (tenant, run, revision, status, version) == ("t1", "r1", 1, "running", state.schema_version)
    assert state.revision == 1


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


def test_concurrent_saves_of_the_same_revision_have_exactly_one_winner(store: PostgresCheckpointStore) -> None:
    state = waiting_state()
    run_sync(store.save(state))
    first, second = state.model_copy(deep=True), state.model_copy(deep=True)

    async def race() -> list[BaseException | None]:
        outcomes = await asyncio.gather(store.save(first), store.save(second), return_exceptions=True)
        return [o if isinstance(o, BaseException) else None for o in outcomes]

    outcomes = run_sync(race())
    assert sum(isinstance(o, ConcurrencyError) for o in outcomes) == 1 and outcomes.count(None) == 1
    loaded = run_sync(store.load("t1", "r1"))
    assert loaded is not None and loaded.revision == 2


def test_a_tampered_or_swapped_row_is_rejected_on_load(
    database: tuple[str, str], store: PostgresCheckpointStore
) -> None:
    run_sync(store.save(waiting_state("t1", "r1")))
    run_sync(store.save(waiting_state("t2", "r2")))
    # Someone with write access copies the state of t2/r2 over t1/r1
    rows(
        database,
        "UPDATE {table} SET state = (SELECT state FROM {table} WHERE tenant_id = %s) WHERE tenant_id = %s",
        "t2",
        "t1",
    )
    with pytest.raises(InvalidStateError, match="belongs to"):
        run_sync(store.load("t1", "r1"))


def test_a_corrupt_state_is_not_silently_accepted(database: tuple[str, str], store: PostgresCheckpointStore) -> None:
    run_sync(store.save(waiting_state()))
    rows(database, "UPDATE {table} SET state = %s", '{"schema_version": 99}')
    with pytest.raises(Exception, match="schema version"):
        run_sync(store.load("t1", "r1"))


def test_list_runs_filters_orders_and_isolates_tenants(store: PostgresCheckpointStore) -> None:
    done = waiting_state("t1", "done-run")
    done.set_status(RunStatus.DONE)
    for state in (waiting_state("t1", "run-a"), done, waiting_state("t1", "run-b"), waiting_state("t2", "other")):
        run_sync(store.save(state))
    everything = run_sync(store.list_runs("t1"))
    assert {r.run_id for r in everything} == {"run-a", "done-run", "run-b"}
    assert [r.updated_at for r in everything] == sorted((r.updated_at for r in everything), reverse=True)
    only_done = run_sync(store.list_runs("t1", status=RunStatus.DONE))
    assert [(r.run_id, r.status, r.revision) for r in only_done] == [("done-run", RunStatus.DONE, 1)]
    assert len(run_sync(store.list_runs("t1", limit=2))) == 2
    assert [r.run_id for r in run_sync(store.list_runs("t2"))] == ["other"]
    assert run_sync(store.list_runs("nobody")) == []


def test_delete_run_and_delete_tenant(store: PostgresCheckpointStore) -> None:
    for tenant, run in (("t1", "a"), ("t1", "b"), ("t2", "c")):
        run_sync(store.save(waiting_state(tenant, run)))
    assert run_sync(store.delete_run("t1", "a")) is True
    assert run_sync(store.delete_run("t1", "a")) is False
    assert run_sync(store.load("t1", "a")) is None
    assert run_sync(store.delete_tenant("t1")) == 1
    assert run_sync(store.load("t1", "b")) is None and run_sync(store.load("t2", "c")) is not None


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


# ------------------------------------------------------------------ A real run through the store


def test_an_approval_resumes_in_another_process(database: tuple[str, str], store: PostgresCheckpointStore) -> None:
    write = WriteFile()
    first_agent, _, _ = make_agent(
        [tool_response(("write_file", {"path": "a.txt"}))], [write], autonomy=Autonomy.SUGGEST, checkpoints=store
    )
    waiting = run_sync(start(first_agent, "write it", run_id="approval-1"))
    assert waiting.status is RunStatus.WAITING_FOR_APPROVAL and write.written == []
    shown = waiting.pending_id

    dsn, schema = database  # a second process: its own store object, its own agent, only the database in common
    other_store = PostgresCheckpointStore(dsn, schema=schema)
    other_agent, _, _ = make_agent(
        [text_response("written")], [write], autonomy=Autonomy.SUGGEST, checkpoints=other_store
    )
    loaded = run_sync(other_store.load("default", "approval-1"))
    assert loaded is not None and loaded.pending_id == shown
    done = run_sync(resume(other_agent, loaded, approved=True, pending_id=shown))
    assert done.status is RunStatus.DONE and write.written == ["a.txt"]
    assert [r.status for r in run_sync(store.list_runs("default"))] == [RunStatus.DONE]


class Crash(BaseException):  # like a killed process: the runner does not turn it into a status
    pass


def test_a_crashed_run_is_recovered_from_the_database(
    database: tuple[str, str], store: PostgresCheckpointStore
) -> None:
    crashing, _, _ = make_agent([Crash()], checkpoints=store)
    with pytest.raises(Crash):
        run_sync(start(crashing, "hello", run_id="crash-1"))
    assert [r.status for r in run_sync(store.list_runs("default"))] == [
        RunStatus.RUNNING
    ]  # what a recovery job looks for

    dsn, schema = database
    other_store = PostgresCheckpointStore(dsn, schema=schema)
    recovering, _, _ = make_agent([text_response("recovered")], checkpoints=other_store)
    stored = run_sync(other_store.load("default", "crash-1"))
    assert stored is not None
    done = run_sync(resume(recovering, stored, recover=True))
    assert done.status is RunStatus.DONE and done.last_text == "recovered"
