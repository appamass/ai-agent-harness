"""SqliteCheckpointStore tests. Verifies SQLite persistence, optimistic locking, and operational queries."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest

from ai_agent_harness.core import ConcurrencyError, ConfigError
from ai_agent_harness.persistence_sql import status_index_name
from ai_agent_harness.persistence_sqlite import SqliteCheckpointStore, schema_sql

from .conftest import run_sync, waiting_state

TABLE = "ai_agent_harness_checkpoints"


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SqliteCheckpointStore]:
    created = SqliteCheckpointStore(tmp_path / "test_checkpoints.sqlite")
    run_sync(created.setup())
    try:
        yield created
    finally:
        created.close()


# ------------------------------------------------------------------ Configuration & Validation


def test_cannot_pass_both_database_and_connection() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        with pytest.raises(ConfigError, match="at most one"):
            SqliteCheckpointStore("x.sqlite", connection=connection)
    finally:
        connection.close()


@pytest.mark.parametrize("name", ["x; drop table y", "Bad", "", "a" * 64, "1abc", 'a"b', "a b", "ä"])
def test_table_names_are_validated(name: str) -> None:
    with pytest.raises(ConfigError, match="identifier"):
        SqliteCheckpointStore(":memory:", table=name)
    with pytest.raises(ConfigError, match="identifier"):
        schema_sql(name)


def test_schema_sql_describes_the_table_and_its_lock_column() -> None:
    text = schema_sql("runs")
    assert "runs" in text and "PRIMARY KEY (tenant_id, run_id)" in text and "revision" in text
    assert "CREATE TABLE IF NOT EXISTS" in text and "(tenant_id, status)" in text


def test_directory_is_created_automatically(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "er" / "runs.sqlite"
    with SqliteCheckpointStore(target) as created:
        run_sync(created.setup())
        run_sync(created.save(waiting_state()))
    assert target.exists()


def test_in_memory_mode() -> None:
    with SqliteCheckpointStore(":memory:") as created:
        run_sync(created.setup())
        state = waiting_state()
        run_sync(created.save(state))
        loaded = run_sync(created.load("t1", "r1"))
        assert loaded is not None and loaded.revision == 1


def test_context_manager(tmp_path: Path) -> None:
    with SqliteCheckpointStore(tmp_path / "cm.sqlite") as created:
        run_sync(created.setup())
        run_sync(created.save(waiting_state()))
    with pytest.raises(ConfigError, match="closed"):
        run_sync(created.load("t1", "r1"))


@pytest.mark.anyio
async def test_async_context_manager(tmp_path: Path) -> None:
    async with SqliteCheckpointStore(tmp_path / "acm.sqlite") as created:
        await created.setup()
        state = waiting_state()
        await created.save(state)

    with pytest.raises(ConfigError, match="closed"):
        await created.save(state)


# ------------------------------------------------------------------ Store contract


def test_roundtrip_save_and_load(store: SqliteCheckpointStore) -> None:
    state = waiting_state()
    assert run_sync(store.load("t1", "r1")) is None
    run_sync(store.save(state))
    run_sync(store.save(state))
    loaded = run_sync(store.load("t1", "r1"))
    assert loaded is not None
    assert loaded.revision == 2 == state.revision
    assert loaded.messages == state.messages


def test_concurrent_initial_insert_conflict(store: SqliteCheckpointStore) -> None:
    first, second = waiting_state(), waiting_state()  # both think the run is new (revision 0)

    async def race() -> list[BaseException | None]:
        outcomes = await asyncio.gather(store.save(first), store.save(second), return_exceptions=True)
        return [o if isinstance(o, BaseException) else None for o in outcomes]

    outcomes = run_sync(race())
    assert sum(isinstance(o, ConcurrencyError) for o in outcomes) == 1 and outcomes.count(None) == 1


def test_concurrent_workers_on_same_file(tmp_path: Path) -> None:
    path = tmp_path / "shared.sqlite"
    one, two = SqliteCheckpointStore(path), SqliteCheckpointStore(path)
    try:
        run_sync(one.setup())
        state = waiting_state()
        run_sync(one.save(state))
        copy_one, copy_two = state.model_copy(deep=True), state.model_copy(deep=True)

        async def race() -> list[BaseException | None]:
            outcomes = await asyncio.gather(one.save(copy_one), two.save(copy_two), return_exceptions=True)
            return [o if isinstance(o, BaseException) else None for o in outcomes]

        outcomes = run_sync(race())
        assert sum(isinstance(o, ConcurrencyError) for o in outcomes) == 1 and outcomes.count(None) == 1
        loaded = run_sync(two.load("t1", "r1"))
        assert loaded is not None and loaded.revision == 2
    finally:
        one.close()
        two.close()


# ------------------------------------------------------------------ Integrity, durability, cancellation


def test_the_database_uses_wal_mode(store: SqliteCheckpointStore, tmp_path: Path) -> None:
    with closing(sqlite3.connect(tmp_path / "test_checkpoints.sqlite")) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_closing_the_store_leaves_a_borrowed_connection_open() -> None:
    borrowed = sqlite3.connect(":memory:")
    try:
        SqliteCheckpointStore(connection=borrowed).close()
        assert borrowed.execute("SELECT 1").fetchone() == (1,)
    finally:
        borrowed.close()


def test_updated_at_moves_when_a_run_is_saved_again(store: SqliteCheckpointStore, tmp_path: Path) -> None:
    state = waiting_state()
    run_sync(store.save(state))
    with closing(sqlite3.connect(tmp_path / "test_checkpoints.sqlite")) as connection:
        connection.execute(f"UPDATE {TABLE} SET updated_at = '2000-01-01T00:00:00.000Z'")
        connection.commit()

    run_sync(store.save(state))

    assert run_sync(store.list_runs("t1"))[0].updated_at.year > 2000


def test_cancellation_during_a_write_keeps_the_revisions_in_sync(
    store: SqliteCheckpointStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = SqliteCheckpointStore.__dict__["_write_sync"]
    written, release = threading.Event(), threading.Event()

    def slow_write(self: SqliteCheckpointStore, record: object) -> None:
        original(self, record)
        written.set()  # the row is stored, the in-memory revision is not updated yet
        release.wait(5)

    monkeypatch.setattr(SqliteCheckpointStore, "_write_sync", slow_write)
    state = waiting_state()

    async def main() -> None:
        task = asyncio.create_task(store.save(state))
        await asyncio.to_thread(written.wait, 5)
        task.cancel()
        await asyncio.sleep(0.02)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    run_sync(main())
    stored = run_sync(store.load("t1", "r1"))
    assert stored is not None and stored.revision == 1 == state.revision  # saving 'cancelled' next works


def test_status_index_names_differ_for_tables_with_a_shared_prefix() -> None:
    first, second = "a" * 45 + "x", "a" * 45 + "y"
    assert status_index_name(first) != status_index_name(second)
    assert len(status_index_name(first)) <= 63


def test_the_busy_timeout_setting_is_honored(tmp_path: Path) -> None:
    path = tmp_path / "busy.sqlite"
    locked, release = threading.Event(), threading.Event()

    def hold_the_write_lock() -> None:
        with closing(sqlite3.connect(path, isolation_level=None)) as other:
            other.execute("BEGIN IMMEDIATE")
            locked.set()
            release.wait(10)
            other.execute("COMMIT")

    with SqliteCheckpointStore(path, busy_timeout_ms=50) as impatient:
        run_sync(impatient.setup())
        holder = threading.Thread(target=hold_the_write_lock)
        holder.start()
        try:
            assert locked.wait(5)
            started = time.monotonic()
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                run_sync(impatient.save(waiting_state()))
            assert time.monotonic() - started < 2  # gave up after about 50 ms, not after the 5 s default
        finally:
            release.set()
            holder.join()
