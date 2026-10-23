"""What every SQL checkpoint store must do the same way. Each test runs on SQLite and on PostgreSQL.

PostgreSQL tests need `HARNESS_TEST_POSTGRES_DSN` (AGENTS.md) and are skipped without it.
Tests that belong to one backend only stay in `test_sqlite_store.py` and `test_postgres_store.py`.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from ai_agent_harness import Autonomy, RunState, RunStatus, resume, start
from ai_agent_harness.core import ConcurrencyError, InvalidStateError
from ai_agent_harness.persistence_postgres import PostgresCheckpointStore
from ai_agent_harness.persistence_sql import FINISHED_STATUSES
from ai_agent_harness.persistence_sqlite import SqliteCheckpointStore
from ai_agent_harness.testing import text_response, tool_response

from .conftest import Crash, SqlBackend, WriteFile, make_agent, run_sync, waiting_state

SqlStore = SqliteCheckpointStore | PostgresCheckpointStore


def test_ids_are_validated_before_any_sql(sql_store: SqlStore) -> None:
    with pytest.raises(ValueError, match="Invalid"):
        run_sync(sql_store.load("../x", "r"))
    with pytest.raises(ValueError, match="Invalid"):
        run_sync(sql_store.delete_run("t", "a/b"))
    with pytest.raises(ValueError, match="Invalid"):
        run_sync(sql_store.list_runs("bad tenant"))
    with pytest.raises(ValueError, match="Invalid"):
        run_sync(sql_store.delete_tenant("bad tenant"))


def test_setup_is_idempotent_and_the_row_mirrors_the_state(sql_backend: SqlBackend, sql_store: SqlStore) -> None:
    run_sync(sql_store.setup())  # second call: nothing happens
    state = waiting_state()
    run_sync(sql_store.save(state))
    ((tenant, run, revision, status, version),) = sql_backend.execute(
        "SELECT tenant_id, run_id, revision, status, schema_version FROM {table}"
    )
    assert (tenant, run, revision, status, version) == ("t1", "r1", 1, "running", state.schema_version)
    assert state.revision == 1


def test_stale_copy_cannot_overwrite(sql_store: SqlStore) -> None:
    state = waiting_state()
    run_sync(sql_store.save(state))
    stale = state.model_copy(deep=True)
    run_sync(sql_store.save(state))  # moves the stored revision on
    with pytest.raises(ConcurrencyError):
        run_sync(sql_store.save(stale))
    loaded = run_sync(sql_store.load("t1", "r1"))
    assert loaded is not None and loaded.revision == 2


def test_concurrent_saves_of_the_same_revision_have_exactly_one_winner(sql_store: SqlStore) -> None:
    state = waiting_state()
    run_sync(sql_store.save(state))
    first, second = state.model_copy(deep=True), state.model_copy(deep=True)

    async def race() -> list[BaseException | None]:
        outcomes = await asyncio.gather(sql_store.save(first), sql_store.save(second), return_exceptions=True)
        return [o if isinstance(o, BaseException) else None for o in outcomes]

    outcomes = run_sync(race())
    assert sum(isinstance(o, ConcurrencyError) for o in outcomes) == 1 and outcomes.count(None) == 1
    loaded = run_sync(sql_store.load("t1", "r1"))
    assert loaded is not None and loaded.revision == 2


def test_list_runs_filters_orders_and_isolates_tenants(sql_backend: SqlBackend, sql_store: SqlStore) -> None:
    done = waiting_state("t1", "done-run")
    done.set_status(RunStatus.DONE)
    for state in (waiting_state("t1", "run-a"), done, waiting_state("t1", "run-b"), waiting_state("t2", "other")):
        run_sync(sql_store.save(state))
    # Distinct timestamps, so the newest-first order is checked and not left to clock resolution
    for run_id, stamp in (("run-a", "10:00"), ("done-run", "11:00"), ("run-b", "12:00")):
        sql_backend.execute(
            "UPDATE {table} SET updated_at = %s WHERE run_id = %s", f"2026-01-01T{stamp}:00.000Z", run_id
        )

    everything = run_sync(sql_store.list_runs("t1"))
    assert [r.run_id for r in everything] == ["run-b", "done-run", "run-a"]
    assert [r.updated_at for r in everything] == sorted((r.updated_at for r in everything), reverse=True)
    only_done = run_sync(sql_store.list_runs("t1", status=RunStatus.DONE))
    assert [(r.run_id, r.status, r.revision) for r in only_done] == [("done-run", RunStatus.DONE, 1)]
    assert [r.run_id for r in run_sync(sql_store.list_runs("t1", limit=2))] == ["run-b", "done-run"]
    assert [r.run_id for r in run_sync(sql_store.list_runs("t2"))] == ["other"]
    assert run_sync(sql_store.list_runs("nobody")) == []
    for bad in (0, -1, 10_001):
        with pytest.raises(ValueError, match="limit"):
            run_sync(sql_store.list_runs("t1", limit=bad))


def test_delete_run_and_delete_tenant(sql_store: SqlStore) -> None:
    for tenant, run in (("t1", "a"), ("t1", "b"), ("t2", "c")):
        run_sync(sql_store.save(waiting_state(tenant, run)))
    assert run_sync(sql_store.delete_run("t1", "a")) is True
    assert run_sync(sql_store.delete_run("t1", "a")) is False
    assert run_sync(sql_store.load("t1", "a")) is None
    assert run_sync(sql_store.delete_tenant("t1")) == 1
    assert run_sync(sql_store.load("t1", "b")) is None and run_sync(sql_store.load("t2", "c")) is not None


def test_a_row_swapped_between_tenants_is_rejected_on_load(sql_backend: SqlBackend, sql_store: SqlStore) -> None:
    run_sync(sql_store.save(waiting_state("t1", "r1")))
    run_sync(sql_store.save(waiting_state("t2", "r2")))
    # Someone with write access copies the state of t2/r2 over t1/r1
    sql_backend.execute(
        "UPDATE {table} SET state = (SELECT state FROM {table} WHERE tenant_id = %s) WHERE tenant_id = %s", "t2", "t1"
    )
    with pytest.raises(InvalidStateError, match="belongs to"):
        run_sync(sql_store.load("t1", "r1"))


def test_an_approval_resumes_in_another_process(sql_backend: SqlBackend, sql_store: SqlStore) -> None:
    write = WriteFile()
    first_agent, _, _ = make_agent(
        [tool_response(("write_file", {"path": "a.txt"}))], [write], autonomy=Autonomy.SUGGEST, checkpoints=sql_store
    )
    waiting = run_sync(start(first_agent, "write it", run_id="approval-1"))
    assert waiting.status is RunStatus.WAITING_FOR_APPROVAL and write.written == []
    shown = waiting.pending_id

    # A second process: its own store object, its own agent, only the database in common
    other_store = sql_backend.reopen()
    other_agent, _, _ = make_agent(
        [text_response("written")], [write], autonomy=Autonomy.SUGGEST, checkpoints=other_store
    )
    loaded = run_sync(other_store.load("default", "approval-1"))
    assert loaded is not None and loaded.pending_id == shown
    done = run_sync(resume(other_agent, loaded, approved=True, pending_id=shown))
    assert done.status is RunStatus.DONE and write.written == ["a.txt"]
    assert [r.status for r in run_sync(sql_store.list_runs("default"))] == [RunStatus.DONE]


def test_a_crashed_run_is_recovered_from_the_database(sql_backend: SqlBackend, sql_store: SqlStore) -> None:
    crashing, _, _ = make_agent([Crash()], checkpoints=sql_store)
    with pytest.raises(Crash):
        run_sync(start(crashing, "hello", run_id="crash-1"))
    assert [r.status for r in run_sync(sql_store.list_runs("default"))] == [
        RunStatus.RUNNING
    ]  # what a recovery job looks for

    other_store = sql_backend.reopen()
    running = run_sync(other_store.list_runs("default", status=RunStatus.RUNNING))
    assert [r.run_id for r in running] == ["crash-1"]
    recovering, _, _ = make_agent([text_response("recovered")], checkpoints=other_store)
    stored = run_sync(other_store.load("default", "crash-1"))
    assert stored is not None
    done = run_sync(resume(recovering, stored, recover=True))
    assert done.status is RunStatus.DONE and done.last_text == "recovered"


def test_finished_statuses_matches_run_status() -> None:
    all_statuses = {s.value for s in RunStatus}
    waiting = {RunStatus.WAITING_FOR_INPUT.value, RunStatus.WAITING_FOR_APPROVAL.value}
    running = {RunStatus.RUNNING.value}
    assert set(FINISHED_STATUSES) == all_statuses - waiting - running
    assert set(FINISHED_STATUSES).isdisjoint(waiting | running)


def test_purge_finished_deletes_only_old_terminal_runs(sql_backend: SqlBackend, sql_store: SqlStore) -> None:
    from ai_agent_harness.core.state import PendingBatch, PendingCall
    from ai_agent_harness.core.types import Message, ToolCall

    states: list[RunState] = []
    terminal_statuses = [
        RunStatus.DONE,
        RunStatus.REFUSED,
        RunStatus.BLOCKED,
        RunStatus.LIMIT_REACHED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    ]

    for status in terminal_statuses:
        st = waiting_state("t1", f"term-{status.value}")
        st.set_status(status)
        states.append(st)

    # Non-terminal states: RUNNING, WAITING_FOR_INPUT, WAITING_FOR_APPROVAL
    running_st = waiting_state("t1", "nonterm-running")
    states.append(running_st)

    from typing import Literal

    kinds: tuple[Literal["input", "approval"], ...] = ("input", "approval")
    for kind in kinds:
        wait_st = RunState(agent_name="a", tenant_id="t1", run_id=f"nonterm-waiting_{kind}")
        wait_st.add_user_text("hi")
        call = ToolCall(id=f"c-{kind}", name="echo")
        wait_st.add_assistant(Message(role="assistant", content=(call,)))
        wait_st.pending = PendingBatch(waiting=PendingCall(kind=kind, call=call, prompt="?"))
        wait_st.set_status(RunStatus.WAITING_FOR_INPUT if kind == "input" else RunStatus.WAITING_FOR_APPROVAL)
        states.append(wait_st)

    # Recent terminal run that should NOT be purged
    recent = waiting_state("t1", "term-recent")
    recent.set_status(RunStatus.DONE)
    states.append(recent)

    # Terminal run for another tenant (t2)
    t2_done = waiting_state("t2", "term-t2")
    t2_done.set_status(RunStatus.DONE)
    states.append(t2_done)

    for st in states:
        run_sync(sql_store.save(st))

    # Move timestamps: old runs to 2 hours ago
    old_stamp = "2026-01-01T10:00:00.000Z"
    for st in states:
        if st.run_id != "term-recent":
            sql_backend.execute(
                "UPDATE {table} SET updated_at = %s WHERE tenant_id = %s AND run_id = %s",
                old_stamp,
                st.tenant_id,
                st.run_id,
            )

    # Invalid arguments rejected
    with pytest.raises(ValueError, match="positive"):
        run_sync(sql_store.purge_finished(older_than=timedelta(0)))
    with pytest.raises(ValueError, match="positive"):
        run_sync(sql_store.purge_finished(older_than=timedelta(seconds=-1)))
    with pytest.raises(ValueError, match="Invalid tenant_id"):
        run_sync(sql_store.purge_finished(older_than=timedelta(hours=1), tenant_id="bad/tenant"))

    # Purge t1:
    purged_t1 = run_sync(sql_store.purge_finished(older_than=timedelta(hours=1), tenant_id="t1"))
    assert purged_t1 == len(terminal_statuses)

    # Check that non-terminal runs and recent run still exist in t1
    assert run_sync(sql_store.load("t1", "nonterm-running")) is not None
    assert run_sync(sql_store.load("t1", "nonterm-waiting_input")) is not None
    assert run_sync(sql_store.load("t1", "nonterm-waiting_approval")) is not None
    assert run_sync(sql_store.load("t1", "term-recent")) is not None
    assert run_sync(sql_store.load("t2", "term-t2")) is not None

    # Now purge all tenants without tenant_id:
    purged_all = run_sync(sql_store.purge_finished(older_than=timedelta(hours=1)))
    assert purged_all == 1  # term-t2 purged

    assert run_sync(sql_store.load("t2", "term-t2")) is None
