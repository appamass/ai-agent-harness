"""What the shutdown and recovery recipe in `docs/guide.md` relies on, with real signals to a real process."""

from __future__ import annotations

import asyncio
import contextlib
import signal
import subprocess
import sys
import time
from collections.abc import Generator
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ai_agent_harness import (
    Autonomy,
    CheckpointStore,
    Decision,
    Hook,
    HookPipeline,
    InMemoryCheckpointStore,
    KillSwitch,
    ModelRequest,
    Risk,
    RunInfo,
    RunState,
    RunStatus,
    SqliteCheckpointStore,
    resume,
    start,
    tool,
)
from ai_agent_harness.core import ConcurrencyError, InvalidStateError
from ai_agent_harness.runtime.runner import INTERRUPTED
from ai_agent_harness.testing import FakeProvider, text_response, tool_response

from .conftest import REPO_ROOT, Crash, SqlBackend, make_agent, results_of, run_sync
from .shutdown_worker import RUN_ID, SHUTDOWN, STORE_FILE, TENANT, worker_agent
from .test_docs import run_doc_blocks
from .test_sql_store_contract import SqlStore

TIMEOUT_S = 60.0


@contextlib.contextmanager
def running_worker(work_dir: Path, mode: str) -> Generator[subprocess.Popen[bytes]]:
    """Starts the worker and returns once its tool or model call is running."""
    command = [sys.executable, "-m", "tests.shutdown_worker", str(work_dir), mode]
    process = subprocess.Popen(command, cwd=REPO_ROOT)
    try:
        deadline = time.monotonic() + TIMEOUT_S
        while not (work_dir / "ready").exists():
            assert process.poll() is None, "the worker ended before it was ready"
            assert time.monotonic() < deadline, "the worker did not get ready"
            time.sleep(0.01)
        yield process
    finally:
        process.kill()
        process.wait()


def after_restart(work_dir: Path, *, recover: bool = False) -> tuple[RunStatus, str | None, RunState]:
    """What a new worker finds in the store, and the run after the new worker continued it."""

    async def go() -> tuple[RunStatus, str | None, RunState]:
        with SqliteCheckpointStore(work_dir / STORE_FILE) as store:
            found = await store.load(TENANT, RUN_ID)
            assert found is not None
            status, reason = found.status, found.status_reason
            agent = worker_agent(work_dir, store, FakeProvider([text_response("Done.")]), asyncio.Event())
            return status, reason, await resume(agent, found, recover=recover)

    return run_sync(go())


def ledger_lines(work_dir: Path) -> int:
    ledger = work_dir / "ledger"
    return len(ledger.read_text(encoding="utf-8").splitlines()) if ledger.exists() else 0


def test_sigterm_lets_the_running_tool_finish_and_the_run_continues(tmp_path: Path) -> None:
    with running_worker(tmp_path, "tool") as worker:
        worker.send_signal(signal.SIGTERM)
        assert worker.wait(TIMEOUT_S) == 0

    status, reason, finished = after_restart(tmp_path)

    assert (status, reason) == (RunStatus.BLOCKED, SHUTDOWN)
    assert finished.status is RunStatus.DONE and finished.last_text == "Done."
    booked, stopped = results_of(finished)
    assert booked.content == "booked" and not booked.is_error  # the real result, not "outcome unknown"
    assert stopped.is_error and SHUTDOWN in stopped.content  # the second call of the batch never ran
    assert ledger_lines(tmp_path) == 1


def test_sigterm_during_a_model_call_runs_none_of_the_tools_it_asked_for(tmp_path: Path) -> None:
    with running_worker(tmp_path, "model") as worker:
        worker.send_signal(signal.SIGTERM)
        assert worker.wait(TIMEOUT_S) == 0

    status, reason, finished = after_restart(tmp_path)

    assert (status, reason) == (RunStatus.BLOCKED, SHUTDOWN)
    assert finished.status is RunStatus.DONE
    assert [SHUTDOWN in result.content for result in results_of(finished)] == [True]
    assert ledger_lines(tmp_path) == 0


def test_sigkill_during_a_tool_stays_running_and_recovery_reports_the_unknown_outcome(tmp_path: Path) -> None:
    with running_worker(tmp_path, "tool") as worker:
        worker.kill()
        worker.wait(TIMEOUT_S)

    status, _, finished = after_restart(tmp_path, recover=True)

    assert status is RunStatus.RUNNING
    assert finished.status is RunStatus.DONE
    assert all(result.content == INTERRUPTED for result in results_of(finished))
    assert ledger_lines(tmp_path) == 1  # reported, not run again


def test_recovering_a_live_run_repeats_the_tool_and_the_live_worker_fails_only_afterwards() -> None:
    """The limit behind the age check of the recipe: `recover=True` cannot tell a live run from a crashed one."""

    async def go() -> list[int]:
        started, release = asyncio.Event(), asyncio.Event()
        calls: list[int] = []

        @tool(risk=Risk.WRITE)
        async def book() -> str:
            """Books once. Only the first call is slow."""
            calls.append(1)
            if len(calls) == 1:
                started.set()
                await release.wait()
            return "booked"

        store = InMemoryCheckpointStore()
        options: dict[str, Any] = {"autonomy": Autonomy.AUTONOMOUS, "checkpoints": store}
        live, _, _ = make_agent([tool_response(("book", {})), text_response("Live.")], [book], **options)
        task = asyncio.create_task(start(live, "Book it", run_id="r1"))
        await started.wait()

        state = await store.load("default", "r1")
        assert state is not None and state.status is RunStatus.RUNNING
        job, _, _ = make_agent([tool_response(("book", {})), text_response("Recovered.")], [book], **options)
        assert (await resume(job, state, recover=True)).status is RunStatus.DONE

        release.set()
        with pytest.raises(ConcurrencyError):
            await task
        return calls

    assert len(run_sync(go())) == 2


def test_two_recovery_jobs_with_the_same_state_continue_the_run_once() -> None:
    async def go() -> None:
        store = InMemoryCheckpointStore()
        await crashed_run(store)
        first, second = await store.load("default", "r1"), await store.load("default", "r1")
        assert first is not None and second is not None

        job, _, _ = make_agent([text_response("Done.")], checkpoints=store)
        assert (await resume(job, first, recover=True)).status is RunStatus.DONE

        late, provider, _ = make_agent([text_response("Never.")], checkpoints=store)
        with pytest.raises(ConcurrencyError):
            await resume(late, second, recover=True)
        assert provider.requests == []

    run_sync(go())


def test_a_run_started_during_shutdown_is_blocked_and_has_nothing_to_continue() -> None:
    """Why the recipe stops taking new work first: the input of this run was never stored."""

    async def go() -> None:
        store = InMemoryCheckpointStore()
        switch = HookPipeline([KillSwitch(lambda state: True, reason=SHUTDOWN)])
        agent, provider, _ = make_agent([text_response("Never.")], hooks=switch, checkpoints=store)
        state = await start(agent, "Go", run_id="r1")
        assert (state.status, state.status_reason) == (RunStatus.BLOCKED, SHUTDOWN)
        assert state.messages == []

        restarted, _, _ = make_agent([text_response("Never.")], checkpoints=store)
        with pytest.raises(InvalidStateError, match="Nothing to resume"):
            await resume(restarted, state)
        assert provider.requests == []

    run_sync(go())


# ------------------------------------------------------------------ The recovery job as printed in the guide

AGED = timedelta(seconds=-1)  # as if every run were older than the limit
FRESH = timedelta(hours=1)


@pytest.fixture(scope="module")
def guide() -> ModuleType:
    return run_doc_blocks(REPO_ROOT / "docs" / "guide.md")


class Guard(Hook):
    async def before_model(self, state: RunState, request: ModelRequest) -> Decision:
        return Decision.block("Not allowed")


async def crashed_run(store: CheckpointStore) -> None:
    crashing, _, _ = make_agent([Crash()], checkpoints=store)
    with pytest.raises(Crash):
        await start(crashing, "Go", run_id="r1")


async def status_of(store: CheckpointStore) -> RunStatus:
    state = await store.load("default", "r1")
    assert state is not None
    return state.status


async def assert_job_skips(guide: ModuleType, store: CheckpointStore, *, agent_name: str = "test") -> None:
    job, provider, _ = make_agent([text_response("Never.")], checkpoints=store)
    assert await guide.continue_runs(replace(job, name=agent_name), "default", AGED) == 0
    assert provider.requests == []


def test_the_job_leaves_a_recent_running_run_and_continues_an_old_one(guide: ModuleType, sql_store: SqlStore) -> None:
    async def go() -> None:
        await crashed_run(sql_store)
        job, provider, _ = make_agent([text_response("Done.")], checkpoints=sql_store)

        assert await guide.continue_runs(job, "default", FRESH) == 0
        assert await status_of(sql_store) is RunStatus.RUNNING and provider.requests == []

        assert await guide.continue_runs(job, "default", AGED) == 1
        assert await status_of(sql_store) is RunStatus.DONE

    run_sync(go())


def test_the_job_keeps_the_decision_of_a_guard(guide: ModuleType, sql_store: SqlStore) -> None:
    async def go() -> None:
        guarded, _, _ = make_agent([text_response("Never.")], hooks=HookPipeline([Guard()]), checkpoints=sql_store)
        await start(guarded, "Go", run_id="r1")

        await assert_job_skips(guide, sql_store)
        assert await status_of(sql_store) is RunStatus.BLOCKED

    run_sync(go())


def test_the_job_leaves_the_runs_of_another_agent(guide: ModuleType, sql_store: SqlStore) -> None:
    async def go() -> None:
        await crashed_run(sql_store)

        await assert_job_skips(guide, sql_store, agent_name="other")
        assert await status_of(sql_store) is RunStatus.RUNNING

    run_sync(go())


def test_the_job_skips_a_run_that_never_stored_its_input(guide: ModuleType, sql_store: SqlStore) -> None:
    async def go() -> None:
        switch = HookPipeline([KillSwitch(lambda state: True, reason=guide.SHUTDOWN)])
        late, _, _ = make_agent([text_response("Never.")], hooks=switch, checkpoints=sql_store)
        await start(late, "Go", run_id="r1")

        await assert_job_skips(guide, sql_store)

    run_sync(go())


def test_the_job_skips_a_run_that_moved_on_after_the_list(guide: ModuleType) -> None:
    class LiveWorkerSavesDuringTheList(SqliteCheckpointStore):
        async def list_runs(
            self, tenant_id: str, *, status: RunStatus | None = None, limit: int = 100
        ) -> list[RunInfo]:
            found = await super().list_runs(tenant_id, status=status, limit=limit)
            state = await self.load(tenant_id, "r1")
            assert state is not None
            await self.save(state)
            return found

    async def go() -> None:
        with LiveWorkerSavesDuringTheList(":memory:") as store:
            await store.setup()
            await crashed_run(store)
            await assert_job_skips(guide, store)

    run_sync(go())


def test_two_jobs_at_once_continue_a_run_once(guide: ModuleType, sql_backend: SqlBackend) -> None:
    async def go() -> None:
        await crashed_run(sql_backend.store)
        stores = (sql_backend.store, sql_backend.reopen())
        jobs = [make_agent([text_response("Done.")], checkpoints=store)[0] for store in stores]

        counts = await asyncio.gather(*(guide.continue_runs(job, "default", AGED) for job in jobs))

        assert sum(counts) == 1
        assert await status_of(sql_backend.store) is RunStatus.DONE

    run_sync(go())
