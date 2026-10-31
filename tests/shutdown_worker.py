"""A worker process for `test_shutdown.py`: SIGTERM flips a `KillSwitch`, as the guide shows.

Run as `python -m tests.shutdown_worker <work_dir> <tool|model>`. The file `ready` appears while the tool or the model
call is running, and both wait for the signal, so no test depends on timing. The tool writes one `ledger` line per run.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path
from typing import Any

from ai_agent_harness import (
    Agent,
    Autonomy,
    BaseTool,
    Hook,
    KillSwitch,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    Risk,
    SqliteCheckpointStore,
    start,
    tool,
)
from ai_agent_harness.providers import ModelFn
from ai_agent_harness.testing import FakeProvider, tool_response

SHUTDOWN = "Worker is restarting: this call did not run, so call it again"
TENANT = "t1"
RUN_ID = "run-1"
STORE_FILE = "runs.db"


def booking_tool(work_dir: Path, stopping: asyncio.Event) -> BaseTool[Any]:
    @tool(risk=Risk.WRITE, timeout_s=120)
    async def book() -> str:
        """Books once and stays busy until the shutdown signal."""
        with (work_dir / "ledger").open("a", encoding="utf-8") as ledger:
            ledger.write("booked\n")
        (work_dir / "ready").touch()
        await stopping.wait()
        return "booked"

    return book


def worker_agent(
    work_dir: Path,
    store: SqliteCheckpointStore,
    provider: ModelProvider | ModelFn,
    stopping: asyncio.Event,
    *hooks: Hook,
) -> Agent:
    return Agent.create(
        provider,
        "You book things.",
        name="worker",
        tools=[booking_tool(work_dir, stopping)],
        autonomy=Autonomy.AUTONOMOUS,
        hooks=hooks,
        checkpoints=store,
    )


async def work(work_dir: Path, mode: str) -> None:
    stopping = asyncio.Event()

    async def slow_model(request: ModelRequest) -> ModelResponse:
        (work_dir / "ready").touch()
        await stopping.wait()
        return tool_response(("book", {}))

    two_bookings = FakeProvider([tool_response(("book", {}), ("book", {}))])
    with SqliteCheckpointStore(work_dir / STORE_FILE) as store:
        await store.setup()
        switch = KillSwitch(lambda state: stopping.is_set(), reason=SHUTDOWN)
        provider = slow_model if mode == "model" else two_bookings
        agent = worker_agent(work_dir, store, provider, stopping, switch)
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stopping.set)
        await start(agent, "Book it", tenant_id=TENANT, run_id=RUN_ID)


if __name__ == "__main__":
    asyncio.run(work(Path(sys.argv[1]), sys.argv[2]))
