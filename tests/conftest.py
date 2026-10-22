"""Shared test helpers. Tests run without pytest plugins: async code executes via `run_sync`."""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import os
import sys
import uuid
from collections.abc import Coroutine, Generator, Sequence
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import pytest
from pydantic import BaseModel

from ai_agent_harness import (
    Agent,
    Autonomy,
    BaseTool,
    ContextBuilder,
    Event,
    HookPipeline,
    Limits,
    ModelResponse,
    ModelRouter,
    RetryPolicy,
    Risk,
    RunState,
    ToolRegistry,
    restore_state,
)
from ai_agent_harness.core.types import ToolCall, ToolResult
from ai_agent_harness.testing import FakeProvider

T = TypeVar("T")


POSTGRES_DSN_ENV = "HARNESS_TEST_POSTGRES_DSN"


@contextlib.contextmanager
def fresh_postgres_schema() -> Generator[tuple[str, str]]:
    """A throwaway schema in the database named by `HARNESS_TEST_POSTGRES_DSN`; skips the test if it is not set."""
    dsn = os.environ.get(POSTGRES_DSN_ENV)
    if not dsn:
        pytest.skip(f"{POSTGRES_DSN_ENV} is not set")
    import psycopg
    from psycopg import sql

    schema = f"t_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        yield dsn, schema
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def load_example(name: str = "repo_assistant") -> Any:
    """Loads `examples/<name>.py` by path (examples are scripts, not an importable package)."""
    path = Path(__file__).resolve().parent.parent / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # pydantic resolves the tool input models through sys.modules
    spec.loader.exec_module(module)
    return module


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


async def no_sleep(_: float) -> None:
    return None


class Crash(BaseException):
    """Hard process termination: not an `Exception`, so the runner does not turn it into a status."""


def waiting_state(tenant: str = "t1", run: str = "r1") -> RunState:
    """A minimal valid state for store tests: one user message."""
    state = RunState(agent_name="a", tenant_id=tenant, run_id=run)
    state.add_user_text("hi")
    return state


class EmptyInput(BaseModel):
    pass


class PathInput(BaseModel):
    path: str
    content: str = ""


class Echo(BaseTool[EmptyInput]):
    name: ClassVar[str] = "echo"
    description: ClassVar[str] = "Returns 'echo'."
    input_model: ClassVar[type[BaseModel]] = EmptyInput
    risk: ClassVar[Risk] = Risk.READ

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, args: EmptyInput, state: RunState) -> str:
        self.calls += 1
        return "echo"


class WriteFile(BaseTool[PathInput]):
    name: ClassVar[str] = "write_file"
    description: ClassVar[str] = "Writes a file (test stub)."
    input_model: ClassVar[type[BaseModel]] = PathInput
    risk: ClassVar[Risk] = Risk.WRITE

    def __init__(self) -> None:
        self.written: list[str] = []

    async def run(self, args: PathInput, state: RunState) -> str:
        self.written.append(args.path)
        return f"written: {args.path}"


class SendMail(BaseTool[EmptyInput]):
    name: ClassVar[str] = "send_mail"
    description: ClassVar[str] = "Sends an email (test stub)."
    input_model: ClassVar[type[BaseModel]] = EmptyInput
    risk: ClassVar[Risk] = Risk.IRREVERSIBLE

    def __init__(self) -> None:
        self.sent = 0

    async def run(self, args: EmptyInput, state: RunState) -> str:
        self.sent += 1
        return "sent"


def make_agent(
    script: Sequence[ModelResponse | BaseException],
    tools: Sequence[BaseTool[Any]] = (),
    *,
    hooks: HookPipeline | None = None,
    autonomy: Autonomy = Autonomy.READ_ONLY,
    limits: Limits | None = None,
    **kwargs: Any,
) -> tuple[Agent, FakeProvider, list[Event]]:
    provider = FakeProvider(script)
    events: list[Event] = []
    agent = Agent(
        name="test",
        router=ModelRouter(
            provider, retry=RetryPolicy(max_attempts=2, base_delay_s=0.0, max_delay_s=0.0), sleep=no_sleep
        ),
        context=ContextBuilder("You are a test agent."),
        tools=ToolRegistry(tools),
        hooks=hooks or HookPipeline(),
        autonomy=autonomy,
        limits=limits or Limits(),
        **kwargs,
    )

    async def collect(event: Event) -> None:
        events.append(event)

    agent.events.subscribe(collect)
    return agent, provider, events


def results_of(state: RunState) -> list[ToolResult]:
    return [r for m in state.messages for r in m.tool_results]


def calls_of(state: RunState) -> list[ToolCall]:
    return [c for m in state.messages for c in m.tool_calls]


def assert_consistent(state: RunState) -> None:
    """Every final state must satisfy the protocol — this is the core invariant."""
    state.check_protocol()
    restored = restore_state(state.model_dump_json(), tenant_id=state.tenant_id, run_id=state.run_id)
    assert restored.model_dump() == state.model_dump()
