"""Shared test helpers. Tests run without pytest plugins: async code executes via `run_sync`."""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import os
import sys
import uuid
from collections.abc import Callable, Coroutine, Generator, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, LiteralString

if TYPE_CHECKING:
    from ai_agent_harness.encryption import AesGcmCodec, StateCodec

import pytest
from pydantic import BaseModel

from ai_agent_harness import (
    Agent,
    Autonomy,
    BaseTool,
    CheckpointStore,
    ContextBuilder,
    Event,
    HookPipeline,
    InMemoryCheckpointStore,
    JsonFileCheckpointStore,
    Limits,
    ModelResponse,
    ModelRouter,
    RetryPolicy,
    Risk,
    RunState,
    ToolRegistry,
    restore_state,
)
from ai_agent_harness.core.hooks import ToolInvocation
from ai_agent_harness.core.types import ToolCall, ToolResult
from ai_agent_harness.testing import FakeProvider

if TYPE_CHECKING:
    from ai_agent_harness.persistence_postgres import PostgresCheckpointStore
    from ai_agent_harness.persistence_sqlite import SqliteCheckpointStore


REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


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
    path = REPO_ROOT / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # pydantic resolves the tool input models through sys.modules
    spec.loader.exec_module(module)
    return module


def run_sync[T](coro: Coroutine[Any, Any, T]) -> T:
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


def invocation(args: BaseModel, *, name: str = "w", risk: Risk = Risk.WRITE) -> ToolInvocation:
    """What a tool hook sees for a call of `name` with validated `args`."""
    return ToolInvocation(
        call=ToolCall(id="1", name=name, arguments=args.model_dump()), tool_name=name, risk=risk, args=args
    )


# ------------------------------------------------------------------ Checkpoint stores


@dataclass
class SqlBackend:
    """One SQL store plus what a contract test needs to look behind it."""

    store: SqliteCheckpointStore | PostgresCheckpointStore
    execute: Callable[..., list[tuple[Any, ...]]]
    """Runs one statement against the store's table. Write `{table}` for it and `%s` for each parameter."""
    reopen: Callable[[], SqliteCheckpointStore | PostgresCheckpointStore]
    """A second store object on the same database, like another process that only shares the database."""


def make_test_codec() -> AesGcmCodec:
    from ai_agent_harness.encryption import AesGcmCodec

    return AesGcmCodec({"k1": b"\x01" * 32}, active="k1")


@contextlib.contextmanager
def _sqlite_backend(tmp_path: Path, codec: StateCodec | None = None) -> Generator[SqlBackend]:
    import sqlite3
    from contextlib import closing

    from ai_agent_harness.persistence_sqlite import SqliteCheckpointStore

    path = tmp_path / f"checkpoints{'_enc' if codec else ''}.sqlite"
    opened = [SqliteCheckpointStore(path, codec=codec)]
    run_sync(opened[0].setup())

    def execute(query: LiteralString, *params: object) -> list[tuple[Any, ...]]:
        with closing(sqlite3.connect(path)) as connection:
            rows = connection.execute(
                query.format(table="ai_agent_harness_checkpoints").replace("%s", "?"), params
            ).fetchall()
            connection.commit()
            return rows

    def reopen() -> SqliteCheckpointStore:
        opened.append(SqliteCheckpointStore(path, codec=codec))
        return opened[-1]

    try:
        yield SqlBackend(opened[0], execute, reopen)
    finally:
        for created in opened:
            created.close()


@contextlib.contextmanager
def _postgres_backend(codec: StateCodec | None = None) -> Generator[SqlBackend]:
    """Needs `HARNESS_TEST_POSTGRES_DSN`; the test is skipped without it."""
    with fresh_postgres_schema() as (dsn, schema):
        import psycopg
        from psycopg import sql

        from ai_agent_harness.persistence_postgres import PostgresCheckpointStore

        created = PostgresCheckpointStore(dsn, schema=schema, codec=codec)
        run_sync(created.setup())

        def execute(query: LiteralString, *params: object) -> list[tuple[Any, ...]]:
            with psycopg.connect(dsn, autocommit=True) as connection:
                statement = sql.SQL(query).format(table=sql.Identifier(schema, "ai_agent_harness_checkpoints"))
                cursor = connection.execute(statement, params)
                return cursor.fetchall() if cursor.description else []

        yield SqlBackend(created, execute, lambda: PostgresCheckpointStore(dsn, schema=schema, codec=codec))


@pytest.fixture
def anyio_backend() -> str:
    """The harness runs on asyncio. Older anyio versions also run each async test on trio, which is not installed."""
    return "asyncio"


@pytest.fixture(params=["sqlite", "postgres", "sqlite_enc", "postgres_enc"])
def sql_backend(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[SqlBackend]:
    """Both SQL stores, plain and encrypted. Postgres is skipped without a database."""
    codec = make_test_codec() if "enc" in request.param else None
    if "sqlite" in request.param:
        with _sqlite_backend(tmp_path, codec=codec) as backend:
            yield backend
    else:
        with _postgres_backend(codec=codec) as backend:
            yield backend


@pytest.fixture
def sql_store(sql_backend: SqlBackend) -> SqliteCheckpointStore | PostgresCheckpointStore:
    return sql_backend.store


@pytest.fixture(
    params=[
        "memory",
        "file",
        "postgres",
        "sqlite",
        "memory_enc",
        "file_enc",
        "postgres_enc",
        "sqlite_enc",
    ]
)
def any_store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[CheckpointStore]:
    """Every shipped checkpoint store, plain and encrypted. Postgres is skipped without a database."""
    codec = make_test_codec() if "enc" in request.param else None
    if "memory" in request.param:
        yield InMemoryCheckpointStore(codec=codec)
    elif "file" in request.param:
        yield JsonFileCheckpointStore(tmp_path / f"cp_{request.param}", codec=codec)
    elif "sqlite" in request.param:
        with _sqlite_backend(tmp_path, codec=codec) as backend:
            yield backend.store
    else:
        with _postgres_backend(codec=codec) as backend:
            yield backend.store
