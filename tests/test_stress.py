"""Randomized stress test: verifies core invariants under arbitrary combinations of faults.

Each scenario randomly combines (with a fixed seed, so fully reproducible) provider errors,
broken hooks, denied approvals, clarifying questions, timeouts, hard process crashes mid-tool,
both checkpoint stores, and all autonomy levels. After every step:

1. The tool-call protocol is intact, status is never `running`, and limits hold.
2. No tool call is executed more than once — not even after a crash and recovery.
3. `IRREVERSIBLE` runs only after human approval; `WRITE` never runs in `read_only`, and in
   `suggest`/`act_with_approval` only after approval.
4. No tool bypasses `before_tool` hooks, and no tool runs if a hook blocked it or crashed
   while checking it (fail-closed).
5. A tool that completed keeps its real result after a later crash; only the genuinely
   interrupted tool is reported as "unknown".
6. Every executed call appears in the history — even after a crash mid-tool (write-ahead).
   Otherwise the model would not know about the side effect and might repeat it.
7. The persisted state matches the returned state.

Default: 300 scenarios (a few seconds). For extended runs: `HARNESS_STRESS=5000 pytest tests/test_stress.py`.
Any failure prints the seed; `run_scenario(seed)` reproduces it deterministically.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import os
import random
import tempfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel

from ai_agent_harness import (
    Agent,
    AskUser,
    Autonomy,
    BaseTool,
    Capabilities,
    CheckpointStore,
    ContextBuilder,
    Decision,
    Hook,
    HookPipeline,
    InMemoryCheckpointStore,
    JsonFileCheckpointStore,
    Limits,
    Message,
    ModelRequest,
    ModelResponse,
    ModelRouter,
    RetryPolicy,
    Risk,
    RunState,
    RunStatus,
    SqliteCheckpointStore,
    StopReason,
    ToolCall,
    ToolError,
    ToolInvocation,
    ToolRegistry,
    ToolResult,
    resume,
    start,
)
from ai_agent_harness.core import ProviderError, TextBlock, ThinkingBlock, Usage
from ai_agent_harness.runtime.runner import INTERRUPTED

from .conftest import POSTGRES_DSN_ENV, Crash, fresh_postgres_schema, no_sleep

SCENARIOS = int(os.environ.get("HARNESS_STRESS", "300"))
TOOL_NAMES = ("echo", "write", "mail", "flaky", "crashy", "ask_user", "ghost")  # ghost = unknown tool


@dataclass
class Tracker:
    """Observes a scenario from the outside — independent of the code under test."""

    executed: Counter[str] = field(default_factory=Counter[str])
    approved: set[str] = field(default_factory=set[str])
    risk_of: dict[str, Risk] = field(default_factory=dict[str, Risk])
    seen_args: dict[int, tuple[BaseModel, str]] = field(default_factory=dict[int, tuple[BaseModel, str]])
    bypassed_hooks: int = 0
    completed: set[str] = field(default_factory=set[str])  # tool returned normally
    vetoed: set[str] = field(default_factory=set[str])  # blocked by a hook or hook crashed


class Empty(BaseModel):
    pass


class Number(BaseModel):
    x: int


# ------------------------------------------------------------------ Tools


class _Tracked(BaseTool[BaseModel]):
    timeout_s: ClassVar[float] = 0.05
    behaviour: ClassVar[str] = "plain"

    def __init__(self, tracker: Tracker, rng: random.Random) -> None:
        self.tracker = tracker
        self.rng = rng

    async def run(self, args: BaseModel, state: RunState) -> str:
        seen = self.tracker.seen_args.get(id(args))
        if seen is None or seen[0] is not args:
            self.tracker.bypassed_hooks += 1
            return "bypassed hooks"
        roll = self.rng.random()
        if self.behaviour == "flaky":
            if roll < 0.2:
                raise ToolError("broken")
            if roll < 0.3:
                raise RuntimeError("secret detail")
            if roll < 0.35:
                await asyncio.sleep(1)  # longer than timeout_s
        self.tracker.executed[seen[1]] += 1
        if self.behaviour == "crashy" and roll < 0.5:
            raise Crash  # side effect happened, result is lost
        self.tracker.completed.add(seen[1])
        return f"ok:{self.name}" + "." * 250  # long enough for context compaction to clear it later


class EchoTool(_Tracked):
    name: ClassVar[str] = "echo"
    description: ClassVar[str] = "reads"
    input_model: ClassVar[type[BaseModel]] = Empty
    risk: ClassVar[Risk] = Risk.READ


class WriteTool(_Tracked):
    name: ClassVar[str] = "write"
    description: ClassVar[str] = "writes"
    input_model: ClassVar[type[BaseModel]] = Number
    risk: ClassVar[Risk] = Risk.WRITE


class MailTool(_Tracked):
    name: ClassVar[str] = "mail"
    description: ClassVar[str] = "sends"
    input_model: ClassVar[type[BaseModel]] = Number
    risk: ClassVar[Risk] = Risk.IRREVERSIBLE


class FlakyTool(_Tracked):
    name: ClassVar[str] = "flaky"
    description: ClassVar[str] = "flaky"
    input_model: ClassVar[type[BaseModel]] = Empty
    risk: ClassVar[Risk] = Risk.READ
    behaviour: ClassVar[str] = "flaky"


class CrashyTool(_Tracked):
    name: ClassVar[str] = "crashy"
    description: ClassVar[str] = "crashes"
    input_model: ClassVar[type[BaseModel]] = Empty
    risk: ClassVar[Risk] = Risk.WRITE
    behaviour: ClassVar[str] = "crashy"


# ------------------------------------------------------------------ Hooks


class TrackingHook(Hook):
    """First hook: records which argument object belongs to which call."""

    def __init__(self, tracker: Tracker) -> None:
        self.tracker = tracker

    async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
        self.tracker.seen_args[id(invocation.args)] = (invocation.args, invocation.call.id)
        self.tracker.risk_of[invocation.call.id] = invocation.risk
        return Decision()


class ChaosHook(Hook):
    def __init__(self, rng: random.Random, tracker: Tracker) -> None:
        self.rng = rng
        self.tracker = tracker

    async def on_user_input(self, state: RunState, text: str) -> Decision:
        return Decision.block("no") if self.rng.random() < 0.05 else Decision()

    async def before_model(self, state: RunState, request: ModelRequest) -> Decision:
        roll = self.rng.random()
        if roll < 0.03:
            raise ValueError("hook error")
        return Decision.block("budget") if roll < 0.06 else Decision()

    async def after_model(self, state: RunState, response: ModelResponse) -> Decision:
        roll = self.rng.random()
        if roll < 0.1:
            return Decision.retry("format")
        if roll < 0.13:
            return Decision.block("forbidden")
        if roll < 0.15:
            return Decision.require_approval("not allowed at this stage")  # → must block
        return Decision()

    async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
        roll = self.rng.random()
        if roll < 0.08:
            self.tracker.vetoed.add(invocation.call.id)
            return Decision.block("nope")
        if roll < 0.16:
            return Decision.require_approval("review")
        if roll < 0.18:
            self.tracker.vetoed.add(invocation.call.id)  # fail-closed: must act like a block
            raise RuntimeError("hook error")
        return Decision()

    async def after_tool(self, state: RunState, invocation: ToolInvocation, result: ToolResult) -> Decision:
        return Decision.block("secret") if self.rng.random() < 0.05 else Decision()

    async def on_run_end(self, state: RunState) -> None:
        if self.rng.random() < 0.1:
            raise RuntimeError("error at end")


# ------------------------------------------------------------------ Provider


class ChaosProvider:
    def __init__(self, rng: random.Random, provider_id: str, ids: Iterator[int]) -> None:
        self.rng = rng
        self._id = provider_id
        self._ids = ids

    @property
    def id(self) -> str:
        return self._id

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(tools=True, vision=True)

    def _call(self) -> ToolCall:
        name = self.rng.choice(TOOL_NAMES)
        args: dict[str, Any] = {}
        if name == "ask_user":
            args = {"question": "?"} if self.rng.random() < 0.9 else {}
        elif name in ("write", "mail") and self.rng.random() < 0.85:
            args = {"x": self.rng.randint(0, 9)}
        elif self.rng.random() < 0.1:
            args = {"unknown_field": 1}
        call_id = f"{self._id}-{next(self._ids)}"
        if self.rng.random() < 0.05:  # model returned malformed JSON
            return ToolCall(id=call_id, name=name, arguments_error="Expecting value: line 1 column 1")
        return ToolCall(id=call_id, name=name, arguments=args)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        roll = self.rng.random()
        if roll < 0.04:
            raise ProviderError("429", retryable=True)
        if roll < 0.06:
            raise ProviderError("400", retryable=False)
        if roll < 0.07:
            raise RuntimeError("adapter error")
        blocks: list[TextBlock | ThinkingBlock | ToolCall] = []
        if self.rng.random() < 0.2:
            blocks.append(ThinkingBlock(text="hm", provider_data={"signature": "x"}))
        kind = self.rng.random()
        if kind < 0.45:
            blocks += [self._call() for _ in range(self.rng.randint(1, 3))]
            stop = self.rng.choice([StopReason.TOOL_USE] * 6 + [StopReason.END, StopReason.MAX_TOKENS])
        elif kind < 0.55:
            blocks.append(TextBlock(text="..."))
            stop = StopReason.TOOL_USE  # announced tool use, but no call block
        elif kind < 0.62:
            blocks.append(TextBlock(text="part"))
            stop = StopReason.MAX_TOKENS
        elif kind < 0.66:
            blocks.append(TextBlock(text="no"))
            stop = StopReason.REFUSAL
        else:
            blocks.append(TextBlock(text="done"))
            stop = StopReason.END
        output = self.rng.randint(1, 50)
        usage = Usage(
            input_tokens=self.rng.randint(1, 50),
            output_tokens=output,
            cache_read_tokens=self.rng.randint(0, 200),
            reasoning_tokens=self.rng.randint(0, output),
        )
        return ModelResponse(message=Message(role="assistant", content=tuple(blocks)), stop_reason=stop, usage=usage)


def _build_agent(rng: random.Random, tracker: Tracker, store: CheckpointStore, autonomy: Autonomy) -> Agent:
    ids = itertools.count()  # unique call IDs across both providers
    tools: list[BaseTool[Any]] = [
        EchoTool(tracker, rng),
        WriteTool(tracker, rng),
        MailTool(tracker, rng),
        FlakyTool(tracker, rng),
        CrashyTool(tracker, rng),
        AskUser(),
    ]
    return Agent(
        name="stress",
        router=ModelRouter(
            ChaosProvider(rng, "p1", ids),
            [ChaosProvider(rng, "p2", ids)],
            RetryPolicy(max_attempts=2, base_delay_s=0.0, max_delay_s=0.0, request_timeout_s=1),
            sleep=no_sleep,
        ),
        context=ContextBuilder("stress"),
        tools=ToolRegistry(tools),
        autonomy=autonomy,
        hooks=HookPipeline([TrackingHook(tracker), ChaosHook(rng, tracker)]),
        checkpoints=store,
        limits=Limits(
            max_steps=rng.randint(2, 12),
            max_total_tokens=rng.randint(100, 2000),
            max_consecutive_retries=rng.randint(0, 3),
            max_context_chars=rng.choice([400_000, 400_000, 1_500, 3_000]),
            max_context_tokens=rng.choice([None, None, 800]),
        ),
        compact_context=rng.random() < 0.5,
    )


# ------------------------------------------------------------------ Invariants


def _check(state: RunState, agent: Agent, tracker: Tracker, seed: int) -> None:
    where = f"Seed {seed}"
    state.check_protocol()
    assert state.status is not RunStatus.RUNNING, f"{where}: status 'running' after return"
    assert state.step <= agent.limits.max_steps, f"{where}: step limit exceeded"
    assert tracker.bypassed_hooks == 0, f"{where}: tool bypassed before_tool"
    assert agent.compact_context or state.context_floor == 0, f"{where}: context cleared without compact_context"
    stored = [r.content for message in state.messages for r in message.tool_results]
    assert not any("Result removed to save space" in text for text in stored), f"{where}: stored result was cleared"
    in_history = {call.id for message in state.messages for call in message.tool_calls}
    untracked = set(tracker.executed) - in_history
    assert not untracked, f"{where}: side effect without trace in history (write-ahead missing): {untracked}"
    lost = {
        result.call_id
        for message in state.messages
        for result in message.tool_results
        if result.call_id in tracker.completed and result.content == INTERRUPTED
    }
    assert not lost, f"{where}: completed result reported as 'unknown' after crash: {lost}"
    vetoed_but_run = set(tracker.executed) & tracker.vetoed
    assert not vetoed_but_run, f"{where}: blocked call ran anyway: {vetoed_but_run}"
    for call_id, count in tracker.executed.items():
        assert count <= 1, f"{where}: call {call_id} executed {count}x"
        risk = tracker.risk_of.get(call_id)
        approved = call_id in tracker.approved
        if risk is Risk.IRREVERSIBLE:
            assert approved, f"{where}: irreversible call {call_id} ran without approval"
        if risk is Risk.WRITE:
            assert agent.autonomy is not Autonomy.READ_ONLY, f"{where}: WRITE {call_id} ran in read_only"
            if agent.autonomy in (Autonomy.SUGGEST, Autonomy.ACT_WITH_APPROVAL):
                assert approved, f"{where}: WRITE {call_id} ran in {agent.autonomy} without approval"


async def _next_step(agent: Agent, state: RunState, rng: random.Random, tracker: Tracker) -> RunState | None:
    """Reacts like a human or worker. Returns None when the scenario ends."""
    if state.status is RunStatus.WAITING_FOR_INPUT:
        return await resume(agent, state, answer=rng.choice(["yes", "no", "a.txt"]), pending_id=state.pending_id)
    if state.status is RunStatus.WAITING_FOR_APPROVAL:
        approve = rng.random() < 0.6
        if approve and state.pending is not None:
            tracker.approved.add(state.pending.waiting.call.id)
        return await resume(
            agent, state, approved=approve, answer=rng.choice([None, "Comment"]), pending_id=state.pending_id
        )
    last = state.last_message
    if last is not None and last.role == "user" and rng.random() < 0.5:
        return await resume(agent, state)
    if rng.random() < 0.6:
        return await resume(agent, state, user_input="continue")
    return None


async def run_scenario(seed: int, stats: Counter[str] | None = None) -> None:
    stats = stats if stats is not None else Counter[str]()
    rng = random.Random(seed)
    tracker = Tracker()
    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        roll = rng.random()
        if roll < 0.33:
            store: CheckpointStore = JsonFileCheckpointStore(tmp)
        elif roll < 0.66:
            sqlite_store = SqliteCheckpointStore(Path(tmp) / "stress.sqlite")
            await sqlite_store.setup()
            stack.callback(sqlite_store.close)
            store = sqlite_store
        else:
            store = InMemoryCheckpointStore()
        if os.environ.get(POSTGRES_DSN_ENV) and seed % 4 == 0:  # every fourth scenario runs on a real PostgreSQL
            from ai_agent_harness.persistence_postgres import PostgresCheckpointStore

            dsn, schema = stack.enter_context(fresh_postgres_schema())
            postgres = PostgresCheckpointStore(dsn, schema=schema)
            await postgres.setup()
            store = postgres
        agent = _build_agent(rng, tracker, store, rng.choice(list(Autonomy)))
        run_id = f"r{seed}"
        state: RunState | None
        try:
            state = await start(agent, "go", run_id=run_id)
        except Crash:
            stats["crash"] += 1
            state = None
        for _ in range(12):
            if state is None:  # after crash: load from the store like a fresh worker
                loaded = await store.load("default", run_id)
                assert loaded is not None, f"Seed {seed}: crash without persisted state"
                if loaded.status is not RunStatus.RUNNING:
                    state = loaded
                    continue
                try:
                    state = await resume(agent, loaded, recover=True)
                    stats["recovered"] += 1
                except Crash:
                    stats["crash"] += 1
                continue

            _check(state, agent, tracker, seed)
            stats[str(state.status)] += 1
            loaded = await store.load("default", run_id)
            assert loaded is not None
            assert loaded.model_dump() == state.model_dump(), f"Seed {seed}: store diverged from state"
            try:
                state = await _next_step(agent, loaded, rng, tracker)
            except Crash:
                stats["crash"] += 1
                state = None
                continue
            if state is None:
                return
        if state is not None:
            _check(state, agent, tracker, seed)


@pytest.fixture(autouse=True)
def _quiet_logs() -> Iterator[None]:
    """Scenarios intentionally trigger many error logs; suppress them here."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def test_invariants_hold_under_chaos() -> None:
    stats = Counter[str]()

    async def main() -> None:
        for seed in range(SCENARIOS):
            await run_scenario(seed, stats)

    asyncio.run(main())
    # Guards the stress test itself: without hitting these paths, it would not be testing enough.
    for key in ("crash", "recovered", "done", "blocked", "failed", "limit_reached", "waiting_for_approval"):
        assert stats[key] > 0, f"Path '{key}' was never reached: {dict(stats)}"
