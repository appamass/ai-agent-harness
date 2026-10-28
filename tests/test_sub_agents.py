"""Sub-agents as read-only tools (ADR 0030): delegation never widens privileges and never strands a run."""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel

from ai_agent_harness import (
    Agent,
    AskUser,
    Autonomy,
    BaseTool,
    ContextBuilder,
    Decision,
    Event,
    EventType,
    Hook,
    HookPipeline,
    InMemoryCheckpointStore,
    InputRequest,
    Limits,
    ModelResponse,
    ModelRouter,
    RetryPolicy,
    Risk,
    RunState,
    RunStatus,
    SqliteCheckpointStore,
    ToolCall,
    ToolExecutor,
    ToolInvocation,
    ToolRegistry,
    cancel,
    restore_state,
    start,
    sub_agent,
)
from ai_agent_harness.core import ConfigError
from ai_agent_harness.core.types import ToolResult
from ai_agent_harness.runtime.delegation import MAX_SUB_AGENT_DEPTH
from ai_agent_harness.testing import FakeProvider, text_response, tool_response

from .conftest import Echo, EmptyInput, WriteFile, no_sleep, results_of, run_sync


def _agent(
    name: str,
    script: Sequence[ModelResponse | BaseException],
    tools: Sequence[BaseTool[Any]] = (),
    **kwargs: Any,
) -> tuple[Agent, FakeProvider, list[Event]]:
    provider = FakeProvider(script, id=f"fake-{name}")
    events: list[Event] = []
    agent = Agent(
        name=name,
        router=ModelRouter(provider, retry=RetryPolicy(max_attempts=1, request_timeout_s=5.0), sleep=no_sleep),
        context=ContextBuilder(f"You are the {name} agent."),
        tools=ToolRegistry(tools),
        **kwargs,
    )

    async def collect(event: Event) -> None:
        events.append(event)

    agent.events.subscribe(collect)
    return agent, provider, events


def _research(specialist: Agent, **kwargs: Any) -> BaseTool[Any]:
    return sub_agent(specialist, name="research", description="Has a specialist research a topic.", **kwargs)


def test_read_only_parent_delegates_and_gets_the_answer_framed_as_data() -> None:
    specialist, specialist_provider, _ = _agent("specialist", [text_response("The pool size is 20.")])
    parent, _, _ = _agent(
        "parent",
        [tool_response(("research", {"task": "Find the pool size"})), text_response("Pool size: 20.")],
        [_research(specialist)],
    )
    state = run_sync(start(parent, "How big is the pool?", tenant_id="t1"))

    assert state.status is RunStatus.DONE
    [result] = results_of(state)
    assert not result.is_error
    assert result.content.startswith('<untrusted source="research">')
    assert "The pool size is 20." in result.content
    assert specialist_provider.requests[0].messages[0].text == "Find the pool size"


def test_a_sub_run_that_does_not_finish_becomes_a_tool_error_and_the_parent_continues() -> None:
    specialist, _, _ = _agent("specialist", [tool_response(("echo", {}))], [Echo()], limits=Limits(max_steps=1))
    parent, _, _ = _agent(
        "parent",
        [tool_response(("research", {"task": "Look everywhere"})), text_response("I could not find it.")],
        [_research(specialist)],
    )
    state = run_sync(start(parent, "Find it"))

    assert state.status is RunStatus.DONE
    [result] = results_of(state)
    assert result.is_error
    assert "limit_reached" in result.content


@pytest.mark.parametrize("autonomy", [Autonomy.SUGGEST, Autonomy.ACT_WITH_APPROVAL, Autonomy.AUTONOMOUS])
def test_only_read_only_specialists_can_become_sub_agents(autonomy: Autonomy) -> None:
    specialist, _, _ = _agent("specialist", [], [WriteFile()], autonomy=autonomy)
    with pytest.raises(ConfigError, match="read_only"):
        _research(specialist)


def test_a_specialist_that_can_ask_the_user_is_rejected_unless_ask_user_is_hidden() -> None:
    asking, _, _ = _agent("asking", [], [AskUser(), Echo()])
    with pytest.raises(ConfigError, match="ask_user"):
        _research(asking)

    hidden, _, _ = _agent("hidden", [], [AskUser(), Echo()], allowed_tools=frozenset({"echo"}))
    assert _research(hidden).name == "research"


def test_the_timeout_must_cover_the_specialists_time_limit() -> None:
    specialist, _, _ = _agent("specialist", [], limits=Limits(max_seconds=60.0))
    with pytest.raises(ConfigError, match="timeout_s"):
        _research(specialist, timeout_s=60.0)
    # Default: the time limit plus one worst-case model call (the limit is only checked before each call)
    assert _research(specialist).timeout_s >= 60.0 + 5.0


def test_the_tool_name_defaults_to_the_agent_name_and_must_be_valid() -> None:
    specialist, _, _ = _agent("doc-search", [])
    assert sub_agent(specialist, description="Searches the docs.").name == "doc-search"
    with pytest.raises(ConfigError, match="name"):
        sub_agent(specialist, name="no spaces", description="Searches the docs.")


def test_a_specialist_tool_that_asks_a_question_ends_as_an_error_instead_of_a_stranded_run() -> None:
    class Clarify(BaseTool[EmptyInput]):
        name: ClassVar[str] = "clarify"
        description: ClassVar[str] = "Asks the user a question."
        input_model: ClassVar[type[BaseModel]] = EmptyInput
        risk: ClassVar[Risk] = Risk.READ

        async def run(self, args: EmptyInput, state: RunState) -> InputRequest:
            return InputRequest(question="Which one?")

    specialist, _, _ = _agent("specialist", [tool_response(("clarify", {}))], [Clarify()])
    parent, _, _ = _agent(
        "parent", [tool_response(("research", {"task": "Pick one"})), text_response("ok")], [_research(specialist)]
    )
    state = run_sync(start(parent, "Go"))

    assert state.status is RunStatus.DONE
    [result] = results_of(state)
    assert result.is_error
    assert "waiting_for_input" in result.content
    assert "cannot ask" in result.content


def test_sub_run_ids_are_unique_per_call_and_keep_the_tenant() -> None:
    specialist, _, events = _agent("specialist", [text_response("A"), text_response("B")])
    parent, _, _ = _agent(
        "parent",
        [tool_response(("research", {"task": "one"}), ("research", {"task": "two"})), text_response("ok")],
        [_research(specialist)],
    )
    run_sync(start(parent, "Go", tenant_id="tenant-7", run_id="parent-run"))
    started = [e for e in events if e.type is EventType.RUN_STARTED]

    assert len({e.run_id for e in started}) == 2
    assert all(e.tenant_id == "tenant-7" and e.run_id.startswith("sub-") for e in started)


def test_the_same_caller_state_always_yields_the_same_sub_run_id() -> None:
    """What a resumed or recovered caller relies on: its persisted state names the same sub-run again."""
    specialist, _, events = _agent("specialist", [text_response("A"), text_response("B")])
    executor = ToolExecutor([_research(specialist)])
    state = RunState(agent_name="parent", tenant_id="t1", run_id="r1")
    call = ToolCall(id="call_1", name="research", arguments={"task": "one"})
    reloaded = restore_state(state.model_dump_json(), tenant_id="t1", run_id="r1")  # as a store would load it
    for caller in (state.model_copy(deep=True), reloaded):
        run_sync(executor.execute(call, caller))

    first, second = (e.run_id for e in events if e.type is EventType.RUN_STARTED)
    assert first == second


def test_the_same_parent_call_never_starts_a_second_sub_run() -> None:
    store = InMemoryCheckpointStore()
    specialist, provider, _ = _agent("specialist", [text_response("A"), text_response("B")], checkpoints=store)
    executor = ToolExecutor([_research(specialist)])
    state = RunState(agent_name="parent", tenant_id="t1", run_id="r1")
    call = ToolCall(id="call_1", name="research", arguments={"task": "one"})

    first = run_sync(executor.execute(call, state.model_copy(deep=True)))
    again = run_sync(executor.execute(call, state.model_copy(deep=True)))

    assert not getattr(first, "is_error", True)
    assert getattr(again, "is_error", False)
    assert len(provider.requests) == 1


def test_a_specialist_that_delegates_back_to_itself_is_refused() -> None:
    agent, provider, _ = _agent("loop", [tool_response(("myself", {"task": "again"})), text_response("stopped")])
    agent.tools.register(sub_agent(agent, name="myself", description="Delegates to itself."))
    state = run_sync(start(agent, "Go"))

    assert state.status is RunStatus.DONE
    [result] = results_of(state)
    assert result.is_error
    assert "cycle" in result.content
    assert provider.remaining == 0


def test_delegation_depth_is_bounded() -> None:
    count = MAX_SUB_AGENT_DEPTH + 2
    agents: list[tuple[Agent, FakeProvider]] = []
    for index in range(count):
        script = [tool_response(("next", {"task": f"level {index + 1}"})), text_response(f"done {index}")]
        agent, provider, _ = _agent(f"level{index}", script if index < count - 1 else [text_response("deepest")])
        agents.append((agent, provider))
    for (agent, _), (child, _) in itertools.pairwise(agents):
        agent.tools.register(sub_agent(child, name="next", description="Delegates one level deeper."))

    state = run_sync(start(agents[0][0], "Go"))

    assert state.status is RunStatus.DONE
    deepest_allowed, refused = agents[MAX_SUB_AGENT_DEPTH], agents[MAX_SUB_AGENT_DEPTH + 1]
    assert refused[1].requests == []
    last_request = deepest_allowed[1].requests[-1]
    [refusal] = [r for m in last_request.messages for r in m.tool_results]
    assert refusal.is_error and "depth" in refusal.content


def test_a_parent_timeout_cancels_the_running_sub_run() -> None:
    cancelled = asyncio.Event()

    class Slow(BaseTool[EmptyInput]):
        name: ClassVar[str] = "slow"
        description: ClassVar[str] = "Takes a long time."
        input_model: ClassVar[type[BaseModel]] = EmptyInput
        risk: ClassVar[Risk] = Risk.READ
        timeout_s: ClassVar[float] = 30.0

        async def run(self, args: EmptyInput, state: RunState) -> str:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return "late"

    specialist, _, _ = _agent("specialist", [tool_response(("slow", {}))], [Slow()], limits=Limits(max_seconds=0.05))
    parent, _, _ = _agent(
        "parent",
        [tool_response(("research", {"task": "slow"})), text_response("gave up")],
        [_research(specialist, timeout_s=0.2)],
    )
    state = run_sync(start(parent, "Go"))

    assert state.status is RunStatus.DONE
    [result] = results_of(state)
    assert result.is_error and "timeout" in result.content
    assert cancelled.is_set()


def test_sub_run_events_carry_metadata_only() -> None:
    secret = "AT611904300234573201"
    specialist, _, specialist_events = _agent("specialist", [text_response(f"IBAN {secret}")])
    parent, _, parent_events = _agent(
        "parent",
        [tool_response(("research", {"task": f"Check {secret}"})), text_response("ok")],
        [_research(specialist)],
    )
    run_sync(start(parent, "Go"))

    assert specialist_events and parent_events
    assert not [e for e in specialist_events + parent_events if secret in e.model_dump_json()]


def test_a_cycle_back_to_the_top_level_agent_is_refused() -> None:
    top, top_provider, _ = _agent("top", [tool_response(("helper", {"task": "help"})), text_response("done")])
    helper, _, _ = _agent("helper", [tool_response(("top", {"task": "again"})), text_response("helped")])
    top.tools.register(sub_agent(helper, description="Helps."))
    helper.tools.register(sub_agent(top, description="Delegates back to the top agent."))
    state = run_sync(start(top, "Go"))

    assert state.status is RunStatus.DONE
    assert len(top_provider.requests) == 2  # the top agent never ran as its own sub-run
    helper_request = helper.router.providers[0]
    assert isinstance(helper_request, FakeProvider)
    [refusal] = [r for m in helper_request.requests[-1].messages for r in m.tool_results]
    assert refusal.is_error and "cycle" in refusal.content


def test_two_callers_that_share_a_run_id_get_different_sub_runs() -> None:
    store = InMemoryCheckpointStore()
    specialist, provider, events = _agent("specialist", [text_response("A"), text_response("B")], checkpoints=store)
    results: list[ToolResult] = []
    for name in ("triage", "answer"):
        caller, _, _ = _agent(
            name, [tool_response(("research", {"task": "look"})), text_response("ok")], [_research(specialist)]
        )
        results += results_of(run_sync(start(caller, "Go", tenant_id="t", run_id="ticket-1")))

    assert [r.is_error for r in results] == [False, False]
    assert len({e.run_id for e in events if e.type is EventType.RUN_STARTED}) == 2
    assert provider.remaining == 0


def test_a_blank_task_is_an_argument_error_for_the_model() -> None:
    specialist, provider, _ = _agent("specialist", [text_response("unused")])
    parent, _, _ = _agent(
        "parent", [tool_response(("research", {"task": "   "})), text_response("ok")], [_research(specialist)]
    )
    [result] = results_of(run_sync(start(parent, "Go")))

    assert result.is_error and "Invalid arguments" in result.content
    assert provider.requests == []


def test_a_sub_run_that_would_wait_for_approval_is_cancelled_and_reported_as_an_error() -> None:
    class AskFirst(Hook):
        async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
            return Decision.require_approval("Check with a human first")

    store = InMemoryCheckpointStore()
    specialist, _, specialist_events = _agent(
        "specialist", [tool_response(("echo", {}))], [Echo()], hooks=HookPipeline([AskFirst()]), checkpoints=store
    )
    parent, _, _ = _agent(
        "parent", [tool_response(("research", {"task": "Echo"})), text_response("ok")], [_research(specialist)]
    )
    state = run_sync(start(parent, "Go"))

    assert state.status is RunStatus.DONE
    [result] = results_of(state)
    assert result.is_error and "waiting_for_approval" in result.content
    [sub_run_id] = {e.run_id for e in specialist_events}
    stored = run_sync(store.load("default", sub_run_id))
    assert stored is not None and stored.status is RunStatus.CANCELLED and stored.pending is None


def test_ask_user_registered_after_wrapping_is_refused_before_the_sub_run_starts() -> None:
    specialist, provider, _ = _agent("specialist", [text_response("unused")], [Echo()])
    research = _research(specialist)
    specialist.tools.register(AskUser())
    parent, _, _ = _agent("parent", [tool_response(("research", {"task": "Go"})), text_response("ok")], [research])
    [result] = results_of(run_sync(start(parent, "Go")))

    assert result.is_error and "ask_user" in result.content
    assert provider.requests == []


async def close_crashed_sub_runs(specialist: Agent, tenant_id: str, older_than: timedelta) -> int:
    """A copy of the recipe in docs/guide.md (keep both the same): a `running` sub-run older than the tool's timeout was
    left by a crash."""
    store = specialist.checkpoints
    if not isinstance(store, SqliteCheckpointStore):
        raise ValueError("The specialist needs a SQLite or PostgreSQL checkpoint store")
    cutoff = datetime.now(UTC) - older_than
    closed = 0
    # Newest first and the crashed ones are the oldest: ask for the maximum (10,000) and run the job repeatedly
    for info in await store.list_runs(tenant_id, status=RunStatus.RUNNING, limit=10_000):
        if info.run_id.startswith("sub-") and info.updated_at < cutoff:
            state = await store.load(tenant_id, info.run_id)
            if state is not None and state.agent_name == specialist.name:
                await cancel(specialist, state, recover=True)
                closed += 1
    return closed


def test_a_sub_run_left_running_by_a_crash_is_closed_by_the_recovery_recipe_and_then_purged(tmp_path: Path) -> None:
    class Dies(BaseTool[EmptyInput]):
        name: ClassVar[str] = "dies"
        description: ClassVar[str] = "The process dies while this runs."
        input_model: ClassVar[type[BaseModel]] = EmptyInput
        risk: ClassVar[Risk] = Risk.READ

        async def run(self, args: EmptyInput, state: RunState) -> str:
            raise SystemExit

    database = tmp_path / "runs.sqlite"
    specialist, _, _ = _agent("specialist", [tool_response(("dies", {}))], [Dies()])

    async def crash() -> None:
        # A real delegation that crashes, in its own event loop like the other crash tests
        with SqliteCheckpointStore(database) as store:
            await store.setup()
            crashing = dataclasses.replace(specialist, checkpoints=store)
            parent, _, _ = _agent("parent", [tool_response(("research", {"task": "x"}))], [_research(crashing)])
            await start(parent, "Go", tenant_id="t1")

    async def recovery_job() -> tuple[int, RunStatus, int]:
        with SqliteCheckpointStore(database) as store:
            await asyncio.sleep(0.01)
            recovering = dataclasses.replace(specialist, checkpoints=store)
            closed = await close_crashed_sub_runs(recovering, "t1", older_than=timedelta(milliseconds=1))
            [info] = await store.list_runs("t1")
            await asyncio.sleep(0.01)
            purged = await store.purge_finished(older_than=timedelta(milliseconds=1), tenant_id="t1")
            return closed, info.status, purged

    with pytest.raises(SystemExit):
        run_sync(crash())
    assert run_sync(recovery_job()) == (1, RunStatus.CANCELLED, 1)
