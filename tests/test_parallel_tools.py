"""Tests for opt-in parallel execution of parallel_safe read-only tools (ADR 0027)."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from ai_agent_harness import (
    Agent,
    Autonomy,
    BaseTool,
    Decision,
    Hook,
    HookPipeline,
    InMemoryCheckpointStore,
    KillSwitch,
    Risk,
    RunState,
    RunStatus,
    resume,
    start,
    tool,
)
from ai_agent_harness.core.errors import ConfigError
from ai_agent_harness.core.hooks import ToolInvocation
from ai_agent_harness.runtime.runner import INTERRUPTED, stream_start
from ai_agent_harness.testing import text_response, tool_response
from ai_agent_harness.tools.base import InputRequest

from .conftest import EmptyInput, WriteFile, make_agent, run_sync
from .test_observability import make_listener


@pytest.fixture
def otel_setup() -> tuple[TracerProvider, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


class ReadDocInput(BaseModel):
    doc_id: str


class SlowRead(BaseTool[ReadDocInput]):
    name: ClassVar[str] = "slow_read"
    description: ClassVar[str] = "Reads a doc with configurable delay."
    input_model: ClassVar[type[BaseModel]] = ReadDocInput
    risk: ClassVar[Risk] = Risk.READ
    parallel_safe: ClassVar[bool] = True

    def __init__(self, delays: dict[str, float] | None = None) -> None:
        self.delays = delays or {}
        self.started: list[str] = []
        self.finished: list[str] = []

    async def run(self, args: ReadDocInput, state: Any) -> str:
        self.started.append(args.doc_id)
        delay = self.delays.get(args.doc_id, 0.01)
        if delay > 0:
            await asyncio.sleep(delay)
        self.finished.append(args.doc_id)
        return f"content of {args.doc_id}"


def test_parallel_safe_configuration_and_validation() -> None:
    # parallel_safe must be a boolean
    class BadTool1(BaseTool[EmptyInput]):
        name: ClassVar[str] = "bad1"
        description: ClassVar[str] = "bad"
        input_model: ClassVar[type[BaseModel]] = EmptyInput
        risk: ClassVar[Risk] = Risk.READ
        parallel_safe: ClassVar[Any] = "yes"

        async def run(self, args: EmptyInput, state: Any) -> str:
            return ""

    with pytest.raises(ConfigError, match=r"'parallel_safe' must be a bool"):
        BadTool1.validate_definition()

    # parallel_safe is only allowed for Risk.READ
    class BadTool2(BaseTool[EmptyInput]):
        name: ClassVar[str] = "bad2"
        description: ClassVar[str] = "bad"
        input_model: ClassVar[type[BaseModel]] = EmptyInput
        risk: ClassVar[Risk] = Risk.WRITE
        parallel_safe: ClassVar[bool] = True

        async def run(self, args: EmptyInput, state: Any) -> str:
            return ""

    with pytest.raises(ConfigError, match=r"parallel_safe is only allowed for Risk\.READ"):
        BadTool2.validate_definition()

    # @tool decorator validation
    with pytest.raises(ConfigError, match=r"parallel_safe is only allowed for Risk\.READ"):

        @tool(risk=Risk.WRITE, parallel_safe=True)
        def bad_tool(path: str) -> str:
            """Bad tool docstring."""
            return path

    # Valid @tool decorator
    @tool(risk=Risk.READ, parallel_safe=True)
    def good_tool(doc_id: str) -> str:
        """Good tool docstring."""
        return doc_id

    @tool(risk=Risk.READ, parallel_safe=False)
    def regular_tool(doc_id: str) -> str:
        """Regular tool docstring."""
        return doc_id

    assert good_tool.parallel_safe is True
    assert good_tool.digest() != ""
    assert good_tool.digest() != regular_tool.digest()

    assert regular_tool.parallel_safe is False

    async def dummy(req: Any) -> str:
        return "ok"

    # Agent max_parallel_tools validation
    with pytest.raises(ConfigError, match=r"Agent\.max_parallel_tools must be a positive integer"):
        Agent.create(dummy, max_parallel_tools=0)

    with pytest.raises(ConfigError, match=r"Agent\.max_parallel_tools must be a positive integer"):
        Agent.create(dummy, max_parallel_tools=-2)


def test_parallel_read_tools_overlap() -> None:
    """Verifies that two parallel_safe read calls demonstrably overlap in time using a barrier."""
    entered = asyncio.Event()
    can_proceed = asyncio.Event()
    overlap_confirmed = False

    class BarrierTool(BaseTool[ReadDocInput]):
        name: ClassVar[str] = "barrier_tool"
        description: ClassVar[str] = "Tool with a barrier to verify concurrent execution."
        input_model: ClassVar[type[BaseModel]] = ReadDocInput
        risk: ClassVar[Risk] = Risk.READ
        parallel_safe: ClassVar[bool] = True

        async def run(self, args: ReadDocInput, state: Any) -> str:
            nonlocal overlap_confirmed
            if args.doc_id == "first":
                entered.set()
                # Wait until the second call has entered
                await asyncio.wait_for(can_proceed.wait(), timeout=1.0)
            elif args.doc_id == "second":
                # Ensure the first call is currently waiting inside its run()
                if entered.is_set():
                    overlap_confirmed = True
                can_proceed.set()
            return f"result_{args.doc_id}"

    barrier = BarrierTool()
    response_calls = tool_response(
        ("barrier_tool", {"doc_id": "first"}),
        ("barrier_tool", {"doc_id": "second"}),
    )
    agent, _, _ = make_agent(
        [response_calls, text_response("done")],
        [barrier],
        max_parallel_tools=2,
    )
    state = run_sync(start(agent, "start"))
    assert state.status is RunStatus.DONE
    assert overlap_confirmed is True


def test_results_in_call_order_even_when_completion_order_is_reversed() -> None:
    """If call 0 finishes after call 1, results in messages and batch_done must preserve call order."""
    slow = SlowRead(delays={"doc_0": 0.05, "doc_1": 0.005})
    response_calls = tool_response(
        ("slow_read", {"doc_id": "doc_0"}),
        ("slow_read", {"doc_id": "doc_1"}),
    )
    agent, _, _ = make_agent(
        [response_calls, text_response("done")],
        [slow],
        max_parallel_tools=2,
    )
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.DONE
    # slow finished doc_1 first, then doc_0
    assert slow.finished == ["doc_1", "doc_0"]

    # But message results must strictly match call order (doc_0, then doc_1)
    tool_results = state.messages[-2].tool_results
    assert len(tool_results) == 2
    assert tool_results[0].content == "content of doc_0"
    assert tool_results[1].content == "content of doc_1"


def test_semaphore_bounds_concurrency() -> None:
    """Verifies that max_parallel_tools limits active concurrency."""
    active = 0
    max_active = 0

    class CountingTool(BaseTool[ReadDocInput]):
        name: ClassVar[str] = "counting_tool"
        description: ClassVar[str] = "Measures active concurrent calls."
        input_model: ClassVar[type[BaseModel]] = ReadDocInput
        risk: ClassVar[Risk] = Risk.READ
        parallel_safe: ClassVar[bool] = True

        async def run(self, args: ReadDocInput, state: Any) -> str:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            active -= 1
            return args.doc_id

    counter = CountingTool()
    calls = tool_response(
        ("counting_tool", {"doc_id": "1"}),
        ("counting_tool", {"doc_id": "2"}),
        ("counting_tool", {"doc_id": "3"}),
        ("counting_tool", {"doc_id": "4"}),
    )
    agent, _, _ = make_agent([calls, text_response("done")], [counter], max_parallel_tools=2)
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.DONE
    assert max_active == 2


def test_write_tool_breaks_parallel_group() -> None:
    """A write tool (or parallel_safe=False tool) splits groups into separate batches."""
    slow = SlowRead(delays={"1": 0.01, "2": 0.01, "3": 0.01, "4": 0.01})
    writer = WriteFile()
    calls = tool_response(
        ("slow_read", {"doc_id": "1"}),
        ("slow_read", {"doc_id": "2"}),
        ("write_file", {"path": "out.txt"}),
        ("slow_read", {"doc_id": "3"}),
        ("slow_read", {"doc_id": "4"}),
    )
    agent, _, _ = make_agent(
        [calls, text_response("done")],
        [slow, writer],
        autonomy=Autonomy.AUTONOMOUS,
        max_parallel_tools=2,
    )
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.DONE
    results = [r.content for r in state.messages[-2].tool_results]
    assert results == [
        "content of 1",
        "content of 2",
        "written: out.txt",
        "content of 3",
        "content of 4",
    ]


def test_approval_mid_group() -> None:
    """When a member of a parallel group requires approval, the first such call pauses the run."""

    class RequireDoc2Approval(Hook):
        async def before_tool(self, state: Any, invocation: ToolInvocation) -> Decision:
            if invocation.call.name == "slow_read" and getattr(invocation.args, "doc_id", None) == "doc_2":
                return Decision.require_approval("Doc 2 is sensitive")
            return Decision()

    slow = SlowRead()
    calls = tool_response(
        ("slow_read", {"doc_id": "doc_1"}),
        ("slow_read", {"doc_id": "doc_2"}),
        ("slow_read", {"doc_id": "doc_3"}),
    )
    agent, _, _ = make_agent(
        [calls, text_response("final done")],
        [slow],
        hooks=HookPipeline([RequireDoc2Approval()]),
        max_parallel_tools=3,
    )
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.WAITING_FOR_APPROVAL
    assert state.pending is not None
    assert state.pending.waiting.call.name == "slow_read"
    # Completed results for doc_1 and doc_3 are in batch.done
    done_ids = [r.call_id for r in state.pending.done]
    open_calls = state.open_tool_calls()
    assert len(done_ids) == 2
    assert done_ids == [open_calls[0].id, open_calls[2].id]

    # Resume with approval
    pending_id = state.pending_id
    assert pending_id is not None
    state2 = run_sync(resume(agent, state, approved=True, pending_id=pending_id))
    assert state2.status is RunStatus.DONE
    assert state2.last_text == "final done"
    # Check that tool results are all in order in the user message
    results = [r.content for r in state2.messages[-2].tool_results]
    assert results == ["content of doc_1", "content of doc_2", "content of doc_3"]


def test_input_request_mid_group() -> None:
    """When a member of a parallel group requests user input, it pauses the run."""

    class AskingTool(BaseTool[ReadDocInput]):
        name: ClassVar[str] = "asking_tool"
        description: ClassVar[str] = "Asks a question."
        input_model: ClassVar[type[BaseModel]] = ReadDocInput
        risk: ClassVar[Risk] = Risk.READ
        parallel_safe: ClassVar[bool] = True

        async def run(self, args: ReadDocInput, state: Any) -> Any:
            if args.doc_id == "ask":
                return InputRequest(question="What is your name?")
            return f"read_{args.doc_id}"

    tool_inst = AskingTool()
    calls = tool_response(
        ("asking_tool", {"doc_id": "1"}),
        ("asking_tool", {"doc_id": "ask"}),
        ("asking_tool", {"doc_id": "2"}),
    )
    agent, _, _ = make_agent([calls, text_response("welcome")], [tool_inst], max_parallel_tools=3)
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.WAITING_FOR_INPUT
    assert state.pending is not None
    assert state.pending.waiting.prompt == "What is your name?"

    pending_id = state.pending_id
    assert pending_id is not None
    state2 = run_sync(resume(agent, state, answer="Alice", pending_id=pending_id))
    assert state2.status is RunStatus.DONE
    results = [r.content for r in state2.messages[-2].tool_results]
    assert results == ["read_1", "Alice", "read_2"]


def test_crash_recovery_in_parallel_group() -> None:
    """Interrupted executions during a crash report outcome unknown (INTERRUPTED)."""

    class AbortingStore(InMemoryCheckpointStore):
        def __init__(self) -> None:
            super().__init__()
            self.count = 0
            self.fail = True
            self.last_run_id: str | None = None

        async def save(self, state: RunState) -> None:
            self.last_run_id = state.run_id
            self.count += 1
            if self.fail and self.count == 2:
                await super().save(state)
                raise RuntimeError("simulated sudden crash")
            if self.fail and self.count > 2:
                raise RuntimeError("process dead")
            await super().save(state)

    store = AbortingStore()
    slow = SlowRead(delays={"1": 0.01, "2": 0.01})
    calls = tool_response(
        ("slow_read", {"doc_id": "1"}),
        ("slow_read", {"doc_id": "2"}),
    )
    agent, _, _ = make_agent([calls, text_response("recovered")], [slow], checkpoints=store, max_parallel_tools=2)

    with pytest.raises(RuntimeError, match=r"process dead|simulated sudden crash"):
        run_sync(start(agent, "start"))

    store.fail = False
    # Now recover the crashed run
    assert store.last_run_id is not None
    crashed_state = run_sync(store.load("default", store.last_run_id))
    assert crashed_state is not None
    assert crashed_state.status is RunStatus.RUNNING

    recovered = run_sync(resume(agent, crashed_state, recover=True))
    assert recovered.status is RunStatus.DONE
    # The interrupted calls were reported as outcome unknown
    results = [r.content for r in recovered.messages[-2].tool_results]
    assert all(INTERRUPTED in r for r in results)


def test_cancellation_in_parallel_group() -> None:
    """Cancelling a parallel group cancels active tasks and sets CANCELLED status."""
    gate = asyncio.Event()

    class HangingTool(BaseTool[ReadDocInput]):
        name: ClassVar[str] = "hanging_tool"
        description: ClassVar[str] = "Hangs until cancelled."
        input_model: ClassVar[type[BaseModel]] = ReadDocInput
        risk: ClassVar[Risk] = Risk.READ
        parallel_safe: ClassVar[bool] = True

        async def run(self, args: ReadDocInput, state: Any) -> str:
            gate.set()
            await asyncio.sleep(10.0)
            return "done"

    hang = HangingTool()
    calls = tool_response(
        ("hanging_tool", {"doc_id": "1"}),
        ("hanging_tool", {"doc_id": "2"}),
    )
    agent, _, _ = make_agent([calls, text_response("never")], [hang], max_parallel_tools=2)

    async def run_cancel() -> RunState:
        handle = stream_start(agent, "start")
        await gate.wait()
        handle.cancel()
        await handle.aclose()
        return handle.state

    state = run_sync(run_cancel())
    assert state.status is RunStatus.CANCELLED
    assert state.status_reason == "Run was cancelled"


def test_kill_switch_with_parallel_group() -> None:
    """KillSwitch blocks execution cleanly when activated."""
    killed = {"value": False}
    ks = KillSwitch(lambda _: killed["value"], reason="Emergency stop")

    slow = SlowRead(delays={"1": 0.01, "2": 0.01})
    calls = tool_response(
        ("slow_read", {"doc_id": "1"}),
        ("slow_read", {"doc_id": "2"}),
    )
    agent, _, _ = make_agent([calls, text_response("done")], [slow], hooks=HookPipeline([ks]), max_parallel_tools=2)

    # Activate killswitch before start
    killed["value"] = True
    state = run_sync(start(agent, "start"))
    assert state.status is RunStatus.BLOCKED
    assert "Emergency stop" in (state.status_reason or "")


def test_two_parallel_spans_of_same_tool(
    otel_setup: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    """Two concurrent calls of the same tool create distinct OpenTelemetry spans using call_id."""
    listener, exporter = make_listener(otel_setup)
    slow = SlowRead(delays={"doc_a": 0.01, "doc_b": 0.01})
    calls_resp = tool_response(
        ("slow_read", {"doc_id": "doc_a"}),
        ("slow_read", {"doc_id": "doc_b"}),
    )

    agent, _, _ = make_agent(
        [calls_resp, text_response("done")],
        [slow],
        max_parallel_tools=2,
    )
    agent.events.subscribe(listener)
    state = run_sync(start(agent, "start"))
    assert state.status is RunStatus.DONE

    spans = exporter.get_finished_spans()
    tool_spans = [s for s in spans if s.name == "execute_tool slow_read"]
    assert len(tool_spans) == 2
    call_ids = {str(s.attributes.get("gen_ai.tool.call_id")) for s in tool_spans if s.attributes}
    assert call_ids == {"call_1", "call_2"}
