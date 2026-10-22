"""Tests for streaming events and incremental text deltas (ADR 0025).

Verifies:
- Standard streaming yielding lifecycle Events and TextDeltas
- Safe cancellation mid-stream persisting RunStatus.CANCELLED
- Mid-stream provider failure persisting RunStatus.FAILED
- Protocol integrity: partial stream tokens are NEVER committed to RunState
- Backpressure support for slow consumers
- Tool execution during streaming
- Pause and stream_resume flow
- Structured final output during streaming
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, ClassVar, cast

import pytest
from pydantic import BaseModel, ConfigDict, Field

from ai_agent_harness import (
    Agent,
    Autonomy,
    BaseTool,
    Capabilities,
    Event,
    EventType,
    InMemoryCheckpointStore,
    Message,
    ModelRequest,
    ModelResponse,
    ModelRouter,
    RetryPolicy,
    Risk,
    RunState,
    RunStatus,
    StreamDelta,
    StreamHandle,
    TextDelta,
    resume,
    start,
    stream_resume,
    stream_start,
)
from ai_agent_harness.core.errors import ConcurrencyError, ConfigError, ProviderError
from ai_agent_harness.testing import FakeProvider, Streamed, text_response, tool_response

from .conftest import Crash, WriteFile, make_agent, no_sleep


class SimpleEchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str = Field(min_length=1)


class SimpleEchoTool(BaseTool[SimpleEchoInput]):
    name: ClassVar[str] = "echo_tool"
    description: ClassVar[str] = "Echoes input back"
    input_model: ClassVar[type[BaseModel]] = SimpleEchoInput
    risk: ClassVar[Risk] = Risk.READ

    async def run(self, args: SimpleEchoInput, state: RunState) -> str:
        return f"echo: {args.value}"


class ApprovalActionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_name: str


class IrreversibleTool(BaseTool[ApprovalActionInput]):
    name: ClassVar[str] = "remediate"
    description: ClassVar[str] = "Performs irreversible remediation"
    input_model: ClassVar[type[BaseModel]] = ApprovalActionInput
    risk: ClassVar[Risk] = Risk.IRREVERSIBLE

    async def run(self, args: ApprovalActionInput, state: RunState) -> str:
        return f"remediated: {args.action_name}"


_REQUEST = ModelRequest(system="s", messages=(Message.user("hi"),))


async def _collect(stream: StreamHandle) -> tuple[list[EventType], list[str]]:
    """Reads the stream to its end: the event types and the text deltas, in order."""
    events: list[EventType] = []
    deltas: list[str] = []
    async for item in stream:
        if isinstance(item, Event):
            events.append(item.type)
        elif isinstance(item, TextDelta):
            deltas.append(item.text)
    return events, deltas


async def _wait_done(stream: StreamHandle) -> None:
    for _ in range(100):
        if stream.done():
            break
        await asyncio.sleep(0.05)


async def _saved_status(store: InMemoryCheckpointStore, stream: StreamHandle) -> RunStatus:
    saved = await store.load("default", stream.state.run_id)
    assert saved is not None
    return saved.status


@pytest.mark.anyio
async def test_stream_basic_text_deltas_and_events() -> None:
    store = InMemoryCheckpointStore()
    provider = FakeProvider([text_response("Hello brave new streaming world")])
    agent = Agent.create(provider, "You are helpful.", name="stream_agent", checkpoints=store)

    stream = stream_start(agent, "Hi there")
    assert isinstance(stream, StreamHandle)

    events, deltas = await _collect(stream)

    assert "".join(deltas) == "Hello brave new streaming world"
    assert EventType.RUN_STARTED in events
    assert EventType.MODEL_REQUEST in events
    assert EventType.MODEL_RESPONSE in events
    assert EventType.RUN_FINISHED in events

    state = stream.state
    assert state.status is RunStatus.DONE
    assert state.last_text == "Hello brave new streaming world"
    state.check_protocol()

    # Verify state saved in checkpoint store
    assert await _saved_status(store, stream) is RunStatus.DONE


@pytest.mark.anyio
async def test_stream_mid_stream_cancellation_persists_cancelled() -> None:
    store = InMemoryCheckpointStore()

    class PausingStreamingProvider:
        id = "pausing"
        capabilities = Capabilities(tools=True)

        async def complete(self, request: ModelRequest) -> ModelResponse:
            return text_response("Full text")

        async def stream(self, request: ModelRequest) -> AsyncIterator[StreamDelta | ModelResponse]:
            yield StreamDelta(text="First part ")
            yield StreamDelta(text="Second part ")
            # Pauses so the consumer can cancel mid-stream
            await asyncio.sleep(10.0)
            yield StreamDelta(text="Third part ")
            yield text_response("Full text")

    agent = Agent.create(PausingStreamingProvider(), "Prompt", name="cancel_agent", checkpoints=store)

    deltas_received: list[str] = []
    stream = stream_start(agent, "Start stream")

    async with stream:
        async for item in stream:
            if isinstance(item, TextDelta):
                deltas_received.append(item.text)
                if len(deltas_received) == 2:
                    # Consumer cancels the stream mid-flight while provider is waiting
                    await stream.aclose()
                    break

    state = stream.state
    assert state.status is RunStatus.CANCELLED
    # Crucial guarantee: partial text is NEVER committed to messages
    state.check_protocol()
    assert state.last_message is not None
    assert state.last_message.role == "user"

    # Checkpoint store also reflects CANCELLED
    assert await _saved_status(store, stream) is RunStatus.CANCELLED


@pytest.mark.anyio
async def test_stream_mid_stream_provider_failure_persists_failed() -> None:
    store = InMemoryCheckpointStore()
    # Provider yields one delta then dies with ProviderError
    provider = FakeProvider(
        [Streamed(StreamDelta(text="Chunk before crash "), ProviderError("Network connection reset", retryable=False))]
    )
    agent = Agent.create(provider, "Prompt", name="crash_agent", checkpoints=store)

    stream = stream_start(agent, "Test fail")
    events, deltas = await _collect(stream)

    assert deltas == ["Chunk before crash "]
    assert EventType.RUN_FINISHED in events

    state = stream.state
    assert state.status is RunStatus.FAILED
    assert "Model call failed" in (state.status_reason or "")
    # Invariant: protocol must be strictly valid
    state.check_protocol()

    assert await _saved_status(store, stream) is RunStatus.FAILED


@pytest.mark.anyio
async def test_stream_backpressure_with_slow_consumer() -> None:
    # A tiny queue_size=2 ensures backpressure is exercised
    provider = FakeProvider([text_response("one two three four five six seven eight nine ten")])
    agent = Agent.create(provider, "Prompt", name="slow_agent")

    stream = stream_start(agent, "Test backpressure", queue_size=2)
    consumed_deltas: list[str] = []

    async for item in stream:
        if isinstance(item, TextDelta):
            consumed_deltas.append(item.text)
            # Simulate slow processing per chunk
            await asyncio.sleep(0.005)

    assert len(consumed_deltas) == 10
    assert "".join(consumed_deltas) == "one two three four five six seven eight nine ten"
    assert stream.state.status is RunStatus.DONE


@pytest.mark.anyio
async def test_stream_with_tool_execution() -> None:
    store = InMemoryCheckpointStore()
    provider = FakeProvider(
        [
            tool_response(("echo_tool", {"value": "harness-stream"})),
            text_response("Tool executed successfully."),
        ]
    )
    agent = Agent.create(provider, "Prompt", name="tool_agent", tools=[SimpleEchoTool()], checkpoints=store)

    stream = stream_start(agent, "Run tool")
    events, deltas = await _collect(stream)

    assert EventType.TOOL_STARTED in events
    assert EventType.TOOL_FINISHED in events
    assert "".join(deltas) == "Tool executed successfully."

    state = stream.state
    assert state.status is RunStatus.DONE
    state.check_protocol()
    assert state.step == 2


@pytest.mark.anyio
async def test_stream_pause_and_stream_resume() -> None:
    store = InMemoryCheckpointStore()
    provider = FakeProvider(
        [
            tool_response(("remediate", {"action_name": "rollback-v1"})),
            text_response("Remediation complete."),
        ]
    )
    agent = Agent.create(
        provider,
        "Prompt",
        name="approval_agent",
        tools=[IrreversibleTool()],
        autonomy=Autonomy.ACT_WITH_APPROVAL,
        checkpoints=store,
    )

    # Initial stream pauses for approval
    stream1 = stream_start(agent, "Do rollback")
    events1, _ = await _collect(stream1)

    state1 = stream1.state
    assert state1.status is RunStatus.WAITING_FOR_APPROVAL
    assert state1.pending_id is not None
    assert EventType.RUN_FINISHED in events1

    # Resume the stream with approval
    stream2 = stream_resume(agent, state1, approved=True, pending_id=state1.pending_id)
    _, deltas2 = await _collect(stream2)

    state2 = stream2.state
    assert state2.status is RunStatus.DONE
    assert "".join(deltas2) == "Remediation complete."
    state2.check_protocol()


@pytest.mark.anyio
async def test_stream_structured_output() -> None:
    class FinalScore(BaseModel):
        model_config = ConfigDict(extra="forbid")
        score: int
        verdict: str

    provider = FakeProvider([text_response('{"score": 98, "verdict": "excellent"}')])
    agent = Agent.create(provider, "Prompt", name="struct_agent")

    state = await stream_start(agent, "Evaluate", output_model=FinalScore)
    assert state.status is RunStatus.DONE
    typed_output = state.output_as(FinalScore)
    assert typed_output.score == 98
    assert typed_output.verdict == "excellent"
    state.check_protocol()


# ------------------------------------------------------------------ Review fixes: closing and abandoning a stream


def _many_agent(store: InMemoryCheckpointStore) -> Agent:
    """Streams far more deltas than any queue holds: the run is blocked on a full queue."""
    deltas = [StreamDelta(text=f"w{i} ") for i in range(500)]
    return Agent.create(
        FakeProvider([Streamed(*deltas, text_response("done"))]), "Prompt", name="many_agent", checkpoints=store
    )


@pytest.mark.anyio
async def test_aclose_with_full_queue_and_stalled_consumer_does_not_hang() -> None:
    store = InMemoryCheckpointStore()
    stream = stream_start(_many_agent(store), "go", queue_size=2)
    await asyncio.sleep(0.1)  # the consumer reads nothing: the queue is full and the run is blocked

    await asyncio.wait_for(stream.aclose(), timeout=3)

    assert stream.state.status is RunStatus.CANCELLED
    assert await _saved_status(store, stream) is RunStatus.CANCELLED
    stream.state.check_protocol()


@pytest.mark.anyio
async def test_consumer_that_stops_reading_without_closing_cancels_the_run() -> None:
    store = InMemoryCheckpointStore()
    stream = stream_start(_many_agent(store), "go", queue_size=2, idle_timeout_s=0.2)
    async for _ in stream:
        break  # abandons the stream, never calls aclose()

    await _wait_done(stream)

    assert stream.done(), "the run task must not leak"
    assert await _saved_status(store, stream) is RunStatus.CANCELLED


@pytest.mark.anyio
async def test_stream_handle_cancel_with_full_queue_does_not_hang() -> None:
    store = InMemoryCheckpointStore()
    stream = stream_start(_many_agent(store), "go", queue_size=2)
    await asyncio.sleep(0.1)
    stream.cancel()
    await _wait_done(stream)
    assert stream.done()
    assert stream.state.status is RunStatus.CANCELLED


def test_stream_arguments_are_validated() -> None:
    agent = _many_agent(InMemoryCheckpointStore())
    with pytest.raises(ConfigError, match="queue_size"):
        stream_start(agent, "go", queue_size=0)
    with pytest.raises(ConfigError, match="idle_timeout_s"):
        stream_start(agent, "go", idle_timeout_s=0)


@pytest.mark.anyio
async def test_router_stream_enforces_request_timeout() -> None:
    class Stalling:
        id = "stalling"
        capabilities = Capabilities(tools=True)

        async def complete(self, request: ModelRequest) -> ModelResponse:
            return text_response("x")

        async def stream(self, request: ModelRequest) -> AsyncIterator[StreamDelta | ModelResponse]:
            await asyncio.sleep(30)
            yield text_response("never")

    router = ModelRouter(Stalling(), retry=RetryPolicy(max_attempts=1, request_timeout_s=0.1), sleep=no_sleep)
    with pytest.raises(ProviderError, match="timed out"):
        await asyncio.wait_for(router.stream(_REQUEST), timeout=3)


@pytest.mark.anyio
async def test_stream_and_plain_run_end_in_the_same_state() -> None:
    def make() -> Agent:
        return Agent.create(FakeProvider([text_response("same answer")]), "Prompt", name="parity")

    plain = await start(make(), "hello")
    streamed = await stream_start(make(), "hello")

    assert streamed.status is plain.status is RunStatus.DONE
    assert [m.text for m in streamed.messages] == [m.text for m in plain.messages]


# ------------------------------------------------------------------ Cancel, early close, await, slow consumer


@pytest.mark.anyio
async def test_iteration_after_cancel_with_full_queue_terminates() -> None:
    store = InMemoryCheckpointStore()
    stream = stream_start(_many_agent(store), "go", queue_size=2)
    await asyncio.sleep(0.1)
    stream.cancel()

    async def drain() -> None:
        async for _ in stream:
            pass

    await asyncio.wait_for(drain(), timeout=3)  # ends once the buffered items are read
    assert stream.done()
    assert stream.state.status is RunStatus.CANCELLED


@pytest.mark.anyio
async def test_aclose_right_after_start_still_persists_cancelled() -> None:
    store = InMemoryCheckpointStore()
    stream = stream_start(_many_agent(store), "go")
    await asyncio.wait_for(stream.aclose(), timeout=3)  # the task has not even started yet

    assert stream.state.status is RunStatus.CANCELLED
    assert await _saved_status(store, stream) is RunStatus.CANCELLED


@pytest.mark.anyio
async def test_awaiting_the_handle_returns_the_final_state_without_iterating() -> None:
    store = InMemoryCheckpointStore()
    stream = stream_start(_many_agent(store), "go", queue_size=2)  # far more items than the queue holds

    state = await asyncio.wait_for(stream, timeout=5)

    assert state.status is RunStatus.DONE


@pytest.mark.anyio
async def test_slow_consumer_does_not_count_against_the_provider_timeout() -> None:
    async def slow(_: StreamDelta) -> None:
        await asyncio.sleep(0.15)  # 3 x 0.15 s > request_timeout_s

    fast = FakeProvider([text_response("a b c")], id="fast")  # streams "a ", "b ", "c", then the answer
    router = ModelRouter(fast, retry=RetryPolicy(max_attempts=1, request_timeout_s=0.3))
    response, provider_id = await router.stream(_REQUEST, on_delta=slow)

    assert response.message.text == "a b c"
    assert provider_id == "fast"


# ------------------------------------------------------------------ Resume closed early, heartbeat, deadline


@pytest.mark.anyio
async def test_stream_resume_closed_before_it_starts_leaves_a_waiting_run_untouched() -> None:
    store = InMemoryCheckpointStore()
    write = WriteFile()
    agent, _, _ = make_agent(
        [tool_response(("write_file", {"path": "a.txt"})), text_response("done")],
        [write],
        autonomy=Autonomy.SUGGEST,
        checkpoints=store,
    )
    waiting = await start(agent, "write it", run_id="w1")
    assert waiting.status is RunStatus.WAITING_FOR_APPROVAL
    shown = waiting.pending_id
    before = waiting.model_dump_json()

    handle = stream_resume(agent, waiting, approved=True, pending_id=shown)
    await asyncio.wait_for(handle.aclose(), timeout=3)

    assert waiting.model_dump_json() == before  # nothing was claimed, nothing was overwritten
    assert write.written == []
    saved = await store.load("default", "w1")
    assert saved is not None and saved.status is RunStatus.WAITING_FOR_APPROVAL
    done = await resume(agent, saved, approved=True, pending_id=shown)  # the approval still works afterwards
    assert done.status is RunStatus.DONE and write.written == ["a.txt"]


@pytest.mark.anyio
async def test_stream_resume_recover_closed_before_it_starts_keeps_the_outcome_unknown_marker() -> None:
    store = InMemoryCheckpointStore()
    crashing, _, _ = make_agent([Crash()], checkpoints=store)
    with pytest.raises(Crash):
        await start(crashing, "hello", run_id="c1")
    stored = await store.load("default", "c1")
    assert stored is not None and stored.status is RunStatus.RUNNING

    recovering, _, _ = make_agent([text_response("recovered")], checkpoints=store)
    handle = stream_resume(recovering, stored, recover=True)
    await asyncio.wait_for(handle.aclose(), timeout=3)
    still = await store.load("default", "c1")
    assert still is not None and still.status is RunStatus.RUNNING  # still recoverable

    done = await resume(recovering, still, recover=True)
    assert done.status is RunStatus.DONE and done.last_text == "recovered"


@pytest.mark.anyio
async def test_a_cancelled_read_does_not_swallow_the_next_item() -> None:
    class Words:
        id = "words"
        capabilities = Capabilities(tools=True)

        async def complete(self, request: ModelRequest) -> ModelResponse:
            return text_response("a b c")

        async def stream(self, request: ModelRequest) -> AsyncIterator[StreamDelta | ModelResponse]:
            for word in ("a ", "b ", "c"):
                await asyncio.sleep(0.15)
                yield StreamDelta(text=word)
            yield text_response("a b c")

    agent = Agent.create(Words(), "Prompt", name="words")
    handle = stream_start(agent, "go")
    seen: list[str] = []
    while True:
        try:
            item = await asyncio.wait_for(handle.__anext__(), timeout=0.05)  # a heartbeat that gives up often
        except TimeoutError:
            continue
        except StopAsyncIteration:
            break
        if isinstance(item, TextDelta):
            seen.append(item.text)

    assert "".join(seen) == "a b c"


@pytest.mark.anyio
async def test_a_consumer_stalling_longer_than_the_provider_deadline_is_not_a_provider_timeout() -> None:
    async def stalled(_: StreamDelta) -> None:
        await asyncio.sleep(0.4)  # longer than request_timeout_s

    one_chunk = FakeProvider([Streamed(StreamDelta(text="hi"), text_response("hi"))], id="one")
    router = ModelRouter(one_chunk, retry=RetryPolicy(max_attempts=1, request_timeout_s=0.1))
    response, _ = await router.stream(_REQUEST, on_delta=stalled)
    assert response.message.text == "hi"


@pytest.mark.anyio
async def test_empty_deltas_do_not_block_retry_and_failover() -> None:
    flaky = FakeProvider(
        [
            Streamed(StreamDelta(), ProviderError("dropped", retryable=True, provider_id="flaky")),
            Streamed(StreamDelta(), text_response("ok")),
        ],
        id="flaky",
    )

    seen: list[StreamDelta] = []

    async def collect(delta: StreamDelta) -> None:
        seen.append(delta)

    router = ModelRouter(flaky, retry=RetryPolicy(max_attempts=2, base_delay_s=0.0, max_delay_s=0.0), sleep=no_sleep)
    response, _ = await router.stream(_REQUEST, on_delta=collect)
    assert response.message.text == "ok" and len(flaky.requests) == 2
    assert seen == []  # nothing reached the consumer, so the retry could not show a second answer


@pytest.mark.anyio
async def test_the_stream_consumer_sees_an_error_of_the_run() -> None:
    store = InMemoryCheckpointStore()
    agent = Agent.create(
        FakeProvider([tool_response(("write_file", {"path": "a"}))]),
        tools=[WriteFile()],
        autonomy=Autonomy.SUGGEST,
        checkpoints=store,
    )
    waiting = await start(agent, "go")
    stale = waiting.model_copy(deep=True)
    await store.save(waiting)  # someone else moved the revision on

    handle = stream_resume(agent, stale, approved=True, pending_id=stale.pending_id)
    with pytest.raises(ConcurrencyError):
        await handle


@pytest.mark.anyio
async def test_a_consumer_that_gave_up_finds_nothing_left_to_read() -> None:
    stream = stream_start(_many_agent(InMemoryCheckpointStore()), "go", queue_size=2, idle_timeout_s=0.2)
    async for _ in stream:
        break  # reads one item, then abandons the stream without closing it
    await asyncio.sleep(1.0)  # five times the idle timeout: the run has been cancelled by now
    assert stream.done()

    leftovers = [item async for item in stream]

    assert leftovers == []  # the buffered items were dropped when the run was abandoned


@pytest.mark.anyio
async def test_the_provider_deadline_still_applies_after_a_slow_consumer() -> None:
    class HangsAfterOneChunk:
        id = "hangs"
        capabilities = Capabilities(tools=True)

        async def complete(self, request: ModelRequest) -> ModelResponse:
            return text_response("x")

        async def stream(self, request: ModelRequest) -> AsyncIterator[StreamDelta | ModelResponse]:
            yield StreamDelta(text="first")
            await asyncio.sleep(30)  # the provider stalls after the consumer was slow once
            yield text_response("never")

    async def slow(_: StreamDelta) -> None:
        await asyncio.sleep(0.05)

    router = ModelRouter(HangsAfterOneChunk(), retry=RetryPolicy(max_attempts=1, request_timeout_s=0.3), sleep=no_sleep)

    with pytest.raises(ProviderError, match="timed out"):
        await asyncio.wait_for(router.stream(_REQUEST, on_delta=slow), timeout=5)


def test_fake_provider_refuses_a_bare_delta_as_a_script_item() -> None:
    with pytest.raises(TypeError, match="Streamed"):
        FakeProvider([cast(Any, StreamDelta(text="x"))])  # a typo a type checker would catch


def test_streamed_refuses_an_unexpected_item() -> None:
    with pytest.raises(TypeError, match="str"):
        Streamed(cast(Any, "text"))
