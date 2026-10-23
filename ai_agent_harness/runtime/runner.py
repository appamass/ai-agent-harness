"""The agent loop: `start` and `resume`.

Guarantees (each backed by tests):

- Every call ends with a well-defined status — even on provider outages,
  errors in hooks, tools, or context sources, and on task cancellation.
  Exception: `ConcurrencyError` (another process owns the run) and
  errors when saving a checkpoint are propagated to the caller.
- The message history never violates the tool-call protocol. Every tool call
  receives a result, even if it is rejected, blocked, or cancelled.
- Clarifying questions (`ask_user`) and approvals pause the run and are resumed via
  `resume(...)`. The human response becomes the result of the waiting tool call.
  `resume` requires the `pending_id` of the shown pause: a stale approval or answer
  can never apply to a different call (ADR 0014).
- Limits apply across all resumes and cannot be bypassed. This includes a
  context budget: an oversized request ends as `limit_reached`, not as a provider error.
- Human inputs are bounded in length (`Limits.max_input_chars`).
- With a checkpoint store, every `resume` claims the run immediately (optimistic lock):
  two workers cannot resume the same run twice and cannot execute an action twice.
- Write-ahead: state is saved before every tool execution. A crash mid-tool is never
  lost and is detected on recovery.
- A crashed run (status `running` in the checkpoint) can be resumed with
  `resume(..., recover=True)`. Interrupted tool calls are honestly reported to the
  model as "outcome unknown", never silently re-executed.

Status transitions:

    start ──► running ──► done | refused | blocked | limit_reached | failed | cancelled
                 │  ▲
                 ▼  │ resume(answer=… / approved=…, pending_id=…)
       waiting_for_input | waiting_for_approval

    done/refused/blocked/limit_reached/failed/cancelled ──resume(user_input=…)──► running
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from ..context import ContextMeter, estimate_tokens
from ..core import state as state_schema
from ..core.errors import ConcurrencyError, ConfigError, InvalidStateError, ProviderError
from ..core.events import Event, EventType
from ..core.hooks import Action, Decision
from ..core.state import CONTINUE_PROMPT, PendingBatch, PendingCall, RunState, RunStatus
from ..core.types import (
    FINAL_OUTPUT_TOOL_NAME,
    Message,
    Risk,
    StopReason,
    ThinkingBlock,
    ToolCall,
    ToolResult,
    canonical_digest,
    is_positive_number,
)
from ..providers.base import StreamDelta
from ..tools.base import BaseTool, json_schema, make_tool
from ..tools.executor import ToolExecutor, describe_validation
from .agent import Agent

log = logging.getLogger("ai_agent_harness.runner")

INTERRUPTED = (
    "Execution was interrupted (restart). Whether the tool already executed "
    "is unknown — check the current state before retrying."
)


@dataclass(frozen=True)
class TextDelta:
    """Incremental text chunk emitted during model streaming.

    Deltas are live, unfiltered model output. `after_model` hooks and guards only see the complete
    response afterwards, so a response they block has already been shown to the stream consumer
    (ADR 0025). Do not forward deltas to end users when that matters.
    """

    text: str


StreamItem = Event | TextDelta


class _StreamSink:
    """Bounded hand-over from the run task to the stream consumer.

    While open, a full queue blocks the run (backpressure), but never longer than `idle_timeout_s`:
    a consumer that stopped reading cancels the run instead of leaking it. Once closed, writes never
    block, so cancellation and the `CANCELLED` checkpoint cannot hang on a full queue.
    """

    def __init__(self, queue_size: int, idle_timeout_s: float) -> None:
        if not is_positive_number(queue_size, integer=True):
            raise ConfigError("queue_size must be a positive integer")
        if not is_positive_number(idle_timeout_s):
            raise ConfigError("idle_timeout_s must be > 0")
        self.queue: asyncio.Queue[StreamItem | None] = asyncio.Queue(maxsize=queue_size)
        self.idle_timeout_s = float(idle_timeout_s)
        self.closed = False
        self.started = False  # the run task has begun executing

    def start(self) -> bool:
        """Marks the run task as started. False if the stream was closed before (the body must not run)."""
        self.started = True
        return not self.closed

    def cancel(self, task: asyncio.Task[RunState]) -> None:
        """Closes the sink and cancels the run task once it has started.

        A task cancelled before it ran would skip `_Run.guarded`; it sees `closed` in `start()` instead.
        """
        self.closed = True
        if self.started and not task.done():
            task.cancel()

    def _put_nowait(self, item: StreamItem | None) -> None:
        with contextlib.suppress(asyncio.QueueFull):
            self.queue.put_nowait(item)

    def _abandon(self) -> None:
        """The consumer stopped reading: drop what is buffered and end its iteration."""
        self.closed = True
        while not self.queue.empty():
            self.queue.get_nowait()
        self._put_nowait(None)

    async def _deliver(self, item: StreamItem | None) -> bool:
        """Queues `item`. False if the consumer stopped reading (the sink was abandoned instead)."""
        if self.closed or not self.queue.full():  # the common case: no timer needed
            self._put_nowait(item)
            return True
        try:
            await asyncio.wait_for(self.queue.put(item), self.idle_timeout_s)
        except TimeoutError:
            self._abandon()
            return False
        return True

    async def put(self, item: StreamItem) -> None:
        if not await self._deliver(item):
            # Treated like a cancellation by `_Run.guarded`: persists `CANCELLED` and ends the task.
            raise asyncio.CancelledError("stream consumer stopped reading")

    async def finish(self) -> None:
        await self._deliver(None)


class StreamHandle:
    """Async iterator over run events and incremental text deltas with backpressure and cancellation support.

    Iterate it, or `await` it to get the final `RunState` (unread items are discarded). Use it as `async with`
    or call `aclose()` to stop early. A bare `break` does not stop the run: it waits for the consumer for
    `idle_timeout_s` and is then cancelled.
    """

    def __init__(self, task: asyncio.Task[RunState], sink: _StreamSink, state: RunState) -> None:
        self._task = task
        self._sink = sink
        self._state = state
        self._closed = False
        self._getter: asyncio.Future[StreamItem | None] | None = None  # the pending read of `_next_item`

    @property
    def state(self) -> RunState:
        """The RunState of the execution."""
        return self._state

    def __aiter__(self) -> StreamHandle:
        return self

    async def __anext__(self) -> StreamItem:
        if self._closed:
            raise StopAsyncIteration
        item = await self._next_item()
        if item is None:
            self._closed = True
            if not self._task.cancelled():
                exc = self._task.exception()
                if exc is not None:
                    raise exc
            raise StopAsyncIteration
        return item

    async def _next_item(self) -> StreamItem | None:
        """The next buffered item; None once the run task is done and everything was read.

        Does not rely on the end marker alone: a closed sink may have had to drop it.
        """
        queue = self._sink.queue
        while True:
            # One getter lives across calls. It is never cancelled while an item could still reach it, so a
            # consumer that is cancelled (a heartbeat timeout) cannot lose an item to a half-finished read.
            if self._getter is not None and self._getter.done():
                item = self._getter.result()
                self._getter = None
                return item
            try:
                return queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            if self._task.done():  # nothing is buffered and nothing more will come
                self._drop_getter()
                return None
            if self._getter is None:
                self._getter = asyncio.ensure_future(queue.get())
            waiting: set[asyncio.Future[Any]] = {self._getter, self._task}
            await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)

    def _drop_getter(self) -> None:
        if self._getter is not None:
            self._getter.cancel()
            self._getter = None

    async def __aenter__(self) -> StreamHandle:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.aclose()

    def __await__(self) -> Any:
        return self._result().__await__()

    async def _result(self) -> RunState:
        """Waits for the run and returns its state, reading (and discarding) what nobody consumed."""
        async for _ in self:
            pass
        return await self._task

    async def aclose(self) -> None:
        """Cancels the run and waits for cancellation cleanup (persisting RunStatus.CANCELLED)."""
        if self._closed:
            return
        self._closed = True
        self.cancel()
        self._drop_getter()
        if not self._task.done():
            with contextlib.suppress(asyncio.CancelledError, ConcurrencyError):
                await self._task

    def done(self) -> bool:
        """True once the run task has ended (finished, failed, or cancelled)."""
        return self._task.done()

    def cancel(self) -> None:
        """Synchronously cancels the running task. Iteration continues until what was buffered is read."""
        self._sink.cancel(self._task)


async def _submit_output(self: BaseTool[BaseModel], args: BaseModel, state: RunState) -> str:
    state.set_output(args)
    return "Final output accepted."


def _final_output_tool(model: type[BaseModel]) -> BaseTool[BaseModel]:
    """The reserved `final_output` tool: validated like any tool, executed through the gate (ADR 0023)."""
    return make_tool(
        BaseTool[BaseModel],
        "_FinalOutputTool",
        __module__=__name__,
        name=FINAL_OUTPUT_TOOL_NAME,
        description="Submit the final structured output to complete the task.",
        input_model=model,
        risk=Risk.READ,
        strict_schema=False,  # the caller's own model: any field types are fine
        run=_submit_output,
    )


def _extract_json_text(text: str) -> str:
    """The JSON object in a model answer: everything from the first `{` to the last `}` (drops fences and prose)."""
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if 0 <= start < end else text.strip()


@dataclass(frozen=True)
class _Calibration:
    """What a provider reported for one request: `prompt_tokens` for a request of `chars` characters (ADR 0026).

    `(0, 0)` means no measurement. The token estimate is then based on the size alone.
    """

    provider_id: str | None = None
    chars: int = 0
    prompt_tokens: int = 0

    def after(self, provider_id: str, chars: int, prompt_tokens: int) -> _Calibration:
        """The measurement after a response. Another provider than before, or no reported tokens: start over."""
        if prompt_tokens <= 0 or (self.provider_id is not None and self.provider_id != provider_id):
            return _Calibration(provider_id)
        return _Calibration(provider_id, chars, prompt_tokens)


@dataclass(frozen=True)
class _ResumeInput:
    """What the caller passes to `resume` / `stream_resume` for the waiting call or a new turn."""

    answer: str | None = None
    approved: bool | None = None
    user_input: str | None = None
    recover: bool = False
    pending_id: str | None = None


def _check_output_binding(state: RunState, output_model: type[BaseModel] | None) -> None:
    """A run started with an `output_model` keeps it (ADR 0023). The stored digest detects a changed model."""
    if state.output_schema_digest is None:
        return
    if output_model is None:
        raise InvalidStateError(
            f"Run requires structured output ({state.output_model_name}): pass the same output_model "
            "again on every resume (ADR 0023)"
        )
    if canonical_digest(json_schema(output_model)) != state.output_schema_digest:
        raise InvalidStateError(
            f"output_model {output_model.__name__} does not match the schema this run was started with"
        )


def _bind(run: _Run, deps: object) -> _Run:
    """The last step of setting up a run: nothing after it can fail, so a failed setup leaves the state as it was."""
    state, model = run.state, run.output_model
    if model is not None and run.output_schema is not None and state.output_schema_digest is None:
        state.output_schema_digest = canonical_digest(run.output_schema)
        state.output_model_name = model.__name__
    state.bind_deps(deps)
    return run


def _build_start(
    agent: Agent,
    user_input: str,
    *,
    tenant_id: str,
    run_id: str | None,
    deps: object,
    output_model: type[BaseModel] | None,
    sink: _StreamSink | None = None,
) -> _Run:
    _require_text(user_input, "user_input", agent.limits.max_input_chars)
    fields: dict[str, Any] = {"agent_name": agent.name, "tenant_id": tenant_id}
    if run_id is not None:
        fields["run_id"] = run_id
    state = RunState(**fields)
    return _bind(_Run(agent, state, output_model=output_model, stream_sink=sink), deps)


def _build_resume(
    agent: Agent,
    state: RunState,
    args: _ResumeInput,
    *,
    deps: object,
    output_model: type[BaseModel] | None,
    sink: _StreamSink | None = None,
) -> _Run:
    if state.agent_name != agent.name:
        raise InvalidStateError(f"Run belongs to agent '{state.agent_name}', not '{agent.name}'")
    if state.schema_version != state_schema.SCHEMA_VERSION:
        raise InvalidStateError(
            f"State has schema version {state.schema_version}, expected {state_schema.SCHEMA_VERSION}: "
            "always load persisted states via restore_state (ADR 0010)"
        )
    state.check_protocol()
    _check_resume_args(state, args, agent.limits.max_input_chars)
    _check_output_binding(state, output_model)
    return _bind(_Run(agent, state, output_model=output_model, stream_sink=sink), deps)


def _spawn(run: _Run, body: Coroutine[Any, Any, None], sink: _StreamSink, *, new_run: bool) -> StreamHandle:
    async def _task_body() -> RunState:
        try:
            if sink.start():
                return await run.guarded(body)
            body.close()  # closed before the task ran: the body never started
            if new_run:  # a new run has nothing to protect: it ends as `cancelled`
                return await run.guarded(_cancelled())
            # A resumed run was not claimed yet. Finishing it now would overwrite real results of its pending batch
            # with "Not executed", so the state and the checkpoint stay exactly as they were.
            run.state.bind_deps(None)
            return run.state
        finally:
            await sink.finish()

    return StreamHandle(asyncio.create_task(_task_body()), sink, run.state)


async def _cancelled() -> None:
    raise asyncio.CancelledError("stream closed before the run started")


async def start(
    agent: Agent,
    user_input: str,
    *,
    tenant_id: str = "default",
    run_id: str | None = None,
    deps: object = None,
    output_model: type[BaseModel] | None = None,
) -> RunState:
    """Starts a new run. `deps`: runtime dependencies for tools/hooks, never persisted.

    `output_model`: the run must end with a validated instance of this Pydantic model in `state.output` (ADR 0023).
    """
    run = _build_start(agent, user_input, tenant_id=tenant_id, run_id=run_id, deps=deps, output_model=output_model)
    return await run.guarded(run.start_body(user_input))


async def resume(
    agent: Agent,
    state: RunState,
    *,
    answer: str | None = None,
    approved: bool | None = None,
    user_input: str | None = None,
    recover: bool = False,
    deps: object = None,
    pending_id: str | None = None,
    output_model: type[BaseModel] | None = None,
) -> RunState:
    """Resumes a run.

    - `waiting_for_input`:    `answer` (response to the clarifying question) and `pending_id`
    - `waiting_for_approval`: `approved` (True/False) and `pending_id`, optional `answer` as a comment
    - finished runs:          optional `user_input` (new message); without it, only allowed
                              when the model still owes a reply (e.g. after `failed`)
    - `running` in checkpoint (crash): only with `recover=True`

    `pending_id` is `state.pending_id` as it was shown to the human. If the run now waits for a
    different call (another worker resumed it in the meantime), `InvalidStateError` is raised.
    Never read it from a state you just loaded to handle a click: it must come from the request
    the human saw (dialog or button payload), otherwise the check approves whatever is pending now.
    `deps` are never persisted and must be passed again on every resume.
    A run started with `output_model` needs the same `output_model` on every resume (ADR 0023).
    """
    args = _ResumeInput(answer=answer, approved=approved, user_input=user_input, recover=recover, pending_id=pending_id)
    run = _build_resume(agent, state, args, deps=deps, output_model=output_model)
    return await run.guarded(run.resume_body(args))


def stream_start(
    agent: Agent,
    user_input: str,
    *,
    tenant_id: str = "default",
    run_id: str | None = None,
    deps: object = None,
    output_model: type[BaseModel] | None = None,
    queue_size: int = 64,
    idle_timeout_s: float = 300.0,
) -> StreamHandle:
    """Starts a new run, returning a StreamHandle that yields Events and TextDeltas.

    A full queue pauses the run (backpressure). A consumer that stops reading for `idle_timeout_s`
    cancels it. Close the handle (`async with` or `aclose()`) when you stop early.
    """
    sink = _StreamSink(queue_size, idle_timeout_s)
    run = _build_start(
        agent, user_input, tenant_id=tenant_id, run_id=run_id, deps=deps, output_model=output_model, sink=sink
    )
    return _spawn(run, run.start_body(user_input), sink, new_run=True)


def stream_resume(
    agent: Agent,
    state: RunState,
    *,
    answer: str | None = None,
    approved: bool | None = None,
    user_input: str | None = None,
    recover: bool = False,
    deps: object = None,
    pending_id: str | None = None,
    output_model: type[BaseModel] | None = None,
    queue_size: int = 64,
    idle_timeout_s: float = 300.0,
) -> StreamHandle:
    """Resumes a run, returning a StreamHandle that yields Events and TextDeltas (see `stream_start`)."""
    sink = _StreamSink(queue_size, idle_timeout_s)
    args = _ResumeInput(answer=answer, approved=approved, user_input=user_input, recover=recover, pending_id=pending_id)
    run = _build_resume(agent, state, args, deps=deps, output_model=output_model, sink=sink)
    return _spawn(run, run.resume_body(args), sink, new_run=False)


def _stamp_thinking(message: Message, provider_id: str) -> Message:
    """Stamps thinking blocks with the provider that actually answered (for failover)."""
    if not any(isinstance(b, ThinkingBlock) for b in message.content):
        return message
    blocks = tuple(
        b.model_copy(update={"provider_id": provider_id}) if isinstance(b, ThinkingBlock) else b
        for b in message.content
    )
    return message.model_copy(update={"content": blocks})


def _check_length(value: str, label: str, max_chars: int) -> None:
    if len(value) > max_chars:
        raise ValueError(f"'{label}' is too long ({len(value)} chars, maximum is {max_chars})")


def _require_text(value: str | None, label: str, max_chars: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{label}' must be a non-empty string")
    _check_length(value, label, max_chars)
    return value


def _check_resume_args(state: RunState, args: _ResumeInput, max_chars: int) -> None:
    status = state.status
    answer, approved, user_input, pending_id = args.answer, args.approved, args.user_input, args.pending_id
    if status.is_waiting:
        if pending_id is None or pending_id != state.pending_id:
            raise InvalidStateError(
                "Missing or stale 'pending_id': the run is waiting for a different call than the one shown. "
                "Show the current request (state.status_reason, state.pending_id) to the human again."
            )
    elif pending_id is not None:
        raise InvalidStateError(f"'pending_id' is only allowed while the run is waiting, got '{status}'")
    for label, value in (("answer", answer), ("user_input", user_input)):
        if value is not None:
            _check_length(value, label, max_chars)
    if status is RunStatus.RUNNING:
        if not args.recover:
            raise InvalidStateError(
                "Run has status 'running': it is either still running or crashed. "
                "Only resume after a crash with recover=True."
            )
        if answer is not None or approved is not None or user_input is not None:
            raise InvalidStateError("recover=True does not allow additional arguments")
        return
    if args.recover:
        raise InvalidStateError(f"recover=True is only allowed when status is 'running', got '{status}'")
    if status is RunStatus.WAITING_FOR_INPUT:
        _require_text(answer, "answer", max_chars)
        if approved is not None or user_input is not None:
            raise InvalidStateError("When waiting for input, pass only 'answer'")
        return
    if status is RunStatus.WAITING_FOR_APPROVAL:
        if not isinstance(approved, bool):
            raise InvalidStateError("When waiting for approval, pass 'approved=True/False'")
        if user_input is not None:
            raise InvalidStateError("When waiting for approval, do not pass 'user_input'")
        return
    if answer is not None or approved is not None:
        raise InvalidStateError(f"Run with status '{status}' is not waiting for an answer or approval")
    if user_input is not None:
        _require_text(user_input, "user_input", max_chars)
        return
    last = state.last_message
    if last is None or last.role != "user":
        raise InvalidStateError("Nothing to resume: pass 'user_input' to continue")


class _Run:
    """A single execution (`start` or `resume`) of a run."""

    def __init__(
        self,
        agent: Agent,
        state: RunState,
        *,
        output_model: type[BaseModel] | None = None,
        stream_sink: _StreamSink | None = None,
    ) -> None:
        self.agent = agent
        self.state = state
        self.output_model = output_model
        # From the model itself: a resumed run was checked against the stored schema, a new one is bound after setup.
        self.output_schema = json_schema(output_model) if output_model is not None else None
        self._stream_sink = stream_sink
        self._output_tier = self._pick_output_tier()
        tools = agent.tools
        policy = agent.policy
        if self._output_tier == "tool" and output_model is not None:
            tools = agent.tools.extended(_final_output_tool(output_model))
            if policy.allowed_tools is not None:
                policy = replace(policy, allowed_tools=policy.allowed_tools | {FINAL_OUTPUT_TOOL_NAME})
        self.executor = ToolExecutor(
            tools,
            policy,
            agent.hooks,
            self.emit,
            max_result_chars=agent.max_result_chars,
            truncation_mode=agent.truncation_mode,
        )
        self._specs = [tool.spec() for tool in self.executor.visible_tools()]  # fixed for the life of a run
        self._output_instructions = self._build_output_instructions()
        self._meter = ContextMeter()
        self._calibration = _Calibration()
        self._elapsed_base = state.elapsed_s
        self._t0 = time.monotonic()
        self._finishing = False

    def _build_output_instructions(self) -> str:
        """The system prompt suffix for the prompt and tool tiers (built once: the schema never changes in a run)."""
        if self._output_tier == "prompt":
            schema_json = json.dumps(self.output_schema, indent=2)
            return f"\n\nYou must return a valid JSON object matching this schema:\n{schema_json}\nOutput JSON only."
        if self._output_tier == "tool":
            return (
                f"\n\nTo complete the task, you must call the tool '{FINAL_OUTPUT_TOOL_NAME}' "
                "with the structured output."
            )
        return ""

    def _pick_output_tier(self) -> Literal["none", "native", "tool", "prompt"]:
        """How the final output is requested. Uses what ALL routed models support, so failover stays valid."""
        if self.output_model is None:
            return "none"
        providers = self.agent.router.providers
        if all(p.capabilities.structured_output for p in providers):
            return "native"
        if all(p.capabilities.tools for p in providers):
            return "tool"
        return "prompt"

    # ------------------------------------------------------------ Frame

    async def guarded(self, body: Coroutine[Any, Any, None]) -> RunState:
        try:
            await body
        except asyncio.CancelledError:
            if not self._finishing:
                try:
                    await asyncio.shield(self._finish(RunStatus.CANCELLED, "Run was cancelled"))
                except ConcurrencyError:
                    # Another worker took over the run in the meantime: do not overwrite, but still
                    # propagate the cancellation expected by the caller (asyncio.timeout, TaskGroup).
                    log.warning("Run %s was taken over by another process during cancellation", self.state.run_id)
            raise
        except ConcurrencyError:
            raise  # another process owns the run — do not write anything
        except ProviderError as exc:
            await self._finish(RunStatus.FAILED, f"Model call failed: {exc}")
        except Exception as exc:
            if self._finishing:
                raise
            log.exception("Unexpected error in run %s", self.state.run_id)
            await self._finish(RunStatus.FAILED, f"Internal error ({type(exc).__name__})")
        finally:
            self.state.bind_deps(None)  # DB session, token, etc. are scoped to this execution only
        return self.state

    async def emit(self, type_: EventType, data: dict[str, Any]) -> None:
        s = self.state
        try:
            event = Event(type=type_, run_id=s.run_id, tenant_id=s.tenant_id, step=s.step, data=data)
        except ValidationError as exc:  # events are for observability: never fail a run because of them
            problems = [(e["loc"], e["type"]) for e in exc.errors(include_url=False, include_input=False)]
        else:
            await self.agent.events.emit(event)
            if self._stream_sink is not None:
                await self._stream_sink.put(event)
            return
        # Deliberately without traceback: the rejected values are the exact ones that might contain content.
        log.error("Event %s dropped due to invalid data: %s", type_, problems)

    async def _on_text_delta(self, delta: StreamDelta) -> None:
        if self._stream_sink is not None and delta.text:
            await self._stream_sink.put(TextDelta(text=delta.text))

    def elapsed(self) -> float:
        return self._elapsed_base + (time.monotonic() - self._t0)

    # ------------------------------------------------------------ Entry

    async def start_body(self, user_input: str) -> None:
        await self.emit(EventType.RUN_STARTED, {"agent": self.agent.name})
        if await self._accept_user_text(user_input):
            await self._save()  # run is now visible and recoverable after a crash
            await self._loop()

    async def resume_body(self, args: _ResumeInput) -> None:
        state = self.state
        previous = state.status
        batch = state.pending
        state.pending = None
        if batch is not None:
            state.batch_done = batch.done  # preserved in the checkpoint even if a crash happens next
        state.consecutive_retries = 0
        state.set_status(RunStatus.RUNNING)
        # Claim the run immediately: a second worker with the same revision fails here with
        # ConcurrencyError — before any tool (such as an approved action) can execute twice.
        # If the process crashes afterward, the checkpoint has 'running' → resume(recover=True).
        await self._save()
        await self.emit(EventType.RUN_RESUMED, {"from": str(previous)})

        if previous is RunStatus.RUNNING:  # crash recovery
            # Completed results are preserved; only genuinely interrupted calls are marked "unknown"
            state.close_open_tool_calls(INTERRUPTED, known=state.batch_done)
            state.batch_done = ()
        elif batch is not None:
            if not await self._resolve_waiting(batch.waiting, args, rest=batch.remaining):
                return
            if await self._finish_if_output_submitted() or await self._run_batch(batch.remaining):
                return
        elif args.user_input is not None and not await self._accept_user_text(args.user_input):
            return
        await self._loop()

    async def _resolve_waiting(self, waiting: PendingCall, args: _ResumeInput, *, rest: Sequence[ToolCall]) -> bool:
        """Resolves the waiting tool call. Returns False if the run ended or paused again."""
        call_id, answer = waiting.call.id, args.answer
        if waiting.kind == "input":
            assert answer is not None  # verified by _check_resume_args
            if await self._blocked(await self.agent.hooks.on_user_input(self.state, answer)):
                return False
            self.state.record_result(ToolResult(call_id=call_id, content=answer))
            return True

        if args.approved:
            outcome = await self.executor.execute(waiting.call, self.state, approved=True)
            if isinstance(outcome, PendingCall):
                await self._pause(outcome, rest)
                return False
            self._discard_rejected_output(waiting.call, outcome)
            return True

        note = "Approval was denied."
        if answer and answer.strip():
            decision = await self.agent.hooks.on_user_input(self.state, answer)
            if decision.action is Action.CONTINUE:
                note = f"Approval was denied. Comment: {answer}"
        self.state.record_result(ToolResult(call_id=call_id, content=note, is_error=True))
        return True

    # ------------------------------------------------------------ Loop

    async def _loop(self) -> None:
        state, agent = self.state, self.agent
        while True:
            reason = agent.limits.exceeded(state, self.elapsed())
            if reason:
                await self._finish(RunStatus.LIMIT_REACHED, reason)
                return

            specs = self._specs
            request = await agent.context.build(state, specs)
            if self._output_tier == "native":
                request = request.model_copy(update={"output_schema": self.output_schema})
            elif self._output_instructions:
                request = request.model_copy(update={"system": request.system + self._output_instructions})

            size = self._meter.measure(request).total
            calibration = self._calibration
            tokens = estimate_tokens(size, calibration.chars, calibration.prompt_tokens)
            reason = self._context_limit(size, tokens)
            if reason:
                await self._finish(RunStatus.LIMIT_REACHED, reason)
                return
            if await self._blocked(await agent.hooks.before_model(state, request)):
                return

            await self.emit(
                EventType.MODEL_REQUEST,
                {
                    "messages": len(request.messages),
                    "tools": len(specs),
                    "context_chars": size,
                    "context_tokens_est": tokens,
                },
            )
            if self._stream_sink is not None:
                response, provider_id = await agent.router.stream(
                    request,
                    self.emit,
                    on_delta=self._on_text_delta,
                )
            else:
                response, provider_id = await agent.router.complete(request, self.emit)
            state.step += 1
            state.usage = state.usage + response.usage
            self._calibration = calibration.after(provider_id, size, response.usage.prompt_tokens)
            await self.emit(
                EventType.MODEL_RESPONSE,
                {
                    "provider": provider_id,
                    "model": response.model,
                    "stop_reason": str(response.stop_reason),
                    "usage": response.usage.model_dump(),
                },
            )
            state.add_assistant(_stamp_thinking(response.message, provider_id))

            decision = await agent.hooks.after_model(state, response)
            if await self._blocked(decision):
                return
            if decision.action is Action.RETRY:
                if not await self._reject(decision.reason or "Response not accepted"):
                    return
                continue

            calls = response.message.tool_calls
            stop = response.stop_reason
            if stop is StopReason.REFUSAL:
                await self._finish(RunStatus.REFUSED, response.message.text or "The model refused the request")
                return
            if stop is StopReason.MAX_TOKENS:
                if calls:
                    if not await self._reject(
                        "The response was truncated due to the token limit, so the tool calls in it "
                        "were not executed. Please call again more concisely."
                    ):
                        return
                    continue
                state.add_user_text(CONTINUE_PROMPT)
            elif calls:  # END or TOOL_USE with calls: execute (saves checkpoints itself, see _run_batch)
                if await self._run_batch(calls):
                    return
                continue
            elif stop is StopReason.TOOL_USE:
                if not await self._reject("Your response indicated a tool call, but did not include one."):
                    return
                continue
            else:
                problem = self._accept_final_text(response.message.text or "")
                if problem is not None:
                    if not await self._reject(problem):
                        return
                    continue
                await self._finish(RunStatus.DONE)
                return

            await self._checkpoint()

    def _context_limit(self, chars: int, tokens: int) -> str | None:
        """Why the request is too large for `Limits`, or None."""
        limits = self.agent.limits
        if chars > limits.max_context_chars:
            return f"Context too large (approx. {chars} chars, limit {limits.max_context_chars})"
        if limits.max_context_tokens is not None and tokens > limits.max_context_tokens:
            return f"Context too large (approx. {tokens} tokens, limit {limits.max_context_tokens})"
        return None

    def _accept_final_text(self, text: str) -> str | None:
        """Stores the structured output found in a final text answer. Returns why it was rejected, or None."""
        if self.output_model is None:
            return None
        try:
            validated = self.output_model.model_validate_json(_extract_json_text(text))
        except ValidationError as exc:
            if self._output_tier == "tool":
                hint = f"Call the '{FINAL_OUTPUT_TOOL_NAME}' tool with the structured output."
            else:
                reason = describe_validation(exc, root="(root)")
                hint = f"Output must be a valid JSON object matching the required schema: {reason}"
            return f"Final output could not be validated: {hint}"
        self.state.set_output(validated)
        return None

    async def _run_batch(self, calls: Sequence[ToolCall]) -> bool:
        """Executes tool calls sequentially. Returns True if the run was paused or finished.

        Write-ahead before EVERY tool: the checkpoint contains the planned calls and all results
        completed so far. If the process dies mid-tool, `recover` reports only that tool as
        "outcome unknown" — completed tools keep their actual result. After the batch, state is saved
        again so a crash during the next model call loses nothing either.
        """
        final_rejected = False
        for index, call in enumerate(calls):
            await self._checkpoint()
            outcome = await self.executor.execute(call, self.state)
            if isinstance(outcome, PendingCall):
                await self._pause(outcome, calls[index + 1 :])
                return True
            if self._discard_rejected_output(call, outcome):
                final_rejected = True
            elif await self._finish_if_output_submitted():
                return True  # `_finish` answers the calls after it: "Not executed: run finished"
        self.state.add_tool_results(self.state.batch_done)
        self.state.batch_done = ()
        if final_rejected:
            # An invalid final output counts like any rejected response (the tool error already told the model why).
            if not await self._count_retry("Model repeatedly returned unusable final output"):
                return True
        else:
            self.state.consecutive_retries = 0
        await self._checkpoint()
        return False

    def _discard_rejected_output(self, call: ToolCall, outcome: PendingCall | ToolResult) -> bool:
        """`final_output` stores its output while it runs; the gate may still withhold the result afterwards
        (`after_tool` hook, cancellation). Only a result that passed the whole gate counts. True if discarded."""
        if call.name != FINAL_OUTPUT_TOOL_NAME or not isinstance(outcome, ToolResult) or not outcome.is_error:
            return False
        self.state.clear_output()
        return True

    async def _finish_if_output_submitted(self) -> bool:
        """Ends the run as `done` once the `final_output` tool has stored a validated output."""
        if self.state.output is None:
            return False
        await self._finish(RunStatus.DONE)
        return True

    async def _reject(self, reason: str) -> bool:
        """Rejects the last response with feedback. Returns False if retry limit reached (run finished)."""
        state = self.state
        if not await self._count_retry(f"Model repeatedly returned unusable responses: {reason}"):
            return False
        feedback = f"Your last response was rejected: {reason}"
        if state.open_tool_calls():
            state.close_open_tool_calls(f"Not executed: {reason}", known=state.batch_done, text=feedback)
            state.batch_done = ()
        else:
            state.add_user_text(feedback)
        return True

    async def _count_retry(self, failure: str) -> bool:
        """Counts one unusable response. False if the retry limit is reached (the run finished as failed)."""
        self.state.consecutive_retries += 1
        if self.state.consecutive_retries > self.agent.limits.max_consecutive_retries:
            await self._finish(RunStatus.FAILED, failure)
            return False
        return True

    async def _accept_user_text(self, text: str) -> bool:
        if await self._blocked(await self.agent.hooks.on_user_input(self.state, text)):
            return False
        self.state.add_user_text(text)
        return True

    async def _blocked(self, decision: Decision) -> bool:
        """Ends the run as `blocked` if a hook said so."""
        if decision.action is not Action.BLOCK:
            return False
        await self._finish(RunStatus.BLOCKED, decision.reason)
        return True

    # ------------------------------------------------------------ Completion

    async def _pause(self, pending: PendingCall, remaining: Sequence[ToolCall]) -> None:
        self.state.pending = PendingBatch(done=self.state.batch_done, waiting=pending, remaining=tuple(remaining))
        self.state.batch_done = ()
        status = RunStatus.WAITING_FOR_INPUT if pending.kind == "input" else RunStatus.WAITING_FOR_APPROVAL
        await self._finish(status, pending.prompt)

    async def _finish(self, status: RunStatus, reason: str | None = None) -> None:
        self._finishing = True
        state = self.state
        if not status.is_waiting:
            state.pending = None
            state.close_open_tool_calls(f"Not executed: run finished ({status}).", known=state.batch_done)
            state.batch_done = ()
        state.set_status(status, reason)
        state.elapsed_s = self.elapsed()
        if not status.is_waiting:
            await self.agent.hooks.on_run_end(state)
        await self._save()
        waiting = self.state.pending.waiting if self.state.pending is not None else None
        # Metadata only: reasons and previews may contain sensitive content and stay in RunState.
        await self.emit(
            EventType.RUN_FINISHED,
            {
                "status": str(status),
                "waiting_for": waiting.kind if waiting else None,
                "tool": waiting.call.name if waiting else None,
            },
        )

    async def _checkpoint(self) -> None:
        if self.agent.checkpoints is not None:
            self.state.elapsed_s = self.elapsed()
            await self._save()
            await self.emit(EventType.CHECKPOINT_SAVED, {"revision": self.state.revision})

    async def _save(self) -> None:
        if self.agent.checkpoints is not None:
            await self.agent.checkpoints.save(self.state)
