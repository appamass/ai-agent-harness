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
import logging
import time
from collections.abc import Coroutine, Sequence
from typing import Any

from pydantic import ValidationError

from ..core import state as state_schema
from ..core.errors import ConcurrencyError, InvalidStateError, ProviderError
from ..core.events import Event, EventType
from ..core.hooks import Action
from ..core.state import CONTINUE_PROMPT, PendingBatch, PendingCall, RunState, RunStatus
from ..core.types import Message, StopReason, ThinkingBlock, ToolCall, ToolResult
from ..tools.executor import ToolExecutor
from .agent import Agent

log = logging.getLogger("ai_agent_harness.runner")

INTERRUPTED = (
    "Execution was interrupted (restart). Whether the tool already executed "
    "is unknown — check the current state before retrying."
)


async def start(
    agent: Agent,
    user_input: str,
    *,
    tenant_id: str = "default",
    run_id: str | None = None,
    deps: object = None,
) -> RunState:
    """Starts a new run. `deps`: runtime dependencies for tools/hooks, never persisted."""
    _require_text(user_input, "user_input", agent.limits.max_input_chars)
    fields: dict[str, Any] = {"agent_name": agent.name, "tenant_id": tenant_id}
    if run_id is not None:
        fields["run_id"] = run_id
    state = RunState(**fields)
    state.bind_deps(deps)
    run = _Run(agent, state)
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
    """
    if state.agent_name != agent.name:
        raise InvalidStateError(f"Run belongs to agent '{state.agent_name}', not '{agent.name}'")
    if state.schema_version != state_schema.SCHEMA_VERSION:
        raise InvalidStateError(
            f"State has schema version {state.schema_version}, expected {state_schema.SCHEMA_VERSION}: "
            "always load persisted states via restore_state (ADR 0010)"
        )
    state.check_protocol()
    _check_resume_args(
        state,
        answer=answer,
        approved=approved,
        user_input=user_input,
        recover=recover,
        pending_id=pending_id,
        max_chars=agent.limits.max_input_chars,
    )
    state.bind_deps(deps)
    run = _Run(agent, state)
    return await run.guarded(run.resume_body(answer=answer, approved=approved, user_input=user_input))


def _stamp_thinking(message: Message, provider_id: str) -> Message:
    """Stamps thinking blocks with the provider that actually answered (for failover)."""
    if not any(isinstance(b, ThinkingBlock) for b in message.content):
        return message
    blocks = tuple(
        b.model_copy(update={"provider_id": provider_id}) if isinstance(b, ThinkingBlock) else b
        for b in message.content
    )
    return message.model_copy(update={"content": blocks})


def _require_text(value: str | None, label: str, max_chars: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{label}' must be a non-empty string")
    if len(value) > max_chars:
        raise ValueError(f"'{label}' is too long ({len(value)} chars, maximum is {max_chars})")
    return value


def _check_resume_args(
    state: RunState,
    *,
    answer: str | None,
    approved: bool | None,
    user_input: str | None,
    recover: bool,
    pending_id: str | None,
    max_chars: int,
) -> None:
    status = state.status
    if status.is_waiting:
        if pending_id is None or pending_id != state.pending_id:
            raise InvalidStateError(
                "Missing or stale 'pending_id': the run is waiting for a different call than the one shown. "
                "Show the current request (state.status_reason, state.pending_id) to the human again."
            )
    elif pending_id is not None:
        raise InvalidStateError(f"'pending_id' is only allowed while the run is waiting, got '{status}'")
    for label, value in (("answer", answer), ("user_input", user_input)):
        if value is not None and len(value) > max_chars:
            raise ValueError(f"'{label}' is too long ({len(value)} chars, maximum is {max_chars})")
    if status is RunStatus.RUNNING:
        if not recover:
            raise InvalidStateError(
                "Run has status 'running': it is either still running or crashed. "
                "Only resume after a crash with recover=True."
            )
        if answer is not None or approved is not None or user_input is not None:
            raise InvalidStateError("recover=True does not allow additional arguments")
        return
    if recover:
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

    def __init__(self, agent: Agent, state: RunState) -> None:
        self.agent = agent
        self.state = state
        self.executor = ToolExecutor(
            agent.tools, agent.policy, agent.hooks, self.emit, max_result_chars=agent.max_result_chars
        )
        self._elapsed_base = state.elapsed_s
        self._t0 = time.monotonic()
        self._finishing = False

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
            return
        # Deliberately without traceback: the rejected values are the exact ones that might contain content.
        log.error("Event %s dropped due to invalid data: %s", type_, problems)

    def elapsed(self) -> float:
        return self._elapsed_base + (time.monotonic() - self._t0)

    # ------------------------------------------------------------ Entry

    async def start_body(self, user_input: str) -> None:
        await self.emit(EventType.RUN_STARTED, {"agent": self.agent.name})
        if await self._accept_user_text(user_input):
            await self._save()  # run is now visible and recoverable after a crash
            await self._loop()

    async def resume_body(self, *, answer: str | None, approved: bool | None, user_input: str | None) -> None:
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
            await self._loop()
            return

        if batch is not None:
            if not await self._resolve_waiting(batch.waiting, answer=answer, approved=approved, rest=batch.remaining):
                return
            if await self._run_batch(batch.remaining):
                return
            await self._loop()
            return

        if user_input is not None and not await self._accept_user_text(user_input):
            return
        await self._loop()

    async def _resolve_waiting(
        self, waiting: PendingCall, *, answer: str | None, approved: bool | None, rest: Sequence[ToolCall]
    ) -> bool:
        """Resolves the waiting tool call. Returns False if the run ended or paused again."""
        call_id = waiting.call.id
        if waiting.kind == "input":
            assert answer is not None  # verified by _check_resume_args
            decision = await self.agent.hooks.on_user_input(self.state, answer)
            if decision.action is Action.BLOCK:
                await self._finish(RunStatus.BLOCKED, decision.reason)
                return False
            self._record(ToolResult(call_id=call_id, content=answer))
            return True

        if approved:
            outcome = await self.executor.execute(waiting.call, self.state, approved=True)
            if isinstance(outcome, PendingCall):
                await self._pause(outcome, rest)
                return False
            return True

        note = "Approval was denied."
        if answer and answer.strip():
            decision = await self.agent.hooks.on_user_input(self.state, answer)
            if decision.action is Action.CONTINUE:
                note = f"Approval was denied. Comment: {answer}"
        self._record(ToolResult(call_id=call_id, content=note, is_error=True))
        return True

    # ------------------------------------------------------------ Loop

    async def _loop(self) -> None:
        state, agent = self.state, self.agent
        while True:
            reason = agent.limits.exceeded(state, self.elapsed())
            if reason:
                await self._finish(RunStatus.LIMIT_REACHED, reason)
                return

            specs = [tool.spec() for tool in self.executor.visible_tools()]
            request = await agent.context.build(state, specs)
            size = request.estimated_chars
            if size > agent.limits.max_context_chars:
                await self._finish(
                    RunStatus.LIMIT_REACHED,
                    f"Context too large (approx. {size} chars, limit {agent.limits.max_context_chars})",
                )
                return
            decision = await agent.hooks.before_model(state, request)
            if decision.action is Action.BLOCK:
                await self._finish(RunStatus.BLOCKED, decision.reason)
                return

            await self.emit(EventType.MODEL_REQUEST, {"messages": len(request.messages), "tools": len(specs)})
            response, provider_id = await agent.router.complete(request, self.emit)
            state.step += 1
            state.usage = state.usage + response.usage
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
            if decision.action is Action.BLOCK:
                await self._finish(RunStatus.BLOCKED, decision.reason)
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
                state.consecutive_retries = 0
                if await self._run_batch(calls):
                    return
                continue
            elif stop is StopReason.TOOL_USE:
                if not await self._reject("Your response indicated a tool call, but did not include one."):
                    return
                continue
            else:
                await self._finish(RunStatus.DONE)
                return

            await self._checkpoint()

    async def _run_batch(self, calls: Sequence[ToolCall]) -> bool:
        """Executes tool calls sequentially. Returns True if the run was paused.

        Write-ahead before EVERY tool: the checkpoint contains the planned calls and all results
        completed so far. If the process dies mid-tool, `recover` reports only that tool as
        "outcome unknown" — completed tools keep their actual result. After the batch, state is saved
        again so a crash during the next model call loses nothing either.
        """
        for index, call in enumerate(calls):
            await self._checkpoint()
            outcome = await self.executor.execute(call, self.state)
            if isinstance(outcome, PendingCall):
                await self._pause(outcome, calls[index + 1 :])
                return True
        self.state.add_tool_results(self.state.batch_done)
        self.state.batch_done = ()
        await self._checkpoint()
        return False

    def _record(self, result: ToolResult) -> None:
        self.state.batch_done = (*self.state.batch_done, result)

    async def _reject(self, reason: str) -> bool:
        """Rejects the last response with feedback. Returns False if retry limit reached (run finished)."""
        state = self.state
        state.consecutive_retries += 1
        if state.consecutive_retries > self.agent.limits.max_consecutive_retries:
            await self._finish(RunStatus.FAILED, f"Model repeatedly returned unusable responses: {reason}")
            return False
        feedback = f"Your last response was rejected: {reason}"
        calls = state.open_tool_calls()
        if calls:
            skipped = [ToolResult(call_id=c.id, content=f"Not executed: {reason}", is_error=True) for c in calls]
            state.add_tool_results(skipped, text=feedback)
        else:
            state.add_user_text(feedback)
        return True

    async def _accept_user_text(self, text: str) -> bool:
        decision = await self.agent.hooks.on_user_input(self.state, text)
        if decision.action is Action.BLOCK:
            await self._finish(RunStatus.BLOCKED, decision.reason)
            return False
        self.state.add_user_text(text)
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
