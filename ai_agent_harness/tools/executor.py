"""Tool executor: the single execution gate. The only place where tools are executed.

Every call passes through the exact same steps, cheapest and safest checks first:

1. Look up tool                           → unknown: error returned to model
2. Validate arguments with Pydantic       → invalid: concise error returned to model
3. Policy (allowlist, autonomy × risk)    → deny: error returned to model
4. `before_tool` hooks (validated args)   → block / require_approval
5. Approval required and not yet granted  → pause run (`PendingCall`)
6. Execute with timeout                   → errors become messages, never crashes
7. Process output                         → truncate; `InputRequest` pauses the run
8. `after_tool` hooks                     → block withholds the result

Error messages returned to the model contain only safe information.
Exception tracebacks go to the log, never into the model context.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from ..context import as_untrusted
from ..core.errors import ToolError
from ..core.events import Emit, EventType
from ..core.hooks import Action, Hook, HookPipeline, ToolInvocation
from ..core.state import PendingCall, RunState
from ..core.types import ToolCall, ToolResult
from ..policy import PermissionPolicy, Verdict
from .base import BaseTool, InputRequest
from .registry import ToolRegistry

log = logging.getLogger("ai_agent_harness.tools")

ToolOutcome = ToolResult | PendingCall
"""Outcome of the execution gate: either a finished `ToolResult` (already in `state.batch_done`) or a `PendingCall`."""


async def _noop_emit(type_: EventType, data: dict[str, Any]) -> None:
    return None


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry | Iterable[BaseTool[Any]],
        policy: PermissionPolicy | None = None,
        hooks: HookPipeline | Sequence[Hook] = (),
        emit: Emit | None = None,
        *,
        max_result_chars: int = 20_000,
    ) -> None:
        self._registry = registry if isinstance(registry, ToolRegistry) else ToolRegistry(registry)
        self._policy = policy if policy is not None else PermissionPolicy()
        self._hooks = hooks if isinstance(hooks, HookPipeline) else HookPipeline(hooks)
        self._emit = emit or _noop_emit
        self._max_result_chars = max_result_chars

    def visible_tools(self) -> list[BaseTool[Any]]:
        """Tools offered to the model: everything not denied by the policy."""
        return [t for t in self._registry if self._policy.check(t.name, t.risk) is not Verdict.DENY]

    async def execute(self, call: ToolCall, state: RunState | None = None, *, approved: bool = False) -> ToolOutcome:
        active_state = state if state is not None else RunState(agent_name="standalone")

        def error(message: str) -> ToolResult:
            return ToolResult(call_id=call.id, content=message, is_error=True)

        def fail(message: str) -> ToolResult:
            result = error(message)
            active_state.batch_done = (*active_state.batch_done, result)
            return result

        # 1. Look up tool
        tool = self._registry.get(call.name)
        if tool is None:
            visible = ", ".join(sorted(t.name for t in self.visible_tools())) or "none"
            return fail(f"Unknown tool '{call.name}'. Available: {visible}")

        # 2. Validate arguments
        if call.arguments_error is not None:
            return fail(
                f"Arguments for '{tool.name}' were not valid JSON ({call.arguments_error}). "
                "Please retry the call with valid JSON."
            )
        try:
            args = tool.input_model.model_validate(call.arguments)
        except ValidationError as exc:
            return fail(f"Invalid arguments for '{tool.name}': {_describe(exc)}")

        # 3. Policy
        verdict = self._policy.check(tool.name, tool.risk)
        if verdict is Verdict.DENY:
            denied = fail(f"'{tool.name}' is not allowed at autonomy level '{self._policy.autonomy}'.")
            await self._emit(EventType.TOOL_BLOCKED, {"tool": tool.name, "source": "policy"})
            return denied

        # 4. Pre-execution hooks
        invocation = ToolInvocation(call=call, tool_name=tool.name, risk=tool.risk, args=args)
        decision = await self._hooks.before_tool(active_state, invocation)
        if decision.action is Action.BLOCK:
            blocked = fail(f"Call to '{tool.name}' blocked: {decision.reason}")
            await self._emit(EventType.TOOL_BLOCKED, {"tool": tool.name, "source": "before_tool"})
            return blocked

        # 5. Approval
        reasons = [
            r
            for r in (
                "Policy requires approval" if verdict is Verdict.NEEDS_APPROVAL else None,
                decision.reason if decision.action is Action.REQUIRE_APPROVAL else None,
            )
            if r
        ]
        if reasons and not approved:
            prompt = _preview(tool, args, reasons)
            await self._emit(EventType.APPROVAL_REQUESTED, {"tool": tool.name, "risk": str(tool.risk)})
            return PendingCall(kind="approval", call=call, prompt=prompt)

        # 6. Execute
        await self._emit(EventType.TOOL_STARTED, {"tool": tool.name})
        try:
            # Tools are external code: verify return type at runtime, not just via type annotations
            output: object = await asyncio.wait_for(tool.run(args, active_state), timeout=tool.timeout_s)
        except asyncio.CancelledError:
            fail(
                f"Execution of '{tool.name}' was cancelled. Whether the tool already executed "
                "is unknown — check the current state before retrying."
            )
            raise
        except TimeoutError:
            limit = f"{tool.timeout_s:g}"
            return await self._complete(
                active_state, invocation, error(f"'{tool.name}' exceeded the timeout of {limit} s.")
            )
        except ToolError as exc:
            return await self._complete(active_state, invocation, error(f"'{tool.name}': {exc}"))
        except Exception as exc:
            log.exception("Tool %s failed (call_id=%s)", tool.name, call.id)
            kind = type(exc).__name__
            return await self._complete(
                active_state, invocation, error(f"'{tool.name}' failed with an internal error ({kind}).")
            )

        # 7. Process output
        if isinstance(output, InputRequest):
            await self._emit(EventType.INPUT_REQUESTED, {"tool": tool.name})
            return PendingCall(kind="input", call=call, prompt=output.question)
        if not isinstance(output, str):
            log.error("Tool %s returned %s instead of str", tool.name, type(output).__name__)
            return await self._complete(active_state, invocation, error(f"'{tool.name}' returned an invalid result."))
        content = self._truncate(output)
        if tool.untrusted_output:
            content = as_untrusted(tool.name, content)
        return await self._complete(active_state, invocation, ToolResult(call_id=call.id, content=content))

    async def _complete(self, state: RunState, invocation: ToolInvocation, result: ToolResult) -> ToolResult:
        # 8. Post-execution hooks
        try:
            decision = await self._hooks.after_tool(state, invocation, result)
        except asyncio.CancelledError:
            withheld = f"'{invocation.tool_name}' was executed, post-check (after_tool) cancelled — result withheld."
            state.batch_done = (*state.batch_done, ToolResult(call_id=result.call_id, content=withheld, is_error=True))
            raise
        if decision.action is Action.BLOCK:
            withheld = f"Result withheld: {decision.reason}"
            result = ToolResult(call_id=result.call_id, content=withheld, is_error=True)
        state.batch_done = (*state.batch_done, result)
        if decision.action is Action.BLOCK:
            await self._emit(EventType.TOOL_BLOCKED, {"tool": invocation.tool_name, "source": "after_tool"})
        await self._emit(EventType.TOOL_FINISHED, {"tool": invocation.tool_name, "is_error": result.is_error})
        return result

    def _truncate(self, content: str) -> str:
        if len(content) <= self._max_result_chars:
            return content
        cut = len(content) - self._max_result_chars
        return f"{content[: self._max_result_chars]}\n… [truncated, {cut} chars omitted]"


def _describe(exc: ValidationError) -> str:
    """Formats validation errors without echoing input values (which may be sensitive)."""
    parts: list[str] = []
    for err in exc.errors(include_input=False, include_url=False):
        loc = ".".join(str(p) for p in err["loc"]) or "(arguments)"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def _preview(tool: BaseTool[Any], args: BaseModel, reasons: list[str]) -> str:
    return (
        f"Tool: {tool.name} (Risk: {tool.risk})\n"
        f"Arguments: {args.model_dump_json(indent=2)}\n"
        f"Reason for approval: {'; '.join(reasons)}"
    )
