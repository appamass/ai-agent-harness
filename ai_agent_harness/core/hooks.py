"""Hooks: the extension mechanism for guardrails and cross-cutting concerns.

Each hook point only allows decisions that are actually enforced at that point:

| Hook point       | Allowed decisions                     | Effect                                             |
|------------------|---------------------------------------|----------------------------------------------------|
| on_user_input    | continue, block                       | Check user input (including answers to questions)  |
| before_model     | continue, block                       | Check request before calling the model             |
| after_model      | continue, block, retry                | Check response; retry = discard + feedback         |
| before_tool      | continue, block, require_approval     | Check tool call with validated arguments           |
| after_tool       | continue, block                       | Check result; block = withhold result              |
| on_run_end       | (no decision)                         | Cleanup: memory extraction, feedback, metrics      |

Fail-closed: If a hook raises an exception or returns a decision not allowed at
that hook point, it is treated as `block`. A broken guardrail never lets
anything slip through. `on_run_end` is the exception: errors there are only
logged because the run has already finished.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, model_validator

from .types import Risk, ToolCall, ToolResult

if TYPE_CHECKING:
    from .state import RunState
    from .types import ModelRequest, ModelResponse

log = logging.getLogger("ai_agent_harness.hooks")


class Action(StrEnum):
    CONTINUE = "continue"
    BLOCK = "block"
    RETRY = "retry"
    REQUIRE_APPROVAL = "require_approval"


class Decision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    action: Action = Action.CONTINUE
    reason: str | None = None

    @model_validator(mode="after")
    def _reason_required(self) -> Self:
        if self.action is not Action.CONTINUE and not (self.reason and self.reason.strip()):
            raise ValueError("Every decision other than 'continue' requires a reason")
        return self

    @classmethod
    def block(cls, reason: str) -> Decision:
        return cls(action=Action.BLOCK, reason=reason)

    @classmethod
    def retry(cls, reason: str) -> Decision:
        return cls(action=Action.RETRY, reason=reason)

    @classmethod
    def require_approval(cls, reason: str) -> Decision:
        return cls(action=Action.REQUIRE_APPROVAL, reason=reason)


CONTINUE = Decision()


@dataclass(frozen=True)
class ToolInvocation:
    """What tool hooks inspect: the raw call plus validated arguments and risk level."""

    call: ToolCall
    tool_name: str
    risk: Risk
    args: BaseModel


class Hook:
    """Base class: every method allows execution by default. Override only what you need."""

    @property
    def name(self) -> str:
        return type(self).__name__

    async def on_user_input(self, state: RunState, text: str) -> Decision:
        return CONTINUE

    async def before_model(self, state: RunState, request: ModelRequest) -> Decision:
        return CONTINUE

    async def after_model(self, state: RunState, response: ModelResponse) -> Decision:
        return CONTINUE

    async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
        return CONTINUE

    async def after_tool(self, state: RunState, invocation: ToolInvocation, result: ToolResult) -> Decision:
        return CONTINUE

    async def on_run_end(self, state: RunState) -> None:
        return None


_ALLOWED: dict[str, frozenset[Action]] = {
    "on_user_input": frozenset({Action.CONTINUE, Action.BLOCK}),
    "before_model": frozenset({Action.CONTINUE, Action.BLOCK}),
    "after_model": frozenset({Action.CONTINUE, Action.BLOCK, Action.RETRY}),
    "before_tool": frozenset({Action.CONTINUE, Action.BLOCK, Action.REQUIRE_APPROVAL}),
    "after_tool": frozenset({Action.CONTINUE, Action.BLOCK}),
}


class HookPipeline:
    """Executes hooks in order.

    `block` and `retry` stop evaluation immediately. `require_approval` is
    recorded while remaining hooks continue to run — a later `block` wins.
    """

    def __init__(self, hooks: Sequence[Hook] = ()) -> None:
        self._hooks: tuple[Hook, ...] = tuple(hooks)

    @property
    def hooks(self) -> tuple[Hook, ...]:
        return self._hooks

    async def on_user_input(self, state: RunState, text: str) -> Decision:
        return await self._decide("on_user_input", lambda h: h.on_user_input(state, text))

    async def before_model(self, state: RunState, request: ModelRequest) -> Decision:
        return await self._decide("before_model", lambda h: h.before_model(state, request))

    async def after_model(self, state: RunState, response: ModelResponse) -> Decision:
        return await self._decide("after_model", lambda h: h.after_model(state, response))

    async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
        return await self._decide("before_tool", lambda h: h.before_tool(state, invocation))

    async def after_tool(self, state: RunState, invocation: ToolInvocation, result: ToolResult) -> Decision:
        return await self._decide("after_tool", lambda h: h.after_tool(state, invocation, result))

    async def on_run_end(self, state: RunState) -> None:
        for hook in self._hooks:
            try:
                await hook.on_run_end(state)
            except Exception:
                log.exception("Hook %s failed in on_run_end", hook.name)

    async def _decide(self, point: str, invoke: Callable[[Hook], Awaitable[Decision]]) -> Decision:
        allowed = _ALLOWED[point]
        approval: Decision | None = None
        for hook in self._hooks:
            try:
                decision = await invoke(hook)
            except Exception:
                log.exception("Hook %s failed in %s — fail-closed", hook.name, point)
                return Decision.block(f"Guardrail '{hook.name}' failed")
            if not isinstance(decision, Decision) or decision.action not in allowed:
                action = decision.action if isinstance(decision, Decision) else type(decision).__name__
                log.error("Hook %s returned an invalid decision in %s: %s", hook.name, point, action)
                return Decision.block(f"Guardrail '{hook.name}' returned an invalid decision")
            if decision.action in (Action.BLOCK, Action.RETRY):
                return decision
            if decision.action is Action.REQUIRE_APPROVAL and approval is None:
                approval = decision
        return approval or CONTINUE
