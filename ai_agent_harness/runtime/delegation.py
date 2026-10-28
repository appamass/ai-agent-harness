"""Sub-agents: a specialist agent wrapped as a read-only tool of another agent (ADR 0030).

To the calling agent, the specialist is an ordinary `Risk.READ` tool: it runs through the execution gate like every
other tool, and its answer comes back framed as untrusted data. The rules that keep delegation safe are checked when
the tool is built, not left to the model:

- The specialist runs at `Autonomy.READ_ONLY`. It can never write, never needs an approval, and so never holds more
  rights than a read tool of the caller. An approval in the caller can never cover writes the specialist chose.
- The specialist cannot ask the user (`ask_user` must not be visible to it). A sub-run that still waits for input or
  approval, or ends without `done` for any other reason, is reported to the caller as a tool error.
- The sub-run ID is derived from the caller's tenant, agent, run, step, and position in the tool batch. With a
  checkpoint store on the specialist, the same call can never start a second sub-run.
- Cycles (including back to the top-level agent) and deep chains are refused (`MAX_SUB_AGENT_DEPTH`).

Tokens and events stay with the sub-run (its own `RunState`, the specialist's `EventBus`). `deps` are not passed on.
A sub-run that ends waiting (for example a hook that requires approval) is cancelled at once (`cancel`, ADR 0031), so
every sub-run ends in a final status. A sub-run that a crash of the caller interrupts stays `running`; it is surely dead
once it is older than the tool's `timeout_s`, and the recovery job cancels it with `cancel(..., recover=True)`.
"""

from __future__ import annotations

import asyncio
import hashlib
from contextvars import ContextVar
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, create_model

from ..core.errors import ConfigError, ToolError
from ..core.state import RunState, RunStatus
from ..core.types import Risk, is_positive_number
from ..policy import Autonomy, Verdict
from ..tools.base import BaseTool, ToolOutput, make_tool
from ..tools.builtin import AskUser
from .agent import Agent
from .runner import cancel, start

MAX_SUB_AGENT_DEPTH = 4
"""How many sub-runs may be nested below a top-level run. A deeper delegation is refused as a tool error."""


class _Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str


_WAITING = frozenset({RunStatus.WAITING_FOR_INPUT, RunStatus.WAITING_FOR_APPROVAL})
_chain: ContextVar[tuple[str, ...]] = ContextVar("ai_agent_harness_sub_agent_chain", default=())
"""Names of the specialists whose sub-runs are active in the current task, outermost first."""


def sub_agent(
    specialist: Agent,
    *,
    name: str | None = None,
    description: str,
    timeout_s: float | None = None,
) -> BaseTool[Any]:
    """Wraps `specialist` as a `Risk.READ` tool that runs it on one task and returns its final text.

    `name` defaults to `specialist.name`. `description` tells the calling model when to delegate.
    `timeout_s` must be larger than `specialist.limits.max_seconds`. The default adds one worst-case model call and
    the slowest tool of the specialist, because the time limit is only checked before each model call. It is an
    estimate: a response with many slow tool calls can take longer, and the caller then cancels the sub-run.
    """
    if specialist.autonomy is not Autonomy.READ_ONLY:
        raise ConfigError(
            f"Sub-agent '{specialist.name}' must use autonomy 'read_only', got '{specialist.autonomy}': "
            "a delegated run can never wait for an approval or write under the caller's rights"
        )
    if _offers_ask_user(specialist):
        raise ConfigError(
            f"Sub-agent '{specialist.name}' must not offer ask_user: a delegated run cannot ask the user "
            "(remove it or leave it out of allowed_tools)"
        )
    max_seconds = specialist.limits.max_seconds
    if timeout_s is None:
        slowest_tool = max((tool.timeout_s for tool in specialist.tools), default=0.0)
        timeout_s = max_seconds + specialist.router.worst_case_s() + slowest_tool
    elif not is_positive_number(timeout_s) or timeout_s <= max_seconds:
        raise ConfigError(
            f"Sub-agent '{specialist.name}': timeout_s must be larger than the specialist's "
            f"Limits.max_seconds ({max_seconds:g}), got {timeout_s!r}"
        )
    tool_name = name if name is not None else specialist.name
    input_model: type[_Task] = create_model(
        f"{tool_name}_Input",
        __base__=_Task,
        task=(
            str,
            Field(
                min_length=1,
                pattern=r"\S",
                max_length=specialist.limits.max_input_chars,
                description="The complete task for the specialist, including every detail it needs.",
            ),
        ),
    )

    async def run(self: BaseTool[BaseModel], args: BaseModel, state: RunState) -> ToolOutput:
        lineage = _chain.get() or (state.agent_name,)  # the top-level agent first, then every active specialist
        if specialist.name in lineage:
            raise ToolError(f"refused: delegation cycle ('{specialist.name}' is already running in this chain)")
        if len(lineage) > MAX_SUB_AGENT_DEPTH:
            raise ToolError(f"refused: delegation depth limit ({MAX_SUB_AGENT_DEPTH}) reached")
        if _offers_ask_user(specialist):  # tools can be registered after the build-time check
            raise ToolError("refused: the sub-agent offers ask_user, and sub-agents cannot ask the user")
        token = _chain.set((*lineage, specialist.name))
        try:
            sub = await start(
                specialist,
                cast("_Task", args).task,
                tenant_id=state.tenant_id,
                run_id=_sub_run_id(state, tool_name),
            )
        finally:
            _chain.reset(token)
        if sub.status is RunStatus.DONE:
            return sub.last_text
        if sub.status in _WAITING:
            waited = sub.status
            # Shielded: a timeout of this tool must not leave the sub-run waiting halfway through its final save
            await asyncio.shield(cancel(specialist, sub, reason="A sub-agent run cannot wait for input or approval"))
            raise ToolError(f"sub-agent ended with status '{waited}': sub-agents cannot ask for input or approval")
        raise ToolError(f"sub-agent ended with status '{sub.status}' and gave no answer")

    return make_tool(
        BaseTool[BaseModel],
        "_SubAgentTool",
        run=run,
        name=tool_name,
        description=description,
        input_model=input_model,
        risk=Risk.READ,
        timeout_s=timeout_s,
        untrusted_output=True,
    )


def _offers_ask_user(agent: Agent) -> bool:
    return any(
        (tool.name == AskUser.name or isinstance(tool, AskUser))
        and agent.policy.check(tool.name, tool.risk) is not Verdict.DENY
        for tool in agent.tools
    )


def _sub_run_id(state: RunState, tool_name: str) -> str:
    """The run ID of the sub-run that one call of a sub-agent tool starts: the same call always yields the same ID.

    Agent name and creation time tell apart callers that share a `run_id` (like `pending_id`). Sub-agent tools run
    one after the other, so the number of finished results in the batch tells calls of one step apart.
    """
    parts = (
        state.tenant_id,
        state.agent_name,
        state.run_id,
        state.created_at.isoformat(),
        str(state.step),
        str(len(state.batch_done)),
        tool_name,
    )
    return "sub-" + hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:40]
