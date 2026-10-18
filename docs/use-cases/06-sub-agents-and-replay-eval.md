# Use Case 6: Specialist Sub-Agents as Tools & Single-Step Replay Evals

> **Difficulty: Medium.** Two high-leverage patterns that require zero changes to the core harness:
> (A) An orchestrator delegates focused research tasks to a read-only specialist sub-agent wrapped as a tool.
> (B) Before deploying a new model or system prompt, evaluate what it would do next at recorded
> production decision points — without executing a single tool.

All Python blocks starting with `# executable` are executed by `tests/test_docs.py`.

| Component | Status |
|---|---|
| `start`/`resume`, `Limits`, checkpoints, `restore_state`, `ContextBuilder.build`, `ModelRouter` | ✅ In core harness |
| `Research` tool (sub-agent wrapped as a `BaseTool`), `next_action` (single-step replay eval) | 🧩 Reference implementation in this document |
| Golden dataset of recorded checkpoints, scoring rubric, release-gate workflow | 🔲 Project-specific |

---

## 1. Part A: Specialist Sub-Agent as a Tool

The specialist is a standard `Agent` with its own model, focused toolset, and tight `Limits`. To the
orchestrator, it looks like an ordinary `Risk.READ` tool. If the specialist hits a limit, refuses, or
fails, the orchestrator receives a clean `ToolError` message rather than crashing.

**Rules that keep this pattern safe:**
- **The specialist must never pause:** If a sub-run paused for human approval or user input, the
  orchestrator would have no way to forward the pause, leaving the sub-run stranded forever.
  `Autonomy.READ_ONLY` prevents approvals, but **does not prevent clarifying questions** because
  `ask_user` is a `Risk.READ` tool. Therefore, enforce `Autonomy.READ_ONLY`, forbid `ask_user` (and any
  tool returning `InputRequest`), validate this in `__init__`, and treat any sub-run status other than
  `RunStatus.DONE` as a `ToolError`.
- **Deterministic sub-run ID:** Derive `run_id` from the parent's `run_id`, `step`, and a hash of the
  query (rather than an in-memory counter that resets on process restart), and pass the parent's `tenant_id`.
- **Size `timeout_s` generously:** Set the wrapper tool's `timeout_s` larger than the specialist's
  `Limits.max_seconds` **plus** `request_timeout_s * max_attempts` from its `RetryPolicy` (since
  `max_seconds` is checked before each model call, a single slow model call can exceed `max_seconds`).
- **Separate token accounting:** The specialist's token usage is recorded on its own `RunState` and
  `EventBus`, not added to the orchestrator's `state.usage`. Subscribe your cost listener to both agents
  (they share the same `tenant_id`).

```python
# executable
import asyncio
import hashlib
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ai_agent_harness import (
    Agent,
    AskUser,
    Autonomy,
    BaseTool,
    ContextBuilder,
    Limits,
    ModelRouter,
    RetryPolicy,
    Risk,
    RunState,
    RunStatus,
    ToolError,
    ToolRegistry,
    start,
)
from ai_agent_harness.testing import FakeProvider, text_response, tool_response


class TopicInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=3, max_length=200)


class SearchDocs(BaseTool[TopicInput]):
    name: ClassVar[str] = "search_docs"
    description: ClassVar[str] = "Searches internal runbooks and documentation."
    input_model: ClassVar[type[BaseModel]] = TopicInput
    risk: ClassVar[Risk] = Risk.READ

    async def run(self, args: TopicInput, state: RunState) -> str:
        return f"Match for '{args.topic}': Runbook RB-12, Section 3"


class Research(BaseTool[TopicInput]):
    """Delegates a documentation research task to a read-only specialist sub-agent."""

    name: ClassVar[str] = "research"
    description: ClassVar[str] = "Has a specialist search internal docs and return a concise summary."
    input_model: ClassVar[type[BaseModel]] = TopicInput
    risk: ClassVar[Risk] = Risk.READ
    timeout_s: ClassVar[float] = 240.0  # > max_seconds (60) + request_timeout_s * max_attempts (60 * 2)

    def __init__(self, specialist: Agent) -> None:
        if specialist.autonomy is not Autonomy.READ_ONLY:
            raise ValueError("Specialist sub-agents must use Autonomy.READ_ONLY so they never pause for approval")
        if "ask_user" in specialist.tools.names():
            raise ValueError("Specialist sub-agents must not include ask_user")
        self.specialist = specialist

    async def run(self, args: TopicInput, state: RunState) -> str:
        topic_hash = hashlib.sha256(args.topic.encode()).hexdigest()[:8]
        sub = await start(
            self.specialist,
            args.topic,
            tenant_id=state.tenant_id,
            run_id=f"{state.run_id[:40]}-s{state.step}-{topic_hash}",  # deterministic across restarts
        )
        if sub.status is not RunStatus.DONE:
            raise ToolError(f"Research aborted ({sub.status}): {sub.status_reason or 'no reason provided'}")
        return sub.last_text
```

| Step | Orchestrator | Specialist Sub-Agent | Outcome |
|---|---|---|---|
| 1 | `research("checkout pool size")` | Searches docs, summarizes | ✅ Summary returned as tool output |
| 2 | `research("full incident history")` | Searches twice, hits `max_steps=2` | ❌ `limit_reached` returned as `ToolError` |
| 3 | Answers using available findings | – | ✅ `done` |

```python
# executable
specialist = Agent(
    name="researcher",
    router=ModelRouter(retry=RetryPolicy(max_attempts=2, request_timeout_s=60), primary=FakeProvider([
        tool_response(("search_docs", {"topic": "checkout pool size"})),
        text_response("RB-12 §3: Never set checkout connection pool size below 20."),
        tool_response(("search_docs", {"topic": "Incidents 2024"})),
        tool_response(("search_docs", {"topic": "Incidents 2025"})),  # hits max_steps=2 after this turn
    ])),
    context=ContextBuilder("You search internal documentation and summarize findings concisely."),
    tools=ToolRegistry([SearchDocs()]),
    autonomy=Autonomy.READ_ONLY,
    limits=Limits(max_steps=2, max_total_tokens=20_000, max_seconds=60),
)
orchestrator = Agent(
    name="orchestrator",
    router=ModelRouter(FakeProvider([
        tool_response(("research", {"topic": "checkout pool size"})),
        tool_response(("research", {"topic": "full incident history"})),
        text_response("Pool size must be at least 20 (RB-12). Full incident history exceeded step budget."),
    ])),
    context=ContextBuilder("You answer operational questions using the research specialist."),
    tools=ToolRegistry([Research(specialist)]),
    autonomy=Autonomy.READ_ONLY,
)

state = asyncio.run(start(orchestrator, "What is the minimum pool size for checkout?", tenant_id="ops"))
assert state.status is RunStatus.DONE
first, second = [r for m in state.messages for r in m.tool_results]
assert first.content == "RB-12 §3: Never set checkout connection pool size below 20." and not first.is_error
assert second.is_error and "limit_reached" in second.content
assert state.usage.total == 3 * 20  # counts only the orchestrator's 3 turns; specialist usage is tracked separately

# A specialist configured with ask_user is rejected at registration time
asking = Agent(name="asking", router=ModelRouter(FakeProvider([])), context=ContextBuilder("x"),
               tools=ToolRegistry([SearchDocs(), AskUser()]), autonomy=Autonomy.READ_ONLY)
try:
    Research(asking)
    raise AssertionError("Specialist with ask_user should have been rejected")
except ValueError:
    pass
```

---

## 2. Part B: Single-Step Replay Evaluation on Recorded Checkpoints

A saved checkpoint contains the exact conversation history the production model saw right before making
a critical decision. That makes it possible to evaluate a candidate model or a revised system prompt on
**real production decision points**: slice the history up to the decision turn, build the `ModelRequest`
using the agent's `ContextBuilder`, call `model.complete(request)` **once**, and compare the proposed
action. Zero tools execute and zero state is mutated.

**What this is — and what it is not:** This is a single-step replay evaluation, not a multi-turn
simulation. If a candidate model chooses a different tool on Turn $N$, Turn $N+1$ would require tool
outputs that were never recorded. Single-step replay evaluates one decision at a time across hundreds
of real checkpoints.

```python
# executable
from ai_agent_harness import ModelProvider, restore_state, resume
from ai_agent_harness.core import ToolCall
from ai_agent_harness.policy import Verdict

PROD_SYSTEM = "You are the Incident Responder. Propose at most one remediation action."


class Rollback(BaseTool[TopicInput]):
    name: ClassVar[str] = "rollback_deploy"
    description: ClassVar[str] = "Rolls a service back to an earlier deployment."
    input_model: ClassVar[type[BaseModel]] = TopicInput
    risk: ClassVar[Risk] = Risk.IRREVERSIBLE

    async def run(self, args: TopicInput, state: RunState) -> str:
        raise AssertionError("Tools must never execute during replay evaluation")


class Restart(BaseTool[TopicInput]):
    name: ClassVar[str] = "restart_service"
    description: ClassVar[str] = "Restarts a service."
    input_model: ClassVar[type[BaseModel]] = TopicInput
    risk: ClassVar[Risk] = Risk.IRREVERSIBLE

    async def run(self, args: TopicInput, state: RunState) -> str:
        raise AssertionError("Tools must never execute during replay evaluation")


def incident_agent(model: ModelProvider, system: str = PROD_SYSTEM) -> Agent:
    return Agent(
        name="incident",
        router=ModelRouter(model),
        context=ContextBuilder(system),
        tools=ToolRegistry([SearchDocs(), Rollback(), Restart()]),
        autonomy=Autonomy.ACT_WITH_APPROVAL,
    )


async def next_action(agent: Agent, checkpoint: str, model: ModelProvider) -> ToolCall | str:
    """Queries `model` once to see what it would have proposed right before the recorded decision."""
    recorded = restore_state(checkpoint)
    history = recorded.messages[:-1]  # history right before the final assistant decision turn
    before = RunState(agent_name=recorded.agent_name, tenant_id=recorded.tenant_id, messages=list(history))
    before.check_protocol()
    visible = [t.spec() for t in agent.tools if agent.policy.check(t.name, t.risk) is not Verdict.DENY]
    request = await agent.context.build(before, visible)  # exact same ModelRequest the runner builds
    response = await model.complete(request)  # direct call bypassing runner: no tools, no checkpoints
    calls = response.message.tool_calls
    return calls[0] if calls else response.message.text


def decision(call: ToolCall | str) -> str:
    return f"{call.name}({call.arguments.get('topic')})" if isinstance(call, ToolCall) else "text-only response"


# A recorded production run: the existing model proposed a rollback and paused for approval
old_model = FakeProvider([
    tool_response(("search_docs", {"topic": "checkout errors"})),
    tool_response(("rollback_deploy", {"topic": "checkout to d-1042"})),
])
live = asyncio.run(start(incident_agent(old_model), "ALERT checkout", tenant_id="ops", run_id="inc-9001"))
assert live.status is RunStatus.WAITING_FOR_APPROVAL
checkpoint = live.model_dump_json()  # production: raw JSON from checkpoint store or golden eval dataset

# Candidate 1 matches the recorded decision; Candidate 2 proposes a service restart instead
candidate_1 = FakeProvider([tool_response(("rollback_deploy", {"topic": "checkout to d-1042"}))], id="candidate-1")
candidate_2 = FakeProvider([tool_response(("restart_service", {"topic": "checkout"}))], id="candidate-2")
recorded_call = live.messages[-1].tool_calls[0]

verdicts = {}
for candidate in (candidate_1, candidate_2):
    proposal = asyncio.run(next_action(incident_agent(candidate), checkpoint, candidate))
    verdicts[candidate.id] = decision(proposal) == decision(recorded_call)

assert verdicts == {"candidate-1": True, "candidate-2": False}
assert candidate_1.requests[0].system == PROD_SYSTEM            # identical request construction
assert candidate_1.requests[0].messages == tuple(live.messages[:-1])
assert live.model_dump_json() == checkpoint                      # replay eval never mutates the live run

# Meanwhile, the live run continues normally
done = asyncio.run(resume(incident_agent(FakeProvider([text_response("Denied, skipping rollback.")])), live,
                          approved=False, answer="Eval run, do not execute",
                          pending_id=live.pending_id))
assert done.status is RunStatus.DONE
```

This turns subjective impressions ("the new model feels smarter") into concrete release metrics — e.g.,
across 200 recorded incidents, Candidate B chose the same action 94% of the time, a more conservative
read action 5% of the time, and a riskier action 1% of the time — allowing engineers to inspect those
1% regressions before switching traffic.

---

## 3. Honest Boundaries

- **Single step, not full trajectory:** Once a candidate model picks a different tool, there are no
  recorded tool outputs for that branch. Multi-step trajectory evals require mocked environment tools.
- **Stochastic outputs:** Live models are non-deterministic even at low temperature. Run each decision
  point $k$ times (e.g. `pass@k` or majority vote) and compare distributions.
- **Production data governance:** Checkpoints contain customer data. Golden eval datasets derived from
  production checkpoints must follow the same privacy, access-control, and retention rules as production
  (or be scrubbed/synthesized first).
- **Sub-agents cannot pause:** Any action that may require human approval or user clarification belongs
  in the top-level orchestrator, not inside a specialist sub-agent tool.

## 4. What the Core Guarantees — and What Your Project Must Provide

**Guaranteed by the core harness (tested):** a sub-agent run always terminates with a defined `RunStatus`
(even on budget exhaustion or provider failure) so the parent orchestrator stays in control; `Limits`
apply independently per agent; saved checkpoints load with full schema migration and protocol checks via
`restore_state`; `ContextBuilder.build` deterministically reconstructs the exact `ModelRequest` from `RunState`.

**Must be provided per project:** curation of representative decision-point checkpoints, domain-specific
scoring rubrics (exact match vs. risk-weighted equivalence), data scrubbing for eval datasets, and
aggregating sub-agent token costs via `EventBus`.
