# How It Works

The idea: **the model proposes, the code decides.** The model is the only probabilistic part. Everything around it is plain, tested code that
builds the context, validates every proposed action, asks a human where needed, saves the state, and ends every run with a status.

## Code map

```
ai_agent_harness/
├── core/                 Zero-dependency core (imports only the standard library and Pydantic)
│   ├── types.py          Canonical message format (TextBlock, ImageBlock, ThinkingBlock, ToolCall, ToolResult)
│   ├── state.py          RunState: state + protocol invariants, RunStatus, PendingBatch, restore_state
│   ├── hooks.py          Hook, HookPipeline (fail-closed), Decision, ToolInvocation
│   ├── events.py         EventBus, Event, EventType (metadata only)
│   └── errors.py         Error hierarchy
├── providers/
│   ├── base.py           ModelProvider protocol, Capabilities, adapter contract, without_foreign_thinking
│   ├── router.py         RetryPolicy, exponential backoff with jitter, timeout, failover, capability filter
│   └── anthropic.py      AnthropicProvider (optional extra `anthropic`; the core never imports it)
├── tools/
│   ├── base.py           BaseTool, SyncTool (isolated thread pool), InputRequest
│   ├── registry.py       ToolRegistry with startup schema and definition validation
│   ├── executor.py       ToolExecutor: the single tool execution gate
│   └── builtin.py        AskUser, ReadFile (reference tools)
├── runtime/
│   ├── agent.py          Agent: validated composition of the building blocks
│   └── runner.py         start / resume: the agent loop
├── context.py            ContextBuilder, ContextSource
├── policy.py             Autonomy levels, allowlist, Limits
├── guardrails.py         ProtectedPaths (anti-reward-hacking), KillSwitch
├── persistence.py        CheckpointStore, InMemoryCheckpointStore, JsonFileCheckpointStore (atomic + locked)
├── persistence_postgres.py  PostgresCheckpointStore (optional extra `postgres`; the core never imports it)
├── observability.py      log_listener
└── testing.py            FakeProvider and response helpers for deterministic tests
```

Dependencies point strictly inward: `core` knows no other subpackage, and `runtime` is the only layer that wires all building blocks together
(enforced by `tests/test_architecture.py`). Outside the package: `examples/` (a runnable agent), `tests/`, `docs/` (this folder).

## The run loop

```
start(agent, text) / resume(agent, state, …)
 └─ on_user_input (hooks)                         block → blocked
 └─ Loop:
     ├─ Check limits                              → limit_reached
     ├─ Build context (only permitted tools visible)
     ├─ Check context budget                      too large → limit_reached
     ├─ before_model (hooks)                      block → blocked
     ├─ Router → Adapter (retry/backoff/failover) outage → failed
     ├─ after_model (hooks)                       retry → reject + feedback (bounded)
     ├─ Stop reason:
     │    Tool calls  → Execution gate per call   approval/question → waiting_for_*
     │    MAX_TOKENS  → Continue text / reject truncated tool calls
     │    REFUSAL     → refused
     │    END         → done
     └─ Checkpoint
 └─ on_run_end (hooks, on a final status) + emit run_finished
```

**Status transitions**

```
start ──► running ──► done | refused | blocked | limit_reached | failed | cancelled
             │  ▲                                      │
             ▼  │ resume(answer=… / approved=…,        │ resume(user_input=…)
             │  │        pending_id=…)                 │
   waiting_for_input | waiting_for_approval            ▼
                                                     running
```

| Status | Resume with |
|---|---|
| `waiting_for_input` | `resume(agent, state, answer="…", pending_id=shown_id)` — the answer becomes the result of `ask_user` |
| `waiting_for_approval` | `resume(agent, state, approved=True/False, answer="optional comment", pending_id=shown_id)` |
| `done`, `refused`, `blocked`, `limit_reached` | `resume(agent, state, user_input="…")` |
| `failed`, `cancelled` | `resume(agent, state)` (for example after a provider outage) or with `user_input="…"`; the diagram above shows only the `user_input` way back |
| `running` in a checkpoint | `resume(agent, state, recover=True)` — after a process crash only |

Invalid argument combinations raise `InvalidStateError` before any state is modified. Every `resume` of a waiting run needs the `pending_id` exactly
as it was shown to the human: if the run now waits for a different call (another approver was faster), the stale approval or answer is rejected
([ADR 0014](adr/0014-bind-resume-to-pending-call.md)). **Never read `pending_id` from a state you just loaded to handle a click** — that approves whatever
the run waits for now. It must come from the request the human saw (for example stored in the dialog or button payload).

## The tool execution gate

Every tool call passes the same steps, cheapest and safest first (`ai_agent_harness/tools/executor.py`):
lookup → Pydantic argument validation → policy → `before_tool` hooks → human approval if required (this pauses the run) → execute with timeout →
truncate output → `after_tool` hooks. No other code calls a tool.

## Autonomy and risk

|                     | READ  | WRITE    | IRREVERSIBLE |
|---------------------|-------|----------|--------------|
| `read_only` (default) | Allow | Deny     | Deny         |
| `suggest`           | Allow | Approval | Deny         |
| `act_with_approval` | Allow | Approval | Approval     |
| `autonomous`        | Allow | Allow    | Approval     |

Denied tools are never even offered to the model. `Agent(allowed_tools=…)` restricts tools further through an explicit allowlist (typos fail at startup).
The table is code, and this check runs in CI:

```python
# executable
from ai_agent_harness import Autonomy, Risk
from ai_agent_harness.policy import PermissionPolicy, Verdict

ALLOW, ASK, DENY = Verdict.ALLOW, Verdict.NEEDS_APPROVAL, Verdict.DENY
matrix = {
    Autonomy.READ_ONLY: (ALLOW, DENY, DENY),
    Autonomy.SUGGEST: (ALLOW, ASK, DENY),
    Autonomy.ACT_WITH_APPROVAL: (ALLOW, ASK, ASK),
    Autonomy.AUTONOMOUS: (ALLOW, ALLOW, ASK),
}
for autonomy, expected in matrix.items():
    policy = PermissionPolicy(autonomy=autonomy)
    assert tuple(policy.check("any_tool", risk) for risk in (Risk.READ, Risk.WRITE, Risk.IRREVERSIBLE)) == expected
assert PermissionPolicy().autonomy is Autonomy.READ_ONLY  # the default is the safest level
```

## Mapping to the harness taxonomy

[`../HARNESS_TAXONOMY.md`](../HARNESS_TAXONOMY.md) maps nine groups of agent patterns to classical software engineering. This is what the core implements and where you plug in.

| Taxonomy section | Implemented in the core | Extension point |
|---|---|---|
| 1. Control flow | Agent loop, termination, stop reasons, human in the loop, clarifying questions, cancellation, resume | Workflows or graphs as a layer on top of `start`/`resume` |
| 2. Context and tools | Tool definitions, execution gate, output truncation, context builder | `ContextSource` (memory, RAG, skills), compaction |
| 3. Validation | Structured types, Pydantic validation, retry with feedback, stop-reason handling | Hooks (`after_model`: schema, grounding, recalculation) |
| 4. Security | Autonomy × risk matrix, allowlist, approvals, tenant IDs, safe error messages | Hooks (injection, PII), sandboxing inside tools |
| 5. Reliability and operations | Checkpoints, limits, events, retry/backoff/failover, timeouts, kill switch | Event listeners (tracing, cost), CI evals |
| 6. Multi-agent | Optimistic locking (`revision`) | A sub-agent wrapped as a `BaseTool` that calls `start` with narrower permissions |
| 7. Scoring | Anti-reward-hacking (`ProtectedPaths`) | Hooks with scorers or judges |
| 8. Multi-model | Canonical message format (including thinking and images), capabilities, router with failover | Per-provider adapters, prompt profiles in `ContextBuilder` |
| 9. UX | Metadata events, approval previews, clarifying questions | Your UI or chat integration |

Deliberately **not** included: empty placeholder modules ([ADR 0008](adr/0008-no-placeholder-modules.md)). Every module is a complete, tested implementation.
