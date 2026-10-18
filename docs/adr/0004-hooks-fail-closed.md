# 0004 – Typed Hooks with Per-Stage Allowed Decisions, Fail-Closed

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-15
- **Affects:** `ai_agent_harness/core/hooks.py`, `ai_agent_harness/tools/executor.py`

## Context

Guardrails are the primary security extension point. In early prototypes, hooks were dispatched by string
names, some returned decisions were silently ignored (e.g. at `after_tool`), and an unhandled exception
inside a hook crashed the entire run.

## Decision

Each hook point is a dedicated, typed method on `Hook`. Every stage explicitly defines which `Action`
values are valid at that stage. Any exception raised by a hook or any disallowed `Action` is converted
into `Decision.block(...)` (fail-closed). `block` and `retry` short-circuit the pipeline immediately;
`require_approval` is remembered while remaining hooks run so a subsequent `block` still wins.
`on_run_end` returns `None`; exceptions there are logged without altering the final status. Tool hooks
receive validated Pydantic arguments (`ToolInvocation`), and the `PermissionPolicy` runs before hooks.

## Alternatives

| Alternative | Why Rejected |
|---|---|
| Fail-open (ignore hook errors) | A buggy guardrail would silently allow dangerous actions through |
| Let hook exceptions crash the run | A bug in a custom guardrail would cause unhandled crashes in production |

## Consequences

- Positive: A guardrail bug can never accidentally permit an action.
- Negative: A buggy guardrail blocks legitimate actions until fixed — requires monitoring on `tool_blocked` and `run_finished` events.
- Verification: `tests/test_failures.py`, `tests/test_tools.py`.
