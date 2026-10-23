# 0004 – Typed Hooks, Fail-Closed

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-15
- **Affects:** `ai_agent_harness/core/hooks.py`, `ai_agent_harness/tools/executor.py`

## Context

Guardrails are the main security extension point. Early hooks used string names, ignored some decisions, and crashed the run when they raised.

## Decision

Each hook point is a typed method on `Hook` with its own set of allowed `Action`s. An exception or a disallowed action becomes `Decision.block(...)`. `block` and `retry` stop the pipeline; `require_approval` is remembered so a later `block` still wins. `on_run_end` returns nothing; its errors are logged. Tool hooks get validated arguments, and `PermissionPolicy` runs before them.

## Rejected

- Fail-open: a buggy guardrail would let dangerous actions through.
- Crash on hook errors: a custom guardrail bug would crash production.

## Consequences

A buggy guardrail blocks legitimate actions until fixed; watch `tool_blocked` events. Tested in `tests/test_failures.py` and `tests/test_tools.py`.
