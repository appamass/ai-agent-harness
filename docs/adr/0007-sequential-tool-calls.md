# 0007 – Execute Tool Calls Within a Response Sequentially

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-15
- **Affects:** `ai_agent_harness/runtime/runner.py`

## Context

Language models can emit multiple tool calls in a single assistant response. Running them in parallel
would reduce latency, but makes execution order, mid-batch pauses, write-ahead checkpoints, and human
approvals ambiguous.

## Decision

Tool calls within a single assistant response execute sequentially in order. If a call pauses for
approval or user input, subsequent calls in the batch wait until `resume(...)` is called; already
completed tool results in the batch are preserved in `batch_done`.

## Alternatives

| Alternative | Why Rejected |
|---|---|
| `asyncio.gather` across all tool calls | Makes it ambiguous which calls have already executed when one call requires approval; breaks order-dependent tool chains |

## Consequences

- Positive: Deterministic execution order and simple, crash-safe resume semantics.
- Negative: Higher latency when a model emits many independent read-only calls. An opt-in extension (running consecutive `Risk.READ` calls in parallel) can be added later with its own ADR.
- Verification: `test_pause_in_the_middle_of_a_batch_keeps_order_and_results` in `tests/test_resume.py`.
