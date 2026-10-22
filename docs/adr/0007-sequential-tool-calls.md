# 0007 – Tool Calls Run Sequentially

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-15
- **Affects:** `ai_agent_harness/runtime/runner.py`

## Context

A response can contain several tool calls. Running them in parallel is faster but makes order, mid-batch pauses, write-ahead checkpoints, and approvals ambiguous.

## Decision

Calls of one response run in order. A call that pauses holds back the rest until `resume(...)`; finished results stay in `batch_done`.

## Rejected

- `asyncio.gather`: unclear which calls ran when one needs approval, and it breaks order-dependent chains.

## Consequences

Slower for many independent reads. A parallel mode for consecutive `Risk.READ` calls would need its own ADR. Tested by `test_pause_in_the_middle_of_a_batch_keeps_order_and_results`.
