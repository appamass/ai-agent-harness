# 0006 – Optimistic Locking and Write-Ahead Checkpoints

- **Status:** Accepted (maintainer, 2026-10-18)
- **Amended by:** [0014](0014-bind-resume-to-pending-call.md): `resume` of a waiting run needs the `pending_id` that was shown
- **Date:** 2026-10-16
- **Affects:** `ai_agent_harness/persistence.py`, `ai_agent_harness/runtime/runner.py`

## Context

Several workers may resume the same run (a double click), and a worker may crash in the middle of a tool with side effects. An approved action could run twice.

## Decision

1. Every `RunState` has a rising `revision`; `CheckpointStore.save` raises `ConcurrencyError` if the stored revision moved on (SQL: `INSERT … ON CONFLICT DO UPDATE … WHERE revision = :expected RETURNING revision`).
2. `start` saves at once; `resume` claims the run with a new revision before any tool or model call.
3. State, including finished results (`batch_done`), is saved before every tool call. After a crash the checkpoint is still `running`; `resume(recover=True)` closes only the interrupted call as "outcome unknown".
4. `JsonFileCheckpointStore` writes atomically (temp file, `fsync`, `os.replace`), locks per tenant with POSIX `fcntl`, and rejects path traversal and case collisions.

## Rejected

- Pessimistic locks with leases: stale locks after crashes.
- Saving only after a full batch: earlier side effects are lost.

## Consequences

One save per tool call. Write tools should still be idempotent. Tested in `tests/test_edge_cases.py` and `tests/test_guardrails_and_persistence.py`.
