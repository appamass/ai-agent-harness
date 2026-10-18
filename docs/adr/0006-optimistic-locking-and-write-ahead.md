# 0006 – Optimistic Locking, Claiming on Resume, and Write-Ahead Checkpoints

- **Status:** Accepted (maintainer, 2026-10-18)
- **Amended by:** [0014](0014-bind-resume-to-pending-call.md) — `resume` of a waiting run requires the `pending_id` that was shown
- **Date:** 2026-10-16
- **Affects:** `ai_agent_harness/persistence.py`, `ai_agent_harness/runtime/runner.py`

## Context

Multiple workers may attempt to resume the same paused run (e.g., when a user double-clicks an approval
button), and worker processes can crash in the middle of executing tools with side effects. Without
coordination, an approved action could execute twice or be silently repeated after a crash.

## Decision

1. Every `RunState` carries a monotonically increasing `revision`; `CheckpointStore.save` raises
   `ConcurrencyError` if the stored revision is newer than expected (in SQL: `UPDATE … WHERE revision = :expected`).
2. `start` saves the initial state immediately; `resume` **claims** the run by saving an incremented
   revision before executing any tool or model call.
3. **Write-ahead checkpointing:** State (including completed tool results in `batch_done`) is saved
   before every tool execution. If a process crashes mid-tool, the checkpoint remains in status `running`
   with open tool calls; `resume(recover=True)` closes only the interrupted tool call as "outcome unknown"
   so the model can verify state before retrying.
4. `JsonFileCheckpointStore` writes atomically (temp file + `fsync` + `os.replace`), uses per-tenant
   POSIX `fcntl` locking, and validates IDs against path traversal and case collisions.

## Alternatives

| Alternative | Why Rejected |
|---|---|
| Pessimistic locks with lease timeouts | Complex to manage; crashed workers leave stale locks until lease expiry |
| Save only after completing a full tool batch | Side effects from earlier tools in the batch are lost if a later tool crashes |

## Consequences

- Positive: At most one worker can advance a run; zero silent duplicate executions after crashes.
- Negative: Saves occur before each tool call in a batch; write tools should still be designed idempotently so the model can safely resolve an "outcome unknown" result.
- Verification: `tests/test_edge_cases.py` (crash inside tool, concurrent approval race), `tests/test_guardrails_and_persistence.py`.
