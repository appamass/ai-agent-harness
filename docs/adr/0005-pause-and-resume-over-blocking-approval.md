# 0005 – Pause and Resume Instead of Blocking Approval Gateways

- **Status:** Accepted (maintainer, 2026-10-18)
- **Amended by:** [0014](0014-bind-resume-to-pending-call.md) — `resume` of a waiting run requires the `pending_id` that was shown
- **Date:** 2026-10-15
- **Affects:** `ai_agent_harness/runtime/runner.py`, `ai_agent_harness/core/state.py`, `ai_agent_harness/tools/base.py`

## Context

Human approvals and clarifying questions can take minutes, hours, or days. Holding an async task open
while waiting on a blocking `ApprovalGateway` ties up worker processes, loses state on restarts or
deployments, and fails to scale. In early prototypes, clarifying questions were tracked via a magic
scratchpad key and broke when resumed across processes.

## Decision

Whenever a tool call requires human approval (via `PermissionPolicy` or a `before_tool` hook) or a tool
returns `InputRequest`, the run pauses cleanly: `PendingBatch` records the already completed tool results
in the current batch, the paused `PendingCall`, and the remaining calls; `state.status` becomes
`waiting_for_approval` or `waiting_for_input`. Calling `resume(...)` later (potentially in a different
worker process) continues the batch: for `waiting_for_input`, the human's `answer` becomes the result
of the paused tool call; for `waiting_for_approval`, `approved=True` executes the tool call (re-running
the `KillSwitch` first). Human answers pass through `on_user_input` hooks.

## Alternatives

| Alternative | Why Rejected |
|---|---|
| Blocking `ApprovalGateway` callback | Holds worker processes open and loses state on process restart |
| Clarifying questions as normal assistant text | Disconnects the tool call from the user's answer and breaks the tool-call protocol |

## Consequences

- Positive: Approvals and questions work reliably across process restarts and multi-day pauses; human reviewers get a structured preview in `state.status_reason`.
- Negative: Downstream applications need a UI or notification queue to display paused runs and invoke `resume(...)`.
- Verification: `tests/test_resume.py`, `tests/test_edge_cases.py`.
