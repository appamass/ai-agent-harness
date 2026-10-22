# 0005 – Pause and Resume Instead of Blocking Approval

- **Status:** Accepted (maintainer, 2026-10-18)
- **Amended by:** [0014](0014-bind-resume-to-pending-call.md): `resume` of a waiting run needs the `pending_id` that was shown
- **Date:** 2026-10-15
- **Affects:** `ai_agent_harness/runtime/runner.py`, `ai_agent_harness/core/state.py`, `ai_agent_harness/tools/base.py`

## Context

Approvals and questions can take days. A blocking gateway holds workers, loses state on restart, and does not scale.

## Decision

When a call needs approval (policy or `before_tool` hook) or a tool returns `InputRequest`, the run pauses: `PendingBatch` stores the finished results, the waiting `PendingCall`, and the remaining calls; the status becomes `waiting_for_approval` or `waiting_for_input`. `resume(...)` continues the batch, possibly in another process. An `answer` becomes the result of the waiting call; `approved=True` executes it (the `KillSwitch` runs first). Answers pass through `on_user_input` hooks.

## Rejected

- Blocking `ApprovalGateway` callback: holds workers, loses state.
- Questions as plain assistant text: breaks the tool-call protocol.

## Consequences

Applications need a UI or queue for paused runs. Tested in `tests/test_resume.py` and `tests/test_edge_cases.py`.
