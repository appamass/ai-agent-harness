# 0014 – Bind `resume` to the Pending Call (`pending_id`)

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-18
- **Affects:** `resume()` signature (public API), `RunState.pending_id`, ADR 0005, ADR 0006

## Context

`resume(agent, state, approved=True)` did not say *which* call was approved. With two approvers this breaks the approval guarantee:

1. The run waits for approval of `pay(alice)`; both approvers see the dialog.
2. Approver 1 approves. The tool runs, the model asks for `pay(mallory)`, the run pauses again.
3. Approver 2 clicks "Approve" on the old dialog. The handler loads the latest state and resumes, so `pay(mallory)` runs unseen.

The optimistic lock (ADR 0006) does not catch this: approver 2 loads a fresh state. Answers to questions have the same gap.

## Decision

- `RunState.pending_id` is derived and read-only: the first 16 hex characters of `sha256("{tenant_id}|{run_id}|{created_at}|{index of last message}|{kind}|{call id}")`, or `None` when the run is not waiting. It is not a field, so nothing is persisted and the schema is unchanged. It is unique per pause: messages only grow, the waiting call is in the last assistant message, and `kind` separates an approval from a later question about the same call.
- `resume(..., pending_id=...)` is **required** while the run waits and must equal `state.pending_id`; otherwise `InvalidStateError` before any change. For other statuses it must be `None`.
- UIs show `pending_id` with `status_reason` and send it back with the click. The value must come from the request the human saw: a handler that passes `loaded.pending_id` brings the bug back.
- The formula is part of the contract. Changing it invalidates open dialogs (fails closed); a test pins the value for the v1 fixture.

## Rejected

- Optional `pending_id`: not fail-closed.
- Binding to the call ID alone: providers and `FakeProvider` reuse IDs like `call_1`.
- A random stored `approval_id`: needs a schema bump for no extra safety.
- Hashing the arguments: no protection against someone who can write the store.

## Consequences

Breaking for every `resume` caller of a waiting run (0.4.0). Someone who can write the checkpoint store can still change the pending call; that needs an access-controlled or encrypted store. Tested in `tests/test_resume.py` and the stress test.
