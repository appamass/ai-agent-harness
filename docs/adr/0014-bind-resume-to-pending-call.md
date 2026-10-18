# 0014 – Bind `resume` to the Pending Call (`pending_id`)

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-18
- **Affects:** `resume()` signature (public API), `RunState.pending_id`, invariant 5, ADR 0005, ADR 0006

## Context

`resume(agent, state, approved=True)` did not say *which* call the human approved. The runner executed
whatever `state.pending` held at that moment. With two approvers (or a mobile and a desktop dialog) this
broke the meaning of invariant 5:

1. The run waits for approval of `pay(alice)`. Both approvers see that dialog.
2. Approver 1 approves. The tool runs, the model asks for `pay(mallory)`, and the run pauses again.
3. Approver 2 clicks "Approve" on the old dialog. The UI handler loads the latest checkpoint and calls
   `resume(approved=True)` — so `pay(mallory)` runs although no human ever saw it.

The optimistic lock (ADR 0006) does not catch this: approver 2 loads a fresh state with a fresh revision.
Answers to clarifying questions have the same gap ("yes" to an old question becomes the answer to a new one).

## Decision

- `RunState.pending_id` is a derived, read-only value that identifies the current pause:
  the first 16 hex characters of `sha256("{tenant_id}|{run_id}|{created_at}|{index of last message}|{kind}|{call id}")`, or `None`
  when the run is not waiting. It is not a field, so it is never persisted and the checkpoint schema is unchanged.
- It is unique per pause: tenant, run ID, and creation time separate runs (also two runs that reuse the
  same `run_id` after deletion); messages only grow; the waiting call is always in the last assistant message;
  call IDs are unique within one message; `kind` separates an approval from a later question of the same call.
- `resume(..., pending_id=...)` is **required** for `waiting_for_input` and `waiting_for_approval` and must equal
  `state.pending_id`. A missing or different value raises `InvalidStateError` before any state is modified.
  For all other statuses `pending_id` must be `None`.
- UIs show `state.pending_id` together with `state.status_reason` and send it back with the button click.
  The value must come from the request the human saw. A handler that loads the state and passes
  `loaded.pending_id` reintroduces the bug; the docs and use cases show the correct pattern.
- The formula is part of the contract: changing it invalidates every open dialog at upgrade (fails closed).
  A test pins the value for the v1 fixture.

## Alternatives

| Alternative | Why Rejected |
|---|---|
| Optional `pending_id` (checked only when passed) | Not fail-closed: every integration that forgets it keeps the gap; invariant 5 must hold by default |
| Bind only on the tool call ID | Some providers and the test `FakeProvider` reuse IDs like `call_1` across turns; the stale approval would still match |
| Random `approval_id` field stored in `PendingCall` | Needs a schema bump and migration (ADR 0010) for no extra safety over the derived value |
| Include a hash of the arguments | Does not add protection against a writer of the checkpoint store (they can recompute it); the derived position is already unique per pause |

## Consequences

- Positive: an approval or answer applies only to the request a human actually saw — also across processes.
- Negative / Cost: breaking change for every caller of `resume` on a waiting run (version 0.4.0).
  It does not protect against someone who can write the checkpoint store (they can change the pending call
  and see the new `pending_id`); that still needs an access-controlled or encrypted store.
- Verification: `tests/test_resume.py` (stale approval regression, missing/wrong id for approval and input,
  id on a non-waiting run, approval → question of the same call, round trip), stress test, executable use cases.
