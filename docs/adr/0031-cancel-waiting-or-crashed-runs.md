# 0031 – Cancel a Waiting or Crashed Run

- **Status:** Accepted (maintainer, 2026-10-28)
- **Date:** 2026-10-28
- **Affects:** `runtime/runner.py`, `runtime/delegation.py`, public API (`cancel`)

## Context

A run that waits for a human, or that a crash left `running`, could only go on with `resume`, which calls the model again. There
was no way to end it for good. Such runs stay in the store forever (`purge_finished` deletes only finished runs) and show up in
approval screens. Sub-agents (ADR 0030) made this common: a sub-run that waits can never be answered, because nobody holds it.

## Decision

`cancel(agent, state, *, recover=False, reason="Run was cancelled")` ends the run as `cancelled`:

1. **Nothing runs:** no model call, no tool, no `before_tool` hook. Only `on_run_end` hooks run, as for every finished run.
2. **Honest answers for open calls:** finished results of the current batch stay. In a waiting run every other open call is
   answered `Not executed: run was cancelled.`; after a crash they are answered with the same "outcome unknown" text as
   `recover` (Invariant 5). The history stays valid (Invariant 2).
3. **Same rules as `resume`:** the state must belong to the agent, have the current schema, and pass `check_protocol`. A waiting
   run can be cancelled directly; a `running` run only with `recover=True`, because it may still be running elsewhere. A finished
   run raises `InvalidStateError`. No `pending_id` is needed: nothing is approved or run, so ADR 0014 is not touched.
4. **One owner:** with a checkpoint store, the save uses the optimistic lock (ADR 0006). A worker that resumes or cancels another
   copy of the same revision afterwards fails with `ConcurrencyError`, so an approval can never run after a cancel. Without a store
   there is no lock, and `cancel` must never get the state of a run that is still executing (stop that run's task instead).
5. **Events:** one `RUN_FINISHED` with status only; the reason stays in `status_reason` (ADR 0011).
6. **Sub-agents:** a sub-run that ends waiting is cancelled at once by the sub-agent tool. A sub-run that a crash of the caller
   interrupted is surely dead once it is older than the tool's `timeout_s`; the recovery job cancels it with `recover=True`
   (recipe in `docs/guide.md`, run repeatedly). After that, `purge_finished` removes it. The tool shields the cancel, so its own
   timeout cannot stop the final save halfway.

## Rejected

- Deleting waiting runs: loses the audit trail and the finished results.
- Ending a waiting run through `resume(approved=False)`: the model runs again and may pause again.
- Cancelling sub-runs automatically on the caller's recovery: the runner would need to know about sub-agents (ADR 0019 bricks).

## Consequences

Every run can be brought to a final status without running anything. Enforced by `tests/test_cancel.py` and
`tests/test_sub_agents.py`; listed as guarantee 22 in `docs/guarantees.md`.
