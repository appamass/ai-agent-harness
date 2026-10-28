# 0030 – Sub-Agents as Read-Only Tools

- **Status:** Accepted (maintainer, 2026-10-28)
- **Date:** 2026-10-28
- **Affects:** `runtime/delegation.py` (new), `providers/router.py`, public API (`sub_agent`)

## Context

Delegating a sub-task to a specialist agent with its own context is a common pattern (`HARNESS_TAXONOMY.md`, section 6). Until v0.6 a
use case showed it as a hand-written tool; that document was removed, and every project had to find the safety rules again. Done
wrong, delegation breaks invariants: a sub-run that waits for a human is stranded (Invariant 1), a sub-run that writes under the
caller's approval acts on calls no human saw (Invariant 4), and a retried call can start the same work twice (Invariant 5).

## Decision

`sub_agent(specialist, *, name=None, description, timeout_s=None)` returns a `Risk.READ` tool built with `make_tool`. It lives in
`runtime/`, the only package that may call `start` (`test_dependencies_point_inward_only_runtime_knows_all_parts`).

1. **Least privilege, checked at build time (`ConfigError`):** the specialist must use `Autonomy.READ_ONLY`, so it can never write
   and never holds more rights than a read tool of the caller. `ask_user` must not be visible to it; this is checked again before
   every sub-run, because tools can be registered later. A blank task is an argument error for the model.
2. **Never stranded:** every sub-run status other than `done` becomes a `ToolError` that names only the status, never
   `status_reason`. A specialist tool that returns `InputRequest`, or a hook that requires approval, ends the sub-run as
   waiting; the tool then cancels it at once ([ADR 0031](0031-cancel-waiting-or-crashed-runs.md)), so every sub-run ends in a
   final status, and the caller sees an error and goes on.
3. **Deterministic sub-run ID:** `"sub-" + sha256(tenant_id, agent_name, run_id, created_at, step, len(batch_done), tool name)[:40]`.
   All parts are persisted, so a resumed or recovered caller names the same sub-run. Agent name and `created_at` tell apart callers
   that share a `run_id` (as `pending_id` does). The tool is not `parallel_safe`, so the position in the batch tells calls apart.
   With a checkpoint store on the specialist, a second start for the same call fails at the store instead of running the work again. The sub-run uses the caller's `tenant_id`.
4. **Time:** `timeout_s` must be larger than the specialist's `Limits.max_seconds`. The default adds `ModelRouter.worst_case_s()`
   (new) and the slowest specialist tool, because the time limit is only checked before each model call. This is an estimate, not
   a bound: a response with many slow tool calls takes longer. A caller timeout cancels the sub-run; the caller reports it like
   any timed-out tool. The caller's `KillSwitch` and `Limits` do not reach into the sub-run.
5. **Cycles and depth:** a `ContextVar` holds the name of the top-level agent and of every active specialist. Delegating to any of
   them, or deeper than `MAX_SUB_AGENT_DEPTH` (4) sub-runs, is refused as a `ToolError` before any model call.
6. **Output is data:** `untrusted_output=True`, so the answer reaches the caller framed by `as_untrusted`.
7. **Accounting stays separate:** the sub-run has its own `RunState` and emits on the specialist's `EventBus` (metadata only,
   Invariant 7). `deps` are not passed on.

## Rejected

- Specialists with write rights: one approval in the caller would cover writes the specialist chose (Invariant 4).
- Forwarding a sub-run's question or approval to the caller: needs nested pending calls in `RunState`, a schema change, for little gain.
- Adding the sub-run's tokens to the caller's `usage`: a tool would change the caller's state outside `RunState`'s own methods.
- A random sub-run ID: a retried call could not be recognized.

## Consequences

Delegation is one line and keeps the invariants. Specialists that must write stay separate runs started by your code. A sub-run
that a crash of the caller interrupted stays `running` until the recovery job cancels it (ADR 0031, recipe in `docs/guide.md`).
Enforced by `tests/test_sub_agents.py`; listed as guarantee 21 in `docs/guarantees.md`.
