# 0011 – Metadata-Only Events, Context and Input Budgets, Kill Switch

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-16
- **Affects:** `ai_agent_harness/core/events.py`, `ai_agent_harness/tools/executor.py`, `ai_agent_harness/runtime/runner.py`,
  `ai_agent_harness/policy.py`, `ai_agent_harness/guardrails.py`

## Context

Before running against a live model, three operational boundaries were missing:

1. Events were shipped to central logs and monitoring systems while carrying raw hook reasons or model
   strings — which could contain bank account numbers, email bodies, or secrets.
2. Conversation history grew without bound. A long run would eventually hit the provider's context-window
   limit and fail with an opaque provider error instead of a clear budget status. Similarly, a single
   multi-megabyte user input could blow out the context immediately.
3. Operators had no central mechanism to immediately halt runs across workers when something went wrong.

## Decision

1. **Events contain metadata only:** `RunStatus`, tool name, `Risk`, token/step counters, and the stage
   that blocked an action (`source`) — never hook reasons, tool arguments, tool outputs, message text,
   or provider error messages (retry/failover events record only provider ID, attempt number, delay, and
   whether the error is retryable). `Event.data` accepts only scalar values (`str`, `int`, `float`, `bool`,
   `None`) and one level of numeric counter dicts; strings are truncated to 200 characters. Invalid event
   payloads are dropped and logged by field name and error type only (no values, no traceback) so an event
   can never crash a run. Human-readable reasons live exclusively in `RunState` (`status_reason`, `messages`),
   which can be stored encrypted.
2. **Context budget:** `Limits.max_context_chars` (default `400_000`). Before every model call, the runner
   checks `ModelRequest.estimated_chars` (system prompt, message text, thinking blocks including provider
   metadata, tool calls, tool results, and tool schemas; excludes binary image data). Exceeding the budget
   ends the run cleanly as `limit_reached` before calling the provider — incurring zero token cost.
3. **Input length limit:** `Limits.max_input_chars` (default `50_000`) for `user_input` and `answer`.
   Oversized inputs raise `ValueError` at the entry point before mutating `RunState`.
4. **Kill switch:** Guardrail `KillSwitch(check)`. Blocks on user input, before every model call, and
   before every tool call (including previously approved actions when resumed). `check` may be sync or
   async and is fail-closed: raising an exception or returning a non-`bool` blocks the run. If triggered
   when resuming an approved action, that approval is consumed.

## Alternatives

| Alternative | Why Rejected |
|---|---|
| Regex redaction of secrets in events | Pattern matching is never 100% complete; omitting content altogether is structurally safe |
| Vendor tokenizers instead of character estimation | Provider-specific and adds heavy dependencies to the core |
| Automatic history compaction in core | Alters what the model remembers; kept as an explicit future opt-in (`docs/next-steps.md`) |
| OS process signals (`SIGTERM`) as kill switch | Does not coordinate across multiple workers or target a single tenant |

## Consequences

- Positive: Event logs can be shipped to central observability platforms without privacy/PII risk; oversized runs stop cleanly before incurring API costs; operators can halt all runs or specific tenants immediately.
- Negative: Debugging why a hook blocked requires inspecting `RunState.status_reason` rather than event payloads. Character estimation is approximate (≈ 4 characters per token) and should be sized for the smallest model in the router.
- Operational boundary: Unexpected internal exception tracebacks (`log.exception`) may still contain local variable values — protect application error logs accordingly.
- Verification: `tests/test_failures.py` (no content in events, including router events; invalid event payloads), `tests/test_budgets_and_killswitch.py`, and `tests/test_stress.py`.
