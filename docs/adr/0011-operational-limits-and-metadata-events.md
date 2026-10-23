# 0011 – Metadata-Only Events, Budgets, Kill Switch

- **Status:** Accepted (maintainer, 2026-10-18). Amended by [0026](0026-token-estimate-and-context-compaction.md): token estimate calibrated by the provider, deterministic compaction
- **Date:** 2026-10-16
- **Affects:** `ai_agent_harness/core/events.py`, `ai_agent_harness/tools/executor.py`, `ai_agent_harness/runtime/runner.py`, `ai_agent_harness/policy.py`, `ai_agent_harness/guardrails.py`

## Context

Events went to central logs while carrying raw hook reasons and model strings, which can hold account numbers, mail bodies, or secrets. History grew without limit and failed late with an opaque provider error. Operators could not halt runs across workers.

## Decision

1. **Events carry metadata only:** status, tool name, `Risk`, counters, the blocking stage (`source`), and for retry/failover only provider ID, attempt, delay, retryable flag. Never reasons, arguments, outputs, message text, or provider errors. `Event.data` allows scalars and one level of numeric counters; strings are cut at 200 characters. Invalid payloads are dropped and logged by field name and error type only. Readable reasons live in `RunState` (`status_reason`, `messages`), which can be stored encrypted.
2. **Context budget:** `Limits.max_context_chars` (default 400,000), checked before every model call against `ModelRequest.estimated_chars` (system, text, thinking incl. provider data, tool calls and results, tool schemas, output schema; no image data). Over budget ends the run as `limit_reached` before any token is spent.
3. **Input limit:** `Limits.max_input_chars` (default 50,000) for `user_input` and `answer`; longer input raises `ValueError` before state changes.
4. **Kill switch:** `KillSwitch(check)` blocks on user input, before each model call, and before each tool call, including approved actions on resume (the approval is then consumed). `check` may be sync or async; an exception or a non-`bool` blocks.

## Rejected

- Regex redaction in events: never complete; leaving content out is safe by construction.
- Vendor tokenizers: provider-specific and heavy.
- Automatic history compaction in core: changes what the model remembers. (ADR 0026 allows one deterministic, opt-in variant that only clears old tool results from the model's view.)
- `SIGTERM` as kill switch: no coordination across workers or tenants.

## Consequences

To see why a hook blocked, read `RunState.status_reason`. The size estimate is rough (`CHARS_PER_TOKEN` characters per token, calibrated by the provider's reported tokens, ADR 0026). Tracebacks logged with `log.exception` may contain local values: protect application logs. Tested in `tests/test_failures.py`, `tests/test_budgets_and_killswitch.py`, `tests/test_stress.py`.
