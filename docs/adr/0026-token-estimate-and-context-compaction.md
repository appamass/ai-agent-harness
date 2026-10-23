# 0026 – Token Estimate and Context Compaction by Clearing Old Tool Results

- **Status:** Proposed
- **Date:** 2026-10-23
- **Affects:** `core/types.py`, `policy.py`, `context.py`, `runtime/runner.py`, `observability.py`

## Context

Long runs end as `limit_reached` because the context budget counts only characters. A character count says little about tokens, and every provider already reports the real input tokens after each call. Amends [ADR 0011](0011-operational-limits-and-metadata-events.md), which rejected a tokenizer and compaction in core.

## Decision

1. **Calibrated estimate, no tokenizer.** After each call the runner keeps `(provider_id, chars, prompt_tokens)`: the size of the request in characters and the prompt tokens the provider reported (`Usage.prompt_tokens`: input plus cache reads and writes). The next request is estimated as `prompt_tokens + (chars_now - chars_then) / CHARS_PER_TOKEN` (`context.estimate_tokens`). Without a measurement the pair is `(0, 0)` and the formula is a plain size estimate. There is one code path. A response from another provider than before (failover), or one without reported tokens, resets the measurement to `(0, 0)`.
2. **`CHARS_PER_TOKEN = 3`** is a named constant in `core/types.py`. It is deliberately low, so the estimate errs on the high side.
3. **`Limits.max_context_tokens: int | None = None`** is the last field of `Limits`. `None` means no limit, so `Limits()` and `Limits(max_steps=...)` work as before. `max_context_chars` stays, and both limits apply. Over a limit, the run ends as `limit_reached` before the provider is called.
4. **Each size is computed once per step.** `context.ContextMeter` keeps the size of every message and of the tool specs. A message that was measured before (the same object) is not measured again. The one number feeds the limit check, the calibration, and the event. `ModelRequest.estimated_chars` stays for users without a runner and returns the same value.
5. **Events.** `MODEL_REQUEST` carries two counters, `context_chars` and `context_tokens_est`. They appear as `harness.request.context_chars` and `harness.request.context_tokens_est` in OpenTelemetry.
6. **No flat charge for images.** It would make `max_context_chars` fire earlier for existing runs with images. The reported prompt tokens include images anyway.

## Rejected

- Vendor tokenizers: provider-specific and heavy (see ADR 0011).
- A second estimation path for the first call: one formula with `(0, 0)` is easier to test.
- Replacing `max_context_chars`: existing configurations would change meaning.

## Consequences

The estimate is only as good as the last measurement. After a failover it is a size-based guess again for one call. `tests/test_budgets_and_killswitch.py` covers the formula, the reset, the limits, and the event counters.
