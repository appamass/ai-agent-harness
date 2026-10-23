# 0026 – Token Estimate and Context Compaction by Clearing Old Tool Results

- **Status:** Proposed
- **Date:** 2026-10-23
- **Affects:** `core/types.py`, `core/state.py`, `policy.py`, `context.py`, `runtime/runner.py`, `runtime/agent.py`, `observability.py`

## Context

Long runs end as `limit_reached` because the context budget counts only characters. A character count says little about tokens, and every provider already reports the real input tokens after each call. Furthermore, long multi-step runs exhaust context budgets quickly without compaction. Amends [ADR 0011](0011-operational-limits-and-metadata-events.md), which rejected a tokenizer and automatic history compaction in core.

## Decision

1. **Calibrated estimate, no tokenizer.** After each call the runner keeps `(provider_id, chars, prompt_tokens)`: the size of the request in characters and the prompt tokens the provider reported (`Usage.prompt_tokens`: input plus cache reads and writes). The next request is estimated as `prompt_tokens + (chars_now - chars_then) / CHARS_PER_TOKEN` (`context.estimate_tokens`). Without a measurement the pair is `(0, 0)` and the formula is a plain size estimate. There is one code path. A response from another provider than before (failover), or one without reported tokens, resets the measurement to `(0, 0)`.
2. **`CHARS_PER_TOKEN = 3`** is a named constant in `core/types.py`. It is deliberately low, so the estimate errs on the high side.
3. **`Limits.max_context_tokens: int | None = None`** is the last field of `Limits`. `None` means no limit, so `Limits()` and `Limits(max_steps=...)` work as before. `max_context_chars` stays, and both limits apply. Over a limit, the run ends as `limit_reached` before the provider is called.
4. **Each size is computed once per step.** `context.ContextMeter` keeps the size of every message and of the tool specs. A message that was measured before (the same object) is not measured again. The one number feeds the limit check, the calibration, and the event. `ModelRequest.estimated_chars` stays for users without a runner and returns the same value.
5. **Deterministic context compaction (opt-in):** `Agent.compact_context: bool = False`. The stored message history (`state.messages`) is **never shortened**. Only the view built for the model replaces the content of old `ToolResult` blocks before `state.context_floor` with `[Result removed to save space (N chars). Call the tool again if you need it.]`. Alternating roles and protocol checks remain intact.
6. **Thresholds and stepping:** Compaction triggers when the view exceeds 80% (`COMPACT_TRIGGER`) of the budget and targets 50% (`COMPACT_TARGET`). The latest 8 messages (4 rounds, `KEEP_RECENT_MESSAGES`) are never cleared. Results shorter than 200 characters (`MIN_CLEARED_CHARS`) are kept. The floor rises in discrete steps to preserve prompt caching.
7. **Schema v5:** `RunState.context_floor: int = Field(default=0, ge=0)` and measurement fields `measured_provider_id`, `measured_chars`, `measured_prompt_tokens`. Migration v4 → v5 documents the change. `check_protocol` validates `context_floor <= len(messages)`. Raising the floor drops the token measurement. A checkpoint is saved before the model call only when the floor changes.
8. **Events:** `MODEL_REQUEST` carries `context_chars` and `context_tokens_est`. `CONTEXT_COMPACTED` carries `floor`, `cleared_results`, `chars_saved`. Unhandled events are logged as span events by `OpenTelemetryListener`.

## Rejected

- Vendor tokenizers: provider-specific and heavy (see ADR 0011).
- LLM summarization for compaction: non-deterministic, expensive, loses auditability, and can hallucinate details.
- Deleting messages from `RunState.messages`: violates audit trail and protocol history.
- Creeping floor adjustments: breaks prompt caching on every step.

## Consequences

The model "forgets" old tool outputs (it must re-invoke tools if details are needed). Checkpoints do not shrink because the full history is preserved; delta checkpoints are future work. Tested in `tests/test_budgets_and_killswitch.py`, `tests/test_compaction.py`, `tests/test_schema.py`, and `tests/test_stress.py`.
