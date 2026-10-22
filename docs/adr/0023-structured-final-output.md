# 0023 – Structured Final Output (`output_model`)

- **Status:** Accepted (maintainer, 2026-10-22)
- **Date:** 2026-10-22
- **Affects:** `core/types.py`, `core/state.py`, `providers/base.py`, `providers/openai.py`, `providers/router.py`, `runtime/runner.py`, `runtime/agent.py`

## Context

Pipelines need a validated, typed result instead of free text. It must keep the protocol, crash, and gate guarantees.

## Decision

- `start(agent, text, output_model=Model)` and `resume(..., output_model=Model)`; the result is `state.output` (JSON-safe dict) and `state.output_as(Model)`.
- **Tiers.** The runner uses what **all** providers of the router support, so failover stays valid: native schema (`Capabilities.structured_output`, sent as `ModelRequest.output_schema`) if every provider has it; otherwise an instructed `final_output` tool if every provider has `tools`; otherwise a prompt instruction with JSON text parsing. "Instructed" means no forced `tool_choice`: a model that answers in text is validated through the text path.
- **`final_output` is a real tool** (Risk.READ, input model = `output_model`) that runs through `ToolExecutor`: policy, hooks, events, write-ahead, and argument validation including `arguments_error` apply (invariant 3). Its output counts only if the result passed the whole gate; if an `after_tool` hook withholds it, the output is discarded and one retry is counted. The name is reserved (`ConfigError` in `Agent`); without a requested `output_model` it is an ordinary unknown tool and cannot end a run. The tool skips the concrete-type check of `validate_definition` (`strict_schema = False`): your own model may use free-form fields (`dict[str, Any]`, `Any`) in every tier.
- **Retries:** invalid output feeds back to the model; it counts toward `max_consecutive_retries` (reset when a batch ends without an invalid `final_output`); exhaustion ends the run `failed`.
- **Fail-closed resume:** the schema is stored in the checkpoint, but only the Pydantic model validates. Resuming a run that has a schema requires the same `output_model` and a matching schema, otherwise `InvalidStateError`. There is no schema-less validation fallback. Output of an earlier turn is cleared on resume, and `RunState.set_status` clears it for every status except `done`.
- **Schema v3** (ADR 0010): fields `output_schema`, `output_model_name`, `output`; migration 2 → 3 sets `None`; the v1 and v2 fixtures stay loadable and `tests/test_schema.py` covers the 2 → 3 migration.
- The router skips providers that lack `structured_output` when `output_schema` is set.

## Rejected

- Forcing the tool via `tool_choice`: needs support in every adapter.
- Validating against the stored schema alone: Pydantic-only core cannot, and "validate against itself" accepts anything.
- A special `final_output` path outside the gate: bypasses policy and hooks.

## Consequences

Tested in `tests/test_structured_output.py` (tiers, gate, hooks, `after_tool`, resume, reserved name, retries) and `tests/test_schema.py`.
