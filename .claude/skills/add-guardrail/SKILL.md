---
name: add-guardrail
description: Build a new guardrail as a Hook (user input, model response, tool call, tool result) including bypass-resistance tests. Use for allowlists, limits, data-exfiltration protection, injection checks, and similar policies.
---

# Adding a New Guardrail

Reference examples: `ai_agent_harness/guardrails.py` (`ProtectedPaths`, `KillSwitch`) and the guardrails in
`docs/use-cases/` (`RecipientAllowlist`, `ExfiltrationGuard`, `PaymentLimits`, `TenantRefundPolicy`).

## 1. Choose the Right Hook Point

| Hook Method | Allowed Decisions | Typical Check |
|---|---|---|
| `on_user_input` | `continue`, `block` | Secrets or prohibited content in user input |
| `before_model` | `continue`, `block` | Custom budgets, kill switch |
| `after_model` | `continue`, `block`, `retry` | Output format, citations/grounding, deterministic recalculation |
| `before_tool` | `continue`, `block`, `require_approval` | Allowlists, amount thresholds, protected paths |
| `after_tool` | `continue`, `block` | Secrets or cross-tenant data in tool output |
| `on_run_end` | – | Metrics, background memory extraction (errors are logged only) |

Any decision not allowed at that hook point, or any uncaught exception, is treated as `block`.
This is intentional (fail-closed) — never wrap hook logic in a `try/except` that silently returns `continue`.

## 2. Implementation Guidelines

- Inherit from `Hook` and override **only** the method you need.
- In `before_tool`, inspect `invocation.args`: these are the already **validated** Pydantic arguments,
  not raw model text. Use `invocation.risk` and `invocation.tool_name` to scope when the guardrail applies.
- Return `Decision.block(...)` when an action is never allowed. Use `Decision.require_approval(...)`
  only when a human can make a meaningful decision — excessive approval prompts train users to click "approve" blindly.
- Keep reasons concise and free of sensitive values; they are sent to the model and shown in approval previews.
- Do not store mutable counters or daily totals on the `Hook` instance across multi-worker processes —
  use shared storage or pass per-run state via `deps`.

## 3. Bypass-Resistance Tests (Mandatory)

A guardrail is only as strong as its adversarial tests. Cover at minimum:

- Case variations, whitespace, Unicode normalization, and alternative separators.
- Values nested inside lists, dictionaries, and non-obvious field names.
- Boundary values: exactly at, just below, and just above the limit.
- Zero false positives on benign inputs (otherwise operators will disable the guardrail).
- End-to-end integration via `start(...)`: a blocked call is reported to the model as an error and the run stays consistent.

## 4. Wrap-Up

Run the `verify-change` skill. Place broadly reusable guardrails in `ai_agent_harness/guardrails.py` and
export them in `ai_agent_harness/__init__.py`; keep domain-specific guardrails in project code.
