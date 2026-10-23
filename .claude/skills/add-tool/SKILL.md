---
name: add-tool
description: Build a new tool for the harness (BaseTool/SyncTool) with risk classification, Pydantic parameters, timeout, safe error messages, and tests. Use when giving an agent a new capability.
---

# Adding a New Tool

Reference implementation: `ai_agent_harness/tools/builtin.py` (`AskUser` = async `BaseTool`, `ReadFile` = `SyncTool`).

## 1. Classify Risk Honestly

| Risk Level | When to Use | Effect |
|---|---|---|
| `Risk.READ` | Read-only, zero side effects | Allowed at every autonomy level |
| `Risk.WRITE` | Modifies state in a reversible way | Requires approval in `suggest` and `act_with_approval` |
| `Risk.IRREVERSIBLE` | Sending, deleting, paying, publishing, merging | **Always** requires human approval |

When in doubt, pick the higher risk level. Downgrading a tool's risk (e.g. small refunds classified as `WRITE`)
is a business decision that requires an ADR and a hard guardrail.

## 2. Writing the Tool

- Define parameters as a Pydantic model with `model_config = ConfigDict(extra="forbid")`, flat structure,
  concrete types (no unconstrained `Any` or `dict[str, Any]`), and tight bounds (`min_length`, `max_length`, `pattern`, `gt`).
- `name` must match `^[A-Za-z0-9_-]{1,64}$`; `description` explains **when** to use the tool and when not to.
- Blocking code (filesystem, `requests`, synchronous DB drivers) must inherit from `SyncTool` and implement `run_sync`.
  An `async def run` method must never block the event loop.
- Set a realistic `timeout_s`.
- Raise `ToolError("...")` for error messages intended for the model; any other exception is logged and
  reported to the model only by its class name. Never leak host paths, foreign tenant IDs, or secrets in error messages.
- Always scope tenant data by `state.tenant_id`; report foreign records as "not found".
- Do not store runtime dependencies (DB sessions, HTTP clients, user tokens) on the tool instance;
  read them via `state.deps_as(MyDeps)`. They are passed per `start`/`resume` via `deps=...` and never persisted.
- Make `WRITE`/`IRREVERSIBLE` tools idempotent (derive an idempotency key from intent, such as an order ID
  or a hash of `run_id` + arguments). After a crash, the harness reports "outcome unknown" — pairing a write
  tool with a read/status tool lets the model check whether the action already took effect.
- Need to ask the user a clarifying question? `return InputRequest(question="...")` pauses the run.

## 3. Mandatory Tests

In `tests/` using `FakeProvider` and helpers from `tests/conftest.py`:

- Happy path via `start(...)`, verifying the tool result in the message history.
- Invalid arguments → error returned to the model without echoing raw input values.
- Policy check: a write tool is not offered or executed in `read_only` mode.
- Timeout enforcement (for `SyncTool`, measure duration inside the run).
- Idempotency: a second identical call does not trigger a duplicate side effect.
- Tenant isolation whenever data is tenant-scoped.

## 4. Wrap-Up

Run the `verify-change` skill. If the tool is broadly useful across projects, export it in
`ai_agent_harness/tools/__init__.py` and `ai_agent_harness/__init__.py` and mention it in `docs/guide.md`.
