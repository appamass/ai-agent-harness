# 0010 – Versioned Checkpoints with Forward Migrations

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-16
- **Affects:** `ai_agent_harness/core/state.py`, `ai_agent_harness/persistence.py`, `tests/fixtures/`
- **Exception (maintainer, 2026-10-18):** before the first public release, the free-text values and the
  example IDs in both fixtures were translated from German to English (`kunde-1` → `customer-1`,
  `anbieter-a` → `provider-a`, prompts and thinking text). Structure, field names, schema versions,
  revisions, call IDs, and timestamps are unchanged, so the fixtures still test the same old formats.
  The rule "never modify existing fixtures" applies again from now on.

## Context

Runs can wait hours or days for human approval or input. During that window, the harness may be upgraded.
If the serialized shape of `RunState` changes, new code would either reject older checkpoints due to
`extra="forbid"` — stranding waiting runs — or, on a rollback, old code might partially parse and overwrite
a newer checkpoint.

## Decision

1. `RunState.schema_version` (currently `SCHEMA_VERSION = 2`, with migration `1 -> 2`) is persisted in
   every checkpoint. Payloads missing the field default to version `1`.
2. Checkpoints are loaded **exclusively** via `restore_state(raw)`: inspect `schema_version` → apply
   `MIGRATIONS[n]` (`n -> n + 1`) sequentially → validate via Pydantic → run `check_protocol()`.
   Custom stores (database, encrypted) must also call `restore_state(raw, tenant_id=…, run_id=…)` with
   the requested key: a checkpoint stored under a mismatched tenant or run ID (copied file, swapped DB row)
   is rejected with `InvalidStateError`. Any migration that fails to return a JSON object raises
   `SchemaVersionError`.
3. Checkpoints with a `schema_version` newer than `SCHEMA_VERSION` raise `SchemaVersionError` immediately —
   no guessing, no overwriting. Additionally, `resume` rejects any `RunState` whose `schema_version` does
   not match `SCHEMA_VERSION`, even if a custom store bypassed `restore_state`.
4. **Rule for every schema change:** Increment `SCHEMA_VERSION`, add a migration from the previous
   version in `MIGRATIONS`, and commit a fixture of the old format in `tests/fixtures/`. Existing fixtures
   are never modified; `tests/test_schema.py` must be able to load, migrate, and resume them.

## Alternatives

| Alternative | Why Rejected |
|---|---|
| `extra="ignore"` instead of versioning | Silently drops fields; a rollback would permanently lose data |
| Drain all waiting runs before every deploy | Approvals can take days; operationally impractical |
| Per-store migration logic | Every store implementation would have to duplicate the same migration chain |

## Consequences

- Positive: Upgrading the harness never strands paused runs; rollbacks fail loudly instead of corrupting state.
- Negative: Every schema change requires writing a forward migration and committing an immutable fixture.
- Verification: `tests/test_schema.py` (loading and resuming v1 and v2 fixtures from disk, future version rejection, multi-step migration chains, broken migrations, `resume` version check) and `tests/test_guardrails_and_persistence.py` (foreign key rejection).
