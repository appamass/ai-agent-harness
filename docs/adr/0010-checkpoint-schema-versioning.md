# 0010 – Versioned Checkpoints with Forward Migrations

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-16
- **Affects:** `ai_agent_harness/core/state.py`, `ai_agent_harness/persistence.py`, `tests/fixtures/`
- **Exception (maintainer, 2026-10-18):** before the first public release, free text and example IDs in the two then-existing fixtures were translated from German to English. Structure, field names, versions, revisions, call IDs, and timestamps are unchanged. Existing fixtures are never modified otherwise.

## Context

Runs wait hours or days. If the shape of `RunState` changes meanwhile, new code would reject old checkpoints (`extra="forbid"`) and strand waiting runs, and old code after a rollback could half-parse and overwrite a newer one.

## Decision

1. `RunState.schema_version` (`SCHEMA_VERSION`) is stored in every checkpoint. A missing field means version 1.
2. Checkpoints load **only** through `restore_state(raw, tenant_id=…, run_id=…)`: read the version, apply `MIGRATIONS[n]` step by step, validate with Pydantic, run `check_protocol()`. A checkpoint under the wrong tenant or run ID is rejected (`InvalidStateError`); a migration that does not return an object raises `SchemaVersionError`. Custom stores must use it too.
3. A newer `schema_version` raises `SchemaVersionError` at once. `resume` rejects any state whose version differs, even from a store that skipped `restore_state`.
4. Every schema change: bump `SCHEMA_VERSION`, add the migration, commit a fixture of the old format.

## Rejected

- `extra="ignore"`: silently drops fields; a rollback loses data.
- Draining waiting runs before each deploy: approvals take days.
- Migration logic per store: every store would repeat it.

## Consequences

Each schema change costs a migration and a fixture. Tested in `tests/test_schema.py` and `tests/test_guardrails_and_persistence.py`.
