# 0018 – PostgreSQL Checkpoint Store (Optional Extra)

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-18
- **Affects:** `ai_agent_harness/persistence_postgres.py`, `persistence.py` and `persistence_sql.py` (shared helpers), `pyproject.toml` (extra `postgres`), CI, tests

## Context

`JsonFileCheckpointStore` fits one POSIX host and stores plaintext. An approval clicked in another pod days later, a recovery job, or several workers need a shared store with the same optimistic locking (ADR 0006). The core stays free of drivers (ADR 0008).

## Decision

- `PostgresCheckpointStore` is an **optional extra** (`pip install "ai-agent-harness[postgres]"`, `psycopg[binary,pool]>=3.2,<4`, async). Without the driver it raises `ConfigError` with the install hint.
- **Table:** primary key `(tenant_id, run_id)`, `revision`, `status`, `schema_version`, `state`, `updated_at`; index `(tenant_id, status)`. `state` is the exact JSON text (not `jsonb`), loaded only through `restore_state` (ADR 0010).
- **Locking in one statement:** `INSERT … ON CONFLICT (tenant_id, run_id) DO UPDATE … WHERE revision = expected RETURNING revision`. A missing row is created; a different revision returns no row and raises `ConcurrencyError`.
- **Shared with the other SQL store:** the statements, `WriteRecord`, and identifier checks (`persistence_sql.py`); cancellation handling (`settle_write`), id and limit checks (`persistence.py`).
- **Connections:** one short-lived connection per operation by default; `pool=` takes an `AsyncConnectionPool` opened in the caller's loop. The store keeps no global state.
- **Schema:** `setup()` runs `CREATE TABLE IF NOT EXISTS` (index names carry a short hash of the table name); `schema_sql()` returns the DDL for migration tools. Schema and table names are validated (`[a-z_][a-z0-9_]{0,62}`) and quoted; values are bound parameters.
- **Operations:** `list_runs(tenant_id, status=, limit=)` for approval queues and recovery jobs (`RunStatus.RUNNING` means a process died), `delete_run`, `delete_tenant`.
- **No encryption in the store.** Use TLS, a least-privilege role, encryption at rest, or wrap the store with per-tenant keys. Our errors never contain the DSN or state; database logs may quote row data, so treat them like checkpoints.
- A commit lost to a dropped connection is visible, not silent: the next save raises `ConcurrencyError` and `recover=True` is the way back.
- **Tests** run against a real PostgreSQL (`HARNESS_TEST_POSTGRES_DSN`, fresh schema per test; skipped without it; CI uses a `postgres:17` service): store contract, one winner under concurrency, tampered rows, cancellation, pool mode, approval across two instances, crash recovery, stress.

## Rejected

- SQLite first: right for one host, not for several workers (it followed in ADR 0020).
- `asyncpg`: untyped and lower-level.
- `SELECT … FOR UPDATE` then `UPDATE`: two round trips and a lock across the network.
- `jsonb`: re-serializes; we want the exact text.
- A store-owned global pool: hidden state; the caller owns it.
- Built-in encryption: key management is the caller's decision.

## Consequences

`setup()` racing on a brand-new database can fail once (PostgreSQL behavior); run it once at deploy time. Tested in `tests/test_postgres_store.py`, the store contract in `tests/test_guardrails_and_persistence.py`, `tests/test_stress.py`, and a mutation check (removing the revision condition fails the concurrency tests).
