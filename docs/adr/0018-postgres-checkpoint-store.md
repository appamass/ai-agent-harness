# 0018 – PostgreSQL Checkpoint Store as an Optional Extra

- **Status:** Proposed
- **Date:** 2026-10-18
- **Affects:** `ai_agent_harness/persistence_postgres.py` (new public module), `ai_agent_harness/persistence.py` (shared helpers), `pyproject.toml` (extra `postgres`), CI, tests, docs

## Context

`JsonFileCheckpointStore` works on one POSIX host and stores plaintext. The main production case — an approval clicked in
another pod days later, a recovery job after a crash, several workers — needs a shared store with the same optimistic locking
(ADR 0006), and every user had to write it themselves. The core must stay free of drivers (invariant 7, ADR 0008).

## Decision

- Ship `PostgresCheckpointStore` as an **optional extra**: `pip install "ai-agent-harness[postgres]"` (`psycopg[binary,pool]>=3.2,<4`,
  async, typed). The module is not imported by default; without the driver it raises `ConfigError` with the install hint.
- **Table** `(tenant_id, run_id)` primary key, `revision`, `status`, `schema_version`, `state`, `updated_at`; index `(tenant_id, status)`.
  `state` is the exact JSON text of `RunState` (not `jsonb`: byte-exact, and loading is only through `restore_state`, ADR 0010).
- **Optimistic locking in one statement:** `INSERT ... ON CONFLICT (tenant_id, run_id) DO UPDATE ... WHERE revision = expected RETURNING revision`.
  A missing row is created (same as the other stores), a different revision returns no row and raises `ConcurrencyError`. No `SELECT` then `UPDATE`.
- **Cancellation** is handled by the same logic as the file store, extracted into `persistence.settle_write` and used by both, so in-memory
  and stored revisions stay in sync (ADR 0006). `check_ids` (formerly private) is shared too.
- **Connections:** by default one short-lived connection per operation (independent of the event loop, simple); `pool=` accepts an
  `AsyncConnectionPool` opened in the caller's loop. The store never creates global state.
- **Schema management:** `setup()` runs `CREATE TABLE IF NOT EXISTS` (index names carry a short hash of the table name, so long table names cannot collide); `schema_sql()` returns the DDL for teams with their own migration tool.
  Schema and table names are validated (`[a-z_][a-z0-9_]{0,62}`) and quoted; all values are bound parameters; IDs are validated before any SQL.
- **Operations helpers:** `list_runs(tenant_id, status=, limit=)` for approval queues and recovery jobs (`RunStatus.RUNNING` = a process died),
  `delete_run` and `delete_tenant` for retention and erasure.
- **No encryption in the store.** Checkpoints contain full conversations: use TLS, a least-privilege role, encryption at rest, or wrap the store
  (Use Case 01). Our error messages never contain the DSN or state; connection errors from the driver contain no password. Errors and logs of
  the database itself (a constraint added by an operator, `log_statement`) may quote row data or parameters: treat them like checkpoints.
- **A failed commit is visible, not silent:** if the connection drops while a commit is in flight, the row may be stored while the in-memory
  revision was not advanced; the next save raises `ConcurrencyError` and `recover=True` is the recovery path.
- **Tests** run against a real PostgreSQL (`HARNESS_TEST_POSTGRES_DSN`, a fresh schema per test): the shared store contract, concurrency (one winner),
  tampered rows, cancellation, pool mode, approval across two store instances, crash recovery, and one in four stress scenarios. Without the variable
  they are skipped; CI sets it with a `postgres:17` service (developed against 18).

## Alternatives

| Alternative | Why Rejected |
|---|---|
| SQLite store first | Good for one host, but Postgres is what several workers need; SQLite can follow (stdlib, no driver) |
| `asyncpg` | Fast, but untyped and lower-level; `psycopg` 3 is typed, has a pool, and supports sync and async |
| `SELECT ... FOR UPDATE` then `UPDATE` | Two round trips and a lock held across a network hop; one conditional upsert is atomic and simpler |
| `jsonb` for the state | Re-serializes (key order, number forms); we want the exact text and load it through `restore_state` anyway |
| Store creates and owns a global pool | Hidden state, event-loop coupling; the caller owns the pool |
| Built-in encryption | Key management is the caller's decision (KMS, per-tenant keys); documented as a wrapper pattern instead |

## Consequences

- Positive: a tested multi-worker store; the cross-process approval flow (ADR 0014) works with it; deletion helpers support retention policies.
- Negative / Cost: one more public module, optional dependency, and a database in CI. `setup()` racing on a brand-new database can fail once
  (known PostgreSQL behavior for concurrent `CREATE TABLE IF NOT EXISTS`): run it once at deploy time.
- Verification: `tests/test_postgres_store.py`, the `store` contract fixture in `tests/test_guardrails_and_persistence.py`, `tests/test_stress.py`,
  and a mutation check (removing the revision condition fails the concurrency and stale-copy tests).
