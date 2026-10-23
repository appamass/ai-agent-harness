# 0020 – SQLite Checkpoint Store

- **Status:** Accepted (maintainer, 2026-10-22)
- **Date:** 2026-10-22
- **Affects:** `ai_agent_harness/persistence_sqlite.py`, `persistence.py` and `persistence_sql.py` (shared helpers), `ai_agent_harness/__init__.py`, `tests/test_sqlite_store.py`

## Context

Stores so far: in memory (tests), file (POSIX only, needs `fcntl`), PostgreSQL (needs a server and an extra). Embedded tools, desktop apps, single-host services, and Windows users need a persistent, crash-safe store with optimistic locking and no infrastructure.

## Decision

- `SqliteCheckpointStore` on the standard library `sqlite3` only (no new dependency), exported from `ai_agent_harness`. It is meant for Windows, Linux, and macOS (CI runs on Linux only; Windows is untested) and creates parent directories.
- Same table and index as the PostgreSQL store (`(tenant_id, run_id)` key, `revision`, `status`, `schema_version`, `state`, `updated_at`; index `(tenant_id, status)`) the same statements and `WriteRecord` (`persistence_sql.py`), and the same `settle_write`, id and limit checks (`persistence.py`).
- **Locking in one statement:** `INSERT … ON CONFLICT (tenant_id, run_id) DO UPDATE … WHERE revision = ? RETURNING revision`; no row back raises `ConcurrencyError`.
- WAL mode, `synchronous = NORMAL`, `busy_timeout = 5000`, applied on first use (a borrowed connection gets them at construction). An internal `threading.Lock` plus `asyncio.to_thread`; `settle_write` keeps revisions in sync on cancellation (ADR 0006).
- `list_runs`, `delete_run`, `delete_tenant` like the PostgreSQL store.

## Rejected

- `aiosqlite`: an extra dependency; `to_thread` is enough.
- PostgreSQL for everything: heavy for single-host use.
- `msvcrt` locks in the file store: different semantics around `os.replace`; SQLite already solves cross-platform locking.

## Consequences

One more store to maintain. Tested in `tests/test_sqlite_store.py`, the store contract in `tests/test_guardrails_and_persistence.py`, and `tests/test_stress.py`.
