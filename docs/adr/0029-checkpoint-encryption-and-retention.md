# 0029 – Optional Checkpoint Encryption and Retention Purge

- **Status:** Accepted (maintainer, 2026-10-23)
- **Date:** 2026-10-23
- **Affects:** `encryption.py`, `persistence.py`, `persistence_sql.py`, `persistence_sqlite.py`, `persistence_postgres.py`, `docs/known-limitations.md`, `docs/guide.md`, `docs/production-checklist.md`

## Context

Checkpoints store full agent histories, including user messages, model reasoning blocks, and tool arguments/results. In multi-tenant systems or production deployments subject to data privacy regulations (e.g. GDPR, HIPAA), storing checkpoints as plaintext JSON exposes sensitive data if storage volumes or database backups are compromised.

Additionally, long-running agent systems require bounded retention. Deleting rows ad-hoc via custom queries risks accidentally purging runs that are actively executing or awaiting human approval (breaking recovery guarantees), and computing retention cutoffs on worker nodes introduces risks from worker clock skew.

## Decision

1. **Envelope format for all stores:**
   - **Prefix:** `enc:v1:` allows $O(1)$ identification via `startswith("enc:v1:")` without parsing.
   - **Cleartext header:** JSON object carrying `kid`, `tenant_id`, `run_id`, `revision`, and `nonce` (12-byte base64-encoded nonce).
   - **Authenticated Additional Data (AAD):** The entire header prefixed with `enc:v1:` (`enc:v1:{"kid":...}`). A swapped tenant ID, altered revision, or tampered nonce fails AES-GCM tag verification.
   - **Ciphertext:** Base64-encoded ciphertext of the serialized `RunState` JSON.
   - **Confidential schema:** `schema_version` resides only inside the encrypted payload.
   - **Fail-closed:** Stores configured with a codec refuse unencrypted entries; stores without a codec refuse encrypted entries.

2. **Unified write and load path:**
   - All stores (`InMemoryCheckpointStore`, `JsonFileCheckpointStore`, `SqliteCheckpointStore`, `PostgresCheckpointStore`) write through `WriteRecord.from_state(state, codec)`.
   - All store loads delegate to `load_state(raw, tenant_id=..., run_id=..., codec=...)` in `persistence.py`: decrypt first, then run `restore_state` (Invariant 9).
   - `JsonFileCheckpointStore` verifies tenant ownership and expected revision from the unencrypted header line, eliminating separate wrapper formats.
   - Encryption and decryption run off the event loop via `asyncio.to_thread`. Payloads are not compressed prior to encryption to avoid length-leakage attacks.

3. **Key management:**
   - `AesGcmCodec(keys: Mapping[str, bytes], *, active: str)` accepts pre-shared 32-byte keys (AES-256-GCM). Key rotation is supported by specifying multiple key IDs with an `active` key used for new writes.
   - The harness never reads filesystem `.env` files or secret managers directly. `repr` and error messages never include secret key material or state text.

4. **Safe-by-construction SQL retention purge:**
   - `purge_finished(*, older_than: timedelta, tenant_id: str | None = None) -> int` on `SqliteCheckpointStore` and `PostgresCheckpointStore`.
   - Only deletes terminal runs (`done`, `refused`, `blocked`, `limit_reached`, `failed`, `cancelled`). Active (`running`) and waiting (`waiting_for_input`, `waiting_for_approval`) runs are never deleted.
   - Cutoffs are computed relative to database clock time (`now` minus seconds) to avoid worker clock skew.

## Consequences

- Full confidentiality and tamper detection for checkpoint storage across all backends.
- Migration recipe: Read using unencrypted store, save using encrypted store (`load` -> `save`).
- Documented limitation: Replay of an older, complete, unmodified row cannot be prevented by encryption alone.
- Storage cost: Base64 encoding adds ~33% overhead, and Postgres TOAST compression does not compress high-entropy ciphertext.
