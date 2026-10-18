# Security Policy

## Architectural Security Guarantees

`ai-agent-harness` is designed around a deterministic security core that never relies on the language model to enforce safety boundaries:

1. **Irreversible actions always require human approval** (`Risk.IRREVERSIBLE`), regardless of the configured `Autonomy` level.
2. **Fail-closed guardrails:** Any `Hook` that raises an exception or returns an action not permitted at that hook point is treated as `Decision.block(...)`.
3. **Single tool execution gate:** Tools are only executed through `ToolExecutor` (`ai_agent_harness/tools/executor.py`), which enforces argument validation, policy checks, `before_tool` hooks, approval checks, timeouts, output truncation, and `after_tool` hooks in order.
4. **Write-ahead checkpointing & optimistic locking:** State is persisted before every tool call, and `resume()` claims the run revision before executing any action so concurrent workers cannot execute an approved action twice.
5. **Metadata-only events (ADR 0011):** Events emitted on `EventBus` never include user prompts, tool arguments, tool outputs, hook reasons, approval previews, or provider error messages.
6. **Runtime dependencies stay out of checkpoints (ADR 0012):** Credentials, database sessions, and user tokens passed via `deps=` are never serialized into checkpoints, deep copies, or pickles, and are cleared when the run returns.
7. **Tenant isolation & path traversal protection:** Checkpoint stores validate `tenant_id` and `run_id` against `^[A-Za-z0-9_-]{1,64}$`, verify resolved paths remain within the root directory, and reject checkpoints whose stored `(tenant_id, run_id)` do not match the requested key.

## Known Operational Boundaries

- **Prompt injection cannot be solved by prompts alone.** Always combine minimal tool permissions (`Autonomy`, `allowed_tools`), human approvals, recipient/egress allowlists, and sandboxes.
- **Shell and filesystem tools:** `ProtectedPaths` inspects tool arguments and dictionary keys, not arbitrary shell command strings. Run shell or code-execution tools inside an isolated container with read-only mounts for tests and CI configuration.
- **`ReadFile` at `read_only`:** `ReadFile` is `Risk.READ`, so it runs at every autonomy level. Its only protection inside the root is a name-based deny list (`DEFAULT_DENIED_READS`: `.env`, `.env.*`, `*.env`, `secrets`, `.git`, `*.pem`, `*.key`). Other secret files, hard links, symlink chains, symlink swaps between check and read, and the targets of denied symlinks (under their own names) are not caught; see `docs/known-limitations.md`. Do not point its root at a directory with secrets.
- **File checkpoint store is POSIX-only:** `JsonFileCheckpointStore` needs `fcntl` for its cross-process lock and raises `ConfigError` elsewhere (e.g. Windows). Use a database-backed store there.
- **Checkpoint encryption:** `JsonFileCheckpointStore` and `PostgresCheckpointStore` write plaintext JSON; checkpoints contain full conversations. For PostgreSQL use TLS (`sslmode=require`), a least-privilege role (at runtime it needs only `SELECT, INSERT, UPDATE, DELETE` on its table; run `setup()` or `schema_sql()` with a separate migration role, because PostgreSQL 15+ does not grant `CREATE` on `public` by default), encryption at rest, and backups with the same retention as the data. For sensitive multi-tenant workloads wrap the store with per-tenant keys in a KMS (see [Use Case 01](docs/use-cases/01-finance-and-communications-assistant.md)); note its limit: it does not detect replay of an older complete row.

## Reporting a Vulnerability

Please **do not** report security vulnerabilities through public GitHub issues.

Instead, use GitHub's [Private Vulnerability Reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability) on this repository. Include:

- Affected module and version
- A minimal reproducible test case demonstrating the bypass or invariant violation
- Impact assessment (which invariant from `AGENTS.md` is violated)
