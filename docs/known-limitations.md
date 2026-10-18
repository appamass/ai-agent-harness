# Known Limitations

Read this before you rely on the harness. Everything here is deliberate or not built yet, and each item says what to do instead.

## Status of the project

- **Version 0.x.** The public API may change between minor versions (see [`../CHANGELOG.md`](../CHANGELOG.md)).
- **Verified live so far:** the Anthropic adapter with Claude Haiku 5.5 (11 opt-in live tests and the example agent, see [`first-real-run.md`](first-real-run.md)).
  **Not verified live:** other providers and models, long-running or high-volume workloads, several real workers against one database in production.
- **Only Anthropic is bundled.** The core ships without LLM SDKs; Anthropic is an optional extra. For other providers write an adapter
  (contract: `ai_agent_harness/providers/base.py`, skill `add-provider-adapter`).
- **Not built yet:** streaming, MCP integration, structured final output, automatic context compaction, parallel tool execution
  (see [`next-steps.md`](next-steps.md)).

## Safety boundaries

- **Prompt injection:** Prompts alone cannot guarantee separation of instructions and data. Real protection comes from least-privilege tool
  permissions, human approvals, argument validation, and egress filtering.
- **`ProtectedPaths`:** For write tools, it inspects path-like argument names, every whitespace-free string, and nested dictionary keys
  (`files={"tests/x.py": …}`). Paths containing spaces under custom field names require `extra_keys`. Because it is fail-closed, single tokens
  like a label `"tests"` or a branch `"fix/tests"` on write tools are blocked — customize `ProtectedPaths(patterns=…)` for your project rather
  than disabling protection.
- **`ProtectedPaths` and symlinks:** The hook compares the path strings in the arguments; it does not resolve symlinks. If `docs/link` is a
  symlink to `tests/`, a write to `docs/link/x.py` passes the hook. Write tools must resolve the target themselves (`Path.resolve()`) and check it,
  or run inside a sandbox with read-only mounts for protected paths. Do not let agents create symlinks.
- **`ReadFile` deny list:** `ReadFile` is `Risk.READ` and therefore allowed at `read_only`. By default it refuses `.env`, `.env.*`, `*.env`,
  `secrets`, `.git`, `*.pem`, and `*.key` anywhere under its root (`DEFAULT_DENIED_READS`, matched per path part, case-insensitive). This is a
  safety net, not a secret scanner. It checks names only, on both the requested and the resolved path: secrets in other files, hard links, chains of
  two or more symlinks, and symlinks swapped between check and read still get through. Blocking a denied name that is a symlink does not protect
  its target: the target is readable under its own name unless it also matches a pattern or lies outside the root. Tools that can create links
  must not share a root with secrets. Common secret files such as `.envrc`, `.netrc`, `.npmrc`, `.aws/`, `.ssh/`, `*.p12`, and `credentials.json`
  are not on the list; add them with `ReadFile(root, denied=(*DEFAULT_DENIED_READS, ".ssh", ...))`.
- **Shell tools:** `ProtectedPaths` inspects structured arguments, not free-form shell command strings. Run shell tools only inside a sandbox
  with read-only mounts for protected paths.

## Runtime behavior

- **Thread timeouts:** Python cannot forcibly kill threads. A `SyncTool` that exceeds its timeout continues running in the background inside a
  dedicated thread pool (`SYNC_TOOL_THREADS`, default 16). While the process is running, this never starves the event loop or the checkpoint store;
  if the pool fills up, new `SyncTool` calls time out cleanly. On process **shutdown**, however, Python waits for hanging threads. Only initialize the
  pool after `fork` (for example Gunicorn `--preload`). For hard isolation, run untrusted or blocking code in a subprocess or sandbox and enforce
  timeouts inside the tool itself.
- **Sequential tool execution:** Tool calls within a single response run one after the other, so execution order, write-ahead checkpoints, and
  approval pauses stay deterministic.
- **Estimated context size:** The context budget is estimated in characters (not exact tokens; images are excluded). Compaction is not built in:
  oversized runs end cleanly with `limit_reached`.
- **Exception tracebacks:** Unexpected internal errors are logged to `ai_agent_harness.runner` with tracebacks that may contain local values.
  Events are content-free, but protect your application logs like production data.

## Storage

- **`JsonFileCheckpointStore`:** uses `fcntl`, so it works only on POSIX (Linux/macOS). On platforms without `fcntl` (for example Windows) it
  raises `ConfigError` rather than running without a cross-process lock. On case-insensitive filesystems (macOS, Windows), `Customer` and `customer`
  map to the same path; the store detects and rejects this collision, so normalize IDs to lowercase. Locking is per tenant. It writes plaintext JSON
  and is meant for development and single-host use.
- **`PostgresCheckpointStore`:** works anywhere PostgreSQL does, but stores plaintext JSON too. Use TLS, a least-privilege role, and encryption at rest,
  or wrap the store with per-tenant keys (Use Case 01, note its limit: it does not detect replay of an older complete row). Database errors and logs may
  quote row data: treat them like checkpoints. See [ADR 0018](adr/0018-postgres-checkpoint-store.md).
