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
- **`ProtectedPaths` and symlinks:** Without `root=`, the hook compares the path strings in the arguments and does not resolve symlinks. Pass
  `ProtectedPaths(root=project_dir)` to also resolve symlinks against the root directory and block targets that land in protected paths or outside
  the root. Write tools should still resolve targets themselves (`Path.resolve()`) and run inside a sandbox with read-only mounts for protected paths.
- **`ReadFile` deny list:** `ReadFile` is `Risk.READ` and therefore allowed at `read_only`. By default it refuses `.env`, `.env.*`, `*.env`, `.envrc`,
  `secrets`, `.git`, `.ssh`, `.aws`, `*.pem`, `*.key`, `*.p12`, `*.pfx`, `.netrc`, `.npmrc`, `.pypirc`, `id_rsa`, `id_ed25519`, and
  `credentials.json` anywhere under its root (`DEFAULT_DENIED_READS`, matched per path part, case-insensitive). Combine `ReadFile` with `SecretGuard`
  to also catch token values inside ordinary files. It checks names on both the requested and the resolved path: hard links, chains of two or more
  symlinks, and symlinks swapped between check and read still get through. Blocking a denied name that is a symlink does not protect its target:
  the target is readable under its own name unless it also matches a pattern or lies outside the root. Tools that can create links must not share a
  root with secrets.
- **`ToolExecutor` used alone:** `ToolExecutor` is exported so you can use the gate without `Agent`. Its `approved=True` argument skips the
  human approval step for any risk level, and a call without `state` has no `pending_id` binding. The approval rules (invariant 5, ADR 0014) are
  enforced by `start` and `resume`. If you call the executor yourself, you own that step.
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

## Built-in guards are heuristics

`SecretGuard`, `EgressGuard`, and `WeakeningGuard` are fast, deterministic pattern checks. They run synchronously inside the event loop; their patterns are bounded so that large inputs scan in linear time. They catch the common cases and make mistakes visible.
They are not data loss prevention and not a sandbox. Use them as one layer.

- **`SecretGuard`:** it finds only the listed formats: private key headers, `hf_`, `sk-`, `sk_live_`/`rk_live_`, `ghp_`-style GitHub tokens, `github_pat_`,
  `glpat-`, `xox*-`, Google `AIza` keys, `npm_` and PyPI (`pypi-AgE…`) tokens, AWS `AKIA`/`ASIA` keys, three-part JWTs, `Bearer` tokens with 20+ characters, and
  `Authorization: Basic` (both case-insensitive), and `scheme://user:password@host`. A long kebab-case name that starts with `sk-` (for example `sk-learn-is-a-long-kebab-case-name`) looks like a key and is blocked. It does not find `password=…` assignments, passwords that contain `/` or `@`, other
  vendors' keys, or secrets that are encoded or split over several messages. As a hook it only blocks (tool arguments, tool results, model
  responses, and user input when enabled); `redact()` is a helper you call yourself.
- **`EgressGuard`:** it looks at the URL as text. It does not follow redirects, does not resolve DNS (a name on your allowlist can point to a private
  address, or to an open redirect), and cannot tell what a tool does with a host. Free text is scanned for `scheme://`, `scheme:/`, `scheme:\\`, and
  `mailto:`/`data:`/`file:`-style URLs. A bare host such as `evil.example/x` is found only in fields named like a URL (`url`, `uri`, `href`,
  `endpoint`, `webhook`, also `base_url` or `callbackUrl`) or listed in `url_keys`. Hosts must be ASCII (use punycode). Text such as `note:/path` or a config line `path:/usr/bin` looks like a URL with an unknown scheme and is blocked (`file:line` and `data:int` are not), and so is a URL-named field that holds a non-URL value (`endpoint="users"`). `allowed_domains` is
  required. With an empty one, every URL in tool arguments and in model answers is blocked, so list the domains your agent may link to, or set
  `check_model_output=False` to scan only tool arguments.
- **`WeakeningGuard`:** it finds `# type: ignore`, `# pyright: ignore`, `# noqa` (also `# ruff: noqa`, `# flake8: noqa`), `# pylint: disable`,
  `# mypy: ignore-errors`, `# nosec`, `# pragma: no cover`, `pytest.mark.skip`/`skipif`/`xfail`, `pytest.skip(`, `pytest.importorskip`,
  `unittest.skip`, `// @ts-ignore`, and `sys.exit(0)`. Other tools' switches are not known to it, and a suppression that is split over several
  write calls is not seen.
- **`ProtectedPaths(root=…)`:** it resolves paths against `root`, while a tool resolves them against its own working directory. Keep both the same.
  The check and the later write are separate steps, so a link created in between is not seen. Each path check reads the file system.
- **`ToolRegistry` pins:** a digest is checked when a tool is registered and when you call `verify()`. The executor does not re-check it on every
  call. Tools without a pin are not checked at all.
- **`as_untrusted`:** the frame tells the model that the content is data. It is a hint, not a boundary. It does not wrap error messages of a failed tool.
