# 0015 – Default Deny List for `ReadFile`; File Store Is POSIX-Only

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-18
- **Affects:** `ai_agent_harness/tools/builtin.py` (`ReadFile`, `DEFAULT_DENIED_READS`), `ai_agent_harness/persistence.py`
  (`JsonFileCheckpointStore`), `ai_agent_harness/policy.py` (`Limits`), public API (`DEFAULT_DENIED_READS`)

## Context

A pre-publication review found three safe-by-default gaps:

1. `ReadFile` is `Risk.READ`, so the policy allows it even at `read_only`. Its only boundary was the root
   directory. A model could read `.env`, `secrets/`, private keys, and `.git/` inside the project.
2. Without `fcntl` (e.g. Windows), `JsonFileCheckpointStore` took no lock at all, while a comment claimed
   in-process locking. Two workers could claim the same run and execute one action twice (invariant 6, ADR 0006).
3. `Limits` accepted NaN: every comparison with NaN is false, so `max_seconds=nan` silently disabled the time limit.

## Decision

- `ReadFile(root, *, max_bytes=..., denied=DEFAULT_DENIED_READS)`. `DEFAULT_DENIED_READS` is
  `(".env", ".env.*", "*.env", "secrets", ".git", "*.pem", "*.key")` and part of the public API. Each pattern is a
  name (no path separators, no outer spaces) and matches any single part of the path relative to the root,
  case-insensitive (`casefold` + `fnmatchcase`). Both the requested path (normalized, not resolved) and the
  resolved path are checked, so `harmless.txt -> .env` and the alias `.env -> vault/prod` are both blocked.
  Blocking a denied alias does not protect its target: `vault/prod` stays readable by its own name unless it
  also matches a pattern or lies outside the root.
  `denied` **replaces** the default; callers extend it with `(*DEFAULT_DENIED_READS, ...)`, `()` disables it.
  Invalid patterns raise `ConfigError` at construction.
- `JsonFileCheckpointStore.__init__` raises `ConfigError` when `fcntl` is missing. There is no Windows lock.
- `Limits` rejects bools, non-numbers, and non-finite floats with `ConfigError`.

## Alternatives

| Alternative | Why Rejected |
|---|---|
| Lock with `msvcrt.locking` on Windows | Cannot be tested in CI today; an untested lock in a crash-safety path is worse than a clear refusal. |
| Thread lock only on Windows | Does not protect two processes — exactly the case that repeats actions. |
| Allowlist of readable files | Safer, but breaks the common "read the project" use; projects can still build one as a hook. |
| Deny list that extends instead of replaces | Then the default cannot be removed for projects that need it; replacing is explicit and auditable. |
| Content-based secret scanning | Heuristic, slow, and gives a false sense of safety; out of scope for a reference tool. |

## Consequences

- Positive: The read-only default no longer exposes common secret files; the file store never runs without
  a cross-process lock; misconfigured limits fail at construction.
- Negative / Cost: Breaking behavior for projects that read `.env*`, `*.key`, etc. via `ReadFile`, and for
  Windows users of `JsonFileCheckpointStore`. The deny list checks names only: hard links, symlink chains,
  symlink swaps between check and read, and targets of denied aliases (by their own names) still get through (README Known Limitations).
- Verification: `tests/test_tools.py::TestReadFile`, `tests/test_guardrails_and_persistence.py::TestCheckpoints::
  test_file_store_refuses_platforms_without_file_locks`, `tests/test_config_and_misc.py::test_limits_*`.
