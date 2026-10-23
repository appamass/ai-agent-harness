# 0015 – Deny List for `ReadFile`; File Store Is POSIX-Only

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-18
- **Affects:** `ai_agent_harness/tools/builtin.py` (`ReadFile`, `DEFAULT_DENIED_READS`), `persistence.py` (`JsonFileCheckpointStore`), `policy.py` (`Limits`)

## Context

A review before publication found three unsafe defaults:

1. `ReadFile` is `Risk.READ`, so it runs even at `read_only`, and its only limit was the root directory. A model could read `.env`, `secrets/`, keys, and `.git/`.
2. Without `fcntl` (Windows) the file store took no lock, so two workers could claim a run and run an action twice.
3. `Limits` accepted NaN, so `max_seconds=nan` silently disabled the time limit.

## Decision

- `ReadFile(root, *, max_bytes=..., denied=DEFAULT_DENIED_READS)`. The default is `(".env", ".env.*", "*.env", "secrets", ".git", "*.pem", "*.key")` (later extended, ADR 0019). A pattern is a plain name and matches any single part of the path relative to the root, case-insensitively. Both the requested path (normalized) and the resolved path are checked, so an alias such as `harmless.txt -> .env` is blocked. The target of a denied alias stays readable under its own name unless it matches too or lies outside the root. `denied` **replaces** the default (`(*DEFAULT_DENIED_READS, ...)` extends it, `()` disables it); invalid patterns raise `ConfigError`.
- `JsonFileCheckpointStore` raises `ConfigError` when `fcntl` is missing.
- `Limits` rejects bools, non-numbers, and non-finite floats with `ConfigError`.

## Rejected

- `msvcrt.locking` on Windows: untestable in CI; an untested lock in a crash-safety path is worse than a refusal.
- Thread lock only: does not protect two processes.
- Allowlist of files: breaks "read the project"; projects can add one as a hook.
- Extending instead of replacing the list: the default could not be removed.
- Content scanning for secrets: heuristic and false comfort.

## Consequences

Breaking for projects that read `.env*` or `*.key` with `ReadFile`, and for Windows users of the file store. Hard links, symlink chains, and swaps between check and read still get through (`docs/known-limitations.md`). Tested in `tests/test_tools.py`, `tests/test_guardrails_and_persistence.py`, `tests/test_config_and_misc.py`.
