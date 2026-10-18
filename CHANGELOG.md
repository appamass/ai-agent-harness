# Changelog

Semantic versioning. Changes to the persisted checkpoint format also list their schema version
(ADR 0010). Prior to `1.0.0`, minor versions may introduce breaking changes to the public API.

## 0.6.0 – 2026-10-18

- Docs: the README is now a short start (about 145 lines): what it is, a no-key demo with a real transcript, a hello-agent block, what is included, status, and a
  documentation map. The detail moved to `docs/`: `guide.md` (build with it), `how-it-works.md` (code map, run loop, gate, autonomy), `guarantees.md` (all 16
  guarantees with the test that enforces each), `production-checklist.md`, `known-limitations.md`. Nothing was dropped.
- Tests: executable blocks in the README and the new guides run in CI; `tests/test_links.py` checks every relative link and anchor; the README demo transcript is
  compared with the real demo output; the README has a length limit.

- New: `ai_agent_harness.persistence_postgres.PostgresCheckpointStore`, a checkpoint store for several workers, as an optional extra:
  `pip install "ai-agent-harness[postgres]"` (ADR 0018). One atomic conditional upsert implements the optimistic locking, loads go
  through `restore_state`, `setup()` / `schema_sql()` create the table, `list_runs` serves approval queues and recovery jobs,
  `delete_run` / `delete_tenant` support retention and erasure. Works with a connection string or an `AsyncConnectionPool`.
- Tests: the store contract tests run against a real PostgreSQL when `HARNESS_TEST_POSTGRES_DSN` is set (CI does); one in four stress
  scenarios runs on PostgreSQL. A mutation check showed the concurrency tests fail if the revision condition is removed.
- Removed (unused, nothing read them): the `BaseTool.idempotent` class attribute and `ModelRouter.providers`. A tool that sets `idempotent` keeps working; the value is ignored.
- Internal: the cancellation-safe write logic of the file store moved to `persistence.settle_write` and is shared; `check_ids` is no longer private.

## 0.5.0 – 2026-10-18

- New: `ai_agent_harness.providers.anthropic.AnthropicProvider` for the Anthropic Messages API, as an optional extra:
  `pip install "ai-agent-harness[anthropic]"` (ADR 0017). The core and `import ai_agent_harness` still need only Pydantic.
  No SDK-level retries (the router owns them), no sampling parameters, thinking signatures round-trip, errors carry only
  status and type, prompt caching on by default.
- New: `examples/repo_assistant.py`, a complete runnable agent (tools, human approval with `pending_id`, limits, kill file,
  checkpoints and `--resume`, `--demo` without an API key), tested offline in `tests/test_examples.py` and opt-in live in
  `tests/live/test_examples_live.py`. `examples/` is checked like the library (ruff, mypy strict, pyright strict) and is not part of the wheel.
- New: opt-in live tests (`pytest -m live`, `tests/live/`); the `live` marker is registered and deselected by default.
- New: `AnthropicProvider.aclose()` closes the connections of a client the adapter created (found in the first live run:
  without it Python 3.14 reports unclosed sockets). Call it when you are done.
- Docs: `docs/first-real-run.md` has a 5-minute Anthropic setup and the first results: the 7 live tests passed against
  Claude Haiku 5.5 on 2026-10-18. Eighteen real responses are recorded as test fixtures
  (`tests/fixtures/anthropic/recorded/`: plain answers, tool calls, thinking together with a tool call, prompt-cache usage, a truncation run with thinking blocks). Fixtures for redacted
  thinking and refusal are still hand-written.
  Only Anthropic was run live (Claude Haiku 5.5). A second live run found a bug in the test helper (it closed the client
  between steps); after the fix the third run passed 7 of 7 without warnings, and the fourth run (with the example agent) 11 of 11.

## 0.4.0 – 2026-10-18

**Breaking:** the import package is renamed from `harness` to `ai_agent_harness` (ADR 0016) — replace
`from harness import ...` with `from ai_agent_harness import ...` and logger names `harness.*` with
`ai_agent_harness.*`. `resume` of a waiting run requires `pending_id` (ADR 0014). `ReadFile` denies secret files by
default and `JsonFileCheckpointStore` refuses platforms without `fcntl` (ADR 0015). Persisted format unchanged (v2).

- Fix (security): a stale approval or answer could apply to a different call. Example: two approvers see
  the dialog for action A; the first approves, the run pauses on action B; the second approver's click
  executed B although nobody had seen it. `resume(..., pending_id=state.pending_id)` is now required for
  `waiting_for_input` and `waiting_for_approval`; a missing or different value raises `InvalidStateError`
  before any state changes. Migration: pass the `pending_id` that was shown with the dialog.
- New: `RunState.pending_id` (derived, never persisted).
- Fix (security, behavior change): `ReadFile` refuses `.env`, `.env.*`, `*.env`, `secrets`, `.git`, `*.pem`, and `*.key`
  under its root by default (`DEFAULT_DENIED_READS`). It is `Risk.READ` and allowed at `read_only`, so it could
  read secrets before. Both the requested and the resolved path are checked (names only; see README Known
  Limitations). New public constant
  `DEFAULT_DENIED_READS`. Migration: pass `denied=(...)` to change the list, `denied=()` for the old behavior.
- Fix (behavior change): `JsonFileCheckpointStore` raises `ConfigError` on platforms without `fcntl` (e.g. Windows).
  Before, it ran there without any cross-process lock, so two workers could execute one action twice.
- Fix: `Limits` rejects NaN, infinity, bools, and non-numbers (`ConfigError`). `max_seconds=nan` silently disabled
  the time limit. Types that are not `int`/`float` subclasses (e.g. `numpy.int64`) are now rejected too.
- Docs: README Known Limitations lists the `ProtectedPaths` symlink gap and the `ReadFile` deny list.
- Docs: corrected Use Case 01 — AAD does not detect replay of an older complete checkpoint row.
- Docs: project status is alpha; no live model provider run yet.
- CI: installs the exact versions from `uv.lock` (`uv sync --locked`), so new tool releases cannot break CI;
  a `lowest` job tests the lower bounds (e.g. `pydantic==2.13`); Dependabot proposes weekly updates.
- Packaging: authors and project URLs (`github.com/appamass/ai-agent-harness`), license holder `appamass`,
  classifiers `3 - Alpha`, `POSIX`, and `MacOS`. All ADRs 0001–0016 accepted; ADR 0013 decided (public PyPI).

## 0.3.0 – 2026-10-16

**Persisted format v2** (automatic migration from v1; older versions cannot read v2).

- New: `start/resume(..., deps=...)` and `state.deps_as(Type)` for runtime dependencies (never persisted).
- New: `Usage.cache_read_tokens`, `cache_write_tokens`, `reasoning_tokens`; `Usage.total` includes them.
- New: `ThinkingBlock.provider_id` and `without_foreign_thinking()` for cross-provider failover.
- New: `ToolCall.arguments_error` — malformed JSON is returned to the model for self-correction instead of terminating the run.
- New: `KillSwitch`, `Limits.max_context_chars`, `Limits.max_input_chars`, `restore_state()` with key verification.
- Fix: After a crash mid-tool-batch, completed tools retain their results (write-ahead before every tool).
- Fix: `last_text` concatenates max-token continuations; `Message.text` no longer glues blocks together without separators.
- Fix: Hanging `SyncTool`s can no longer starve the checkpoint store (dedicated worker thread pool).
- Fix: `ProtectedPaths` detects paths under arbitrary field names and as nested dictionary keys.
- Fix: Cancellation during a checkpoint write always ends as `CancelledError` (even on repeated cancellation
  or when another worker has taken over the run) rather than `ConcurrencyError`.
- Fix: `deps` are unbound after each run and never pickled or deep-copied; overly long `arguments_error` strings are truncated rather than rejected.
- Fix: File checkpoint store no longer leaves a lock file per run and detects case-insensitive filesystem collisions.
- Fix: Events no longer include provider error messages.
- Dependency: `pydantic>=2.13`.
- Docs: Use Cases 04 (incident responder), 05 (provider switch), 06 (sub-agents & next-action eval);
  updated Use Cases 01–03 for v0.3.0 (results after crash, cost by category, `deps`), UC2 + "only done when tests pass",
  UC3 + multi-day clarifying question.

## 0.2.0

First complete release: runner with protocol invariants, hooks, tool execution gate, policy matrix,
router, checkpoints with optimistic locking and write-ahead, executable use cases, and ADRs 0001–0009.
