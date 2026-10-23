---
name: verify-change
description: Verify every change before completion — tests, linter, formatter, typecheckers, architectural invariants, and anti-reward-hacking rules. Run before reporting any task as done.
allowed-tools: Bash(pytest) Bash(pytest *) Bash(ruff check *) Bash(ruff format *) Bash(mypy *) Bash(pyright) Bash(git diff *) Bash(git status)
---

# Verifying a Change

## 1. Automated Checks — All Must Be Green

```bash
ruff format ai_agent_harness examples tests
ruff check ai_agent_harness examples tests
mypy ai_agent_harness examples tests
pyright
pytest
```

If any check fails: **fix the root cause.** Bypassing or weakening a check is never permitted.

## 2. Anti-Reward-Hacking — Audit Your Own Diff

Inspect `git diff` and confirm none of the following occurred (otherwise revert or ask a human):

- Deleted, weakened (`assert` removed or loosened), or skipped (`skip`, `xfail`) tests.
- Added `# type: ignore`, `# noqa`, or `pragma: no cover` comments.
- Relaxed rules in `pyproject.toml` (ignore lists, `strict`, `filterwarnings`).
- Updated expected test values to match broken behavior instead of fixing the bug.

## 3. Architectural Invariants

Check against **all** invariants in `AGENTS.md` (do not rely on memory). Pay special attention when modifying
`ai_agent_harness/runtime/runner.py`, `ai_agent_harness/core/state.py`, or `ai_agent_harness/tools/executor.py`:

- Is there any new path where a run terminates without a defined status?
- Can any tool call remain unanswered in the message history?
- Can any tool execute outside `ToolExecutor` or before the write-ahead checkpoint?
- Can an exception or invalid decision in a hook allow an action through?
- Does any event contain sensitive content (arguments, results, reasons, or message text)?
- Did the persisted schema of `RunState` change? If so, bump `SCHEMA_VERSION`, add a migration,
  and add a fixture in `tests/fixtures/` — never modify existing fixtures.
- Does `tests/test_stress.py` pass? For core changes, also run `HARNESS_STRESS=3000 pytest tests/test_stress.py`.

## 4. Documentation

- Executable blocks (`# executable`) in `README.md`, `docs/guide.md`, and the ai-kit recipes pass (`pytest tests/test_docs.py`).
- `README.md`, `AGENTS.md`, and ADRs remain accurate.

## 5. Independent Architecture Review

When modifying `ai_agent_harness/`, invoke the `architecture-reviewer` subagent at the end and address its findings.
