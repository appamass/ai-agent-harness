# Contributing to `ai-agent-harness`

Thank you for contributing! A bug in this library reaches every project that uses it, so we prioritize **provable invariants, a small surface area, and no untested code**.

Read [`AGENTS.md`](AGENTS.md) first — it is the single source of truth for both human contributors and AI coding agents.

## Quick Setup

**Recommended — with [uv](https://docs.astral.sh/uv/)**, which installs exactly the versions CI uses (from `uv.lock`):

```bash
uv sync --extra dev
source .venv/bin/activate
```

**Without uv** — works too, but you get the newest tool versions, which may differ from CI:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the PostgreSQL tests

The checkpoint store tests need a database; without one they are skipped (CI always runs them). Locally, with Docker:

```bash
docker run --rm -d --name harness-pg -e POSTGRES_USER=harness -e POSTGRES_PASSWORD=harness -e POSTGRES_DB=harness_test -p 5432:5432 postgres:17
export HARNESS_TEST_POSTGRES_DSN=postgresql://harness:harness@localhost:5432/harness_test
pytest
```

Each test creates and drops its own schema, so it is safe to point the variable at any scratch database.

## Changing Dependencies

`uv.lock` pins every package version for CI and local development. CI fails if it does not match `pyproject.toml`.

- After changing dependencies in `pyproject.toml`, run `uv lock` and commit `uv.lock` in the same pull request.
- To upgrade everything to the newest allowed versions, run `uv lock --upgrade`. Dependabot also proposes
  weekly update pull requests (`.github/dependabot.yml`).
- Lower bounds in `pyproject.toml` are tested by the `lowest` CI job. Only lower a bound when that job passes.

## Verification Checklist

Every change must pass all checks before opening a pull request:

```bash
ruff format --check ai_agent_harness examples tests
ruff check ai_agent_harness examples tests
mypy ai_agent_harness examples tests
pyright
pytest
```

For changes touching `ai_agent_harness/core/`, `ai_agent_harness/runtime/`, or `ai_agent_harness/tools/executor.py`, also run the extended chaos stress suite:

```bash
HARNESS_STRESS=3000 pytest tests/test_stress.py
```

## Working Rules

1. **Bug fixes start with a failing test.** First reproduce the bug in `tests/`, then fix it at the producing boundary (adapter, tool schema, or `restore_state`), not via downstream workarounds.
2. **Never weaken checks to get green.** Do not delete, relax, or skip tests. Do not add `# type: ignore`, `# noqa`, or `pragma: no cover` or relax `pyproject.toml` without explicit human approval (`tests/test_architecture.py` enforces this automatically).
3. **Keep `ai_agent_harness/core/` dependency-free.** Only `pydantic` and the Python standard library may be imported in `ai_agent_harness/core/`.
4. **Executable documentation is tested.** Code blocks starting with `# executable` in `README.md`, `docs/guide.md`, `docs/how-it-works.md`, `docs/use-cases/`, and `docs/first-real-run.md` are executed by `tests/test_docs.py`, and `tests/test_links.py` checks every relative link. Keep `README.md` short (the test allows 160 lines): detail belongs in `docs/`.
5. **Architecture Decision Records (ADRs):** Any change to an invariant, the public API (`ai_agent_harness/__init__.py`), runtime dependencies, or security boundaries requires an ADR in `docs/adr/` (see `docs/adr/0000-template.md`).
6. **Checkpoint schema changes (ADR 0010):** If `RunState` serialization changes, increment `SCHEMA_VERSION` in `ai_agent_harness/core/state.py`, add a migration in `MIGRATIONS`, and save a fixture of the previous version in `tests/fixtures/`. Never modify existing fixtures.

## Working with AI Coding Agents

This repository is set up for Claude Code and other AI coding agents. The same risks the harness prevents in production — especially
relaxing tests or linters to get a green build — apply to coding agents working on it:

| File | Purpose |
|---|---|
| `AGENTS.md` | Single source of truth for commands, invariants, and working rules (read by all major agents) |
| `CLAUDE.md` | Imports `AGENTS.md` and adds Claude Code workflow instructions |
| `.claude/skills/` | Step-by-step guides: `add-tool`, `add-guardrail`, `add-provider-adapter`, `write-adr`, `verify-change` |
| `.claude/agents/architecture-reviewer.md` | Independent reviewer subagent for auditing changes to `ai_agent_harness/` |
| `.claude/settings.json` | Pre-approves verification commands; prompts before editing `.github/`, `.claude/`, or `pyproject.toml`; blocks `.env` and `secrets/` |

Markdown rules are context, not enforcement: hard boundaries come from permissions, CI checks, and human review.
