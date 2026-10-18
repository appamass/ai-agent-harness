# AGENTS.md – Rules for AI Coding Agents in This Repository

This file is the **single source of truth** for all coding agents (Claude Code, Codex, Cursor, Copilot, etc.).
Claude Code loads it via `CLAUDE.md` (`@AGENTS.md`). Tool-specific instructions belong only there.

## What This Project Is

A model-agnostic harness for AI agents, published for public use. A bug here lands in every downstream project.
Therefore: prefer one fewer change over an untested one.

Overview: `README.md` · Concepts: `HARNESS_TAXONOMY.md` · Decisions: `docs/adr/` ·
Scenarios: `docs/use-cases/` · First real run: `docs/first-real-run.md`.

## Commands

```bash
uv sync --extra dev                      # exact versions from uv.lock (or: pip install -e ".[dev]")
pytest                                   # all tests including executable docs
ruff check ai_agent_harness examples tests && ruff format --check ai_agent_harness examples tests
mypy ai_agent_harness examples tests              # strict
pyright                                  # strict
# Postgres tests run when HARNESS_TEST_POSTGRES_DSN is set (CONTRIBUTING.md); CI always sets it
```

A change is only complete when every check is green (skill `verify-change`).

## Non-Negotiable Invariants

These properties are enforced by tests. Anyone wanting to change them must write an ADR first.

1. **Every run ends with a status.** Error paths end as `failed`/`blocked`/`cancelled`, never as an unhandled crash.
2. **History is always API-compliant.** Messages are mutated exclusively via `RunState` methods.
3. **Guardrails are fail-closed.** Any hook that raises or returns an invalid decision blocks.
4. **Tools run only through the execution gate** (`tools/executor.py`). No other code calls `tool.run`.
5. **`IRREVERSIBLE` always requires human approval** — at every autonomy level, and an approval or answer
   applies only to the call that was shown (`resume(..., pending_id=...)`, ADR 0014).
6. **Write-ahead:** State is saved before every tool execution along with completed results of the batch;
   `resume` claims the run first. Completed results are never lost on a crash.
7. **The core stays clean:** `ai_agent_harness/core/` imports nothing except Pydantic and the Python standard library.
8. **Events contain metadata only** — never arguments, results, reasons, or message text (ADR 0011).
9. **Persisted states remain readable:** Any schema change to `RunState` requires bumping `SCHEMA_VERSION`,
   adding a migration, and saving a fixture in `tests/fixtures/`. Never modify existing fixtures (ADR 0010).

## Working Rules

- **Bug = failing test first**, then the fix. The test stays as a regression test.
- **Fix at the producing boundary, not downstream:** Validate and normalize data once at the responsible
  boundary (adapter, tool schema, `restore_state`) rather than bolting on defensive workarounds later
  in the call path; remove obsolete code in the same change.
- **Keep the stress test green:** `tests/test_stress.py` verifies invariants under randomized faults.
  On failure, record the reported seed as a dedicated regression test.
- **Never weaken checks to get green:** Do not delete, relax, or skip tests; do not add `# type: ignore`,
  `# noqa`, or `pragma: no cover`, and do not relax configuration in `pyproject.toml` without explicit
  human approval. That exact behavior (reward hacking) is what this harness prevents in its own agents.
- **No placeholder modules.** Add new files only with real implementations and tests (ADR 0008).
- **Architectural decisions → ADR** in `docs/adr/` (skill `write-adr`).
- **Doc code is code:** Blocks starting with `# executable` in `docs/use-cases/` and `docs/first-real-run.md`
  are executed by `tests/test_docs.py`. Anyone changing the API must update them.
- **Public API** lives in `ai_agent_harness/__init__.py`. Changes there are breaking changes for downstream
  projects → document in an ADR, `README.md`, and `CHANGELOG.md`, and bump the version in `pyproject.toml`.
- **Keep context concise:** `AGENTS.md` stays under 100 lines (invariants and non-obvious boundaries
  only, no directory trees); anything statically verifiable belongs in `tests/test_architecture.py`.
- **Language:** Write all documentation, comments, docstrings, error messages, and identifiers in Simple English.

## Where Things Belong

| Task | Location | Guide |
|---|---|---|
| New tool | Project code or `ai_agent_harness/tools/builtin.py` (only if broadly useful) | Skill `add-tool` |
| New guardrail | `ai_agent_harness/guardrails.py` (general) or project code | Skill `add-guardrail` |
| Model adapter | `ai_agent_harness/providers/<provider>.py` + optional dependency | Skill `add-provider-adapter` |
| Architecture decision | `docs/adr/NNNN-title.md` | Skill `write-adr` |
| Example / scenario | `docs/use-cases/NN-title.md` with `# executable` blocks, or a runnable script in `examples/` (tested offline in `tests/test_examples.py`) | – |

## Security

- No secrets in code, tests, docs, or prompts. Never read `.env` or `secrets/`.
- Modify `.github/`, `.claude/`, and `pyproject.toml` only with human approval.
- No new runtime dependencies in the core. Optional dependencies are for adapters only.
