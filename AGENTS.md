# AGENTS.md – Guide for AI Coding Agents

You are reading this repository to learn from it or take patterns into your own project.
If you are contributing code to *this* repository, the commands and working rules at the end of this file apply.

## Three Ways to Adopt This Repository

1. **Use (Dependency):** Pin a release tag ([install line](README.md#install)). Best when building pure Python services that need crash-safe runs, human approval, and fail-closed guardrails. We deliberately do not publish to PyPI ([ADR 0013](docs/adr/0013-package-distribution.md)).
2. **Copy (Standalone Bricks):** Use or copy individual bricks (they work without `Agent`; copy them together with `core/` and `policy.py`; stores also need `persistence.py` and `persistence_sql.py`) ([ADR 0019](docs/adr/0019-modular-bricks-and-security-guards.md)). For example: `ToolExecutor` (the 8-step gate), `@tool(risk=…)` (risk-typed functions), `SecretGuard` / `EgressGuard` / `WeakeningGuard`, or `PostgresCheckpointStore`.
3. **Rebuild (Invariants in Your Stack):** Re-implement our core guarantees inside LangGraph, Pydantic AI, or your own loop using [`ai-kit/`](ai-kit/skills/ai-agent-harness/SKILL.md).

## Non-Negotiable Invariants to Preserve

Whether you use, copy, or rebuild, these properties must survive:

1. **Status:** Every run ends with a defined status (`done`, `refused`, `blocked`, `limit_reached`, `failed`, `cancelled`, or waiting for a human), never an unhandled crash.
2. **Protocol:** Message history strictly alternates (`user` ↔ `assistant`). Open tool calls always receive a matching result.
3. **Gate:** Tools execute only through one gate (`tools/executor.py`). Direct tool execution is forbidden.
4. **Approval binding:** `IRREVERSIBLE` never runs without approval (denied below `act_with_approval`). An approval applies exclusively to the call the human saw (`pending_id`, [ADR 0014](docs/adr/0014-bind-resume-to-pending-call.md)). Stale approvals never execute subsequent calls.
5. **Write-ahead & crash honesty:** State is persisted before every tool call. A crashed run resumes with `recover=True`; interrupted calls are reported as "outcome unknown", never silently re-executed.
6. **Fail-closed:** Any hook or guard that raises or returns an invalid decision blocks execution.
7. **Safe observability:** Events carry metadata only (counters, IDs, statuses), never raw prompts, arguments, or results ([ADR 0011](docs/adr/0011-operational-limits-and-metadata-events.md)).
8. **Clean core:** The core runtime depends only on Pydantic and the Python standard library.
9. **Persisted schemas remain readable:** Any schema change to `RunState` requires bumping `SCHEMA_VERSION`, adding a migration, and saving a fixture in `tests/fixtures/` ([ADR 0010](docs/adr/0010-checkpoint-schema-versioning.md)).

## Code and Pattern Map

| Component | Code Location | What to Take / Reference |
|---|---|---|
| Tool execution gate | `ai_agent_harness/tools/executor.py` | 8-step gate, timeout handling, untrusted wrapping |
| Function decorator | `ai_agent_harness/tools/base.py` | `@tool(risk=…)` parameter schema generation |
| State & protocol | `ai_agent_harness/core/state.py` | `RunState`, `PendingBatch`, optimistic `revision` |
| Guardrails | `ai_agent_harness/guardrails.py` | `SecretGuard`, `EgressGuard`, `ProtectedPaths`, `KillSwitch` |
| Checkpoint stores | `persistence_sqlite.py`, `persistence_postgres.py` | Optimistic locking on SQLite (stdlib) & PostgreSQL |
| Provider failover | `providers/router.py`, `providers/base.py` | Retry and failover loop; stripping foreign reasoning blocks (`without_foreign_thinking`) |
| Sub-agents | `ai_agent_harness/runtime/delegation.py` | `sub_agent(...)`: a read-only specialist as one gated tool ([ADR 0030](docs/adr/0030-sub-agents-as-read-only-tools.md)) |
| Agent project layout | `docs/agent-package.md` | Definition vs. runtime vs. system; prose in files, rules in code (tested) |
| Integration recipes | `ai-kit/skills/ai-agent-harness/references/adopt-in-x.md` | `ToolExecutor` as the tool node of any loop (tested), guardrails, crash recovery |

## Development and Verification Commands

```bash
uv sync --extra dev                      # exact versions from uv.lock
pytest                                   # all tests including executable docs
ruff check ai_agent_harness examples tests && ruff format --check ai_agent_harness examples tests
mypy ai_agent_harness examples tests     # strict
pyright                                  # strict
```

Extended chaos suite (for changes in `core/`, `runtime/`, `tools/executor.py`): `HARNESS_STRESS=3000 pytest tests/test_stress.py`.
PostgreSQL tests are skipped without `HARNESS_TEST_POSTGRES_DSN` (CI always runs them); each test creates and drops its own schema, so any scratch
database works, for example `docker run --rm -d -e POSTGRES_USER=harness -e POSTGRES_PASSWORD=harness -e POSTGRES_DB=harness_test -p 5432:5432 postgres:17`
with `HARNESS_TEST_POSTGRES_DSN=postgresql://harness:harness@localhost:5432/harness_test`.

## Working Rules

1. **Bug fixes start with a failing test**, then the fix at the producing boundary (adapter, tool schema, `restore_state`), not a downstream workaround.
2. **Never weaken checks to get green.** No deleting or skipping tests, no `# type: ignore`, `# noqa`, `pragma: no cover`, no relaxing `pyproject.toml`
   without explicit human approval (`tests/test_architecture.py` enforces this).
3. **Docs are tested.** `# executable` blocks in `README.md`, `docs/guide.md`, and the ai-kit recipes run in `tests/test_docs.py`; `tests/test_links.py`
   checks every relative link. `README.md` stays under 160 lines.
4. **ADRs:** a change to an invariant, the public API (`ai_agent_harness/__init__.py`), runtime dependencies, or a security boundary needs a short ADR in
   `docs/adr/` (`0000-template.md`).
5. **Dependencies:** after editing `pyproject.toml`, run `uv lock` and commit `uv.lock`. Lower bounds are tested by the `lowest` CI job.
6. **No placeholder modules** (ADR 0008). Write docs, comments, and error messages in Simple English.
7. **Secrets and config:** no secrets in code, tests, docs, or prompts; never read `.env` or `secrets/`. Change `.github/`, `.claude/`, and
   `pyproject.toml` only with explicit human approval.
8. **AI coding agents:** `CLAUDE.md` imports this file; `.claude/skills/` has guides (`add-tool`, `add-guardrail`, `add-provider-adapter`, `write-adr`,
   `verify-change`); `.claude/agents/architecture-reviewer.md` reviews changes independently. These are context, not enforcement: hard limits come from
   permissions, CI, and review.

Overview: [`README.md`](README.md) · Invariant proofs: [`docs/guarantees.md`](docs/guarantees.md) · Architecture decisions: [`docs/adr/`](docs/adr/README.md)
