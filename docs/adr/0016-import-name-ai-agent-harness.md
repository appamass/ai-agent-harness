# 0016 – Import Name `ai_agent_harness`

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-18
- **Affects:** Import name of the whole public API, package directory, logger names, tooling commands

## Context

The distribution is published as `ai-agent-harness`, but the import package was called `harness`.
PyPI already has a different package named `harness`. Two installed distributions that both provide a
top-level `harness` package overwrite each other's files without a warning, so a user who installs both
gets a broken environment. A generic top-level name also collides easily with local project folders.
The project is not published yet, so a rename costs nothing for existing users now and a lot later.

## Decision

- The import package is `ai_agent_harness` (it matches the distribution name `ai-agent-harness`).
  `from ai_agent_harness import Agent, start, resume` replaces `from harness import ...`.
- The package directory is `ai_agent_harness/`. Loggers are named `ai_agent_harness.*`
  (e.g. `ai_agent_harness.runner`).
- No compatibility shim `harness` is shipped: it would keep claiming the colliding top-level name.
- Released as part of 0.4.0 (not yet published, already a breaking release).

## Alternatives

| Alternative | Why Rejected |
|---|---|
| Keep `harness` | Collides with the existing PyPI package `harness` |
| `agent_harness` | The PyPI package `agent-harness` already exists and very likely uses that import name |
| Ship a `harness` alias module | Keeps the collision that this decision removes |

## Consequences

- Positive: no collision with other distributions; the import name matches the install name.
- Negative / Cost: longer imports; every downstream import, logger configuration, and tooling path changes.
- Verification: `tests/test_architecture.py` reads the package from `ai_agent_harness/`; CI and all executable
  docs import `ai_agent_harness`.
