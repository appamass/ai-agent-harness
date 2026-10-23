# 0016 – Import Name `ai_agent_harness`

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-18
- **Affects:** Import name of the public API, package directory, logger names

## Context

The distribution is `ai-agent-harness`, but the import package was `harness`. PyPI has another package named `harness`; two distributions with the same top-level package overwrite each other silently. The project was not published yet, so renaming was cheap.

## Decision

The import package is `ai_agent_harness`, matching the distribution name. Loggers are `ai_agent_harness.*`. No `harness` alias is shipped, because it would keep the colliding name. Released in 0.4.0.

## Rejected

- Keep `harness`: collides with the PyPI package.
- `agent_harness`: the PyPI package `agent-harness` likely uses that import name.
- A `harness` alias module: keeps the collision.

## Consequences

Longer imports; every import, logger config, and tool path changed. `tests/test_architecture.py` reads the package from `ai_agent_harness/`.
