# 0032 – Python 3.12 Is the Minimum

- **Status:** Accepted (maintainer, 2026-10-28)
- **Date:** 2026-10-28
- **Affects:** `pyproject.toml` (`requires-python`, classifiers, mypy, ruff, pyright), CI, README

## Context

Python 3.11 reaches its end of life in 2027. It also behaves differently from 3.12+: `asyncio.wait_for` runs each awaited call
in its own task, so cancellation and crash paths need special handling and tests that differ per version.

## Decision

`requires-python = ">=3.12"`. CI tests 3.12, 3.13, and 3.14; the `lowest` job runs on 3.12. Type checkers and ruff target 3.12,
and the generics ruff flags use the PEP 695 syntax (`class BaseTool[InputT: BaseModel]`, `SyncTool`, small helpers).
Generic methods such as `RunState.deps_as` keep their `TypeVar` for now.

## Rejected

- Keeping 3.11 until its end of life: two code paths for async timing in a library whose guarantees depend on it.

## Consequences

Projects on Python 3.11 stay on v0.8.0. `uv.lock` was regenerated for the new range. The module-level `InputT` in
`ai_agent_harness.tools.base` is gone (it was never exported). Type checkers now infer the variance of `BaseTool`'s parameter
(contravariant, because tools only take it as input), so assignments between `BaseTool[...]` types can check differently in your
code; runtime behavior and tool digests are unchanged.
