# 0001 – Custom Lean Core Instead of an External Agent Framework

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-15
- **Affects:** Entire package, dependencies

## Context

This template is shared across multiple production projects. Our primary goals are independence from
individual model vendors and complete auditability of every run. The core of an agent (loop, tool
dispatch, retry, cancellation) is small; the hard parts are persistence, human approvals, and failure paths.

## Decision

We implement our own lean core in pure Python with **Pydantic as the sole runtime dependency**.
No third-party agent frameworks (LangGraph, Google ADK, CrewAI, etc.) are used in the template.
Focused utility libraries (Pydantic in core, vendor SDKs inside optional provider adapters, database
drivers inside optional checkpoint stores) are permitted.

## Alternatives

| Alternative | Why Rejected |
|---|---|
| LangGraph / Google ADK | Dictates architecture, couples projects to external release cycles, makes strict guarantees (protocol invariants, fail-closed hooks) harder to enforce |
| Zero dependencies (no Pydantic) | Custom schema validation and serialization would require more code and provide weaker guarantees |

## Consequences

- Positive: Every line of the core is easy to read and audit; guarantees are directly testable; switching providers only requires an adapter.
- Negative: Advanced orchestration features (multi-day durable execution engines or visual graph editors) must be built when needed or layered on top.
- Verification: `dependencies` in `pyproject.toml` contains only `pydantic`; enforced by `tests/test_architecture.py`.
