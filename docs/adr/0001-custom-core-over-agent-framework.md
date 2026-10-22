# 0001 – Own Core Instead of an Agent Framework

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-15
- **Affects:** Whole package, dependencies

## Context

We want independence from model vendors and full control of every run. The loop itself is small; the hard parts are persistence, approvals, and failure paths.

## Decision

A pure Python core with **Pydantic as the only runtime dependency**. No agent framework. Vendor SDKs live in optional provider adapters, database drivers in optional stores.

## Rejected

- LangGraph, Google ADK: they dictate the architecture and make strict guarantees harder to enforce.
- No Pydantic at all: more code, weaker validation.

## Consequences

Orchestration features (durable engines, graph editors) must be built or layered on top. Enforced by `tests/test_architecture.py`.
