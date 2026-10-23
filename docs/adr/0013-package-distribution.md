# 0013 – Distribution as a Versioned Package

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-16
- **Affects:** Distribution of `ai_agent_harness/`, `pyproject.toml`, releases

## Context

A copied repository needs every fix and hardening ported by hand to every project, and the most carefully tested parts drift apart.

## Decision

A versioned package (import name `ai_agent_harness`, ADR 0016), installed from pinned Git release tags of `github.com/appamass/ai-agent-harness` under the MIT license
(the install line is in the [README](../../README.md#install)).
Projects pin a tag and commit a lockfile. There is no PyPI release.

## Rejected

- Copying the repository: unavoidable drift.
- Git submodule or subtree: awkward pinning and upgrades.

## Consequences

Breaking changes to `ai_agent_harness/__init__.py` or checkpoint schemas need a version bump, an ADR, and a `CHANGELOG.md` entry. Fixes ship as patch releases to all projects.
