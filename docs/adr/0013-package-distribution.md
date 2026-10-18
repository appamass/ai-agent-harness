# 0013 – Package Distribution: Versioned Package vs. Copied Template

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-16
- **Affects:** Distribution of `ai_agent_harness/` to downstream projects, `pyproject.toml`, release workflow

## Context

If this repository is **copied** into multiple downstream projects, every subsequent bug fix or security
hardening in the core (protocol invariants, write-ahead checkpoints, guardrails) must be manually backported
into every project — or gets missed. The exact components that are most rigorously verified here would
gradually diverge across projects.

## Options

| Option | Pros | Cons |
|---|---|---|
| **A: Copy repository** (status quo) | Zero infrastructure, full local freedom per project | Fixes must be ported to every repo; configuration and core drift is inevitable |
| **B: Versioned package + project template** | One fix, one version bump across all projects; clean boundary between core (`ai_agent_harness/`) and project code | Requires a package index (public PyPI or private registry) and strict SemVer / `CHANGELOG.md` discipline |
| **C: Git submodule / subtree** | No package registry needed | Clunky developer ergonomics; version pinning and upgrades are error-prone |

## Decision

**Option B, decided by the maintainer on 2026-10-18.** `ai-agent-harness` is published as a versioned
package on public PyPI under the MIT license (import name `ai_agent_harness`, ADR 0016), developed in the
public repository `github.com/appamass/ai-agent-harness`. Downstream projects depend on a pinned range
instead of copying the code.

## Recommendation (original proposal)

**Option B.** Publish `ai_agent_harness/` as the versioned package `ai-agent-harness` following Semantic Versioning
(on PyPI or an internal package registry such as GitHub Packages / Google Artifact Registry). Provide a
lightweight project template (e.g., via Copier) that scaffolds only the project-specific layer — custom
tools, hooks, evals, `AGENTS.md`, and CI — while depending on `ai-agent-harness>=0.4,<0.5`.

The repository is already structured for Option B: `ai_agent_harness/` is a self-contained package free of
project-specific code, the public API is explicitly exported in `ai_agent_harness/__init__.py`, checkpoint schemas
are versioned with forward migrations (ADR 0010), and all changes are tracked in `CHANGELOG.md`.

## Consequences of Option B

- Breaking changes to `ai_agent_harness/__init__.py` or checkpoint schemas require a major/minor version bump, an ADR, and an entry in `CHANGELOG.md`.
- Downstream projects commit a lockfile (e.g., `uv.lock`) for reproducible builds.
- Security and correctness fixes ship as patch releases to all projects simultaneously.
