# 0008 – No Placeholder Modules

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-15
- **Affects:** Repository structure

## Context

An early draft of this repository contained 37 out of 68 Python files with no working logic (empty
packages, stubs raising `NotImplementedError`, and protocols without implementations). They created a
false sense of completeness and added maintenance overhead.

## Decision

New modules, packages, and interfaces may only be added when accompanied by at least one real,
working implementation and automated tests. Planned extension points are documented in `docs/how-it-works.md`
(Mapping to the harness taxonomy) and demonstrated as executable patterns in `docs/use-cases/`.

## Consequences

- Positive: Every file in `ai_agent_harness/` represents working, tested production code.
- Negative: The directory tree alone does not advertise every future extension — `docs/how-it-works.md` and `docs/use-cases/` serve that role instead.
- Verification: Enforced by `tests/test_architecture.py` (no empty files or `NotImplementedError` stubs).
