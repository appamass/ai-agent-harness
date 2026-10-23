# 0008 – No Placeholder Modules

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-15
- **Affects:** Repository structure

## Context

An early draft had 37 of 68 Python files without working logic (empty packages, `NotImplementedError` stubs). They gave a false sense of completeness.

## Decision

New modules, packages, and interfaces need a real implementation and tests. Ideas for later are not kept as empty files; `HARNESS_TAXONOMY.md` only catalogues concepts.

## Consequences

The file tree does not show future extensions. Enforced by `test_no_placeholder_modules` in `tests/test_architecture.py` (no empty modules, no `NotImplementedError` stubs).
