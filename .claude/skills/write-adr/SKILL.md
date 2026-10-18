---
name: write-adr
description: Create or supersede an Architecture Decision Record in docs/adr/. Use whenever a decision affects architecture, invariants, public API, security, dependencies, or a deliberate trade-off.
---

# Writing an ADR

## When an ADR Is Required

- An invariant in `AGENTS.md` is being added or changed.
- The public API (`ai_agent_harness/__init__.py`) changes.
- A new dependency, subpackage, or extension interface is introduced.
- A security or risk trade-off is made (e.g. classifying a write action at a lower risk level with compensating controls).
- A plausible alternative was seriously considered and rejected — so the discussion does not have to be repeated later.

Do not write an ADR for routine bug fixes, behavior-preserving refactorings, or documentation-only edits.

## Procedure

1. Find the next available four-digit number in `docs/adr/`.
2. Copy `docs/adr/0000-template.md` to `docs/adr/NNNN-short-title.md` and fill it in.
   Set the initial status to `Proposed`.
3. Write **Context** honestly: what problem and competing forces exist? List **Alternatives Considered**
   with the specific reason each was rejected. Include negative **Consequences** as well as positive ones.
4. Add the new ADR to the index table in `docs/adr/README.md`.
5. A human sets the status to `Accepted`. Coding agents never set `Accepted` on their own.
6. Superseding an older ADR: **never delete** the old ADR; update its status to `Superseded by NNNN`
   and reference the old ADR in the new one.
