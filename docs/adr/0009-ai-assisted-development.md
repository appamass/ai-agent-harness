# 0009 – AI-Assisted Development: AGENTS.md, Skills, Independent Reviewer

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-16
- **Affects:** `AGENTS.md`, `CLAUDE.md`, `.claude/`, `docs/`

## Context

This template is built and extended with AI coding agents (Claude Code, Codex, Cursor, etc.). The exact
same risks that the harness prevents at runtime — especially reward hacking (weakening tests or linter
rules to get a green build), violating invariants, or letting documentation drift — also apply to AI
coding agents working on this repository.

## Decision

- `AGENTS.md` is the tool-agnostic single source of truth for rules, commands, and non-negotiable
  invariants. `CLAUDE.md` imports it (`@AGENTS.md`) and adds only Claude-specific instructions.
- Repeatable engineering workflows live as step-by-step playbooks in `.claude/skills/` (`add-tool`,
  `add-guardrail`, `add-provider-adapter`, `write-adr`, `verify-change`).
- A read-only `architecture-reviewer` subagent audits changes independently before work is considered complete.
- `.claude/settings.json` pre-approves verification commands, requires human confirmation before editing
  `.github/`, `.claude/`, or `pyproject.toml`, and blocks access to `.env` and `secrets/`.
- Architectural decisions are recorded in `docs/adr/`; the `Accepted` status is set only by a human.
- Documentation code blocks (`docs/use-cases/`, `docs/first-real-run.md`) are executed by `tests/test_docs.py`
  so documentation cannot silently drift out of sync with the API.

## Alternatives

| Alternative | Why Rejected |
|---|---|
| Only `CLAUDE.md` | Other coding agents used by the team would not read the rules |
| Rules only in `README.md` | Not automatically loaded into every coding agent session |

## Consequences

- Positive: Every coding agent session starts with the exact same invariants; code changes are reviewed by an independent persona rather than self-graded.
- Negative: Markdown instructions are context, not hard enforcement. Hard boundaries require tool permissions (`settings.json`), CI checks (`tests/test_architecture.py`), and human code review.
