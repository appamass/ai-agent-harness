# 0009 – AI-Assisted Development

- **Status:** Accepted (maintainer, 2026-10-18). Amended 2026-10-24: reviewer runs at `effort: high`
- **Date:** 2026-10-16
- **Affects:** `AGENTS.md`, `CLAUDE.md`, `.claude/`, `docs/`

## Context

The repository is built with AI coding agents. The risks the harness stops at runtime (weakening tests or linters to get green, breaking invariants, drifting docs) also apply to the agents that work on it.

## Decision

- `AGENTS.md` is the one tool-agnostic source for rules, commands, and invariants. `CLAUDE.md` imports it and adds only Claude-specific workflow.
- Repeatable workflows live in `.claude/skills/` (`add-tool`, `add-guardrail`, `add-provider-adapter`, `write-adr`, `verify-change`).
- A read-only `architecture-reviewer` subagent checks changes independently. It runs at `effort: high`: the review is the last check before a human sees the change, so depth matters more there than speed. The session effort stays the user's choice. The `CLAUDE_CODE_EFFORT_LEVEL` environment variable still wins, and a model that does not offer `high` falls back to its highest lower level.
- `.claude/settings.json` pre-approves verification commands, asks before editing `.github/`, `.claude/`, or `pyproject.toml`, and blocks `.env` and `secrets/`.
- The `Accepted` status of an ADR is set by a human. Code blocks in the docs run in `tests/test_docs.py`.

## Rejected

- Only `CLAUDE.md`: other agents would not read the rules.
- Rules only in `README.md`: not loaded into agent sessions.

## Consequences

Markdown is context, not enforcement. Hard limits come from permissions, CI (`tests/test_architecture.py`), and human review.
