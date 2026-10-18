@AGENTS.md

## Additional Notes for Claude Code

- **Use skills** in `.claude/skills/` for recurring tasks: `add-tool`, `add-guardrail`,
  `add-provider-adapter`, `write-adr`, `verify-change`.
- **Before finishing** any change to `ai_agent_harness/`, invoke the `architecture-reviewer` subagent.
  It did not write the code and checks against invariants and ADRs — so the work does not grade itself.
- **Core changes** (`ai_agent_harness/core/`, `ai_agent_harness/runtime/`): plan first in plan mode and present the
  plan for review. Use the strongest available model for core changes.
- Permissions for verification commands are pre-approved in `.claude/settings.json`; changes to
  `.github/`, `.claude/`, and `pyproject.toml` always prompt for confirmation.
