# Documentation

Start with the [README](../README.md) for the 30-second overview. Then pick what you need:

| Read | Contents | Audience |
|---|---|---|
| [`guide.md`](guide.md) | How to build with it: tools, runtime `deps`, guardrails, providers, context sources, events, checkpoint stores. Every example runs in CI | Developers |
| [`how-it-works.md`](how-it-works.md) | Code map, the run loop, statuses and `resume`, the tool execution gate, the autonomy matrix, taxonomy mapping | Developers, reviewers |
| [`guarantees.md`](guarantees.md) | The 16 guarantees, each with the test that enforces it | Security, architecture, auditors |
| [`use-cases/`](use-cases/README.md) | Six end-to-end production scenarios; all code blocks run in CI | Teams adopting the harness |
| [`first-real-run.md`](first-real-run.md) | Runbook for the first run against a real model, and the results of the runs so far | Engineers connecting a model |
| [`production-checklist.md`](production-checklist.md) | What to decide and configure before you go live | Operations, engineers |
| [`known-limitations.md`](known-limitations.md) | Project status and every known boundary, with what to do instead | Everyone |
| [`adr/`](adr/README.md) | Architecture Decision Records: context, decision, rejected alternatives | Anyone changing the core |
| [`next-steps.md`](next-steps.md) | Roadmap with a definition of done for each step | Planning, contributors |
| [`../examples/`](../examples/README.md) | A runnable, tested agent you can copy | Developers |
| [`../HARNESS_TAXONOMY.md`](../HARNESS_TAXONOMY.md) | Concepts: agent patterns mapped to classical software engineering | Onboarding, architecture |
| [`../SECURITY.md`](../SECURITY.md) | Security guarantees, operational boundaries, vulnerability reporting | Security, operations |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Contributor workflow, checks, and the AI coding agent setup | Contributors |
| [`../AGENTS.md`](../AGENTS.md) | Rules, commands, and invariants for AI coding agents | Claude Code, Codex, Cursor, Copilot, … |
| [`../CHANGELOG.md`](../CHANGELOG.md) | Release notes and checkpoint schema versions | Anyone upgrading |
| [`../llms.txt`](../llms.txt) | Structured index for AI crawlers and language models | AI agents and crawlers |

## Where things live in the build and review workflow

```
AGENTS.md                     Rules for all coding agents (single source of truth)
CLAUDE.md                     @AGENTS.md + Claude-specific notes
.claude/
├── settings.json             Permissions: check commands allowed, config edits ask first, secrets blocked
├── skills/                   Repeatable playbooks (add-tool, add-guardrail, add-provider-adapter, write-adr, verify-change)
└── agents/architecture-reviewer.md   Independent review before a change counts as complete
docs/                         This folder; `# executable` blocks and links are tested (tests/test_docs.py, tests/test_links.py)
tests/fixtures/               Saved checkpoints from earlier schema versions – never modify (ADR 0010)
.github/workflows/ci.yml      Automated checks on every push and pull request
```

**Core principle:** Markdown rules give context to AI agents; they are not enforcement. Hard boundaries come from permissions
(`.claude/settings.json`), automated CI checks, and human review.
