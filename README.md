# AI Agent Harness

**A small Python harness for AI agents where the model proposes and your code decides.**

A harness is the loop between a model and its tools. Most harnesses make that loop work; this one also decides whether each call may run.
It is not a coding agent like Claude Code or Codex. It is the base for agents inside your own Python services.

Every run ends with a status. Tools run only through one gate. Irreversible actions
never run without a human. State survives crashes and several workers. Events carry only metadata.
You bring the model, the tools, and the policy.

## Try it in 30 seconds (no API key, no cost)

```bash
git clone https://github.com/appamass/ai-agent-harness && cd ai-agent-harness
pip install -e .
python examples/repo_assistant.py --demo
```

An assistant reads a project, then **stops and asks you** before it writes anything (you type `y` or `n`):

```text
You: What is this project? Save a one-line summary as a note.

[run <id> is waiting for your approval]

The agent wants to do this:

Tool: save_note (Risk: write)
Arguments: {
  "name": "summary.md",
  "content": "Adder: a tiny library that adds two numbers."
}
Reason for approval: Policy requires approval

Approve this action? [y/N]
This is Adder, a tiny library that adds two numbers. The summary note exists only if you approved.

[done | run <id> | steps 4 | tokens 80 (in 40, out 40)]
```

With a real model: `pip install -e ".[anthropic]"`, `export ANTHROPIC_API_KEY=...` in your own shell, then `python examples/repo_assistant.py`
(if the extra or the key is missing, the example tells you what to do). With a real model, that line also shows how to continue the run later with
`--resume`. More in [`examples/`](examples/README.md).

## What you get

- **Every run ends with a status** (`done`, `blocked`, `failed`, `waiting_for_approval`, ...), even on outages, broken hooks, or cancellation.
- **Humans stay in control:** an irreversible tool call never runs without approval (it is denied below `act_with_approval` and pauses the run above),
  and so does a write at the `suggest` and `act_with_approval` levels (the default, `read_only`, denies writes). The approval is tied to the exact request that was shown, so an old dialog can never approve a later action.
- **Guardrails fail closed:** a check that crashes blocks the action. It never lets it through by accident.
- **Crash-safe, several workers:** state is saved before every tool call, in a file, SQLite, or PostgreSQL, and two workers cannot resume the same run.
  An approval can arrive days later in another process.
- **Safe logs:** events carry only metadata, never prompts, arguments, or results. Limits on steps, tokens, and time, and a kill switch, hold across restarts.
- **Tested:** [`docs/guarantees.md`](docs/guarantees.md) lists all 22 guarantees and the test for each, including a test that injects random failures.

## How to use this

An executable specification and reference implementation of an agent loop with enforced guarantees. Everyone is welcome to use it, whether as a whole or just for a snippet. The guarantees are enforced by tests; from v1 on, the code base is not promised to stay small or easy to audit in full.

- **Use:** As a pinned Git dependency for pure Python services needing crash safety, optimistic locking, and human approval. We deliberately do not publish to PyPI ([ADR 0013](docs/adr/0013-package-distribution.md)).
- **Copy:** Use or copy standalone bricks (`ToolExecutor`, `@tool(risk=…)`, `SecretGuard`, `RunState`; copy them together with `core/` and `policy.py`) ([ADR 0019](docs/adr/0019-modular-bricks-and-security-guards.md)).
- **Rebuild:** Apply our core invariants (bound `pending_id`, write-ahead checkpoints, outcome-unknown on crash) in LangGraph, Pydantic AI, or custom loops via [`ai-kit/`](ai-kit/skills/ai-agent-harness/SKILL.md).

## Hello agent

A complete run with a scripted model (this block runs in CI, so it stays correct):

```python
# executable
import asyncio

from ai_agent_harness import Agent, AskUser, ContextBuilder, ModelRouter, RunStatus, ToolRegistry, resume, start
from ai_agent_harness.testing import FakeProvider, text_response, tool_response

provider = FakeProvider([
    tool_response(("ask_user", {"question": "Which file should I read?"})),
    text_response("Thanks, all done."),
])
agent = Agent(
    name="demo",
    router=ModelRouter(provider),            # FakeProvider replays scripted answers; use a real provider in production
    context=ContextBuilder("You are a helpful assistant."),
    tools=ToolRegistry([AskUser()]),
)


async def main() -> None:
    state = await start(agent, "Read the file", tenant_id="tenant-42")
    assert state.status is RunStatus.WAITING_FOR_INPUT           # the agent asks a question and waits
    print("Question:", state.status_reason)
    state = await resume(agent, state, answer="report.txt", pending_id=state.pending_id)
    assert state.status is RunStatus.DONE and state.last_text == "Thanks, all done."
    print(state.status, state.last_text)


asyncio.run(main())
```

With a real model, replace the provider (needs `pip install "ai-agent-harness[anthropic]"` and `ANTHROPIC_API_KEY`):

```python
from ai_agent_harness.providers.anthropic import AnthropicProvider

agent = Agent(name="demo", router=ModelRouter(AnthropicProvider("claude-haiku-5-5")), context=ContextBuilder("You are helpful."))
```

## Included, and what you bring

| Part | Included | You bring |
|---|---|---|
| Models | `AnthropicProvider` (Claude), `OpenAIProvider` (GPT/Ollama/vLLM), router with retries, timeout, and fallback models, `FakeProvider` for tests | Adapters for other providers (a small contract, reference adapters) |
| Tools | `AskUser`, `ReadFile` (secrets denied by default), `@tool(risk=…)` for plain functions, the execution gate (also usable alone), approvals | Your tools: a Pydantic model, a risk level, a timeout |
| Guardrails | `ProtectedPaths`, `SecretGuard`, `EgressGuard`, `WeakeningGuard`, `KillSwitch`, fail-closed hooks (pattern checks, see [limits](docs/known-limitations.md#built-in-guards-are-heuristics)) | Your policies as `Hook` classes |
| State | In-memory, file (POSIX), SQLite (stdlib), and PostgreSQL stores with `list_runs`, retention purge, and `AesGcmCodec` | Keys, retention intervals, and access rules for your data |
| Operations | Metadata events, `log_listener`, `OpenTelemetryListener`, `Limits`, crash recovery | Your trace backend, cost tracking, your UI for approvals |

Context compaction, parallel read tools, streaming, MCP, and structured output exist; read their [limits](docs/known-limitations.md#streaming-structured-output-mcp) first.

## Install

```bash
pip install "ai-agent-harness @ git+https://github.com/appamass/ai-agent-harness@v0.8.0"
```

Python 3.11+. The only runtime dependency is Pydantic 2. Optional extras: `[anthropic]` (Claude), `[openai]` (OpenAI/compat), `[postgres]` (PostgreSQL store), `[encryption]` (AES-256-GCM), `[opentelemetry]` (tracing).
Pin a release tag (for example `@v0.8.0`) and commit your lockfile: before 1.0, a minor version may contain breaking changes ([`CHANGELOG.md`](CHANGELOG.md)).
What has been run against a real model, and what has not: [`docs/known-limitations.md`](docs/known-limitations.md).

## Documentation

| Read | When |
|---|---|
| [`docs/guide.md`](docs/guide.md) | You want to build: tools, guardrails, providers, stores, sub-agents (tested examples) |
| [`docs/agent-package.md`](docs/agent-package.md) | You lay out an agent project: what goes in files, what stays in code |
| [`docs/guarantees.md`](docs/guarantees.md) | You want proof: every guarantee and the test behind it |
| [`docs/production-checklist.md`](docs/production-checklist.md) · [`docs/known-limitations.md`](docs/known-limitations.md) | You go to production |
| [`docs/adr/`](docs/adr/README.md) · [`HARNESS_TAXONOMY.md`](HARNESS_TAXONOMY.md) | You want the decisions and the concepts |
| [`AGENTS.md`](AGENTS.md) | You are an AI agent, or you change this repository: invariants, commands, and working rules |

## Contributing

Humans and AI coding agents are welcome. Read [`AGENTS.md`](AGENTS.md) first: every change needs a test, and the checks (`ruff`, `mypy --strict`,
`pyright --strict`, `pytest`) must pass. Report vulnerabilities privately through GitHub's Private Vulnerability Reporting on this repository, never in a public issue; include the affected module and version and a minimal failing test.

## License

[MIT](LICENSE)
