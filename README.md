# AI Agent Harness

**A small Python harness for AI agents where the model proposes and your code decides.**

It runs the agent loop and enforces the rules around it. Every run ends with a status. Tools run only through one gate. Irreversible actions
always wait for a human. State survives crashes and several workers. Nothing sensitive leaks into logs.
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
- **Humans stay in control:** an irreversible tool call always pauses the run, and so does a write at the `suggest` and `act_with_approval` levels
  (the default, `read_only`, denies writes). The approval is tied to the exact request that was shown, so an old dialog can never approve a later action.
- **Guardrails fail closed:** a check that crashes blocks the action. It never lets it through by accident.
- **Crash-safe, several workers:** state is saved before every tool call, in a file or in PostgreSQL, and two workers cannot resume the same run.
  An approval can arrive days later in another process.
- **Safe logs:** events carry only metadata, never prompts, arguments, or results. Limits on steps, tokens, and time, and a kill switch, hold across restarts.
- **Tested:** [`docs/guarantees.md`](docs/guarantees.md) lists all 16 guarantees and the test for each, including a test that injects random failures.

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
| Models | `AnthropicProvider` (Claude), a router with retries, a timeout, and fallback models, `FakeProvider` for tests | Adapters for other providers (a small contract, one reference adapter) |
| Tools | `AskUser`, `ReadFile` (secrets denied by default), `@tool(risk=…)` for plain functions, the execution gate (also usable alone), approvals | Your tools: a Pydantic model, a risk level, a timeout |
| Guardrails | `ProtectedPaths`, `SecretGuard`, `EgressGuard`, `WeakeningGuard`, `KillSwitch`, fail-closed hooks (pattern checks, see [limits](docs/known-limitations.md#built-in-guards-are-heuristics)) | Your policies as `Hook` classes |
| State | In-memory, file (POSIX), and PostgreSQL stores with `list_runs` and deletion | Encryption, retention, and access rules for your data |
| Operations | Metadata events, `log_listener`, `Limits`, crash recovery | Tracing, cost tracking, your UI for approvals |

Not built yet: streaming, MCP, structured final output, parallel tool calls ([`docs/next-steps.md`](docs/next-steps.md)).

## Install

```bash
pip install "ai-agent-harness @ git+https://github.com/appamass/ai-agent-harness@v0.7.0"
```

Python 3.11+. The only runtime dependency is Pydantic 2. Optional extras: `[anthropic]` (Claude), `[postgres]` (PostgreSQL store).
Pin a release tag (for example `@v0.7.0`) and commit your lockfile: before 1.0, a minor version may contain breaking changes ([`CHANGELOG.md`](CHANGELOG.md)).
What has been run against a real model, and what has not: [`docs/known-limitations.md`](docs/known-limitations.md).

## Documentation

| Read | When |
|---|---|
| [`docs/guide.md`](docs/guide.md) | You want to build: tools, guardrails, providers, stores (tested examples) |
| [`docs/how-it-works.md`](docs/how-it-works.md) | You want to understand the run loop, statuses, the gate, and the code map |
| [`docs/guarantees.md`](docs/guarantees.md) | You want proof: every guarantee and the test behind it |
| [`docs/use-cases/`](docs/use-cases/README.md) | You want full scenarios: assistant with a bank account, coding agents, support, incident responder |
| [`docs/first-real-run.md`](docs/first-real-run.md) | You connect a real model for the first time |
| [`docs/production-checklist.md`](docs/production-checklist.md) · [`docs/known-limitations.md`](docs/known-limitations.md) · [`SECURITY.md`](SECURITY.md) | You go to production |
| [`docs/adr/`](docs/adr/README.md) · [`docs/next-steps.md`](docs/next-steps.md) · [`HARNESS_TAXONOMY.md`](HARNESS_TAXONOMY.md) | You want the decisions, the roadmap, and the concepts |
| [`llms.txt`](llms.txt) · [`AGENTS.md`](AGENTS.md) | You are an AI agent: a compact index, and the rules for changing this repository |

## Contributing

Humans and AI coding agents are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) first: every change needs a test, and the checks (`ruff`, `mypy --strict`,
`pyright --strict`, `pytest`) must pass. Report vulnerabilities privately as described in [`SECURITY.md`](SECURITY.md).

## License

[MIT](LICENSE)
