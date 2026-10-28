# Lay Out an Agent Project

The harness is a library. Your agent lives in your own project. This page shows a layout that keeps what the agent *is* under version
control, keeps what *happened* out of it, and keeps every hard rule in code. Nothing here is required by the harness: it is a convention
that fits the harness and the common formats (`SKILL.md` skills, a Markdown manifest with frontmatter).

## Three layers

| Layer | Question | Where it lives |
|---|---|---|
| **Definition** | What is this agent? Prompt, skills, tools, policy, limits | The agent package, in git, reviewed like code |
| **Runtime** | What happened? Run state, memories, traces, audit, costs | Checkpoint stores, your databases, event listeners. Never in the package |
| **System** | How do agents work together? | Code that composes agents: `sub_agent(...)` or your own routing |

Adding an agent adds a package. It does not add folders inside other packages.

## Layout

```text
my-service/
├── agents/
│   └── support/                    one agent package
│       ├── agent.md                manifest: frontmatter (id, version, description) and a short intro
│       ├── agent.py                builds the Agent: model, tools, autonomy, limits, hooks
│       ├── identity.md             role, tone, scope → the system prompt
│       ├── skills/<name>/SKILL.md  instructions loaded only when needed (Agent Skills format)
│       ├── references/             material the agent works with (documents, templates)
│       └── agents/<name>/          sub-agent packages with the same layout
├── docs/adr/                       why the agents are built this way
├── tests/                          does the code work?
└── evals/                          does the agent behave well?
```

Only `agent.md` and `agent.py` are needed. Add a folder when it has content; do not keep empty folders for later.

## Where each concern lives

| Concern | File | Harness piece |
|---|---|---|
| Model | `agent.py` | `ModelRouter`, `AnthropicProvider` / `OpenAIProvider`, `RetryPolicy`, fallback models |
| Loop and limits | `agent.py` | `start` / `resume`, `Limits`, `Autonomy`, `KillSwitch` |
| Capabilities | `agent.py`, `skills/` | `@tool(risk=…)`, `BaseTool`, `ToolRegistry`, `load_mcp_tools`, a `ContextSource` for skills |
| Memory | `agent.py` | a `ContextSource` that reads it (scoped by `state.tenant_id`); run state in a `CheckpointStore` |
| Identity | `identity.md` | `ContextBuilder(system_prompt)` |
| Guardrails | `agent.py` | `autonomy`, `allowed_tools`, tool `risk`, `Hook` classes, `SecretGuard`, `EgressGuard`, `ProtectedPaths` |
| Sub-agents | `agents/<name>/` | `sub_agent(...)` ([ADR 0030](adr/0030-sub-agents-as-read-only-tools.md)) |
| Decisions | `docs/adr/` (project) | ADRs, as in this repository |

## Rules

1. **Hard rules are code, not prose.** A sentence in `identity.md` is a wish. Risk levels, `autonomy`, `allowed_tools`, and hooks are
   enforced by the execution gate. Do not load permissions from a file that a loader trusts: whoever can edit the file can raise the
   autonomy. If you keep policy in a file, let `agent.py` check it against a ceiling set in code.
2. **Runtime data stays out of the package.** Checkpoints, memory data, traces, audit logs, and metrics go to stores and event listeners.
   Events carry metadata only.
3. **Load only what the task needs.** Put skill names and descriptions in the prompt, and the full text behind a read tool.
4. **A change to the package changes behavior.** Review it, run the evals, and raise `version` in `agent.md`.
5. **Tests and evals live next to the packages, not inside them.** The agent is what they test.

## Small and large models

The same package can run on a small local model or a large hosted one. What changes is the model profile in `agent.py`, never the
safety: risk levels, autonomy, approvals, and guards do not depend on the model, so a weaker model can give worse answers but cannot
do anything the policy, approvals, and guards do not allow. Within what they allow, a weaker model errs more often and is easier
to mislead, so give it less. A small model usually needs more help from the code:

- fewer tools at a time (`allowed_tools`), each with a flat schema and a clear description;
- a typed result (`output_model=...`) instead of free text, so the code validates the answer;
- tighter `Limits` (steps, context size) and a router with a larger model as fallback;
- if it needs other wording, a second prompt file per model (`identity.small.md`), chosen in `agent.py` and tested by the same evals.

Let the evals decide which model a package runs on, not the model's reputation (`HARNESS_TAXONOMY.md`, section 8).

## Letting an agent improve itself

An agent may propose changes to its own package (a better skill, a new tool description), but never apply them itself:

- the tool that writes a proposal has `Risk.WRITE` or `Risk.IRREVERSIBLE`, so a human approves it (`suggest` or `act_with_approval`);
- `ProtectedPaths` keeps `tests/`, `evals/`, and CI out of reach of the agent's write tools by default; add the packages with
  `ProtectedPaths(patterns=(*DEFAULT_PROTECTED, "agents", "agents/*"))`. `WeakeningGuard` blocks attempts to switch checks off;
- the proposal goes through the same path as any change: review, tests, evals, then a new `version` in `agent.md`.

Self-improvement can be automated up to the proposal; the change to the trusted definition stays a reviewed commit.

## Example

A package written to a temporary folder, then loaded: the identity becomes the system prompt, skills are listed by name and loaded on
demand, and a sub-agent package becomes a read tool. Everything that decides what may run is in code.

```python
# executable
import asyncio
import tempfile
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ai_agent_harness import (
    Agent,
    BaseTool,
    ContextBuilder,
    Limits,
    ModelRouter,
    Risk,
    RunState,
    RunStatus,
    ToolError,
    ToolRegistry,
    start,
    sub_agent,
)
from ai_agent_harness.testing import FakeProvider, text_response, tool_response


def front_matter(path: Path) -> dict[str, str]:
    """`key: value` lines between the first two `---` lines. Use a YAML parser in your project if you need more."""
    lines = path.read_text(encoding="utf-8").splitlines()
    end = lines.index("---", 1)
    return dict(line.split(": ", 1) for line in lines[1:end])


def body(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return text.split("---", 2)[2].strip() if text.startswith("---") else text.strip()


class SkillIndex:
    """Lists each skill by name and description. The full text stays out of the prompt until it is loaded."""

    def __init__(self, folder: Path) -> None:
        self.files = {front_matter(p)["name"]: p for p in sorted(folder.glob("*/SKILL.md"))} if folder.is_dir() else {}

    async def render(self, state: RunState) -> str | None:
        if not self.files:
            return None
        lines = [f"- {name}: {front_matter(path)['description']}" for name, path in self.files.items()]
        return "Skills you can load with load_skill:\n" + "\n".join(lines)


class SkillName(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9-]{1,64}$")


class LoadSkill(BaseTool[SkillName]):
    name: ClassVar[str] = "load_skill"
    description: ClassVar[str] = "Loads the full instructions of one listed skill."
    input_model: ClassVar[type[BaseModel]] = SkillName
    risk: ClassVar[Risk] = Risk.READ

    def __init__(self, index: SkillIndex) -> None:
        self.index = index

    async def run(self, args: SkillName, state: RunState) -> str:
        path = self.index.files.get(args.name)  # only files found at startup, never a path from the model
        if path is None:
            raise ToolError(f"Unknown skill '{args.name}'")
        return body(path)


def build_agent(package: Path, router: ModelRouter, *tools: BaseTool, limits: Limits | None = None) -> Agent:
    """What each `agent.py` does: read the prose, decide everything else in code. Autonomy stays read_only here."""
    manifest = front_matter(package / "agent.md")
    skills = SkillIndex(package / "skills")
    registry = ToolRegistry([LoadSkill(skills), *tools] if skills.files else list(tools))
    return Agent(
        name=manifest["id"],
        router=router,
        context=ContextBuilder(body(package / "identity.md"), sources=[skills]),
        tools=registry,
        limits=limits or Limits(max_steps=10, max_seconds=120),
    )


FILES = {
    "support/agent.md": "---\nid: support\nversion: 1.2.0\ndescription: Answers customer questions.\n---\n# Support agent\n",
    "support/identity.md": "You are the support agent of a small shop. Answer in two sentences.",
    "support/skills/refunds/SKILL.md": (
        "---\nname: refunds\ndescription: How to handle a refund request.\n---\n"
        "Check the order date first. Look up the refund policy before you promise anything."
    ),
    "support/agents/policy-lookup/agent.md": "---\nid: policy-lookup\nversion: 1.0.0\ndescription: Finds policy text.\n---\n",
    "support/agents/policy-lookup/identity.md": "You quote the shop policy that answers the question.",
}

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    for relative, text in FILES.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text, encoding="utf-8")

    lookup = build_agent(
        root / "support/agents/policy-lookup",
        ModelRouter(FakeProvider([text_response("Policy 4.1: refunds within 30 days of delivery.")], id="lookup")),
        limits=Limits(max_steps=3, max_seconds=60),
    )
    support_model = FakeProvider([
        tool_response(("load_skill", {"name": "refunds"})),
        tool_response(("policy-lookup", {"task": "What is the refund window?"})),
        text_response("You can get a refund within 30 days of delivery."),
    ])
    support = build_agent(
        root / "support",
        ModelRouter(support_model),
        sub_agent(lookup, description="Finds the shop policy that answers a question."),
    )
    state = asyncio.run(start(support, "Can I still return my order from last week?", tenant_id="shop-1"))

system = support_model.requests[0].system
assert system.startswith("You are the support agent") and "- refunds: How to handle a refund request." in system
assert "Check the order date first" not in system  # loaded only on demand
skill, policy = [result for message in state.messages for result in message.tool_results]
assert "Check the order date first" in skill.content
assert "Policy 4.1" in policy.content
assert state.status is RunStatus.DONE
```

## Rebuilding this in another stack

The layout does not depend on Python. The same split works with LangGraph, Pydantic AI, or your own loop: prose files feed the prompt,
and one gate in code decides what runs. See the [ai-kit recipes](../ai-kit/skills/ai-agent-harness/references/adopt-in-x.md).
