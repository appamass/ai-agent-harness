# Build With It

How to add your own pieces. Every Python block marked as executable runs in CI, so these examples stay correct. Start with the
[repo assistant example](../examples/README.md) if you want a complete agent first.

A small helper used by the examples below (`FakeProvider` replays scripted model answers, so nothing here needs an API key):

```python
# executable
import asyncio
from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ai_agent_harness import (
    Agent,
    Autonomy,
    BaseTool,
    ContextBuilder,
    Decision,
    Hook,
    HookPipeline,
    InMemoryCheckpointStore,
    ModelRouter,
    Risk,
    RunState,
    RunStatus,
    ToolError,
    ToolInvocation,
    ToolRegistry,
    resume,
    start,
)
from ai_agent_harness.testing import FakeProvider, text_response, tool_response


def make_agent(script, *tools, **options):
    return Agent(
        name="guide",
        router=ModelRouter(FakeProvider(script)),
        context=ContextBuilder("You help with customer questions."),
        tools=ToolRegistry(list(tools)),
        **options,
    )
```

## A tool

A tool declares its name, a description the model reads, a Pydantic model for its arguments, a risk level, and a timeout. Arguments are validated
before your code runs; a wrong call goes back to the model as an error it can fix.

```python
# executable
CRM = {"C-1001": "Ada Lovelace, customer since 2021"}  # stands in for your database


class LookupInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: str = Field(description="Customer ID, e.g. 'C-1001'")


class LookupCustomer(BaseTool[LookupInput]):
    name: ClassVar[str] = "lookup_customer"
    description: ClassVar[str] = "Reads master data for a customer. Read-only, modifies nothing."
    input_model: ClassVar[type[BaseModel]] = LookupInput
    risk: ClassVar[Risk] = Risk.READ
    timeout_s: ClassVar[float] = 10.0

    async def run(self, args: LookupInput, state: RunState) -> str:
        summary = CRM.get(args.customer_id)  # scope real queries by state.tenant_id
        if summary is None:
            raise ToolError(f"Customer {args.customer_id} does not exist")  # this message goes to the model
        return summary


script = [tool_response(("lookup_customer", {"customer_id": "C-1001"})), text_response("Ada has been a customer since 2021.")]
state = asyncio.run(start(make_agent(script, LookupCustomer()), "Who is C-1001?"))
assert state.status is RunStatus.DONE
assert state.messages[2].tool_results[0].content == "Ada Lovelace, customer since 2021"

unknown = [tool_response(("lookup_customer", {"customer_id": "C-9999"})), text_response("I could not find that customer.")]
state = asyncio.run(start(make_agent(unknown, LookupCustomer()), "Who is C-9999?"))
assert state.messages[2].tool_results[0].is_error  # the model saw the error and answered anyway
```

Rules: `async` tools must **never block the event loop**; use `SyncTool` with `run_sync` for blocking I/O. Raise `ToolError` for messages meant for
the model; any other exception is logged and reported to the model by its class name only. Return `InputRequest(question=…)` to pause the run for a
clarifying question. Make write tools idempotent wherever possible. Declare the risk honestly: `READ`, `WRITE` (reversible), or `IRREVERSIBLE` (send,
delete, pay, publish).

## Runtime dependencies (`deps`)

A database session, HTTP client, or user token must not live on the tool instance and must never reach a checkpoint. Pass them per run:

```python
# executable
@dataclass
class Deps:
    user_token: str


class WhoAmIInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WhoAmI(BaseTool[WhoAmIInput]):
    name: ClassVar[str] = "who_am_i"
    description: ClassVar[str] = "Returns the signed-in user."
    input_model: ClassVar[type[BaseModel]] = WhoAmIInput
    risk: ClassVar[Risk] = Risk.READ

    async def run(self, args: WhoAmIInput, state: RunState) -> str:
        deps = state.deps_as(Deps)  # raises InvalidStateError if missing or of the wrong type
        return f"user for token ending {deps.user_token[-4:]}"


script = [tool_response(("who_am_i", {})), text_response("done")]
state = asyncio.run(start(make_agent(script, WhoAmI()), "who am I?", deps=Deps(user_token="secret-token-1234")))
assert state.status is RunStatus.DONE
assert "secret-token" not in state.model_dump_json()  # never serialized, so never in a checkpoint
assert state.deps is None  # and detached again when the run returns
```

When you resume later (possibly in another process), pass `deps=` again.

## A guardrail

A guardrail is a `Hook`. Override only the method you need. A hook that raises, or returns a decision that is not allowed at that point, **blocks**
(fail-closed). Here a payment above 500 needs a human:

```python
# executable
class PayInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount_eur: int = Field(gt=0, le=100_000)


class Pay(BaseTool[PayInput]):
    name: ClassVar[str] = "pay"
    description: ClassVar[str] = "Pays an amount in EUR."
    input_model: ClassVar[type[BaseModel]] = PayInput
    risk: ClassVar[Risk] = Risk.WRITE
    paid: ClassVar[list[int]] = []

    async def run(self, args: PayInput, state: RunState) -> str:
        self.paid.append(args.amount_eur)
        return "paid"


class MaxAmount(Hook):
    async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
        amount = getattr(invocation.args, "amount_eur", 0)
        if amount > 500:
            return Decision.require_approval(f"Amount {amount} EUR exceeds 500 EUR")
        return Decision()


hooks = HookPipeline([MaxAmount()])
small = asyncio.run(start(make_agent([tool_response(("pay", {"amount_eur": 50})), text_response("ok")], Pay(), autonomy=Autonomy.AUTONOMOUS, hooks=hooks), "pay 50"))
assert small.status is RunStatus.DONE and Pay.paid == [50]

big_agent = make_agent([tool_response(("pay", {"amount_eur": 900})), text_response("ok")], Pay(), autonomy=Autonomy.AUTONOMOUS, hooks=hooks)
big = asyncio.run(start(big_agent, "pay 900"))
assert big.status is RunStatus.WAITING_FOR_APPROVAL and Pay.paid == [50]  # nothing happened yet
done = asyncio.run(resume(big_agent, big, approved=True, pending_id=big.pending_id))
assert done.status is RunStatus.DONE and Pay.paid == [50, 900]
```

See `ai_agent_harness/core/hooks.py` for which decisions are allowed at each hook point (input, before/after model, before/after tool, run end).
Reusable guardrails that ship with the library: `ProtectedPaths` (keeps agents away from tests and CI config) and `KillSwitch` (emergency stop).

## A model provider

**Anthropic (Claude) is included:** `pip install "ai-agent-harness[anthropic]"`, then `AnthropicProvider("claude-haiku-5-5")` (it reads `ANTHROPIC_API_KEY`).
Put it into a `ModelRouter` to add retries, a timeout, and fallbacks:

```python
from ai_agent_harness import ModelRouter, RetryPolicy
from ai_agent_harness.providers.anthropic import AnthropicProvider

router = ModelRouter(
    AnthropicProvider("claude-haiku-5-5"),
    fallbacks=[MyOtherProvider()],                       # any object that follows the contract below
    retry=RetryPolicy(max_attempts=3, request_timeout_s=120),
)
```

For another provider, write a small class: an `id`, its `capabilities`, and `async complete(request) -> ModelResponse`. The full contract (error
classification, stop reasons, own versus foreign thinking blocks, malformed tool-call JSON, `Usage` semantics, prompt caching) is in the docstring of
`ai_agent_harness/providers/base.py`; `ai_agent_harness/providers/anthropic.py` and its tests are the reference. The smallest possible provider:

```python
# executable
from ai_agent_harness.core.types import Message, ModelRequest, ModelResponse, StopReason, TextBlock
from ai_agent_harness.providers.base import Capabilities


class EchoProvider:
    id = "echo"
    capabilities = Capabilities(tools=False, vision=False)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        question = request.messages[-1].text
        return ModelResponse(
            message=Message(role="assistant", content=(TextBlock(text=f"You said: {question}"),)),
            stop_reason=StopReason.END,
        )


agent = Agent(name="echo", router=ModelRouter(EchoProvider()), context=ContextBuilder("Repeat the question."))
state = asyncio.run(start(agent, "hello"))
assert state.status is RunStatus.DONE and state.last_text == "You said: hello"
```

## Context sources (memory, search, skills)

Anything that should appear in the system prompt implements `async render(state) -> str | None`. Always scope queries by `state.tenant_id`. Keep sources
stable between steps: a source that changes on every step defeats prompt caching.

```python
# executable
class Today:
    async def render(self, state: RunState) -> str | None:
        return "Today is 2026-10-18."


provider = FakeProvider([text_response("It is the 18th.")])
agent = Agent(name="ctx", router=ModelRouter(provider), context=ContextBuilder("You are brief.", sources=[Today()]))
asyncio.run(start(agent, "What day is it?"))
assert "Today is 2026-10-18." in provider.requests[0].system
```

## Events and logging

Subscribe a listener to follow a run: `agent.events.subscribe(listener)`. Listeners run inside the loop, so hand slow work to a queue. A failing
listener is logged and never breaks a run. **Events contain metadata only** ([ADR 0011](adr/0011-operational-limits-and-metadata-events.md)): read
human-facing content such as an approval preview from `RunState` (`state.status_reason`, `state.pending`).

```python
# executable
from ai_agent_harness import Event

seen: list[Event] = []


async def collect(event: Event) -> None:
    seen.append(event)


agent = make_agent([text_response("secret answer")])
agent.events.subscribe(collect)
asyncio.run(start(agent, "a private question"))
assert {e.type.value for e in seen} >= {"run_started", "model_request", "run_finished"}
assert all("private" not in str(e.data) and "secret" not in str(e.data) for e in seen)  # no content in events
```

`ai_agent_harness.observability.log_listener` writes events to the standard `logging` module.

## Checkpoint stores

A store keeps the state between steps. That is what lets a run wait days for an approval, survive a crash, and be resumed by another worker.

```python
# executable
store = InMemoryCheckpointStore()  # tests and single process; see below for durable stores
agent = make_agent([tool_response(("pay", {"amount_eur": 900})), text_response("ok")], Pay(), autonomy=Autonomy.AUTONOMOUS, hooks=hooks, checkpoints=store)
waiting = asyncio.run(start(agent, "pay 900", run_id="run-1"))
loaded = asyncio.run(store.load("default", "run-1"))  # what another worker would do
assert loaded is not None and loaded.status is RunStatus.WAITING_FOR_APPROVAL and loaded.pending_id == waiting.pending_id
```

- `JsonFileCheckpointStore(directory)`: one JSON file per run, atomic and locked. POSIX only; for development and a single host.
- **`PostgresCheckpointStore`** (`pip install "ai-agent-harness[postgres]"`): for several workers. Same optimistic locking in one atomic statement.

  ```python
  from ai_agent_harness.persistence_postgres import PostgresCheckpointStore

  store = PostgresCheckpointStore("postgresql://app@db/agents?sslmode=require")
  await store.setup()                                    # once at deploy time (or run schema_sql() in your migrations)
  agent = Agent(..., checkpoints=store)
  queue = await store.list_runs("tenant-42", status=RunStatus.WAITING_FOR_APPROVAL)   # approval queue
  await store.delete_tenant("tenant-42")                 # retention / erasure
  ```

  Checkpoints hold full conversations as plaintext JSON: use TLS, a least-privilege role, and encryption at rest. Details: [ADR 0018](adr/0018-postgres-checkpoint-store.md).
- **Your own store:** implement `save(state)` with optimistic revision checking and `load(tenant_id, run_id)`, and **always** restore through
  `restore_state(raw, tenant_id=tenant_id, run_id=run_id)`, never directly with `RunState.model_validate_json`. That gives you schema migrations,
  ownership checks, and protocol checks ([ADR 0010](adr/0010-checkpoint-schema-versioning.md)).

## Where to go next

[`how-it-works.md`](how-it-works.md) for the run loop and the gate · [`use-cases/`](use-cases/README.md) for complete, tested scenarios ·
[`first-real-run.md`](first-real-run.md) before you connect a real model · [`production-checklist.md`](production-checklist.md) before you go live.
