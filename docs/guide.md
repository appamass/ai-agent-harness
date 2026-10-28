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
Reusable guardrails that ship with the library: `ProtectedPaths` (keeps agents away from tests and CI config, including via symlinks when `root=` is set),
`SecretGuard` (blocks tool arguments and results that contain common API tokens, private keys, and URL credentials; `redact()` is a separate
helper for your own code), `EgressGuard` (HTTPS plus an outbound domain allowlist for URLs in tool arguments and model text), `WeakeningGuard`
(blocks `# type: ignore`, `# noqa`, `pytest.mark.skip`, etc. on writes), and `KillSwitch` (emergency stop).
These guards are fast pattern checks, not complete protection. What they miss is listed in [`known-limitations.md`](known-limitations.md#built-in-guards-are-heuristics).

## Modular bricks (pick and choose)

You do not have to adopt the full agent loop to use this library. Every layer works standalone — you can use only the guardrails, only the tool
execution gate, or a 5-line script agent:

```python
# executable
from ai_agent_harness import (
    EgressGuard,
    ProtectedPaths,
    Risk,
    SecretGuard,
    ToolCall,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
    WeakeningGuard,
    as_untrusted,
    tool,
)

# 1. Guardrails standalone (plain synchronous checks, no RunState or Agent needed)
secrets = SecretGuard()
assert secrets.find_secret("token = hf_abcdefghijklmnopqrstuvwxyz1234") == "api_token"
assert "[REDACTED]" in secrets.redact("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload")

egress = EgressGuard(allowed_domains=["api.example.com", ".trusted.org"])
assert egress.check_url("https://api.example.com/v1/status") is None
assert egress.check_url("https://169.254.169.254/latest/meta-data/") is not None

paths = ProtectedPaths()
assert paths.is_protected("tests/test_auth.py") and not paths.is_protected("src/auth.py")
assert WeakeningGuard().check_text("x = 1  # type: ignore") is not None


# 2. Function tools (@tool, `risk` is always required) + standalone execution gate (ToolExecutor)
@tool(risk=Risk.READ, untrusted_output=True)
def fetch_note(topic: str) -> str:
    """Fetches an external note for a topic."""
    return f"Note on {topic}"


gate = ToolExecutor([fetch_note], hooks=[secrets, egress])
outcome = asyncio.run(gate.execute(ToolCall(id="c1", name="fetch_note", arguments={"topic": "routing"})))
assert isinstance(outcome, ToolResult) and not outcome.is_error
assert outcome.content.startswith('<untrusted source="fetch_note">')

# Pin tool digests (name, description, schema, risk, timeout, untrusted_output) to notice a changed tool at startup
# or later with `pinned.verify()`
pinned = ToolRegistry([fetch_note], expected_digests={"fetch_note": fetch_note.digest()})
assert "fetch_note" in pinned.digests()


# 3. Five-line agent with Agent.create() and an async model function
async def tiny_model(request):
    return f"Handled: {request.messages[-1].text}"


mini = Agent.create(tiny_model, "You are concise.", tools=[fetch_note], hooks=[secrets, egress])
mini_state = asyncio.run(start(mini, "hello"))
assert mini_state.status is RunStatus.DONE and mini_state.last_text == "Handled: hello"
```

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
  ```

  Checkpoints hold full conversations as plaintext JSON by default: use TLS, a least-privilege role, encryption at rest, or pass `codec=AesGcmCodec(...)` (`pip install "ai-agent-harness[encryption]"`). Details: [ADR 0018](adr/0018-postgres-checkpoint-store.md), [ADR 0029](adr/0029-checkpoint-encryption-and-retention.md).
- **Your own store:** implement `save(state)` with optimistic revision checking and `load(tenant_id, run_id)`, and **always** restore through
  `restore_state(raw, tenant_id=tenant_id, run_id=run_id)` (or `load_state`), never directly with `RunState.model_validate_json`. That gives you schema migrations,
  ownership checks, and protocol checks ([ADR 0010](adr/0010-checkpoint-schema-versioning.md)).

### Retention and Deletion

Checkpoints accumulate full conversations over time. Define data retention policies to prune completed runs and comply with privacy regulations (e.g. GDPR right to erasure):

```python
# executable
import asyncio
from datetime import timedelta
from pathlib import Path
import tempfile

from ai_agent_harness import RunState, RunStatus, SqliteCheckpointStore


async def retention_demo() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SqliteCheckpointStore(Path(tmp) / "runs.sqlite")
        await store.setup()

        # 1. Scheduled retention: prune finished runs older than a threshold (e.g. 30 days).
        # Running and waiting runs are NEVER deleted by purge_finished.
        purged = await store.purge_finished(older_than=timedelta(days=30))
        assert purged == 0

        # 2. Deletion on demand (GDPR erasure, tenant offboarding):
        # delete_tenant deletes all runs of a tenant; delete_run deletes a single run.
        st = RunState(agent_name="a", tenant_id="tenant-42", run_id="run-1")
        st.add_user_text("hi")
        await store.save(st)
        assert await store.delete_tenant("tenant-42") == 1
        assert await store.load("tenant-42", "run-1") is None


asyncio.run(retention_demo())
```

- **Scheduled purge:** Call `purge_finished(older_than=timedelta(days=...))` regularly (e.g. daily cron job). It evaluates time against the database clock to avoid worker clock skew.
- **Manual erasure:** Call `delete_tenant(tenant_id)` or `delete_run(tenant_id, run_id)`. These delete runs in any status because a human made an explicit decision.
- **Storage backends:** `SqliteCheckpointStore` and `PostgresCheckpointStore` support `purge_finished`. `JsonFileCheckpointStore` stores runs under `<dir>/<tenant>/<run>.json`. The in-memory store keeps nothing past process termination.
- **Encryption at rest:** Pass `codec=AesGcmCodec(...)` (`pip install "ai-agent-harness[encryption]"`) to encrypt state payloads with AES-256-GCM. Metadata columns (`status`, `updated_at`) remain visible for queries and retention.
- **Logs and backups:** Metadata events are content-free, but application tracebacks may log local context. Destroying encryption keys renders old checkpoints permanently unreadable even inside existing database backups.

## More building blocks

**OpenAI and compatible servers:** `pip install "ai-agent-harness[openai]"`, then `OpenAIProvider("gpt-4o")`. For Ollama, vLLM, or Gemini's compatible endpoint
pass `base_url=` and `structured_output=False` (they often reject `json_schema`); models that reject `max_tokens` need
`max_tokens_param="max_completion_tokens"`.

**A typed result, live text, and a SQLite store** (all run offline with `FakeProvider`):

```python
# executable
import asyncio
import tempfile
from pathlib import Path

from pydantic import BaseModel

from ai_agent_harness import Agent, SqliteCheckpointStore, TextDelta, stream_start
from ai_agent_harness import RunStatus as Status
from ai_agent_harness import start as start_run
from ai_agent_harness.testing import FakeProvider, text_response


class Weather(BaseModel):
    city: str
    celsius: float


async def typed_and_streamed() -> None:
    with tempfile.TemporaryDirectory() as tmp, SqliteCheckpointStore(Path(tmp) / "runs.sqlite") as sqlite_store:
        await sqlite_store.setup()
        provider = FakeProvider([text_response('{"city": "Bern", "celsius": 3.5}'), text_response("live words")])
        agent = Agent.create(provider, "You report weather.", checkpoints=sqlite_store)

        # output_model: the run ends with a validated Weather (resume needs the same model again)
        typed = await start_run(agent, "Weather in Bern?", output_model=Weather)
        assert typed.status is Status.DONE and typed.output_as(Weather) == Weather(city="Bern", celsius=3.5)

        # stream_start: text arrives while the model answers. Close handles with `async with`.
        async with stream_start(agent, "Say something") as handle:
            words = [item.text async for item in handle if isinstance(item, TextDelta)]
        assert "".join(words).strip() == "live words"


asyncio.run(typed_and_streamed())
```

Streamed text is unfiltered live output: `after_model` guards see a response only when it is complete (see [limits](known-limitations.md#streaming-structured-output-mcp)).

**Tool output size:** results above `max_result_chars` keep their head and their tail (`truncation_mode="head_and_tail"`, the default; `"head"` keeps only the start).
Set it on `Agent`, `ToolExecutor`, or a single tool (`@tool(max_result_chars=...)`; a tool can lower the agent's limit, not raise it).

**Tracing and MCP:** `agent.events.subscribe(OpenTelemetryListener())` (`pip install "ai-agent-harness[opentelemetry]"`) emits GenAI spans and metrics from metadata only.
`await load_mcp_tools(MCPServerConfig(server_name=..., command=[...], tool_risks={"tool": Risk.READ}))` turns MCP tools into gated tools; every tool
needs a risk you set locally ([ADR 0024](adr/0024-mcp-integration-and-trust-model.md)).

## Sub-agents

A specialist agent with its own model, prompt, tools, and limits becomes one read tool of another agent with `sub_agent(...)`. The specialist
must be `read_only` and must not offer `ask_user`: a delegated run can never write under the caller's rights and never waits for a human.
Its answer comes back framed as untrusted data. Any other ending (`limit_reached`, `failed`, a question) comes back as a tool error, so
the caller's run always goes on ([ADR 0030](adr/0030-sub-agents-as-read-only-tools.md)).

```python
# executable
from ai_agent_harness import Limits, sub_agent


class TopicInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=3, max_length=200)


class SearchDocs(BaseTool[TopicInput]):
    name: ClassVar[str] = "search_docs"
    description: ClassVar[str] = "Searches the internal runbooks."
    input_model: ClassVar[type[BaseModel]] = TopicInput
    risk: ClassVar[Risk] = Risk.READ

    async def run(self, args: TopicInput, state: RunState) -> str:
        return f"Runbook RB-12 covers '{args.topic}': the pool size is 20."


specialist = Agent(
    name="doc-search",
    router=ModelRouter(
        FakeProvider([
            tool_response(("search_docs", {"topic": "checkout pool"})),
            text_response("RB-12: the checkout pool size is 20."),
            tool_response(("search_docs", {"topic": "incident history"})),
            tool_response(("search_docs", {"topic": "older incidents"})),  # second sub-run: out of steps
        ])
    ),
    context=ContextBuilder("You search the runbooks and answer in one sentence."),
    tools=ToolRegistry([SearchDocs()]),
    limits=Limits(max_steps=2, max_seconds=60),
)
research = sub_agent(specialist, name="research", description="Has a specialist search the runbooks.")

orchestrator = make_agent(
    [
        tool_response(("research", {"task": "What is the checkout pool size?"})),
        tool_response(("research", {"task": "Summarize the full incident history."})),
        text_response("The pool size is 20; the full history was not found."),
    ],
    research,
)
state = asyncio.run(start(orchestrator, "Prepare the incident brief", tenant_id="tenant-42"))
found, gave_up = [result for message in state.messages for result in message.tool_results]
assert state.status is RunStatus.DONE
assert "pool size is 20" in found.content and found.content.startswith('<untrusted source="research">')
assert gave_up.is_error and "limit_reached" in gave_up.content
```

The sub-run uses the caller's `tenant_id` and an ID that starts with `sub-` and is derived from the caller's agent, run, step, and
position in the tool batch, so a resumed caller names the same sub-run again.
Tokens, events, limits, and the kill switch belong to the specialist: subscribe your cost listener to both agents. A specialist that must write is a separate run
that your own code starts, with its own approvals.

**Ending runs that nobody will answer:** `cancel(agent, state)` ends a waiting run as `cancelled` without calling the model or any tool
([ADR 0031](adr/0031-cancel-waiting-or-crashed-runs.md)). Finished results stay; other open calls are answered "not executed". The sub-agent
tool does this itself when a sub-run would wait. A sub-run that a crash of the caller interrupted stays `running`; it is surely dead once it
is older than the tool's `timeout_s`, so let the job that recovers the caller close it as well:

```python
# executable
from datetime import UTC, datetime, timedelta

from ai_agent_harness import SqliteCheckpointStore, cancel


async def close_crashed_sub_runs(specialist, store, tenant_id: str, older_than: timedelta) -> int:
    """Cancels `sub-` runs of `specialist` that a crash left `running` for longer than `older_than` (the tool's timeout_s)."""
    cutoff = datetime.now(UTC) - older_than
    closed = 0
    # Newest first and the crashed ones are the oldest: ask for the maximum (10,000) and run the job repeatedly
    for info in await store.list_runs(tenant_id, status=RunStatus.RUNNING, limit=10_000):
        if info.run_id.startswith("sub-") and info.updated_at < cutoff:
            crashed = await store.load(tenant_id, info.run_id)
            if crashed is not None and crashed.agent_name == specialist.name:
                await cancel(specialist, crashed, recover=True)  # interrupted calls are reported as "outcome unknown"
                closed += 1
    return closed


async def recovery_job() -> int:
    with SqliteCheckpointStore(":memory:") as sub_runs:
        await sub_runs.setup()
        # updated_at comes from the store's clock (PostgreSQL: the database), so add a margin for clock skew
        older_than = timedelta(seconds=research.timeout_s) + timedelta(minutes=5)
        return await close_crashed_sub_runs(specialist, sub_runs, "tenant-42", older_than)


assert asyncio.run(recovery_job()) == 0  # nothing crashed here; tests/test_sub_agents.py runs it against a real crash
```

After `cancel`, the runs are finished, so `purge_finished` removes them with your other old runs.

## Where to go next

[`agent-package.md`](agent-package.md) to lay out an agent project · [`guarantees.md`](guarantees.md) for what is enforced and by which test · [`production-checklist.md`](production-checklist.md) before you go live.
