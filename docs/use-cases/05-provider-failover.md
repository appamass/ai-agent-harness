# Use Case 5: Mid-Run Model Provider Failover

> **Difficulty: High.** A run starts on a reasoning model from Provider A that returns cryptographically
> signed thinking blocks. Halfway through the run, Provider A becomes overloaded and `ModelRouter` fails
> over to Provider B. Provider B does not recognize Provider A's thinking signatures and will return HTTP
> 400 if they are sent. On top of that, Provider B emits malformed JSON for a tool call on its first turn.

All Python blocks starting with `# executable` are executed by `tests/test_docs.py`. The providers here
are **simulated** via a lean `ContractAdapter` that follows the adapter contract in
`ai_agent_harness/providers/base.py`. Validating failover against live APIs requires real adapters backed by
recorded vendor responses (`docs/next-steps.md`, Step 1).

| Component | Status |
|---|---|
| `ModelRouter` with retry, backoff, and failover; `ThinkingBlock.provider_id` stamping; `without_foreign_thinking`; `ToolCall.arguments_error`; normalized `Usage` (cache & reasoning tokens) | ✅ In core harness |
| `ContractAdapter` (reference adapter enforcing the boundary contract) | 🧩 Reference implementation in this document |
| Live vendor SDK adapters for Providers A and B, wire-format translation, prompt caching | 🔲 Project-specific / Step 1 in `docs/next-steps.md` |

---

## 1. What Can Go Wrong During Mid-Run Failover

| # | Failure Mode | Example | Defense Layer |
|---|---|---|---|
| F1 | Foreign thinking signatures in history | Provider B receives Provider A's signed `ThinkingBlock` and rejects the request with HTTP 400 | Runner stamps `provider_id`; adapter filters history with `without_foreign_thinking` (✅ Core + adapter contract) |
| F2 | Run aborts on provider switch | State is lost or tools run twice | `ModelRouter` fails over within a single model call; history and tool results remain intact (✅ Core) |
| F3 | Malformed tool-call JSON | `{"service": "checkout"` missing closing brace | Adapter populates `ToolCall.arguments_error`; `ToolExecutor` returns the parse error to the model for self-correction without aborting the run (✅ Core + adapter contract) |
| F4 | Inaccurate token accounting | Cached input tokens billed as full-price input | Normalized `Usage` fields (✅ Core); adapter maps vendor counters at the boundary |
| F5 | Cost spike after failover | Provider A's warm prompt cache does not carry over to Provider B | ⚠️ Unavoidable across vendors; break down cost metrics by `provider` in event listeners |

---

## 2. Reference Adapter Following the Boundary Contract

A real provider adapter translates `ModelRequest` into the vendor's SDK wire format. This reference
adapter skips HTTP translation, enforces the mandatory boundary rules, and records the exact messages
it would have sent over the wire.

```python
# executable
import asyncio
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ai_agent_harness import (
    BaseTool,
    Capabilities,
    Message,
    ModelRequest,
    ModelResponse,
    Risk,
    RunState,
    StopReason,
    ToolCall,
    without_foreign_thinking,
)
from ai_agent_harness.core import ProviderError, TextBlock, ThinkingBlock, Usage


class ContractAdapter:
    """Enforces the provider adapter contract. `script` stands in for the live vendor API."""

    def __init__(self, provider_id: str, script: list[ModelResponse | Exception]) -> None:
        self._id = provider_id
        self._script = list(script)
        self.sent: list[tuple[Message, ...]] = []  # exact messages that would be sent to the vendor API

    @property
    def id(self) -> str:
        return self._id

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(tools=True)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        messages = without_foreign_thinking(request.messages, self._id)  # contract: strip foreign thinking blocks
        self.sent.append(messages)
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def answer(*blocks: TextBlock | ThinkingBlock | ToolCall, stop: StopReason, usage: Usage) -> ModelResponse:
    return ModelResponse(message=Message(role="assistant", content=blocks), stop_reason=stop, usage=usage)


def broken_json_call(call_id: str, name: str, parser_error: str) -> ToolCall:
    """How an adapter reports malformed tool-call JSON — never by raising ProviderError."""
    return ToolCall(id=call_id, name=name, arguments={}, arguments_error=parser_error)


class ServiceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service: str = Field(pattern=r"^[a-z0-9-]{1,40}$")


class Status(BaseTool[ServiceInput]):
    name: ClassVar[str] = "service_status"
    description: ClassVar[str] = "Checks the health status of a service."
    input_model: ClassVar[type[BaseModel]] = ServiceInput
    risk: ClassVar[Risk] = Risk.READ

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run(self, args: ServiceInput, state: RunState) -> str:
        self.calls.append(args.service)
        return f"{args.service}: healthy"
```

---

## 3. Scenario: Provider A Thinks, Fails, and Provider B Takes Over

| Step | Provider | Event | Mechanism Triggered | Outcome |
|---|---|---|---|---|
| 1 | A | Emits signed `ThinkingBlock` + tool call, 1,800 cached tokens | Runner stamps `provider_id="provider-a"` | Tool executes |
| 2 | A | Overloaded (HTTP 529) for the rest of the run | `ModelRouter`: tries A with retry on each turn, then fails over to B | Fails over to B three times |
| 3 | B | Receives conversation history | Adapter calls `without_foreign_thinking` | A's signature is **never** sent to B; tool call and result remain |
| 4 | B | Emits tool call with malformed JSON | `arguments_error` → `ToolExecutor` | Parse error returned to B as tool result; run continues |
| 5 | B | Emits valid tool call, then final answer | – | ✅ `done`, tool executed twice total |

```python
# executable
from ai_agent_harness import Agent, Autonomy, ContextBuilder, EventType, ModelRouter, RetryPolicy, RunStatus, ToolRegistry, start
from ai_agent_harness.core import Event


async def no_wait(_: float) -> None:
    return None


overloaded = ProviderError("529 overloaded", retryable=True)
provider_a = ContractAdapter("provider-a", [
    answer(
        ThinkingBlock(text="Check checkout status first.", provider_data={"signature": "sig-from-a"}),
        ToolCall(id="a1", name="service_status", arguments={"service": "checkout"}),
        stop=StopReason.TOOL_USE,
        usage=Usage(input_tokens=200, cache_read_tokens=1_800, output_tokens=120, reasoning_tokens=80),
    ),
    *[overloaded] * 6,  # A stays overloaded: each of the 3 subsequent turns tries A twice before failing over
])
provider_b = ContractAdapter("provider-b", [
    answer(broken_json_call("b1", "service_status", "Expecting ',' delimiter: line 1 column 21"),
           stop=StopReason.TOOL_USE, usage=Usage(input_tokens=2_100, output_tokens=30)),
    answer(ToolCall(id="b2", name="service_status", arguments={"service": "payments"}),
           stop=StopReason.TOOL_USE, usage=Usage(input_tokens=2_200, output_tokens=25)),
    answer(TextBlock(text="Both checkout and payments are healthy."),
           stop=StopReason.END, usage=Usage(input_tokens=2_300, output_tokens=15)),
])

status = Status()
switches: list[str] = []


async def watch(event: Event) -> None:
    if event.type is EventType.PROVIDER_FAILOVER:
        switches.append(str(event.data["to"]))  # metadata only: target provider ID, no raw error message


agent = Agent(
    name="failover",
    router=ModelRouter(provider_a, [provider_b], RetryPolicy(max_attempts=2, base_delay_s=1.0), sleep=no_wait),
    context=ContextBuilder("Check service health and report concisely."),
    tools=ToolRegistry([status]),
    autonomy=Autonomy.READ_ONLY,
)
agent.events.subscribe(watch)

state = asyncio.run(start(agent, "Are checkout and payments healthy?"))
assert state.status is RunStatus.DONE
assert switches == ["provider-b"] * 3      # stateless router: each model call tries primary A first
assert len(provider_a.sent) == 7           # 1 initial response + 3 turns * 2 attempts

# F1: Provider A's thinking block is preserved in RunState with its origin...
thinking = [b for m in state.messages for b in m.content if isinstance(b, ThinkingBlock)]
assert [(t.provider_id, t.provider_data["signature"]) for t in thinking] == [("provider-a", "sig-from-a")]
# ...but Provider B never received it. Provider B did receive the tool call and result from Step 1.
first_for_b = provider_b.sent[0]
assert not any(isinstance(b, ThinkingBlock) for m in first_for_b for b in m.content)
assert any(c.id == "a1" for m in first_for_b for c in m.tool_calls)

# F3: Malformed JSON was returned to the model as an error result and the run self-corrected
results = [r for m in state.messages for r in m.tool_results]
assert results[1].is_error and "not valid JSON" in results[1].content
assert status.calls == ["checkout", "payments"]  # no duplicate tool calls, no execution with broken args

# F4: Usage separates cache, input, and output — and total counts all processed tokens toward the budget
assert state.usage.cache_read_tokens == 1_800 and state.usage.reasoning_tokens == 80
assert state.usage.total == 200 + 1_800 + 120 + 2_100 + 30 + 2_200 + 25 + 2_300 + 15
```

---

## 4. Honest Operational Boundaries

- **Switching back to Provider A:** When Provider A recovers on a later turn, it receives its own earlier
  thinking blocks back, while assistant turns generated by Provider B have no thinking blocks. Some vendor
  APIs require every assistant tool-use turn to start with a thinking block when extended thinking is
  enabled. A production adapter must handle mixed histories cleanly (e.g., disabling forced thinking for
  that request) and verify this with recorded API responses.
- **Behavioral differences:** Provider B is a different model. Mid-run failover can shift tone or
  decision quality. For high-stakes workflows, failing cleanly (`failed`) and resuming when the primary
  model recovers may be preferable to falling back to a weaker model (see [Use Case 3](03-multi-tenant-customer-support.md)).
- **No built-in circuit breaker:** `ModelRouter` is stateless across turns. Every model call tries
  Provider A first (including retries and backoff) before failing over to B. During a prolonged outage,
  either keep `RetryPolicy` tight or wrap Provider A in a circuit-breaker `ModelProvider` that immediately
  raises `ProviderError(retryable=False)` while the circuit is open.
- **Cache warming cost:** Prompt caches are vendor-local. After failover, Provider B processes the full
  conversation prefix as uncached input tokens (F5).
- **Error strings omitted from events:** `provider_retry` and `provider_failover` events intentionally
  omit vendor error messages because vendor errors sometimes echo prompt content. When a run succeeds via
  failover, Provider A's error message is not stored in `RunState`; log vendor errors inside your adapter
  to a protected operational log.
- **Legacy v1 thinking blocks:** Checkpoints created before Format v2 have `provider_id=None`.
  `without_foreign_thinking` preserves blocks with `provider_id=None` because their origin is unknown.

## 5. What the Core Guarantees — and What Your Adapter Must Provide

**Guaranteed by the core harness (tested):** mid-call failover without losing conversation history or
tool results; every `ThinkingBlock` created in Format v2 carries its `provider_id`;
`without_foreign_thinking` strips foreign thinking blocks without breaking message alternation or tool
pairing; malformed tool JSON (`arguments_error`) never aborts a run; `Usage` has normalized semantics.

**Must be provided per adapter:** call `without_foreign_thinking(request.messages, self.id)` before
translating requests; report JSON parse errors via `ToolCall.arguments_error`; map vendor token counters
to `Usage`; classify HTTP/SDK errors accurately with `retryable=True/False`; test mixed-provider
histories when extended thinking is enabled.
