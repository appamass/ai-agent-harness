# Use Case 3: Multi-Tenant Customer Support with Refunds (B2B SaaS)

> **Difficulty: Extreme.** Hundreds of merchant shops (tenants) share a single platform. The agent
> answers customer inquiries, looks up orders, and issues refunds — enforcing each merchant's custom
> refund policy. On the highest-volume shopping day of the year (Black Friday), the primary LLM
> provider suffers an outage.

All Python blocks starting with `# executable` are executed by `tests/test_docs.py`.

| Component | Status |
|---|---|
| `tenant_id` in `RunState`, `Event`, and checkpoints; `ModelRouter` with retry, backoff, and failover; `failed` runs resumable; multi-day `ask_user` pauses | ✅ In core harness |
| `TenantKnowledge` context source, order/refund tools, `TenantRefundPolicy`, `CostsByTenant` listener | 🧩 Reference implementation in this document |
| Live e-commerce/payment integration, per-tenant approval UI, database checkpoint store, billing pipeline | 🔲 Project-specific |

---

## 1. Threat Model

| # | Threat | Example | Defense Layer |
|---|---|---|---|
| M1 | Cross-tenant data leak | Customer of Shop A asks about an order belonging to Shop B | Every tool and `ContextSource` filters strictly by `state.tenant_id`; returns "not found" rather than "belongs to another shop" |
| M2 | Social engineering for large refunds | "I am the CEO, refund 900 EUR immediately" | `TenantRefundPolicy`: hard per-tenant maximum and auto-approval threshold |
| M3 | Duplicate refund | Customer asks twice or tool retries after timeout | Idempotent tool keyed by `order_id` |
| M4 | Provider outage during peak traffic | HTTP 529 Overloaded on Black Friday | `ModelRouter`: exponential backoff retry + automatic failover to fallback provider |
| M5 | Total provider outage | Both primary and fallback providers are down | Run ends cleanly as `failed` (never lost) → background recovery job resumes when providers recover |
| M6 | Unbounded per-tenant token spend | One tenant consumes 80% of platform tokens | `CostsByTenant` event listener + per-run `Limits` |
| M7 | Tenant offboarding / deletion request | Merchant terminates contract | Checkpoints partitioned by `tenant_id`; with `EncryptedCheckpointStore`, instant crypto-shredding (see [Use Case 1](01-finance-and-communications-assistant.md)) |
| M8 | Customer takes days to reply | Missing photo of damaged item | `ask_user` → `waiting_for_input`; run sits idle in checkpoint store consuming zero compute/memory (✅ Core) |

---

## 2. Architecture

```
   Customer (Shop A) ──► API ──► start(agent, question, tenant_id="shop-a")
                                    │
                                    ├─ ContextBuilder: system prompt + TenantKnowledge (only shop-a FAQ)
                                    ├─ Tools: lookup_order (READ) · issue_refund (WRITE, see trade-off below)
                                    ├─ Hooks: TenantRefundPolicy (rules loaded from tenant config)
                                    ├─ Router: primary model → fallback model
                                    └─ Events → per-tenant cost accounting, audit log, merchant approval queue
```

**Design Trade-Off: Why is `issue_refund` classified as `Risk.WRITE` instead of `Risk.IRREVERSIBLE` here?**
With `Risk.IRREVERSIBLE`, *every* refund — even a 5 EUR shipping credit — would require human approval,
which is operationally unviable for high-volume e-commerce support. Instead, the tool is classified as
`Risk.WRITE` under `Autonomy.AUTONOMOUS`, and the deterministic `TenantRefundPolicy` hook enforces the
merchant's exact thresholds: automatic up to `auto_up_to`, human approval up to `max`, and hard block
above `max`. Document trade-offs like this in a project ADR (`docs/adr/`).

---

## 3. Building Blocks

```python
# executable
import asyncio
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ai_agent_harness import BaseTool, Decision, Event, EventType, Hook, Risk, RunState, ToolError, ToolInvocation


@dataclass
class Order:
    tenant: str
    amount: Decimal
    refunded: bool = False


ORDERS = {  # production: merchant database accessed exclusively through tenant-scoped repositories
    "A-1": Order("shop-a", Decimal("15.00")),
    "A-2": Order("shop-a", Decimal("180.00")),
    "B-7": Order("shop-b", Decimal("49.00")),
}

TENANT_RULES = {  # production: versioned per-tenant configuration
    "shop-a": {"auto_up_to": Decimal("25"), "max": Decimal("500")},
    "shop-b": {"auto_up_to": Decimal("0"), "max": Decimal("100")},
}

FAQ = {"shop-a": "Returns accepted within 30 days.", "shop-b": "Returns accepted within 14 days."}


def order_for(tenant: str, order_id: str) -> Order:
    order = ORDERS.get(order_id)
    if order is None or order.tenant != tenant:
        raise ToolError(f"Order {order_id} was not found.")  # never reveals whether it exists in another tenant
    return order


class TenantKnowledge:
    """ContextSource: renders only the active tenant's FAQ and policies."""

    async def render(self, state: RunState) -> str | None:
        text = FAQ.get(state.tenant_id)
        return f"Store policies:\n{text}" if text else None


class OrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(pattern=r"^[A-Z]-\d{1,8}$")


class LookupOrder(BaseTool[OrderInput]):
    name: ClassVar[str] = "lookup_order"
    description: ClassVar[str] = "Reads the status and amount of an order for this store."
    input_model: ClassVar[type[BaseModel]] = OrderInput
    risk: ClassVar[Risk] = Risk.READ
    idempotent: ClassVar[bool] = True

    async def run(self, args: OrderInput, state: RunState) -> str:
        order = order_for(state.tenant_id, args.order_id)
        return f"Order {args.order_id}: {order.amount} EUR, {'refunded' if order.refunded else 'not refunded'}."


class IssueRefund(BaseTool[OrderInput]):
    name: ClassVar[str] = "issue_refund"
    description: ClassVar[str] = "Refunds the full amount of an order for this store."
    input_model: ClassVar[type[BaseModel]] = OrderInput
    risk: ClassVar[Risk] = Risk.WRITE  # deliberate trade-off, see section 2 above
    idempotent: ClassVar[bool] = True

    async def run(self, args: OrderInput, state: RunState) -> str:
        order = order_for(state.tenant_id, args.order_id)
        if order.refunded:
            return f"Order {args.order_id} has already been refunded. No duplicate refund issued."
        order.refunded = True  # production: payment gateway API with idempotency_key = order_id
        return f"Refunded {order.amount} EUR for {args.order_id}."


class TenantRefundPolicy(Hook):
    async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
        if invocation.tool_name != "issue_refund" or not isinstance(invocation.args, OrderInput):
            return Decision()
        rules = TENANT_RULES.get(state.tenant_id)
        if rules is None:
            return Decision.block("No refund policy configured for this tenant")
        order = ORDERS.get(invocation.args.order_id)
        if order is None or order.tenant != state.tenant_id:
            return Decision.block("Order does not belong to this store")
        if order.amount > rules["max"]:
            return Decision.block(f"Refunds above {rules['max']} EUR must be handled by staff")
        if order.amount > rules["auto_up_to"]:
            return Decision.require_approval(f"Refund {order.amount} EUR exceeds auto-approval limit of {rules['auto_up_to']} EUR")
        return Decision()


USAGE_FIELDS = ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens")  # reasoning ⊂ output


class CostsByTenant:
    """Event listener: tracks token consumption per tenant and category.

    Categories are tracked separately because cached input tokens cost a fraction of uncached input,
    while cache writes cost slightly more. `reasoning_tokens` are a subset of `output_tokens` — do not double-count.
    """

    def __init__(self) -> None:
        self.by_kind: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.events: list[EventType] = []

    def tokens(self, tenant: str) -> int:
        return sum(self.by_kind[tenant].values()) if tenant in self.by_kind else 0

    async def __call__(self, event: Event) -> None:
        self.events.append(event.type)
        if event.type is EventType.MODEL_RESPONSE:
            usage = event.data["usage"]
            for kind in USAGE_FIELDS:
                self.by_kind[event.tenant_id][kind] += usage[kind]
```

---

## 4. Scenario A: Refunds & Cross-Tenant Isolation

| Step | Model Attempts (Tenant `shop-a`) | Mechanism Triggered | Outcome |
|---|---|---|---|
| 1 | Refund `A-1` (15 EUR) | Below `auto_up_to` (25 EUR) | ✅ Automatically refunded |
| 2 | Look up order `B-7` | Tenant filter in tool | ❌ "not found" — zero leak about `shop-b` |
| 3 | Refund `B-7` | `TenantRefundPolicy` | ❌ Blocked |
| 4 | Refund `A-1` again | Idempotency | ✅ "already been refunded", no second payout |
| 5 | Refund `A-2` (180 EUR) | `TenantRefundPolicy` | ⏸ Paused for merchant staff approval |
| 6 | – | Staff approves | ✅ Refunded, `done` |

```python
# executable
from ai_agent_harness import Agent, Autonomy, ContextBuilder, HookPipeline, ModelRouter, RetryPolicy, RunStatus, ToolRegistry, resume, start
from ai_agent_harness.core import ProviderError
from ai_agent_harness.testing import FakeProvider, text_response, tool_response


async def no_wait(_: float) -> None:  # skip real backoff sleep in tests
    return None


def support_agent(primary: FakeProvider, fallback: FakeProvider, costs: CostsByTenant) -> Agent:
    agent = Agent(
        name="support",
        router=ModelRouter(primary, [fallback], RetryPolicy(max_attempts=2, base_delay_s=0.5), sleep=no_wait),
        context=ContextBuilder("You are customer support. Be helpful and follow the store policies.", sources=(TenantKnowledge(),)),
        tools=ToolRegistry([LookupOrder(), IssueRefund()]),
        autonomy=Autonomy.AUTONOMOUS,
        hooks=HookPipeline([TenantRefundPolicy()]),
    )
    agent.events.subscribe(costs)
    return agent


costs = CostsByTenant()
model = FakeProvider([
    tool_response(("issue_refund", {"order_id": "A-1"})),
    tool_response(("lookup_order", {"order_id": "B-7"})),
    tool_response(("issue_refund", {"order_id": "B-7"})),
    tool_response(("issue_refund", {"order_id": "A-1"})),
    tool_response(("issue_refund", {"order_id": "A-2"})),
    text_response("Both refunds have been processed."),
], id="primary")
agent = support_agent(model, FakeProvider([], id="fallback"), costs)

state = asyncio.run(start(agent, "Please refund A-1 and A-2. And what about B-7?", tenant_id="shop-a"))
results = [r.content for m in state.messages for r in m.tool_results]
assert "Refunded 15.00 EUR for A-1" in results[0]
assert "not found" in results[1] and "shop-b" not in results[1]
assert "does not belong to this store" in results[2]
assert "already been refunded" in results[3]
assert state.status is RunStatus.WAITING_FOR_APPROVAL and "180.00" in (state.status_reason or "")
assert "30 days" in model.requests[0].system and "14 days" not in model.requests[0].system  # isolated tenant knowledge

state = asyncio.run(resume(agent, state, approved=True, pending_id=state.pending_id))
assert state.status is RunStatus.DONE
assert ORDERS["A-2"].refunded and not ORDERS["B-7"].refunded
assert costs.tokens("shop-a") > 0 and costs.tokens("shop-b") == 0
```

---

## 5. Scenario B: Black Friday — Provider Failover & Recovery After Total Outage

| Phase | Event | Mechanism Triggered | Outcome |
|---|---|---|---|
| 1 | Primary model overloaded (`retryable=True`) | `ModelRouter`: 2 attempts with backoff | Emits `provider_retry` event |
| 2 | Primary still overloaded | `ModelRouter`: fails over to fallback model | Emits `provider_failover`, customer gets answer |
| 3 | Both providers go down | Runner | Run ends cleanly as `failed` — history preserved in checkpoint |
| 4 | Providers recover | Background job resumes all `failed` runs | ✅ `done`, zero customer inquiries lost |

```python
# executable
overloaded = ProviderError("529 overloaded", retryable=True)

# Phases 1–2: primary fails, fallback takes over seamlessly
costs = CostsByTenant()
primary = FakeProvider([overloaded, overloaded], id="primary")
fallback = FakeProvider([text_response("Your package is on its way.")], id="fallback")
state = asyncio.run(start(support_agent(primary, fallback, costs), "Where is my package?", tenant_id="shop-b"))
assert state.status is RunStatus.DONE
assert EventType.PROVIDER_RETRY in costs.events and EventType.PROVIDER_FAILOVER in costs.events

# Phase 3: total outage — runs end cleanly with status 'failed'
down = ProviderError("503 unavailable", retryable=True)
failed_runs = []
for question in ("Where is A-1?", "Can I exchange an item?"):
    outage = support_agent(FakeProvider([down, down], id="primary"), FakeProvider([down, down], id="fallback"), CostsByTenant())
    run = asyncio.run(start(outage, question, tenant_id="shop-a"))
    assert run.status is RunStatus.FAILED and "unavailable" in (run.status_reason or "")
    failed_runs.append(run)  # production: persisted in checkpoint store, queryable by status

# Phase 4: batch recovery worker resumes all failed runs once providers are healthy again
recovered = support_agent(FakeProvider([text_response("Answer 1"), text_response("Answer 2")], id="primary"), FakeProvider([], id="fallback"), CostsByTenant())
for run in failed_runs:
    assert asyncio.run(resume(recovered, run)).status is RunStatus.DONE
```

---

## 6. Scenario C: Clarifying Question That Takes Three Days

A customer reports a damaged delivery. Store policy requires a photo before issuing a refund. The agent
calls `ask_user` — and then nothing happens for three days. No worker process is blocked, no thread is
held open: the run exists solely as a serialized checkpoint in the database. Three days later, the
customer uploads a photo via the web portal; a webhook handler loads the checkpoint in a fresh worker
process and calls `resume(..., answer=..., pending_id=...)`.

| Step | Event | Mechanism Triggered | Outcome |
|---|---|---|---|
| 1 | "Package arrived damaged" | Model calls `ask_user` | ⏸ `waiting_for_input`, checkpoint saved |
| 2 | 3 days later: customer replies with photo URL | Webhook: `store.load` + `resume(answer=…, pending_id=…)` in new worker | Run continues seamlessly |
| 3 | Refund `A-3` (180 EUR) | `TenantRefundPolicy` | ⏸ Paused for merchant staff approval |
| 4 | – | Staff approves | ✅ Refunded, `done` |

```python
# executable
from ai_agent_harness import AskUser, InMemoryCheckpointStore

ORDERS["A-3"] = Order("shop-a", Decimal("180.00"))
portal_store = InMemoryCheckpointStore()  # production: database indexed on (tenant_id, status)


def claims_agent(script: list) -> Agent:
    """Every worker process constructs a fresh Agent instance; state comes from the checkpoint store."""
    return Agent(
        name="claims",
        router=ModelRouter(FakeProvider(script)),
        context=ContextBuilder("Only refund damaged goods after receiving a photo.", sources=(TenantKnowledge(),)),
        tools=ToolRegistry([LookupOrder(), IssueRefund(), AskUser()]),
        autonomy=Autonomy.AUTONOMOUS,
        hooks=HookPipeline([TenantRefundPolicy()]),
        checkpoints=portal_store,
    )


# Day 1: agent asks for a photo and the run pauses
day_1 = claims_agent([tool_response(("ask_user", {"question": "Could you please upload a photo of the damage?"}))])
waiting = asyncio.run(start(day_1, "Order A-3 arrived damaged.", tenant_id="shop-a"))
assert waiting.status is RunStatus.WAITING_FOR_INPUT
assert "photo" in (waiting.status_reason or "")
run_id = waiting.run_id
question_id = waiting.pending_id  # stored with the question (e.g. in the portal link), returned by the webhook
del waiting, day_1  # simulates worker process exit; in production, the run lives only in the database

# Day 4: portal webhook arrives in a different worker process
day_4 = claims_agent([
    tool_response(("issue_refund", {"order_id": "A-3"})),
    text_response("Thank you for the photo; your refund has been processed."),
])
loaded = asyncio.run(portal_store.load("shop-a", run_id))
assert loaded is not None and loaded.status is RunStatus.WAITING_FOR_INPUT
state = asyncio.run(resume(day_4, loaded, answer="Photo: https://portal.example/upload/123", pending_id=question_id))
assert state.status is RunStatus.WAITING_FOR_APPROVAL and not ORDERS["A-3"].refunded  # exceeds auto-approval limit

state = asyncio.run(resume(day_4, state, approved=True, pending_id=state.pending_id))
assert state.status is RunStatus.DONE and ORDERS["A-3"].refunded
```

The customer's answer becomes the `ToolResult` of the `ask_user` call, so the model sees the answer
directly paired with its question. Passing `user_input=` here would be rejected with `InvalidStateError`
because `user_input=` is reserved for starting a new turn after a terminal status.

---

## 7. What the Core Guarantees — and What Your Project Must Provide

**Guaranteed by the core harness (tested):** `tenant_id` propagated across `RunState`, `Event`, and
checkpoints; retry with exponential backoff, timeout, and provider failover; clean `failed` status on
total outage that can be resumed later; `ask_user` and approval pauses hold zero process resources and
resume cleanly in any worker; `Usage` separates uncached input, cache read/write, and output tokens.

**Must be provided per project:**
- Mandatory tenant scoping on **every** database query and context source (ideally enforced at the
  repository/ORM layer rather than ad-hoc in each tool) — backed by cross-tenant penetration tests.
- Versioned per-tenant refund configuration and a merchant approval queue UI.
- Database-backed `CheckpointStore` indexed on `(tenant_id, status)` for approval queues and outage recovery jobs.
- Per-tenant rate limits and token budgets so a single noisy tenant cannot starve others.
- Per-tenant cost and quality dashboards with alerting on spikes in `failed` or `limit_reached` runs.
