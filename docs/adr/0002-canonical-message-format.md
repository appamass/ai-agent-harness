# 0002 – Canonical Message Format with Provider Adapters

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-15
- **Affects:** `ai_agent_harness/core/types.py`, `ai_agent_harness/providers/`

## Context

Models from different providers differ in message structure, tool calls, thinking/reasoning blocks,
error codes, and stop reasons. Without a unified internal representation, provider-specific quirks
leak into the agent loop and guardrails.

## Decision

We define a canonical, immutable message format (`Message`, `TextBlock`, `ImageBlock`, `ThinkingBlock`
with opaque `provider_data`, `ToolCall`, `ToolResult`, `StopReason`, `Usage`). Each provider gets an
adapter that translates strictly at the boundary and declares its `Capabilities`. `ModelRouter` filters
providers by capability, retries transient errors, fails over across providers, and enforces request timeouts.

## Alternatives

| Alternative | Why Rejected |
|---|---|
| Use one provider's wire format internally | Couples the core to that vendor and turns every other adapter into a leaky special case |
| Lowest common denominator (text + tool calls only, no thinking/images) | Sacrifices model capabilities and breaks providers that require thinking blocks to be echoed back in multi-turn tool loops |

## Consequences

- Positive: The agent loop, hooks, and tests are completely provider-agnostic; `FakeProvider` enables deterministic testing of every path.
- Negative: Each provider adapter must carefully translate wire formats and classify errors accurately.
- Verification: `extra="forbid"` and validators in `types.py`; tested in `tests/test_types_and_state.py` and `tests/test_router.py`.
