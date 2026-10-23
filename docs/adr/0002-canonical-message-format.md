# 0002 – Canonical Message Format

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-15
- **Affects:** `ai_agent_harness/core/types.py`, `ai_agent_harness/providers/`

## Context

Providers differ in messages, tool calls, reasoning blocks, errors, and stop reasons. Without one internal format, their quirks leak into the loop and the guardrails.

## Decision

One immutable format (`Message`, `TextBlock`, `ImageBlock`, `ThinkingBlock` with opaque `provider_data`, `ToolCall`, `ToolResult`, `StopReason`, `Usage`). Each provider has an adapter that translates at the boundary and declares its `Capabilities`. `ModelRouter` picks capable providers, retries transient errors, fails over, and enforces timeouts.

## Rejected

- One vendor's wire format inside: couples the core to that vendor.
- Text and tool calls only: loses thinking blocks that some providers need echoed back.

## Consequences

Each adapter must translate carefully and classify errors accurately. Tested in `tests/test_types_and_state.py` and `tests/test_router.py`.
