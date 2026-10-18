# 0003 – Enforce Protocol Invariants in RunState

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-15
- **Affects:** `ai_agent_harness/core/state.py`, `ai_agent_harness/runtime/runner.py`

## Context

All major LLM APIs require that message roles alternate (`user` ↔ `assistant`) and that every tool call
in an assistant message is immediately answered by exactly one matching tool result in the following
user message. In early prototypes, retries, truncated responses (`MAX_TOKENS`), and paused runs produced
invalid message histories that were only rejected later by live provider APIs.

## Decision

Messages are mutated exclusively through `RunState` methods (`add_user_text`, `add_assistant`,
`add_tool_results`, `close_open_tool_calls`), which validate the protocol on every step and raise
`ProtocolError` on any violation. `check_protocol()` validates complete histories, including loaded
checkpoints. Every run termination closes any open tool calls (committing already completed results
from `batch_done` and recording the remainder as error results).

## Alternatives

| Alternative | Why Rejected |
|---|---|
| Validate only inside provider adapters | Bugs surface too late and behave differently across providers |
| Unchecked list manipulation | Any new control-flow path in the runner could silently corrupt the conversation history |

## Consequences

- Positive: Invalid message histories are structurally impossible, even after process crashes or checkpoint tampering.
- Negative: Direct `state.messages.append(...)` is forbidden by convention and AST tests (`AGENTS.md`, `tests/test_architecture.py`).
- Verification: `assert_consistent` in `tests/conftest.py` runs across virtually all runner tests.
