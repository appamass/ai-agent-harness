# 0003 – Protocol Invariants Live in `RunState`

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-15
- **Affects:** `ai_agent_harness/core/state.py`, `ai_agent_harness/runtime/runner.py`

## Context

APIs require alternating roles and exactly one matching tool result for every tool call. Retries, truncated responses, and paused runs produced broken histories that only a live API rejected.

## Decision

Messages change only through `RunState` methods (`add_user_text`, `add_assistant`, `add_tool_results`, `close_open_tool_calls`), which raise `ProtocolError` on any violation. `check_protocol()` validates whole histories, including loaded checkpoints. Every run end closes open tool calls: finished results are kept, the rest become error results.

## Rejected

- Validation only in adapters: bugs surface late and differ per provider.
- Plain list manipulation: any new runner path could corrupt the history.

## Consequences

Direct changes of `state.messages` outside `core/state.py` fail `tests/test_architecture.py` (AST test). `assert_consistent` in `tests/conftest.py` checks almost every runner test.
