# 0012 – Checkpoint Format v2, Runtime Dependencies (`deps`), and Pre-Adapter Hardenings

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-16
- **Affects:** `ai_agent_harness/core/types.py`, `ai_agent_harness/core/state.py`, `ai_agent_harness/runtime/runner.py`,
  `ai_agent_harness/tools/executor.py`, `ai_agent_harness/tools/base.py`, `ai_agent_harness/providers/base.py`,
  `ai_agent_harness/guardrails.py`, `ai_agent_harness/persistence.py`

## Context

An architectural audit conducted before implementing the first live provider adapter identified several
gaps. Several affected the **persisted checkpoint schema** — fixing them later would have caused a
breaking schema change across downstream projects. Others were subtle runtime edge cases under load
or process crashes.

## Decision

**Checkpoint Format v2** (`SCHEMA_VERSION = 2`, forward migration `1 -> 2` populates defaults, ADR 0010):

1. `RunState.batch_done`: stores already completed `ToolResult`s of the active tool batch. A checkpoint
   is written before **every** tool execution, not just once per batch. If a crash occurs during Tool 2,
   only Tool 2 is closed as "outcome unknown" on recovery; Tool 1 retains its real result — even when
   Tool 2 runs after an approval pause.
2. `Usage` adds `cache_read_tokens`, `cache_write_tokens`, and `reasoning_tokens` with normalized
   semantics across all providers; `Usage.total` sums all processed and generated tokens.
3. `ThinkingBlock.provider_id`: stamped by the runner with the ID of the provider that produced the
   response. `without_foreign_thinking()` strips thinking blocks from other providers before translating
   requests — ensuring that after a mid-run failover, Provider B never receives cryptographic thinking
   signatures from Provider A.
4. `ToolCall.arguments_error`: adapters report malformed tool-call JSON via this field instead of aborting
   the run with `ProviderError`; `ToolExecutor` returns the error to the model as a tool result so the
   model can self-correct on the next turn.

**Runtime dependencies (`deps`):** `start(..., deps=...)` and `resume(..., deps=...)`. Tools and hooks
retrieve them via `state.deps_as(ExpectedType)`. They live in a `PrivateAttr`, are never serialized into
checkpoints, are detached in `finally` blocks after every run, and are excluded from `pickle`/`deepcopy`.
No breaking change: `tool.run(args, state)` keeps its signature.

**Hardenings:**

- `Message.text` joins multiple `TextBlock`s with `\n`; `RunState.last_text` concatenates consecutive
  assistant messages across `MAX_TOKENS` continuations.
- `SyncTool` runs in a dedicated, bounded `ThreadPoolExecutor` (`SYNC_TOOL_THREADS = 16`). Hanging
  blocking tools can no longer starve the default `asyncio` thread pool used by `JsonFileCheckpointStore`.
- `JsonFileCheckpointStore` uses one lock file per tenant (`<tenant>/.lock`) instead of per run, and
  detects case-insensitive filesystem collisions.
- `ProtectedPaths` inspects path-like keys, every whitespace-free string value regardless of field name,
  nested dictionary keys, and custom `extra_keys` for paths containing spaces.
- Task cancellation during a checkpoint write shields the write to completion and updates `state.revision`
  before re-raising `CancelledError` — preventing the subsequent `cancelled` checkpoint save from failing
  against its own revision lock.

## Alternatives

| Alternative | Why Rejected |
|---|---|
| Keep completed results of a batch only in memory | A crash mid-batch loses earlier tool results and causes the model to repeat side effects |
| Pass `deps` as a third parameter to `tool.run` | Breaking change for every tool and hook signature |
| Pass `deps` via `contextvars` | Implicit, harder to trace, and untyped |
| Convert foreign thinking blocks into plain text | Pollutes conversation history; cryptographic signatures are invalid across vendors anyway |
| Raise `ProviderError` on malformed tool JSON | Aborts the entire run for a mistake the model can easily fix on its next turn |
| Run `ProtectedPaths` substring check on all free text | Would falsely block harmless prose (like email bodies) that merely mentions a filename |

## Consequences

- Positive: Zero checkpoint schema breakage when adding real provider adapters; mid-batch crashes never lose completed tool outputs; tools receive per-run DB sessions and user tokens cleanly without polluting checkpoints.
- Negative: `n + 1` checkpoints per batch of `n` tools (previously 2). A whitespace-free token matching a protected pattern (e.g. `"pyproject.toml"` or `"tests"`) in a non-path field of a write tool is now blocked (fail-closed). Older v1 code cannot read v2 checkpoints (by design, ADR 0010).
- Verification: `tests/test_format_v2.py` (including the v2 mid-batch crash fixture), `tests/test_deps.py`, `tests/test_edge_cases.py`, `tests/test_tools.py`, `tests/test_runner.py`, `tests/test_guardrails_and_persistence.py`, and `tests/test_stress.py`.
