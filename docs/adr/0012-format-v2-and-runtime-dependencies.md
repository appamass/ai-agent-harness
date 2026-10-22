# 0012 – Checkpoint Format v2, Runtime Dependencies, Hardening

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-16
- **Affects:** `ai_agent_harness/core/types.py`, `core/state.py`, `runtime/runner.py`, `tools/executor.py`, `tools/base.py`, `providers/base.py`, `guardrails.py`, `persistence.py`

## Context

An audit before the first live provider adapter found gaps. Some changed the persisted schema, which is cheap before adapters exist and a breaking change afterwards.

## Decision

**Checkpoint format v2** (`SCHEMA_VERSION = 2`, migration 1 → 2 fills defaults, ADR 0010):

1. `RunState.batch_done` keeps the finished results of the current batch. A checkpoint is written before **every** tool call, so after a crash only the interrupted call becomes "outcome unknown".
2. `Usage` gets `cache_read_tokens`, `cache_write_tokens`, `reasoning_tokens`, with one meaning for all providers; `Usage.total` counts everything.
3. `ThinkingBlock.provider_id` is stamped by the runner; `without_foreign_thinking()` removes other providers' thinking blocks so a failover never sends foreign signatures.
4. `ToolCall.arguments_error` carries malformed tool-call JSON. The executor returns it to the model as a tool result so it can correct itself, instead of failing the run.

**Runtime dependencies:** `start(..., deps=...)` and `resume(..., deps=...)`; tools and hooks read them with `state.deps_as(Type)`. They live in a private attribute, are never serialized, pickled, or copied, and are detached after every run. `tool.run(args, state)` keeps its signature.

**Hardening:** `Message.text` joins text blocks with `\n`; `RunState.last_text` joins continuations after `MAX_TOKENS`. `SyncTool` runs in its own bounded thread pool (`SYNC_TOOL_THREADS = 16`). `JsonFileCheckpointStore` uses one lock per tenant and detects case collisions. `ProtectedPaths` checks path-like keys, every whitespace-free string, nested keys, and `extra_keys`. A cancelled checkpoint write finishes and updates `state.revision` before `CancelledError` is re-raised.

## Rejected

- Batch results only in memory: a crash repeats side effects.
- `deps` as a third `run` parameter: breaks every tool and hook. Via `contextvars`: implicit and untyped.
- Foreign thinking as plain text: pollutes history; the signatures are invalid anyway.
- `ProviderError` on malformed tool JSON: aborts the run for something the model can fix.
- `ProtectedPaths` substring check on all text: blocks harmless prose.

## Consequences

`n + 1` checkpoints per batch of `n` tools. A whitespace-free token like `tests` in a non-path field of a write tool is blocked (fail-closed). Tested in `tests/test_format_v2.py`, `test_deps.py`, `test_edge_cases.py`, `test_tools.py`, `test_runner.py`, `test_guardrails_and_persistence.py`, `test_stress.py`.
