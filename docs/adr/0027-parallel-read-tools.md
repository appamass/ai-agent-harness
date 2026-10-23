# 0027 – Optional Parallel Execution of Read-Only Tools

- **Status:** Accepted (maintainer, 2026-10-23)
- **Date:** 2026-10-23
- **Affects:** `tools/base.py`, `tools/executor.py`, `runtime/agent.py`, `runtime/runner.py`, `observability.py`

## Context

[ADR 0007](0007-sequential-tool-calls.md) chose strictly sequential tool execution to guarantee deterministic order, mid-batch pauses, approvals, and crash honesty. For consecutive independent reads (e.g. inspecting multiple files or searching multiple queries), strictly sequential execution introduces unnecessary latency. ADR 0007 explicitly noted: "A parallel mode for consecutive `Risk.READ` calls would need its own ADR."

## Decision

1. **Opt-in tool qualification:** `BaseTool.parallel_safe: ClassVar[bool] = False`. Only tools with `risk=Risk.READ` are allowed to set `parallel_safe=True` (`validate_definition` enforces this). Read tools with side effects (such as cursor advancement or stateful pagination) remain `False`. `BaseTool.digest()` records `"parallel_safe": True` only when set to avoid invalidating existing tool digests.
2. **Opt-in agent concurrency limit:** `Agent.max_parallel_tools: int = 1` (default 1, strictly sequential). Must be a positive integer (`is_positive_number(..., integer=True)`).
3. **Execution gate remains single authority:** Parallel calls route through the exact same `ToolExecutor.execute` gate (Invariant 3). The runner queries `self.executor.parallel_safe(call)` using the executor's registry.
4. **Group formation:** In `runner._run_batch`, consecutive calls where `parallel_safe` is True form a parallel group if `max_parallel_tools > 1` and `len(group) > 1`. Write tools, tools requiring human approval, MCP tools, and `final_output` immediately break the group and run sequentially.
5. **Bounded concurrency and write-ahead:** A parallel group executes under `asyncio.Semaphore(max_parallel_tools)` via `asyncio.gather`. A write-ahead checkpoint is saved **before** the group starts. No checkpoints are saved from within concurrent tasks to avoid revision collisions. A crash during a parallel group reports all interrupted calls as outcome unknown (`INTERRUPTED`, Invariant 5).
6. **Preserved call order:** Completed results in `batch_done` and in messages are sorted to match the exact call sequence of the model's message.
7. **Pauses mid-group:** If a hook requires human approval or a tool requests input (`PendingCall`), the first such call in call order pauses the run. Completed results from the group enter `batch.done`, while unexecuted calls enter `batch.remaining`. `pending_id` remains cryptographically bound to the waiting call (Invariant 4).
8. **Observability:** `TOOL_STARTED` and `TOOL_FINISHED` include `call_id`. `OpenTelemetryListener` keys `_tool_spans` by `(run_id, call_id)` so concurrent calls to the same tool create distinct, non-overlapping spans.

## Consequences

- Faster execution for read-heavy batches.
- Unchanged safety: write operations and irreversible actions cannot run concurrently.
- Hooks running during parallel groups must be stateless or coroutine-safe.
- Amends [ADR 0007](0007-sequential-tool-calls.md).
