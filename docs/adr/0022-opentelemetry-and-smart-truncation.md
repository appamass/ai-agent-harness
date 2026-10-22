# 0022 – OpenTelemetry Listener and Head-and-Tail Truncation

- **Status:** Accepted (maintainer, 2026-10-22)
- **Date:** 2026-10-22
- **Affects:** `ai_agent_harness/observability.py`, `tools/base.py`, `tools/executor.py`, `runtime/agent.py`, `pyproject.toml` (extra `opentelemetry`)

## Context

Teams need traces in their APM without leaking prompts or secrets (ADR 0011). Head-only truncation of tool output drops the end, which holds the traceback or test summary.

## Decision

### OpenTelemetry listener

- `OpenTelemetryListener` is an **optional extra** (`pip install "ai-agent-harness[opentelemetry]"`). The core never imports it; without the SDK the constructor raises `ConfigError` with the install hint.
- **Spans:** `run {agent}` (workflow), `chat {model}` (client, child of the run), `execute_tool {name}` (child of the run). Retry and failover are events on the model span; approval, input, and checkpoint are events on the run span. Names follow the GenAI semantic conventions where they exist (`gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.response.model`, `gen_ai.usage.*`, `gen_ai.tool.name`); everything else is prefixed `harness.`. The conventions are still in development.
- **Metrics:** `gen_ai.client.token.usage` (attribute `gen_ai.token.type` = `input` or `output`) and `gen_ai.client.operation.duration`, on the global meter provider or a `meter=` you pass.
- **Bounded memory:** at most `max_open_runs` (default 1000) open run spans; the oldest are ended as abandoned (runs that never report `RUN_FINISHED`).
- **Metadata only:** no prompts, messages, tool arguments, or secrets in names, attributes, or events.

### Head-and-tail truncation

- `ToolExecutor` defaults to `truncation_mode="head_and_tail"`: 60% head, 40% tail, joined by `… [truncated, N chars omitted]`. **This changes the old head-only default**; `"head"` restores it.
- Configurable on `ToolExecutor`, `Agent`, per tool (`BaseTool.max_result_chars`, `truncation_mode`, `@tool(...)`), and for MCP servers. A tool may only lower the agent's limit, never raise it. Invalid modes raise `ConfigError`.

## Rejected

- OpenTelemetry SDK in the core: breaks the dependency rule.
- Keep head-only: loses the part that usually matters.

## Consequences

Tested in `tests/test_observability.py` (span tree, metrics, no leakage, eviction) and `tests/test_tools.py`.
