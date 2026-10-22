# 0017 – Anthropic Provider Adapter (Optional Extra)

- **Status:** Accepted (maintainer, 2026-10-18)
- **Date:** 2026-10-18
- **Affects:** `ai_agent_harness/providers/anthropic.py`, `pyproject.toml` (extra `anthropic`), tests, docs

## Context

Every guarantee was first shown against the scripted `FakeProvider`, and users had to write the first real adapter themselves. The core must stay free of provider SDKs (ADR 0008). Claude APIs have rules a naive adapter breaks: non-default sampling and manual thinking budgets return 400, prefill returns 400, thinking blocks must come back with their signature, and error texts echo the request.

## Decision

- `AnthropicProvider` is an **optional extra** (`pip install "ai-agent-harness[anthropic]"`, `anthropic>=1.15,<2`). Importing it without the SDK raises `ConfigError` with the install hint. The core never imports it.
- It only translates. The SDK client uses `max_retries=0`; retry, backoff, and failover belong to `ModelRouter`.
- **No sampling parameters.** `ModelRequest.temperature` is rejected with a non-retryable `ProviderError` instead of an HTTP 400. Thinking and effort use model defaults; `effort=` is opt-in.
- **Thinking** is kept in `ThinkingBlock.provider_data` (`thinking` with signature, or `redacted_thinking`) and sent back unchanged. Foreign blocks go through `without_foreign_thinking`; blocks without usable data are dropped.
- **Errors** carry only the HTTP status and API error type, never the SDK text, and are not chained. The exception is built outside the `except` block, so the SDK exception is reachable neither as cause nor as context. Retryable: connection errors, timeouts, 408, 409, 429, all 5xx. `is_retryable_status` in `providers/base.py` holds the rule for all adapters.
- **SDK responses are validated once at the boundary:** missing `content` or `usage` becomes a non-retryable `ProviderError`.
- **Prompt caching** is on through one top-level `cache_control` marker (stable prefix `system` + `tools`). `Usage.reasoning_tokens` comes from `output_tokens_details.thinking_tokens` when present.
- Requests carry an explicit SDK timeout (600 s) so the SDK accepts large `max_tokens`; the router's `RetryPolicy.request_timeout_s` is the real limit. Tool call IDs are made API-safe (`[A-Za-z0-9_-]`) in `tool_use` and `tool_result`, so history from another provider works after failover. `aclose()` closes a client the adapter created.
- **Tests:** `tests/test_anthropic_provider.py` runs the real SDK against an `httpx` mock transport with wire bodies from `tests/fixtures/anthropic/` (no network). `tests/live/` has opt-in live tests (`pytest -m live`). Cases not yet recorded (redacted thinking, refusal) are hand-written from the documented shape.
- Known limit: after a mid-loop failover or when resuming an old checkpoint, thinking blocks that cannot be verified are dropped; with thinking on, the API may reject such a history for some models. No live test covers this.

## Rejected

- Docs only, no adapter: every user repeats the same pitfalls.
- Hard dependency on `anthropic`: forces the SDK on users of other providers.
- SDK retries (`max_retries=2`): hidden, multiplied with the router's, bypass its events and failover.
- Forwarding `temperature` to the API: turns a config mistake into an opaque 400.
- Copying the SDK error text: it echoes the response body (ADR 0011).
- An OpenAI-compatible adapter first: would hide thinking-signature and refusal handling.

## Consequences

One more public module and an optional dependency. Tested in `tests/test_anthropic_provider.py`, `tests/live/`, and the `lowest` CI job.
