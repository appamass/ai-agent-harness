# 0017 – Anthropic Provider Adapter as an Optional Extra

- **Status:** Proposed
- **Date:** 2026-10-18
- **Affects:** `ai_agent_harness/providers/anthropic.py` (new public module), `pyproject.toml` (extra `anthropic`), tests, docs

## Context

Until now every guarantee was shown against the scripted `FakeProvider`. The first real adapter is the biggest
unknown (`docs/next-steps.md`, Step 1) and the biggest hurdle for new users, who had to write one themselves.
The core must stay free of provider SDKs (invariant 7, ADR 0008). Claude models also have API rules that a naive
adapter breaks: non-default sampling parameters and manual thinking budgets return a 400, assistant prefill returns
a 400, thinking blocks must be sent back with their signature, and error texts echo parts of the request.

## Decision

- Ship `ai_agent_harness.providers.anthropic.AnthropicProvider` as an **optional extra**:
  `pip install "ai-agent-harness[anthropic]"` (`anthropic>=1.15,<2`). The module is not imported by
  `ai_agent_harness` or `ai_agent_harness.providers`; importing it without the SDK raises `ConfigError` with the install hint.
- The adapter owns translation only. It creates the SDK client with `max_retries=0`: retry, backoff, and failover stay in `ModelRouter`.
- **No sampling parameters are sent.** `ModelRequest.temperature` is rejected with a non-retryable `ProviderError`
  instead of failing as an HTTP 400. Thinking and effort use the model defaults; `effort=` is opt-in.
- **Thinking** is stored in `ThinkingBlock.provider_data` as `{"kind": "thinking", "signature": ...}` or
  `{"kind": "redacted_thinking", "data": ...}` and sent back unchanged. Foreign blocks are stripped by `without_foreign_thinking`;
  blocks without usable data are dropped.
- **Errors** carry only the HTTP status and the API error type (never the SDK text, which echoes the response body) and are not chained.
  Retryable: connection errors, timeouts, 408, 409, 429, and all 5xx (including 529). Everything else is not.
- **Responses from the SDK are validated once at the boundary** (the SDK is lenient): missing `content` or `usage` becomes a non-retryable `ProviderError`.
- **Prompt caching** is on by default through one top-level `cache_control` marker (stable prefix: `system` + `tools`).
- `Usage.reasoning_tokens` comes from `usage.output_tokens_details.thinking_tokens` when present.
- **Tests:** `tests/test_anthropic_provider.py` runs the real SDK against an `httpx2` mock transport with wire-format bodies from
  `tests/fixtures/anthropic/` (no network, no key). `tests/live/` holds opt-in tests (`pytest -m live`; the marker is registered and
  deselected by default) that call the real API. Responses recorded from the first live run (`HARNESS_RECORD_DIR`) live in
  `tests/fixtures/anthropic/recorded/` and are translated and round-tripped by a test; fixtures for cases not yet recorded
  (redacted thinking, refusal, cache usage; thinking together with a tool call is missing) are hand-written from the documented shape and say so in the fixtures README.
- Requests pass an explicit SDK timeout (600 s) so the SDK does not refuse large non-streaming `max_tokens`; the router's
  `RetryPolicy.request_timeout_s` is the effective limit. Non-streaming is kept on purpose: the runner needs complete messages.
- Tool call IDs are made safe for the API (`[A-Za-z0-9_-]`) in `tool_use` and `tool_result` alike, so history from another
  provider (failover) is accepted.
- The exception raised for an API failure is built outside the `except` block, so the SDK exception (which echoes the
  response body) is reachable neither through `__cause__` nor `__context__`.
- `AnthropicProvider.aclose()` closes the client the adapter created (a client passed in stays the caller's). The first live run
  showed that without it open connections outlive the event loop (`ResourceWarning`).
- Known limit: after a mid-loop failover or when resuming an old checkpoint, assistant turns lose thinking blocks that cannot
  be verified. With thinking on, the API may reject such a history for some models. This is not covered by a live test yet.
- The first live target is `claude-haiku-5-5` (cheapest current model). The adapter is model-agnostic; the model is a required argument.

## Alternatives

| Alternative | Why Rejected |
|---|---|
| Ship no adapter, only docs | Leaves the largest gap open; every user repeats the same API pitfalls |
| Hard dependency on `anthropic` | Breaks "core needs only Pydantic" and forces the SDK on users of other providers |
| Let the SDK retry (`max_retries=2`) | Hidden retries multiply with the router's retries and bypass its events and failover |
| Forward `temperature` and let the API reject it | Turns a configuration mistake into an opaque 400 at run time |
| Copy the SDK error text into `ProviderError` | The text echoes the response body, which can contain prompt content (violates the spirit of ADR 0011) |
| A generic OpenAI-compatible adapter first | Would hide the thinking-signature and refusal behavior that the canonical format has to handle |

## Consequences

- Positive: a working path to a real model in minutes; the canonical format is exercised against the real SDK types in CI.
- Negative / Cost: one more public module and an optional dependency to maintain; the SDK is on the CI dev path (Dependabot proposes updates).
  Part of the wire-format fixtures are hand-written approximations of the real API until they are recorded.
- Verification: `tests/test_anthropic_provider.py` (translation, errors, full run through `start()` with `check_protocol()`),
  `tests/live/test_anthropic_live.py` (real API, opt-in), CI `lowest` job (lowest allowed SDK).
