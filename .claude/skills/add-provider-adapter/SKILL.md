---
name: add-provider-adapter
description: Build an adapter for a model provider (Anthropic, OpenAI, Google, local) that translates the canonical format, classifies errors accurately, and reports capabilities honestly. Use when connecting a new model provider.
---

# Adding a Model Provider Adapter

Contract: module docstring of `ai_agent_harness/providers/base.py`. Reference test double: `ai_agent_harness/testing.py` (`FakeProvider`). Reference adapter: `ai_agent_harness/providers/anthropic.py` with its tests (`tests/test_anthropic_provider.py`, mock transport + wire-format fixtures) and live tests (`tests/live/`).

## 1. Location and Dependency

- Create `ai_agent_harness/providers/<provider>.py` (or place it in downstream project code).
- Import the provider SDK only inside that module — never in `ai_agent_harness/core/`.
- Add the SDK as an **optional** dependency in `pyproject.toml` (`[project.optional-dependencies]`),
  e.g. `anthropic = ["anthropic>=..."]`. Editing `pyproject.toml` requires human approval (see `AGENTS.md`).

## 2. Translation Rules

| Canonical (`ai_agent_harness/core/types.py`) | Adapter Responsibility |
|---|---|
| `ModelRequest.system`, `messages`, `tools` | Translate into the provider's API format; place tool results in the role/block format expected by the API |
| `ThinkingBlock.provider_data` | Round-trip unchanged (e.g. signatures) — **own blocks only**: call `without_foreign_thinking(request.messages, self.id)` before translating; `provider_id` is stamped by the runner, not the adapter |
| `ImageBlock` | Send only when `capabilities.vision` is `True` |
| Response | Return `ModelResponse` with role `assistant` and content blocks in original order |
| Stop reason | Map to `StopReason`; map unknown values to `StopReason.END` |
| Tool call arguments | Parse JSON; on parse failure **never** raise `ProviderError` — return `ToolCall(arguments={}, arguments_error=<parser message without raw text>)` |
| `Usage` | Follow fixed semantics (`core/types.py`): `input_tokens` excluding cache, `cache_read_tokens`/`cache_write_tokens` separate, `output_tokens` including reasoning, `reasoning_tokens` as a subset of output; `model` = actual model snapshot served. **Note for Gemini:** reports reasoning ("thoughts") separately from candidates — set `output_tokens = candidate_tokens + thoughts_tokens`, otherwise `Usage` validation fails. **Note for OpenAI:** `prompt_tokens` includes cached tokens — subtract `cached_tokens` |
| Prompt caching | Place cache breakpoints at the end of the stable prefix (system prompt, tool specs) when required by the provider API |

## 3. Error Classification

| Case | Raise |
|---|---|
| HTTP 408, 409, 429, 5xx (incl. 529 overload), network errors, SDK timeout (`is_retryable_status`) | `ProviderError(..., retryable=True, provider_id=self.id)` |
| HTTP 400, invalid request, auth failure, content policy rejection | `ProviderError(..., retryable=False, provider_id=self.id)` |

Retries, exponential backoff, timeouts, and failover are handled by `ModelRouter` — do not retry inside the adapter.
Never include API keys or secrets in error messages.

## 4. Honest Capabilities

Declare `Capabilities(tools=..., vision=..., structured_output=...)` honestly (OpenAI-compatible servers often reject `json_schema`). The router skips providers that lack capabilities required by a request. Offer `stream()` if the API streams.

## 5. Mandatory Tests

- Translation unit tests using **recorded** API payloads (no network in CI): plain text, single tool call,
  multiple tool calls, thinking block with `provider_data`, truncated response (`MAX_TOKENS`), refusal,
  **malformed tool-call JSON**, and **`Usage` with cache and reasoning tokens**.
- **Failover:** A message history containing thinking blocks from another provider translates cleanly.
- Error mapping for every HTTP status code category above.
- Round-trip: A conversation produced with the adapter passes `RunState.check_protocol()`.
- Optional (marked separately, excluded from default CI): a live smoke test against the real provider API.

## 6. Wrap-Up

Run the `verify-change` skill and document the adapter in `docs/guide.md` (and list it in `README.md` if it ships with the library). When using multiple providers,
run the same eval suite across all adapters (see `HARNESS_TAXONOMY.md`, Section 8).
