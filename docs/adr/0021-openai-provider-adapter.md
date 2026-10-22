# 0021 – OpenAI-Compatible Provider Adapter (Optional Extra)

- **Status:** Accepted (maintainer, 2026-10-22)
- **Date:** 2026-10-22
- **Affects:** `ai_agent_harness/providers/openai.py`, `providers/__init__.py`, `pyproject.toml` (extra `openai`), tests, docs

## Context

`AnthropicProvider` was the only adapter. An OpenAI-compatible one proves the provider contract, reaches OpenAI, OpenRouter, vLLM, Ollama, LM Studio, and Gemini's compatible endpoint (`base_url`), and allows failover between Claude and GPT mid-run. The core stays free of SDKs.

## Decision

- `OpenAIProvider` is an **optional extra** (`pip install "ai-agent-harness[openai]"`, `openai>=1.40,<2`). Without the SDK it raises `ConfigError` with the install hint.
- It implements `ModelProvider` (`id`, `capabilities`, `complete`) plus the optional `stream()` and `aclose()`. The client uses `max_retries=0`; retry and failover belong to `ModelRouter`.
- `without_foreign_thinking` strips other providers' reasoning before translating. `reasoning_content` is kept as a `ThinkingBlock`, also when streaming.
- Malformed tool-call JSON becomes `ToolCall(arguments_error=…)`, never a `ProviderError`, so the model can correct itself.
- **Usage:** `input_tokens = prompt_tokens − cached_tokens`, `cache_read_tokens = cached_tokens`, `output_tokens = completion_tokens`, `reasoning_tokens` from `completion_tokens_details` (a subset of the output).
- **Errors:** 408, 409, 429, and 5xx are retryable; other statuses are not. Messages carry status and error type only, are not chained, and never echo request bodies. A stream that ends without `finish_reason` is a retryable error, not a complete answer.
- **Honest capabilities:** `structured_output` (default `True`) sends `response_format: json_schema`; servers that reject it (Ollama, vLLM, Gemini-compat) need `structured_output=False`, so the runner uses the tool or prompt tier. `max_tokens_param="max_completion_tokens"` is for models that reject `max_tokens`. `strict` schema mode is not used.
- `base_url`, `organization`, and custom headers are supported.

## Rejected

- Raw `httpx` without the SDK: the SDK is typed and standard; an optional extra keeps the core clean.
- One adapter for both vendors: wire formats and errors differ.

## Consequences

One more adapter to test in CI. Tested offline against the real SDK with mocked HTTP in `tests/test_openai_provider.py`, including a failover test; `tests/live/test_openai_live.py` is opt-in and has not been run against compatible servers.
