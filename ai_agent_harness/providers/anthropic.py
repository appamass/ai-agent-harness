"""Anthropic (Claude) provider adapter. Needs the optional SDK: `pip install "ai-agent-harness[anthropic]"`.

Implements the adapter contract of `providers/base.py` for the Messages API. This module is the only place
that imports the SDK; the core and `import ai_agent_harness` never need it.

Choices that follow from the contract and the API (ADR 0017):

- **No retries here.** The SDK client is created with `max_retries=0`; `ModelRouter` owns retry, backoff, and
  failover. If you pass your own `client=`, set `max_retries=0` on it too, otherwise retries multiply.
- **Errors carry only the status code and the API error type.** The SDK error text echoes the response body, which
  can contain parts of the prompt, so it is never copied into a `ProviderError` and never chained (`from None`).
  The API key is read by the SDK and never stored, logged, or put into messages.
- **No sampling parameters.** Current Claude models reject non-default `temperature`/`top_p`/`top_k` with a 400, so
  they are never sent; a request that sets `temperature` fails loudly instead of as an API error. Thinking and effort
  use the model defaults unless you set `effort`.
- **Thinking blocks** keep their `signature` (or the opaque `redacted_thinking` data) in `ThinkingBlock.provider_data`
  and are sent back unchanged. Blocks of other providers are stripped (`without_foreign_thinking`); blocks without
  usable data (for example from an old checkpoint) are dropped: the API rejects thinking blocks it cannot verify.
- **Prompt caching** is on by default (one top-level `cache_control` marker; the stable prefix is `system` + `tools`).
  Prompts below the model's minimum cacheable size are simply not cached.
- **Closing**: `await provider.aclose()` releases the connections of the client the adapter created.
- **Usage**: `input_tokens` excludes cache reads and writes, `output_tokens` includes thinking, and `reasoning_tokens`
  is the reported thinking subset.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Any, Final, Literal, cast

from ..core.errors import ConfigError, ProviderError
from ..core.types import (
    Block,
    ImageBlock,
    Message,
    ModelRequest,
    ModelResponse,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
    Usage,
)
from .base import (
    EMPTY_RESPONSE_NOTE,
    SDK_TIMEOUT_S,
    Capabilities,
    StreamDelta,
    is_retryable_status,
    without_foreign_thinking,
)

try:
    import anthropic
    from anthropic import types as api
except ImportError as exc:
    raise ConfigError("The Anthropic adapter needs the SDK: pip install 'ai-agent-harness[anthropic]'") from exc

Effort = Literal["low", "medium", "high", "xhigh", "max"]

_STOP_REASONS: Final[dict[str, StopReason]] = {
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
    "model_context_window_exceeded": StopReason.MAX_TOKENS,
    "refusal": StopReason.REFUSAL,
}  # everything else (end_turn, stop_sequence, pause_turn, unknown values) ends the turn
_ERROR_TYPE = re.compile(r"[a-z_]{1,64}")
_UNSAFE_ID = re.compile(r"[^A-Za-z0-9_-]")


class AnthropicProvider:
    """`ModelProvider` for the Anthropic Messages API.

    `model` is required (for example `"claude-haiku-5-5"`). `api_key=None` lets the SDK read `ANTHROPIC_API_KEY`.
    `id` must be unique within a `ModelRouter`. `effort` sets `output_config.effort` (models decide when unset).
    `vision=False` hides image support from the router. `cache=False` turns prompt caching off.
    """

    def __init__(
        self,
        model: str,
        *,
        id: str = "anthropic",
        api_key: str | None = None,
        client: anthropic.AsyncAnthropic | None = None,
        effort: Effort | None = None,
        vision: bool = True,
        cache: bool = True,
    ) -> None:
        if not model.strip():
            raise ConfigError("AnthropicProvider: model must not be empty")
        if not id.strip():
            raise ConfigError("AnthropicProvider: id must not be empty")
        if client is not None and api_key is not None:
            raise ConfigError("AnthropicProvider: pass either client or api_key, not both")
        self._model = model
        self._id = id
        self._owns_client = client is None
        self._client = client if client is not None else anthropic.AsyncAnthropic(api_key=api_key, max_retries=0)
        self._effort = effort
        self._capabilities = Capabilities(tools=True, vision=vision)
        self._cache = cache

    @property
    def id(self) -> str:
        return self._id

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    async def aclose(self) -> None:
        """Closes the HTTP connections of the client this adapter created. A client you passed in stays yours.

        Call it when you are done, in the same event loop that used the provider (for example at shutdown).
        Without it, open connections are only released when the loop closes, which shows up as `ResourceWarning`.
        """
        if self._owns_client:
            await self._client.close()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self._reject_unsupported(request)
        response: object = None
        failure: ProviderError | None = None
        try:
            # An explicit timeout lifts the SDK's refusal of big non-streaming requests (the router owns the real one)
            parameters = self._parameters(request)
            response = cast("object", await self._client.messages.create(**parameters, timeout=SDK_TIMEOUT_S))
        except anthropic.APIError as exc:
            failure = self._translate(exc)
        if failure is not None:
            # Raised outside the except block on purpose: the SDK exception echoes the response body and must not
            # stay reachable through __cause__ or __context__
            raise failure
        return _to_response(self._checked(response))

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamDelta | ModelResponse]:
        self._reject_unsupported(request)
        final_msg: object = None
        failure: ProviderError | None = None
        try:
            parameters = self._parameters(request)
            async with self._client.messages.stream(**parameters, timeout=SDK_TIMEOUT_S) as stream_ctx:
                async for text in stream_ctx.text_stream:
                    if text:
                        yield StreamDelta(text=text)
                final_msg = cast(object, await stream_ctx.get_final_message())
        except anthropic.APIError as exc:
            failure = self._translate(exc)
        if failure is not None:
            raise failure
        yield _to_response(self._checked(final_msg))

    def _reject_unsupported(self, request: ModelRequest) -> None:
        if request.temperature is not None:
            raise ProviderError(
                "AnthropicProvider does not support 'temperature': current Claude models reject non-default sampling",
                retryable=False,
                provider_id=self._id,
            )

    def _checked(self, response: object) -> api.Message:
        # The SDK does not validate responses strictly: check the parts we rely on once, here at the boundary
        if not (
            isinstance(response, api.Message)
            and isinstance(response.content, list)
            and isinstance(response.usage, api.Usage)
        ):
            raise ProviderError("Anthropic API returned an unexpected result", retryable=False, provider_id=self._id)
        return response

    def _translate(self, exc: anthropic.APIError) -> ProviderError:
        if isinstance(exc, anthropic.APIStatusError):
            return self._status_error(exc)
        if isinstance(exc, anthropic.APIConnectionError):  # includes timeouts
            return ProviderError("Anthropic API connection failed", retryable=True, provider_id=self._id)
        return ProviderError("Anthropic API returned an invalid response", retryable=False, provider_id=self._id)

    def _status_error(self, exc: anthropic.APIStatusError) -> ProviderError:
        status = exc.status_code
        kind = "unknown"
        body: object = exc.body
        if isinstance(body, dict):
            detail: object = cast("dict[str, object]", body).get("error")
            if isinstance(detail, dict):
                value: object = cast("dict[str, object]", detail).get("type")
                if isinstance(value, str) and _ERROR_TYPE.fullmatch(value):
                    kind = value
        return ProviderError(
            f"Anthropic API error {status} ({kind})", retryable=is_retryable_status(status), provider_id=self._id
        )

    def _parameters(self, request: ModelRequest) -> dict[str, Any]:
        parameters: dict[str, Any] = {
            "model": self._model,
            "max_tokens": request.max_tokens,
            "messages": [_to_api_message(m) for m in without_foreign_thinking(request.messages, self._id)],
        }
        if request.system:
            parameters["system"] = request.system
        if request.tools:
            parameters["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in request.tools
            ]
        if self._cache:
            parameters["cache_control"] = {"type": "ephemeral"}
        if self._effort is not None:
            parameters["output_config"] = {"effort": self._effort}
        return parameters


def _to_api_message(message: Message) -> dict[str, Any]:
    results: list[dict[str, Any]] = []  # the API requires tool results at the start of the user message
    others: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, ToolResult):
            result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": _api_id(block.call_id),
                "content": block.content,
            }
            if block.is_error:
                result["is_error"] = True
            results.append(result)
        else:
            converted = _to_api_block(block)
            if converted is not None:
                others.append(converted)
    content = results + others or [{"type": "text", "text": EMPTY_RESPONSE_NOTE}]
    return {"role": message.role, "content": content}


def _to_api_block(block: Block) -> dict[str, Any] | None:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text} if block.text else None
    if isinstance(block, ImageBlock):
        return {"type": "image", "source": {"type": "base64", "media_type": block.media_type, "data": block.data}}
    if isinstance(block, ToolCall):
        return {"type": "tool_use", "id": _api_id(block.id), "name": block.name, "input": block.arguments}
    if isinstance(block, ThinkingBlock):
        return _thinking_block(block)
    return None


def _api_id(value: str) -> str:
    """The API accepts only [A-Za-z0-9_-] in tool call IDs; IDs from another provider (failover) may not conform."""
    return _UNSAFE_ID.sub("_", value)


def _thinking_block(block: ThinkingBlock) -> dict[str, Any] | None:
    data = block.provider_data
    signature = data.get("signature")
    opaque = data.get("data")
    if data.get("kind") == "thinking" and isinstance(signature, str) and signature:
        return {"type": "thinking", "thinking": block.text, "signature": signature}
    if data.get("kind") == "redacted_thinking" and isinstance(opaque, str) and opaque:
        return {"type": "redacted_thinking", "data": opaque}
    return None  # unverifiable reasoning would be rejected by the API


def _to_response(message: api.Message) -> ModelResponse:
    blocks: list[Block] = []
    for item in message.content:
        if isinstance(item, api.TextBlock):
            blocks.append(TextBlock(text=item.text))
        elif isinstance(item, api.ThinkingBlock):
            blocks.append(
                ThinkingBlock(text=item.thinking, provider_data={"kind": "thinking", "signature": item.signature})
            )
        elif isinstance(item, api.RedactedThinkingBlock):
            blocks.append(ThinkingBlock(provider_data={"kind": "redacted_thinking", "data": item.data}))
        elif isinstance(item, api.ToolUseBlock):
            raw_input: object = item.input  # typed as a dict, but the SDK does not enforce it
            if isinstance(raw_input, dict):
                blocks.append(ToolCall(id=item.id, name=item.name, arguments=dict(cast("dict[str, Any]", raw_input))))
            else:  # the executor reports this to the model so it can retry
                blocks.append(ToolCall(id=item.id, name=item.name, arguments_error="Tool input was not a JSON object"))
    if not blocks:  # for example a refusal without content
        blocks.append(TextBlock(text=""))
    stop = _STOP_REASONS.get(message.stop_reason or "", StopReason.END)
    return ModelResponse(
        message=Message(role="assistant", content=tuple(blocks)),
        stop_reason=stop,
        usage=_usage(message.usage),
        model=message.model,
    )


def _usage(usage: api.Usage) -> Usage:
    details = usage.output_tokens_details
    reasoning = details.thinking_tokens if details is not None else 0
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_input_tokens or 0,
        cache_write_tokens=usage.cache_creation_input_tokens or 0,
        reasoning_tokens=min(reasoning, usage.output_tokens),  # a subset of output; never fail the response on it
    )
