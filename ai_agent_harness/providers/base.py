"""Provider protocol and capabilities.

One adapter per provider translates `ModelRequest` into the provider API format and the
response back into `ModelResponse`. Provider SDKs are imported only inside adapters.

Adapter contract — violations are bugs in the adapter, not in the harness:

- Raise API errors as `ProviderError`: `retryable=True` for timeouts and conflicts (408, 409), rate limits (429),
  overload (5xx, 529), and network errors (`is_retryable_status`); `retryable=False` for everything else
  (400, auth, invalid request). The router handles retry, backoff, and fallback.
- Map provider stop reasons to `StopReason`; map unknown values to `END`.
- Pass `ThinkingBlock.provider_data` through unchanged — but **only own blocks**:
  after a failover, the history contains blocks from other providers whose signatures foreign APIs
  reject. Call `without_foreign_thinking(request.messages, self.id)` before translating.
  The runner stamps `provider_id` onto new blocks; adapters leave that field empty.
- **Tool call arguments** that are not valid JSON must never be raised as `ProviderError` (which ends
  the run), but returned as `ToolCall(arguments={}, arguments_error=<parser message>)`. The executor
  reports this back to the model so it can self-correct. Do not copy raw argument text into the error message.
- Populate **`Usage`** according to the fixed semantics in `core/types.py`: `input_tokens` excluding cache,
  cache read/write separate, `output_tokens` including reasoning, `reasoning_tokens` as a subset of output.
  Providers that report reasoning separately (Gemini "thoughts") add it to `output_tokens`;
  providers whose input includes cache hits (OpenAI `prompt_tokens`) subtract them.
- **Prompt caching** is the adapter's responsibility (e.g. placing cache breakpoints at the end of the
  system prompt and tool specs). The stable prefix is `system` + `tools`; `ContextSource` sections live
  in the system prompt — if a source changes on every step, the cache is invalidated every time.
  Deliver frequently changing data via tools or messages, not via context sources.
- Handle differences in message formats (e.g. tool results as a dedicated role)
  exclusively inside the adapter.
- Report `capabilities` honestly; the router selects providers based on them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ..core.errors import ConfigError
from ..core.types import Block, Message, ModelRequest, ModelResponse, StopReason, TextBlock, ThinkingBlock

EMPTY_RESPONSE_NOTE: Final = "(empty response)"
"""Sent instead of an assistant message that has no content left (the APIs reject empty messages)."""

SDK_TIMEOUT_S: Final = 600.0
"""Explicit SDK timeout: lifts the SDK's refusal of big requests; the router owns the real request deadline."""

_RETRYABLE_STATUS = frozenset({408, 409, 429})


def is_retryable_status(status: int) -> bool:
    """Shared HTTP classification for adapters: rate limits, conflicts, timeouts, and server errors are retryable."""
    return status in _RETRYABLE_STATUS or status >= 500


@dataclass(frozen=True)
class StreamDelta:
    """Incremental chunk produced by a streaming model provider."""

    text: str = ""
    thinking: str = ""


class Capabilities(BaseModel):
    """What a model supports. The router skips models that lack required capabilities."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tools: bool = True
    vision: bool = False
    structured_output: bool = False


@runtime_checkable
class ModelProvider(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def capabilities(self) -> Capabilities: ...

    async def complete(self, request: ModelRequest) -> ModelResponse: ...


ModelFn = Callable[[ModelRequest], Awaitable[ModelResponse | str]]


class CallableProvider:
    """Wraps an async function `(ModelRequest) -> ModelResponse | str` as a `ModelProvider`.

    Useful for small scripts, custom HTTP clients, or connecting external SDKs without writing
    a full adapter class. If the function returns a plain `str`, it is wrapped into a
    `ModelResponse` with `StopReason.END`.
    """

    def __init__(
        self,
        fn: ModelFn,
        *,
        id: str = "callable",
        tools: bool = True,
        vision: bool = False,
    ) -> None:
        if not callable(fn):
            raise ConfigError("CallableProvider requires a callable")
        if not id.strip():
            raise ConfigError("CallableProvider: id must not be empty")
        self._fn = fn
        self._id = id
        self._capabilities = Capabilities(tools=tools, vision=vision)

    @property
    def id(self) -> str:
        return self._id

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    async def complete(self, request: ModelRequest) -> ModelResponse:
        result = await self._fn(request)
        if isinstance(result, str):
            return ModelResponse(
                message=Message(role="assistant", content=(TextBlock(text=result),)),
                stop_reason=StopReason.END,
            )
        return result

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamDelta | ModelResponse]:
        res = await self.complete(request)
        if res.message.text:
            yield StreamDelta(text=res.message.text)
        yield res


FOREIGN_THINKING_NOTE = "(Reasoning from another model omitted)"


def without_foreign_thinking(messages: Sequence[Message], provider_id: str) -> tuple[Message, ...]:
    """Removes thinking blocks that were provably created by a different provider.

    Blocks without `provider_id` (saved before schema v2) are kept — their origin is unknown,
    and the adapter decides. A message never becomes empty; if it consisted solely of foreign reasoning,
    a short placeholder note replaces it. Roles and tool calls remain unchanged so the protocol stays valid.
    """
    cleaned: list[Message] = []
    for message in messages:
        kept: list[Block] = [
            b
            for b in message.content
            if not (isinstance(b, ThinkingBlock) and b.provider_id is not None and b.provider_id != provider_id)
        ]
        if len(kept) == len(message.content):
            cleaned.append(message)
            continue
        cleaned.append(message.model_copy(update={"content": tuple(kept) or (TextBlock(text=FOREIGN_THINKING_NOTE),)}))
    return tuple(cleaned)
