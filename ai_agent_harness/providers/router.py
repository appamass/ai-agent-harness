"""Model selection with retry, exponential backoff, timeout, and fallback.

Flow per request:
1. Candidates = primary model + fallbacks, filtered by capabilities
   (tools in request → model must support tools; images → vision).
2. Up to `max_attempts` attempts per candidate. Retries happen only on
   `ProviderError(retryable=True)` or timeout, with backoff + jitter.
3. Non-retryable error or attempts exhausted → next candidate.
4. All candidates failed → `ProviderError`; the runner then finishes the run
   with status `failed` (resumable once the provider recovers).

Unexpected exceptions from adapters are logged and treated as non-retryable
errors — an adapter bug must never crash the process.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import cast

from ..core.errors import ConfigError, ProviderError
from ..core.events import Emit, EventType, noop_emit
from ..core.types import ModelRequest, ModelResponse
from .base import ModelProvider, StreamDelta

log = logging.getLogger("ai_agent_harness.providers")

Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    request_timeout_s: float = 300.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ConfigError("RetryPolicy.max_attempts must be >= 1")
        if self.base_delay_s < 0 or self.max_delay_s < self.base_delay_s:
            raise ConfigError("RetryPolicy: 0 <= base_delay_s <= max_delay_s")
        if self.request_timeout_s <= 0:
            raise ConfigError("RetryPolicy.request_timeout_s must be > 0")

    def delay(self, attempt: int, jitter: float) -> float:
        """Exponential backoff with jitter in [50%, 100%] of the raw delay."""
        raw = min(self.max_delay_s, self.base_delay_s * 2.0 ** (attempt - 1))
        return raw * (0.5 + jitter / 2)


class ModelRouter:
    def __init__(
        self,
        primary: ModelProvider,
        fallbacks: Sequence[ModelProvider] = (),
        retry: RetryPolicy | None = None,
        *,
        sleep: Sleep = asyncio.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self._providers: tuple[ModelProvider, ...] = (primary, *fallbacks)
        ids = [p.id for p in self._providers]
        if len(ids) != len(set(ids)):
            raise ConfigError(f"Provider IDs must be unique: {ids}")
        self._retry = retry or RetryPolicy()
        self._sleep = sleep
        self._jitter = jitter

    @property
    def providers(self) -> tuple[ModelProvider, ...]:
        return self._providers

    def candidates(self, request: ModelRequest) -> tuple[ModelProvider, ...]:
        has_images = request.has_images
        return tuple(
            p
            for p in self._providers
            if (p.capabilities.tools or not request.tools)
            and (p.capabilities.vision or not has_images)
            and (p.capabilities.structured_output or not request.output_schema)
        )

    async def complete(self, request: ModelRequest, emit: Emit | None = None) -> tuple[ModelResponse, str]:
        """Returns `(response, provider_id_that_answered)`."""
        return await self._route(request, emit, lambda provider: self._complete_once(provider, request))

    async def _complete_once(self, provider: ModelProvider, request: ModelRequest) -> object:
        # Adapters are external code: the result is verified at runtime by `_route`, not trusted from annotations
        return await asyncio.wait_for(provider.complete(request), self._retry.request_timeout_s)

    async def stream(
        self,
        request: ModelRequest,
        emit: Emit | None = None,
        on_delta: Callable[[StreamDelta], Awaitable[None]] | None = None,
    ) -> tuple[ModelResponse, str]:
        """Streams the response, with the same retry and failover as `complete`.

        Calls `on_delta` with each `StreamDelta`. Once a delta has been delivered, a failure is raised as it
        is: a retry or another model would show the consumer a second, different answer.
        Providers without `stream()` answer through `complete()` as a single delta.
        """
        delivered = 0

        async def attempt(provider: ModelProvider) -> object:
            nonlocal delivered
            delivered = 0
            stream_fn = getattr(provider, "stream", None)
            if not callable(stream_fn):
                response = await self._complete_once(provider, request)
                if isinstance(response, ModelResponse) and on_delta is not None and response.message.text:
                    await on_delta(StreamDelta(text=response.message.text))
                    delivered += 1
                return response
            final: ModelResponse | None = None
            loop = asyncio.get_running_loop()
            async with asyncio.timeout(self._retry.request_timeout_s) as deadline:
                async for chunk in cast(AsyncIterator[object], stream_fn(request)):
                    if isinstance(chunk, StreamDelta):
                        if on_delta is not None and (chunk.text or chunk.thinking):
                            # A slow consumer is not a slow provider: the provider's clock stops while it holds us up
                            when = deadline.when()
                            left = None if when is None else when - loop.time()
                            deadline.reschedule(None)
                            await on_delta(chunk)
                            deadline.reschedule(None if left is None else loop.time() + left)
                            if chunk.text:  # reasoning deltas are not shown, so they do not commit us to this answer
                                delivered += 1
                    elif isinstance(chunk, ModelResponse):
                        final = chunk
            if final is None:
                raise ProviderError(
                    "Streaming provider finished without returning ModelResponse",
                    retryable=False,
                    provider_id=provider.id,
                )
            return final

        return await self._route(request, emit, attempt, committed=lambda: delivered > 0)

    async def _route(
        self,
        request: ModelRequest,
        emit: Emit | None,
        attempt_call: Callable[[ModelProvider], Awaitable[object]],
        *,
        committed: Callable[[], bool] = lambda: False,
    ) -> tuple[ModelResponse, str]:
        """Retry, backoff, and failover over the capable models. `committed()`: output already reached the consumer."""
        active_emit = emit or noop_emit
        candidates = self.candidates(request)
        if not candidates:
            raise ProviderError("No configured model satisfies the requirements of this request", retryable=False)

        last_error: ProviderError | None = None
        for index, provider in enumerate(candidates):
            if index > 0:
                await active_emit(
                    EventType.PROVIDER_FAILOVER,
                    {"to": provider.id, "retryable": bool(last_error and last_error.retryable)},
                )
            for attempt in range(1, self._retry.max_attempts + 1):
                try:
                    response = await attempt_call(provider)
                except TimeoutError:
                    last_error = ProviderError("Request timed out", retryable=True, provider_id=provider.id)
                except ProviderError as exc:
                    last_error = exc
                except Exception as exc:  # adapter bug: log it, never crash the process
                    log.exception("Unexpected error in adapter %s", provider.id)
                    last_error = ProviderError(
                        f"Internal error in adapter ({type(exc).__name__})", retryable=False, provider_id=provider.id
                    )
                else:
                    if isinstance(response, ModelResponse):
                        return response, provider.id
                    log.error("Adapter %s returned %s instead of ModelResponse", provider.id, type(response).__name__)
                    last_error = ProviderError(
                        "Adapter returned an invalid response", retryable=False, provider_id=provider.id
                    )
                if committed():
                    raise last_error
                if not last_error.retryable or attempt == self._retry.max_attempts:
                    break
                delay = self._retry.delay(attempt, self._jitter())
                await active_emit(
                    EventType.PROVIDER_RETRY,
                    {
                        "provider": provider.id,
                        "attempt": attempt,
                        "delay_s": round(delay, 2),
                    },  # no error text (ADR 0011)
                )
                await self._sleep(delay)

        if last_error is None:  # pragma: no cover – unreachable as long as candidates exist
            raise ProviderError("No model was called", retryable=False)
        raise ProviderError(
            f"All models failed, last error: {last_error}",
            retryable=last_error.retryable,
            provider_id=last_error.provider_id,
        )
