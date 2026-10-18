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
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from ..core.errors import ConfigError, ProviderError
from ..core.events import Emit, EventType
from ..core.types import ModelRequest, ModelResponse
from .base import ModelProvider

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

    def candidates(self, request: ModelRequest) -> tuple[ModelProvider, ...]:
        return tuple(
            p
            for p in self._providers
            if (p.capabilities.tools or not request.tools) and (p.capabilities.vision or not request.has_images)
        )

    async def complete(self, request: ModelRequest, emit: Emit) -> tuple[ModelResponse, str]:
        """Returns `(response, provider_id_that_answered)`."""
        candidates = self.candidates(request)
        if not candidates:
            raise ProviderError("No configured model satisfies the requirements of this request", retryable=False)

        last_error: ProviderError | None = None
        for index, provider in enumerate(candidates):
            if index > 0:
                await emit(
                    EventType.PROVIDER_FAILOVER,
                    {"to": provider.id, "retryable": bool(last_error and last_error.retryable)},
                )
            for attempt in range(1, self._retry.max_attempts + 1):
                try:
                    # Adapters are external code: verify return type at runtime, not just via type annotations
                    response: object = await asyncio.wait_for(provider.complete(request), self._retry.request_timeout_s)
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
                if not last_error.retryable or attempt == self._retry.max_attempts:
                    break
                delay = self._retry.delay(attempt, self._jitter())
                await emit(
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
