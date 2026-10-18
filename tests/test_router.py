"""Router: retry, backoff, timeout, fallback, and capabilities."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ai_agent_harness import Capabilities, EventType, ModelRequest, ModelResponse, ModelRouter, RetryPolicy
from ai_agent_harness.core import ConfigError, ImageBlock, Message, ProviderError, ToolSpec
from ai_agent_harness.testing import FakeProvider, text_response

from .conftest import run_sync

REQUEST = ModelRequest(system="s", messages=(Message.user("hi"),))


class Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[EventType, dict[str, Any]]] = []
        self.sleeps: list[float] = []

    async def emit(self, type_: EventType, data: dict[str, Any]) -> None:
        self.events.append((type_, data))

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


def router(*providers: FakeProvider, rec: Recorder, **policy: Any) -> ModelRouter:
    kwargs: dict[str, Any] = {"base_delay_s": 1.0, **policy}
    return ModelRouter(providers[0], providers[1:], RetryPolicy(**kwargs), sleep=rec.sleep, jitter=lambda: 1.0)


def test_retryable_errors_are_retried_with_exponential_backoff() -> None:
    rec = Recorder()
    p = FakeProvider([ProviderError("429", retryable=True), ProviderError("529", retryable=True), text_response("ok")])
    response, pid = run_sync(router(p, rec=rec, max_attempts=3).complete(REQUEST, rec.emit))
    assert response.message.text == "ok" and pid == "fake"
    assert rec.sleeps == [1.0, 2.0]
    assert [t for t, _ in rec.events] == [EventType.PROVIDER_RETRY] * 2


def test_non_retryable_error_goes_straight_to_fallback() -> None:
    rec = Recorder()
    primary = FakeProvider([ProviderError("400", retryable=False)], id="a")
    backup = FakeProvider([text_response("backup")], id="b")
    _, pid = run_sync(router(primary, backup, rec=rec).complete(REQUEST, rec.emit))
    assert pid == "b"
    assert rec.sleeps == []
    assert rec.events[0][0] is EventType.PROVIDER_FAILOVER


def test_all_providers_failing_raises_provider_error() -> None:
    rec = Recorder()
    a = FakeProvider([ProviderError("x", retryable=True)] * 2, id="a")
    b = FakeProvider([ProviderError("y", retryable=False)], id="b")
    with pytest.raises(ProviderError, match="All models"):
        run_sync(router(a, b, rec=rec, max_attempts=2).complete(REQUEST, rec.emit))


def test_timeout_is_retryable() -> None:
    class Hanging(FakeProvider):
        async def complete(self, request: ModelRequest) -> ModelResponse:
            await asyncio.sleep(10)
            raise AssertionError

    rec = Recorder()
    backup = FakeProvider([text_response("ok")], id="b")
    r = router(Hanging([], id="a"), backup, rec=rec, max_attempts=2, request_timeout_s=0.01)
    _, pid = run_sync(r.complete(REQUEST, rec.emit))
    assert pid == "b"
    assert len(rec.sleeps) == 1


def test_adapter_returning_garbage_is_handled() -> None:
    class Garbage(FakeProvider):
        async def complete(self, request: ModelRequest) -> ModelResponse:
            return "not a ModelResponse"  # type: ignore[return-value]

    rec = Recorder()
    with pytest.raises(ProviderError, match="invalid response"):
        run_sync(router(Garbage([]), rec=rec).complete(REQUEST, rec.emit))


def test_capabilities_filter_candidates() -> None:
    rec = Recorder()
    no_tools = FakeProvider([], id="text-only", capabilities=Capabilities(tools=False))
    full = FakeProvider([text_response("ok")], id="full", capabilities=Capabilities(tools=True, vision=True))
    request = REQUEST.model_copy(update={"tools": (ToolSpec(name="x", description="d", input_schema={}),)})
    _, pid = run_sync(router(no_tools, full, rec=rec).complete(request, rec.emit))
    assert pid == "full" and no_tools.requests == []


def test_no_capable_provider_fails_fast() -> None:
    rec = Recorder()
    text_only = FakeProvider([], capabilities=Capabilities(vision=False))
    image = Message(role="user", content=(ImageBlock(media_type="image/png", data="aGk="),))
    with pytest.raises(ProviderError, match="satisfies"):
        run_sync(router(text_only, rec=rec).complete(ModelRequest(system="s", messages=(image,)), rec.emit))


def test_backoff_is_capped_and_jittered() -> None:
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=5.0)
    assert policy.delay(10, jitter=1.0) == 5.0
    assert policy.delay(1, jitter=0.0) == 0.5


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ConfigError):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ConfigError):
        ModelRouter(FakeProvider([], id="x"), [FakeProvider([], id="x")])
