"""Router: retry, backoff, timeout, fallback, and capabilities."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ai_agent_harness import (
    Capabilities,
    EventType,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ModelRouter,
    RetryPolicy,
    StreamDelta,
)
from ai_agent_harness.core import ConfigError, ImageBlock, Message, ProviderError, ToolSpec
from ai_agent_harness.testing import FakeProvider, Streamed, text_response

from .conftest import no_sleep, run_sync

REQUEST = ModelRequest(system="s", messages=(Message.user("hi"),))


class Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[EventType, dict[str, Any]]] = []
        self.sleeps: list[float] = []

    async def emit(self, type_: EventType, data: dict[str, Any]) -> None:
        self.events.append((type_, data))

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


def router(*providers: ModelProvider, rec: Recorder, **policy: Any) -> ModelRouter:
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


def test_structured_output_capabilities_filter_candidates() -> None:
    rec = Recorder()
    no_so = FakeProvider([], id="no-so", capabilities=Capabilities(structured_output=False))
    so = FakeProvider([text_response('{"ok": true}')], id="so", capabilities=Capabilities(structured_output=True))
    request = REQUEST.model_copy(update={"output_schema": {"type": "object"}})
    _, pid = run_sync(router(no_so, so, rec=rec).complete(request, rec.emit))
    assert pid == "so" and no_so.requests == []


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


def test_router_stream_yields_deltas_and_returns_response() -> None:
    rec = Recorder()
    deltas: list[str] = []

    async def on_delta(d: StreamDelta) -> None:
        deltas.append(d.text)

    p = FakeProvider([text_response("streaming text response")])
    r = router(p, rec=rec)
    res, pid = run_sync(r.stream(REQUEST, rec.emit, on_delta=on_delta))
    assert res.message.text == "streaming text response"
    assert pid == "fake"
    assert "".join(deltas) == "streaming text response"


def test_router_stream_fallback_when_provider_lacks_stream() -> None:
    rec = Recorder()
    deltas: list[str] = []

    async def on_delta(d: StreamDelta) -> None:
        deltas.append(d.text)

    class OnlyCompleteProvider:
        id = "only_complete"
        capabilities = Capabilities()

        def __init__(self, response: ModelResponse) -> None:
            self.response = response

        async def complete(self, request: ModelRequest) -> ModelResponse:
            return self.response

    p = OnlyCompleteProvider(text_response("non-streaming result"))
    r = router(p, rec=rec)
    res, pid = run_sync(r.stream(REQUEST, rec.emit, on_delta=on_delta))
    assert pid == "only_complete"
    assert res.message.text == "non-streaming result"
    assert "".join(deltas) == "non-streaming result"


def test_router_stream_failover_before_deltas() -> None:
    rec = Recorder()
    primary = FakeProvider([ProviderError("500 Overloaded", retryable=True)], id="primary")
    backup = FakeProvider([text_response("backup response")], id="backup")
    r = router(primary, backup, rec=rec, max_attempts=1)
    res, pid = run_sync(r.stream(REQUEST, rec.emit))
    assert pid == "backup"
    assert res.message.text == "backup response"
    assert rec.events[0][0] is EventType.PROVIDER_FAILOVER


# ------------------------------------------------------------------ Streaming: never a second answer


def test_stream_does_not_fail_over_after_deltas_were_delivered() -> None:
    primary = FakeProvider([Streamed(StreamDelta(text="partial "), ProviderError("reset", retryable=True))], id="p")
    backup = FakeProvider([text_response("other")], id="b")
    router = ModelRouter(primary, [backup], RetryPolicy(max_attempts=2, base_delay_s=0, max_delay_s=0), sleep=no_sleep)
    seen: list[str] = []

    async def on_delta(delta: StreamDelta) -> None:
        seen.append(delta.text)

    request = ModelRequest(system="s", messages=(Message.user("hi"),))
    with pytest.raises(ProviderError):
        run_sync(router.stream(request, on_delta=on_delta))

    assert seen == ["partial "] and backup.requests == [] and len(primary.requests) == 1


def test_stream_without_a_final_response_is_an_error() -> None:
    provider = FakeProvider([Streamed(StreamDelta(text="x"))])
    router = ModelRouter(provider, retry=RetryPolicy(max_attempts=1), sleep=no_sleep)
    request = ModelRequest(system="s", messages=(Message.user("hi"),))
    with pytest.raises(ProviderError, match="without returning"):
        run_sync(router.stream(request))


def test_worst_case_covers_every_provider_attempt_and_backoff() -> None:
    retry = RetryPolicy(max_attempts=3, base_delay_s=1.0, max_delay_s=4.0, request_timeout_s=10.0)
    router = ModelRouter(FakeProvider([], id="a"), [FakeProvider([], id="b")], retry=retry)
    assert router.retry is retry
    assert router.worst_case_s() == 2 * (3 * 10.0 + 2 * 4.0)
