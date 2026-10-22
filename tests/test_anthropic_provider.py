"""Anthropic adapter: translation, error classification, and a full run — against the real SDK.

The SDK is real; only the network is replaced by an `httpx2` mock transport that returns the wire-format
bodies from `tests/fixtures/anthropic/`. So request serialization, response parsing, and error classes are
exercised exactly as in production. Live API calls live in `tests/live/` (`pytest -m live`).
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest

from ai_agent_harness import (
    Agent,
    ContextBuilder,
    ModelRequest,
    ModelRouter,
    RetryPolicy,
    RunStatus,
    ToolRegistry,
    start,
)
from ai_agent_harness.core import ConfigError, ProviderError
from ai_agent_harness.core.types import (
    ImageBlock,
    Message,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from ai_agent_harness.providers.anthropic import AnthropicProvider
from ai_agent_harness.providers.base import EMPTY_RESPONSE_NOTE, FOREIGN_THINKING_NOTE

from .conftest import Echo, no_sleep, run_sync

FIXTURES = Path(__file__).parent / "fixtures" / "anthropic"
API_KEY = "sk-ant-test-key-1234567890"
MODEL = "claude-haiku-5-5"


def fixture(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    return data


def ok(name: str, **changes: Any) -> httpx2.Response:
    return httpx2.Response(200, json={**fixture(name), **changes})


class Wire:
    """Scripted HTTP responses; records every request body the SDK sends."""

    def __init__(self, *script: httpx2.Response | Exception) -> None:
        self._script = list(script)
        self.bodies: list[dict[str, Any]] = []
        self.headers: list[httpx2.Headers] = []

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.bodies.append(json.loads(request.content))
        self.headers.append(request.headers)
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def client(self) -> anthropic.AsyncAnthropic:
        return anthropic.AsyncAnthropic(
            api_key=API_KEY,
            max_retries=0,
            http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(self._handle)),
        )


def provider_for(wire: Wire, **kwargs: Any) -> AnthropicProvider:
    return AnthropicProvider(MODEL, client=wire.client(), **kwargs)


ECHO_SPEC = ToolSpec(name="echo", description="Returns 'echo'.", input_schema={"type": "object", "properties": {}})


def request(*messages: Message, **kwargs: Any) -> ModelRequest:
    return ModelRequest(system="You are a test agent.", messages=messages or (Message.user("hi"),), **kwargs)


def complete(provider: AnthropicProvider, req: ModelRequest | None = None) -> Any:
    return run_sync(provider.complete(req or request()))


# ------------------------------------------------------------------ Response translation


def test_text_response_is_translated_with_usage_semantics() -> None:
    response = complete(provider_for(Wire(ok("text"))))
    assert response.message.role == "assistant" and response.message.text == "Paris is the capital of France."
    assert response.stop_reason is StopReason.END
    assert response.model == MODEL
    usage = response.usage
    assert (usage.input_tokens, usage.output_tokens) == (25, 40)  # input excludes cache, output includes reasoning
    assert (usage.cache_read_tokens, usage.cache_write_tokens, usage.reasoning_tokens) == (900, 30, 25)


def test_tool_use_becomes_tool_call() -> None:
    response = complete(provider_for(Wire(ok("tool_use"))))
    assert response.stop_reason is StopReason.TOOL_USE
    assert response.message.tool_calls == (ToolCall(id="toolu_fixture_1", name="echo", arguments={}),)
    assert response.message.text == "I will read the file."


def test_thinking_block_keeps_signature_and_stays_unstamped() -> None:
    response = complete(provider_for(Wire(ok("thinking_tool_use"))))
    thinking = response.message.content[0]
    assert isinstance(thinking, ThinkingBlock)
    assert thinking.text == "" and thinking.provider_id is None  # the runner stamps provider_id, not the adapter
    assert thinking.provider_data == {"kind": "thinking", "signature": "sig-fixture-abc"}


def test_redacted_thinking_is_kept_opaque() -> None:
    response = complete(provider_for(Wire(ok("redacted_thinking"))))
    thinking = response.message.content[0]
    assert isinstance(thinking, ThinkingBlock)
    assert thinking.provider_data == {"kind": "redacted_thinking", "data": "enc-fixture-xyz"}


@pytest.mark.parametrize(
    ("api_value", "expected"),
    [
        ("end_turn", StopReason.END),
        ("stop_sequence", StopReason.END),
        ("pause_turn", StopReason.END),
        ("tool_use", StopReason.TOOL_USE),
        ("max_tokens", StopReason.MAX_TOKENS),
        ("model_context_window_exceeded", StopReason.MAX_TOKENS),
        ("refusal", StopReason.REFUSAL),
    ],
)
def test_stop_reasons_are_mapped(api_value: str, expected: StopReason) -> None:
    response = complete(provider_for(Wire(ok("text", stop_reason=api_value))))
    assert response.stop_reason is expected


def test_refusal_without_content_still_yields_a_valid_message() -> None:
    response = complete(provider_for(Wire(ok("refusal"))))
    assert response.stop_reason is StopReason.REFUSAL
    assert response.message.text == ""


# ------------------------------------------------------------------ Request translation


def test_request_shape_has_no_sampling_parameters_and_enables_caching() -> None:
    wire = Wire(ok("text"))
    complete(provider_for(wire), request(tools=(ECHO_SPEC,), max_tokens=500))
    body = wire.bodies[0]
    assert body["model"] == MODEL and body["max_tokens"] == 500 and body["system"] == "You are a test agent."
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert body["tools"] == [{"name": "echo", "description": "Returns 'echo'.", "input_schema": ECHO_SPEC.input_schema}]
    assert body["cache_control"] == {"type": "ephemeral"}
    for forbidden in ("temperature", "top_p", "top_k", "thinking", "output_config", "tool_choice"):
        assert forbidden not in body  # Haiku 5.5 rejects non-default sampling; thinking and effort use model defaults


def test_caching_and_effort_are_configurable() -> None:
    wire = Wire(ok("text"))
    complete(provider_for(wire, cache=False, effort="low"))
    body = wire.bodies[0]
    assert "cache_control" not in body
    assert body["output_config"] == {"effort": "low"}


def test_api_key_is_sent_only_as_header() -> None:
    wire = Wire(ok("text"))
    complete(provider_for(wire))
    assert wire.headers[0]["x-api-key"] == API_KEY
    assert API_KEY not in json.dumps(wire.bodies[0])


def test_temperature_fails_loudly_before_any_network_call() -> None:
    wire = Wire(ok("text"))
    with pytest.raises(ProviderError, match="temperature") as caught:
        complete(provider_for(wire), request(temperature=0.2))
    assert caught.value.retryable is False and wire.bodies == []


def test_tool_results_come_first_and_errors_are_flagged() -> None:
    history = (
        Message.user("go"),
        Message(role="assistant", content=(ToolCall(id="toolu_1", name="echo"), ToolCall(id="toolu_2", name="echo"))),
        Message(
            role="user",
            content=(
                TextBlock(text="Your last response was rejected: x"),
                ToolResult(call_id="toolu_2", content="boom", is_error=True),
                ToolResult(call_id="toolu_1", content="echo"),
            ),
        ),
    )
    wire = Wire(ok("text"))
    complete(provider_for(wire), request(*history))
    sent = wire.bodies[0]["messages"]
    assert sent[1]["content"] == [
        {"type": "tool_use", "id": "toolu_1", "name": "echo", "input": {}},
        {"type": "tool_use", "id": "toolu_2", "name": "echo", "input": {}},
    ]
    assert sent[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_2", "content": "boom", "is_error": True},
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "echo"},
        {"type": "text", "text": "Your last response was rejected: x"},
    ]


def test_images_are_sent_as_base64() -> None:
    image = ImageBlock(media_type="image/png", data="aGVsbG8=")
    wire = Wire(ok("text"))
    complete(provider_for(wire), request(Message(role="user", content=(image, TextBlock(text="what is this?")))))
    assert wire.bodies[0]["messages"][0]["content"] == [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}},
        {"type": "text", "text": "what is this?"},
    ]


def test_vision_capability_is_reported_honestly() -> None:
    assert provider_for(Wire(), vision=False).capabilities.vision is False
    assert provider_for(Wire()).capabilities.tools is True


def history_with_assistant(*blocks: Any) -> tuple[Message, ...]:
    return (Message.user("go"), Message(role="assistant", content=tuple(blocks)), Message.user("next"))


def test_own_thinking_is_sent_back_unchanged_before_tool_use() -> None:
    own = ThinkingBlock(text="", provider_data={"kind": "thinking", "signature": "sig-1"}, provider_id="anthropic")
    redacted = ThinkingBlock(provider_data={"kind": "redacted_thinking", "data": "enc-1"}, provider_id="anthropic")
    wire = Wire(ok("text"))
    complete(provider_for(wire), request(*history_with_assistant(own, redacted, TextBlock(text="ok"))))
    assert wire.bodies[0]["messages"][1]["content"] == [
        {"type": "thinking", "thinking": "", "signature": "sig-1"},
        {"type": "redacted_thinking", "data": "enc-1"},
        {"type": "text", "text": "ok"},
    ]


def test_thinking_of_another_provider_is_stripped_and_never_leaves_an_empty_message() -> None:
    foreign = ThinkingBlock(text="secret", provider_data={"signature": "x"}, provider_id="openai")
    wire = Wire(ok("text"))
    complete(provider_for(wire), request(*history_with_assistant(foreign)))
    assert wire.bodies[0]["messages"][1]["content"] == [{"type": "text", "text": FOREIGN_THINKING_NOTE}]
    assert "secret" not in json.dumps(wire.bodies[0])


def test_thinking_of_unknown_origin_without_signature_is_dropped() -> None:
    unusable = ThinkingBlock(text="from an old checkpoint")
    wire = Wire(ok("text"))
    complete(provider_for(wire), request(*history_with_assistant(unusable, TextBlock(text="answer"))))
    assert wire.bodies[0]["messages"][1]["content"] == [{"type": "text", "text": "answer"}]


def test_empty_text_blocks_are_skipped_and_empty_messages_get_a_note() -> None:
    wire = Wire(ok("text"))
    complete(provider_for(wire), request(*history_with_assistant(TextBlock(text=""))))
    assert wire.bodies[0]["messages"][1]["content"] == [{"type": "text", "text": EMPTY_RESPONSE_NOTE}]


# ------------------------------------------------------------------ Real recorded responses

RECORDED = sorted((FIXTURES / "recorded").glob("*.json"))


@pytest.mark.parametrize("path", RECORDED, ids=[p.stem for p in RECORDED])
def test_recorded_real_responses_translate_and_round_trip(path: Path) -> None:
    """Responses recorded from the live API (see the fixtures README): translation and the signature round trip."""
    assert any(  # guard: the signature half of this test must not pass silently on files without thinking
        b["type"] == "thinking" for p in RECORDED for b in json.loads(p.read_text(encoding="utf-8"))["content"]
    )
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    response = complete(provider_for(Wire(httpx2.Response(200, json=raw))))

    assert response.model == raw["model"]
    assert (response.usage.input_tokens, response.usage.output_tokens) == (
        raw["usage"]["input_tokens"],
        raw["usage"]["output_tokens"],
    )
    assert (response.usage.cache_read_tokens, response.usage.cache_write_tokens) == (
        raw["usage"].get("cache_read_input_tokens") or 0,
        raw["usage"].get("cache_creation_input_tokens") or 0,
    )
    details: dict[str, int] = raw["usage"].get("output_tokens_details") or {}
    thinking_tokens = details.get("thinking_tokens", 0)
    assert response.usage.reasoning_tokens == thinking_tokens <= response.usage.output_tokens
    expected_stop = {"end_turn": StopReason.END, "max_tokens": StopReason.MAX_TOKENS, "tool_use": StopReason.TOOL_USE}
    assert response.stop_reason is expected_stop[raw["stop_reason"]]
    raw_calls = [b for b in raw["content"] if b["type"] == "tool_use"]
    assert [(c.id, c.name, c.arguments) for c in response.message.tool_calls] == [
        (b["id"], b["name"], b["input"]) for b in raw_calls
    ]

    # Send the assistant message back as history: thinking blocks must keep their exact signature, in order
    stamped = Message(
        role="assistant",
        content=tuple(
            b.model_copy(update={"provider_id": "anthropic"}) if isinstance(b, ThinkingBlock) else b
            for b in response.message.content
        ),
    )
    wire = Wire(ok("text"))
    complete(provider_for(wire), request(Message.user("go"), stamped, Message.user("next")))
    sent = wire.bodies[0]["messages"][1]["content"]
    raw_thinking = [b for b in raw["content"] if b["type"] in ("thinking", "redacted_thinking")]
    assert [b for b in sent if b["type"] in ("thinking", "redacted_thinking")] == [
        {k: v for k, v in b.items() if k != "type"} | {"type": b["type"]} for b in raw_thinking
    ]
    assert [b["text"] for b in sent if b["type"] == "text"] == [
        b["text"] for b in raw["content"] if b["type"] == "text" and b["text"]
    ]


# ------------------------------------------------------------------ Errors


def error_body(kind: str) -> dict[str, Any]:
    return {"type": "error", "error": {"type": kind, "message": "SECRET echo of the prompt"}}


@pytest.mark.parametrize(
    ("status", "kind", "retryable"),
    [
        (400, "invalid_request_error", False),
        (401, "authentication_error", False),
        (403, "permission_error", False),
        (404, "not_found_error", False),
        (413, "request_too_large", False),
        (422, "invalid_request_error", False),
        (408, "timeout_error", True),
        (409, "conflict_error", True),
        (429, "rate_limit_error", True),
        (500, "api_error", True),
        (503, "api_error", True),
        (504, "timeout_error", True),
        (529, "overloaded_error", True),
    ],
)
def test_http_errors_are_classified_and_leak_nothing(status: int, kind: str, retryable: bool) -> None:
    wire = Wire(httpx2.Response(status, json=error_body(kind)))
    with pytest.raises(ProviderError) as caught:
        complete(provider_for(wire, id="claude"))
    exc = caught.value
    assert exc.retryable is retryable and exc.provider_id == "claude"
    assert str(status) in str(exc) and kind in str(exc)
    assert "SECRET" not in str(exc) and API_KEY not in str(exc)  # no echoed prompt text, no key


def test_non_json_error_body_is_still_classified() -> None:
    wire = Wire(httpx2.Response(502, text="<html>Bad gateway</html>"))
    with pytest.raises(ProviderError) as caught:
        complete(provider_for(wire))
    assert caught.value.retryable is True and "Bad gateway" not in str(caught.value)


@pytest.mark.parametrize(
    "bad",
    [
        httpx2.Response(200, text="<html>not json</html>", headers={"content-type": "text/html"}),
        httpx2.Response(200, json={"unexpected": True}),
        httpx2.Response(200, json={**fixture("text"), "content": "a string instead of blocks"}),
        httpx2.Response(200, json={k: v for k, v in fixture("text").items() if k != "usage"}),
    ],
)
def test_malformed_success_responses_become_non_retryable_provider_errors(bad: httpx2.Response) -> None:
    with pytest.raises(ProviderError) as caught:
        complete(provider_for(Wire(bad)))
    assert caught.value.retryable is False and caught.value.provider_id == "anthropic"


def test_large_max_tokens_work_without_streaming() -> None:
    """The SDK refuses big non-streaming requests unless a timeout is given; the router owns the real timeout."""
    wire = Wire(ok("text"))
    complete(provider_for(wire), request(max_tokens=64_000))
    assert wire.bodies[0]["max_tokens"] == 64_000


def test_provider_errors_do_not_keep_the_sdk_exception_reachable() -> None:
    wire = Wire(httpx2.Response(400, json=error_body("invalid_request_error")))
    with pytest.raises(ProviderError) as caught:
        complete(provider_for(wire))
    exc = caught.value
    assert exc.__cause__ is None and exc.__context__ is None  # the SDK exception holds the echoed response body


def test_tool_input_that_is_not_an_object_becomes_arguments_error() -> None:
    body = fixture("tool_use")
    body["content"][1]["input"] = "not an object"
    response = complete(provider_for(Wire(httpx2.Response(200, json=body))))
    call = response.message.tool_calls[0]
    assert call.arguments == {} and call.arguments_error


def test_tool_call_ids_are_made_safe_for_the_api_consistently() -> None:
    history = (
        Message.user("go"),
        Message(role="assistant", content=(ToolCall(id="call.1:abc", name="echo"),)),
        Message(role="user", content=(ToolResult(call_id="call.1:abc", content="echo"),)),
    )
    wire = Wire(ok("text"))
    complete(provider_for(wire), request(*history))
    sent = wire.bodies[0]["messages"]
    assert sent[1]["content"][0]["id"] == "call_1_abc"  # the API only accepts [A-Za-z0-9_-]
    assert sent[2]["content"][0]["tool_use_id"] == "call_1_abc"


NETWORK_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.mark.parametrize(
    "failure",
    [
        httpx2.ConnectError("no route to host", request=NETWORK_REQUEST),
        httpx2.ReadTimeout("timed out", request=NETWORK_REQUEST),
    ],
)
def test_network_problems_are_retryable(failure: Exception) -> None:
    with pytest.raises(ProviderError) as caught:
        complete(provider_for(Wire(failure)))
    assert caught.value.retryable is True


# ------------------------------------------------------------------ Construction


def test_own_client_never_retries_because_the_router_owns_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class Recorder:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(anthropic, "AsyncAnthropic", Recorder)
    AnthropicProvider(MODEL, api_key="k")
    assert captured["max_retries"] == 0 and captured["api_key"] == "k"


def test_aclose_closes_only_a_client_the_adapter_created(monkeypatch: pytest.MonkeyPatch) -> None:
    injected = Wire().client()  # built before the patch below replaces the SDK client class
    closed: list[bool] = []

    class Recorder:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(anthropic, "AsyncAnthropic", Recorder)
    run_sync(AnthropicProvider(MODEL, api_key="k").aclose())
    assert closed == [True]  # its own client: closed

    run_sync(AnthropicProvider(MODEL, client=injected).aclose())
    assert injected.is_closed() is False  # a client the caller passed in stays the caller's


def test_missing_sdk_gives_an_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)  # makes `import anthropic` fail like an uninstalled extra
    monkeypatch.delitem(sys.modules, "ai_agent_harness.providers.anthropic")
    with pytest.raises(ConfigError, match=r"ai-agent-harness\[anthropic\]"):
        importlib.import_module("ai_agent_harness.providers.anthropic")


def test_model_must_not_be_empty() -> None:
    with pytest.raises(ConfigError):
        AnthropicProvider(" ", client=Wire().client())


# ------------------------------------------------------------------ Full run


def test_full_run_keeps_the_protocol_and_returns_the_signature() -> None:
    wire = Wire(ok("thinking_tool_use"), ok("text"))
    agent = Agent(
        name="live-like",
        router=ModelRouter(provider_for(wire), retry=RetryPolicy(max_attempts=1), sleep=no_sleep),
        context=ContextBuilder("You are a test agent."),
        tools=ToolRegistry([Echo()]),
    )
    state = run_sync(start(agent, "use the tool"))
    assert state.status is RunStatus.DONE and state.last_text == "Paris is the capital of France."
    state.check_protocol()

    first_assistant = state.messages[1].content[0]
    assert isinstance(first_assistant, ThinkingBlock) and first_assistant.provider_id == "anthropic"
    second_request = wire.bodies[1]["messages"]
    assert [m["role"] for m in second_request] == ["user", "assistant", "user"]
    assert second_request[1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig-fixture-abc"}
    assert second_request[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_fixture_2", "content": "echo"}
    ]
    assert [t["name"] for t in wire.bodies[1]["tools"]] == ["echo"]
