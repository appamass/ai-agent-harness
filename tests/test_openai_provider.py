"""OpenAI adapter: translation, error classification, and full agent runs against the real SDK.

The SDK is real; only the network is replaced by an `httpx` mock transport that returns the wire-format
bodies from `tests/fixtures/openai/`.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any, ClassVar, cast

import httpx
import openai
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
from ai_agent_harness.providers.base import EMPTY_RESPONSE_NOTE
from ai_agent_harness.providers.openai import OpenAIProvider

from .conftest import Echo, no_sleep, run_sync

FIXTURES = Path(__file__).parent / "fixtures" / "openai"
API_KEY = "sk-openai-test-key-1234567890"
MODEL = "gpt-4o"


def fixture(name: str) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8")))


def ok(name: str, **changes: Any) -> httpx.Response:
    data = {**fixture(name), **changes}
    return httpx.Response(200, json=data)


class Wire:
    """Scripted HTTP responses; records every request body the SDK sends."""

    def __init__(self, *script: httpx.Response | Exception) -> None:
        self._script = list(script)
        self.bodies: list[dict[str, Any]] = []

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def client(self) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            api_key=API_KEY,
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self._handle)),
        )


def provider_for(wire: Wire, **kwargs: Any) -> OpenAIProvider:
    return OpenAIProvider(MODEL, client=wire.client(), **kwargs)


ECHO_SPEC = ToolSpec(name="echo", description="Returns 'echo'.", input_schema={"type": "object", "properties": {}})


def request(*messages: Message, **kwargs: Any) -> ModelRequest:
    return ModelRequest(system="You are a test agent.", messages=messages or (Message.user("hi"),), **kwargs)


def complete(provider: OpenAIProvider, req: ModelRequest | None = None) -> Any:
    return run_sync(provider.complete(req or request()))


# ------------------------------------------------------------------ Configuration & Validation


def test_missing_sdk_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)
    monkeypatch.delitem(sys.modules, "ai_agent_harness.providers.openai", raising=False)
    with pytest.raises(ConfigError, match=r"ai-agent-harness\[openai\]"):
        importlib.import_module("ai_agent_harness.providers.openai")


@pytest.mark.parametrize("bad_model", ["", "   "])
def test_empty_model_name_rejected(bad_model: str) -> None:
    with pytest.raises(ConfigError, match="model must not be empty"):
        OpenAIProvider(bad_model)


@pytest.mark.parametrize("bad_id", ["", "   "])
def test_empty_id_rejected(bad_id: str) -> None:
    with pytest.raises(ConfigError, match="id must not be empty"):
        OpenAIProvider("gpt-4o", id=bad_id)


def test_cannot_pass_client_and_api_key() -> None:
    wire = Wire()
    with pytest.raises(ConfigError, match="pass either client or connection parameters"):
        OpenAIProvider("gpt-4o", client=wire.client(), api_key="sk-another-key")


# ------------------------------------------------------------------ Response Translation


def test_text_response_translation() -> None:
    wire = Wire(ok("text"))
    provider = provider_for(wire)

    response = complete(provider)
    assert response.message.role == "assistant"
    assert response.message.text == "Paris is the capital of France."
    assert response.stop_reason == StopReason.END
    assert response.model == "gpt-4o"

    # prompt_tokens=50, cached_tokens=20 -> input_tokens=30
    assert response.usage.input_tokens == 30
    assert response.usage.cache_read_tokens == 20
    assert response.usage.output_tokens == 15
    assert response.usage.reasoning_tokens == 0


def test_tool_calls_response_translation() -> None:
    wire = Wire(ok("tool_calls"))
    provider = provider_for(wire)

    response = complete(provider, request(tools=(ECHO_SPEC,)))
    assert response.stop_reason == StopReason.TOOL_USE
    calls = response.message.tool_calls
    assert len(calls) == 1
    call = calls[0]
    assert call.id == "call_abc123"
    assert call.name == "echo"
    assert call.arguments == {"message": "hello"}
    assert call.arguments_error is None


def test_malformed_tool_call_json_reports_error_without_raising() -> None:
    # Malformed JSON in function arguments
    broken_payload = fixture("tool_calls")
    broken_payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "invalid json{"

    wire = Wire(httpx.Response(200, json=broken_payload))
    provider = provider_for(wire)

    response = complete(provider)
    assert response.stop_reason == StopReason.TOOL_USE
    calls = response.message.tool_calls
    assert len(calls) == 1
    call = calls[0]
    assert call.arguments == {}
    assert call.arguments_error is not None
    assert "Invalid JSON" in call.arguments_error


def test_reasoning_content_translation() -> None:
    wire = Wire(ok("reasoning"))
    provider = provider_for(wire)

    response = complete(provider)
    assert response.message.text == "The answer is 42."
    thinking_blocks = [b for b in response.message.content if isinstance(b, ThinkingBlock)]
    assert len(thinking_blocks) == 1
    assert thinking_blocks[0].text == "Let me calculate step by step..."

    assert response.usage.input_tokens == 30  # 60 - 30 cached
    assert response.usage.output_tokens == 100
    assert response.usage.reasoning_tokens == 80


def test_max_tokens_stop_reason() -> None:
    wire = Wire(ok("max_tokens"))
    provider = provider_for(wire)

    response = complete(provider)
    assert response.stop_reason == StopReason.MAX_TOKENS


def test_refusal_stop_reason() -> None:
    wire = Wire(ok("refusal"))
    provider = provider_for(wire)

    response = complete(provider)
    assert response.stop_reason == StopReason.REFUSAL


# ------------------------------------------------------------------ Request Translation & Protocol


def test_system_prompt_and_tools_sent_to_api() -> None:
    wire = Wire(ok("text"))
    provider = provider_for(wire)

    complete(provider, ModelRequest(system="System prompt here.", messages=(Message.user("hi"),), tools=(ECHO_SPEC,)))

    sent = wire.bodies[0]
    assert sent["model"] == "gpt-4o"
    assert sent["messages"][0] == {"role": "system", "content": "System prompt here."}
    assert sent["messages"][1] == {"role": "user", "content": "hi"}
    assert "tools" in sent
    assert sent["tools"][0]["function"]["name"] == "echo"


def test_tool_result_translated_as_tool_role() -> None:
    wire = Wire(ok("text"))
    provider = provider_for(wire)

    req = ModelRequest(
        system="",
        messages=(
            Message.user("hi"),
            Message(
                role="assistant",
                content=(TextBlock(text="let me call echo"), ToolCall(id="c1", name="echo", arguments={})),
            ),
            Message(role="user", content=(TextBlock(text="next"), ToolResult(call_id="c1", content="echoed output"))),
        ),
    )
    complete(provider, req)

    sent = wire.bodies[0]["messages"]
    # In OpenAI, the ToolResult is emitted as a message with role="tool"
    tool_msgs = [m for m in sent if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0] == {"role": "tool", "tool_call_id": "c1", "content": "echoed output"}


def test_image_blocks_translated_as_image_url() -> None:
    wire = Wire(ok("text"))
    provider = provider_for(wire)

    image = ImageBlock(media_type="image/png", data="base64data")
    req = ModelRequest(system="", messages=(Message(role="user", content=(TextBlock(text="see this"), image)),))
    complete(provider, req)

    sent_user = wire.bodies[0]["messages"][0]
    assert sent_user["role"] == "user"
    assert isinstance(sent_user["content"], list)
    assert sent_user["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,base64data"},
    }


def test_empty_messages_use_empty_response_note() -> None:
    wire = Wire(ok("text"))
    provider = provider_for(wire)

    # Empty user message and assistant message
    req = ModelRequest(
        system="",
        messages=(
            Message(role="user", content=(TextBlock(text=""),)),
            Message(role="assistant", content=(TextBlock(text=""),)),
        ),
    )
    complete(provider, req)

    sent = wire.bodies[0]["messages"]
    assert sent[0]["content"] == EMPTY_RESPONSE_NOTE
    assert sent[1]["content"] == EMPTY_RESPONSE_NOTE


def test_foreign_thinking_blocks_stripped_for_openai() -> None:
    wire = Wire(ok("text"))
    provider = provider_for(wire, id="openai_target")

    # Message contains a foreign thinking block from Anthropic
    req = ModelRequest(
        system="",
        messages=(
            Message.user("hi"),
            Message(
                role="assistant",
                content=(
                    ThinkingBlock(
                        text="claude internal thought",
                        provider_id="anthropic",
                        provider_data={"kind": "thinking", "signature": "sig123"},
                    ),
                    TextBlock(text="answer"),
                ),
            ),
        ),
    )
    complete(provider, req)

    sent = wire.bodies[0]["messages"]
    # The assistant message must have TextBlock only; foreign thinking is stripped
    assistant_msg = sent[1]
    assert assistant_msg["content"] == "answer"


# ------------------------------------------------------------------ Error Handling


@pytest.mark.parametrize("status_code", [408, 409, 429, 500, 502, 503, 504])
def test_retryable_errors(status_code: int) -> None:
    wire = Wire(httpx.Response(status_code, json={"error": {"code": "rate_limit_exceeded"}}))
    provider = provider_for(wire)

    with pytest.raises(ProviderError) as exc_info:
        complete(provider)

    assert exc_info.value.retryable is True
    assert exc_info.value.provider_id == "openai"
    assert str(status_code) in str(exc_info.value)


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
def test_non_retryable_errors(status_code: int) -> None:
    wire = Wire(httpx.Response(status_code, json={"error": {"type": "invalid_request_error"}}))
    provider = provider_for(wire)

    with pytest.raises(ProviderError) as exc_info:
        complete(provider)

    assert exc_info.value.retryable is False
    assert exc_info.value.provider_id == "openai"
    assert str(status_code) in str(exc_info.value)


def test_connection_errors_are_retryable() -> None:
    wire = Wire(httpx.ConnectError("connection refused"))
    provider = provider_for(wire)

    with pytest.raises(ProviderError) as exc_info:
        complete(provider)

    assert exc_info.value.retryable is True
    assert exc_info.value.provider_id == "openai"


def test_error_messages_do_not_leak_secrets() -> None:
    # Error message with sensitive info in body
    secret_body = {"error": {"message": f"key {API_KEY} expired", "code": "invalid_api_key"}}
    wire = Wire(httpx.Response(401, json=secret_body))
    provider = provider_for(wire)

    with pytest.raises(ProviderError) as exc_info:
        complete(provider)

    # API key must NOT be present in exception string
    assert API_KEY not in str(exc_info.value)


# ------------------------------------------------------------------ Full Agent Run & Failover


def test_full_agent_run_with_openai_provider() -> None:
    # Scripted two-turn run: tool call then final answer
    wire = Wire(ok("tool_calls"), ok("text"))
    provider = provider_for(wire)

    agent = Agent(
        name="test_openai_agent",
        router=ModelRouter(provider, retry=RetryPolicy(max_attempts=1)),
        context=ContextBuilder("You are an agent."),
        tools=ToolRegistry([Echo()]),
    )

    state = run_sync(start(agent, "call echo"))
    assert state.status == RunStatus.DONE
    state.check_protocol()
    assert len(wire.bodies) == 2


def test_mid_run_failover_from_anthropic_to_openai() -> None:
    # Primary provider fails with 500 error; Fallback (OpenAI) succeeds
    fallback_wire = Wire(ok("text"))

    # We simulate a primary provider that fails with a retryable ProviderError
    from ai_agent_harness.testing import FakeProvider

    failing_primary = FakeProvider([ProviderError("Anthropic overloaded", retryable=True, provider_id="claude")])
    openai_fallback = provider_for(fallback_wire, id="openai_fallback")

    router = ModelRouter(
        failing_primary,
        fallbacks=(openai_fallback,),
        retry=RetryPolicy(max_attempts=1, base_delay_s=0.0, max_delay_s=0.0),
        sleep=no_sleep,
    )

    agent = Agent(
        name="failover_agent",
        router=router,
        context=ContextBuilder("You are an agent."),
    )

    state = run_sync(start(agent, "hello"))
    assert state.status == RunStatus.DONE
    state.check_protocol()
    assert len(fallback_wire.bodies) == 1
    assert state.messages[-1].text == "Paris is the capital of France."


# ------------------------------------------------------------------ Capabilities, token parameter, streaming


def test_structured_output_capability_can_be_switched_off_for_compatible_servers() -> None:
    wire = Wire(ok("text"))
    provider = provider_for(wire, structured_output=False)
    assert provider.capabilities.structured_output is False

    complete(provider, request(output_schema={"type": "object", "properties": {"a": {"type": "string"}}}))
    assert "response_format" not in wire.bodies[0]  # never sent to a server that cannot take it


def test_structured_output_is_on_by_default_and_sent() -> None:
    wire = Wire(ok("text"))
    provider = provider_for(wire)
    assert provider.capabilities.structured_output is True

    complete(provider, request(output_schema={"type": "object", "properties": {"a": {"type": "string"}}}))
    assert wire.bodies[0]["response_format"]["type"] == "json_schema"


def test_max_tokens_parameter_name_is_configurable() -> None:
    wire = Wire(ok("text"), ok("text"))
    complete(provider_for(wire), request(max_tokens=77))
    complete(provider_for(wire, max_tokens_param="max_completion_tokens"), request(max_tokens=77))

    assert wire.bodies[0]["max_tokens"] == 77
    assert "max_completion_tokens" not in wire.bodies[0]
    assert wire.bodies[1]["max_completion_tokens"] == 77
    assert "max_tokens" not in wire.bodies[1]


def test_invalid_max_tokens_parameter_rejected() -> None:
    with pytest.raises(ConfigError, match="max_tokens_param"):
        OpenAIProvider(MODEL, api_key=API_KEY, max_tokens_param=cast("Any", "tokens"))


def _sse(*chunks: dict[str, Any]) -> httpx.Response:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


def _chunk(delta: dict[str, Any], finish: str | None = None, usage: dict[str, Any] | None = None) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}] if delta or finish else [],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def test_streaming_keeps_reasoning_like_the_non_streaming_path() -> None:
    from ai_agent_harness import StreamDelta
    from ai_agent_harness.core.types import ModelResponse

    wire = Wire(
        _sse(
            _chunk({"role": "assistant", "reasoning_content": "step 1, "}),
            _chunk({"reasoning_content": "step 2"}),
            _chunk({"content": "The answer"}),
            _chunk({"content": " is 42."}, finish="stop"),
            _chunk({}, usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
        )
    )
    provider = provider_for(wire)

    async def collect() -> list[Any]:
        return [item async for item in provider.stream(request())]

    items = run_sync(collect())
    deltas = [i for i in items if isinstance(i, StreamDelta)]
    final = items[-1]
    assert isinstance(final, ModelResponse)
    assert [d.thinking for d in deltas if d.thinking] == ["step 1, ", "step 2"]
    assert "".join(d.text for d in deltas) == "The answer is 42."
    thinking = [b for b in final.message.content if isinstance(b, ThinkingBlock)]
    assert [b.text for b in thinking] == ["step 1, step 2"]
    assert final.message.text == "The answer is 42."
    assert final.usage.input_tokens == 10


def test_stream_without_finish_reason_is_a_retryable_error_not_a_complete_answer() -> None:
    wire = Wire(_sse(_chunk({"role": "assistant", "content": "half an ans"})))  # connection ends mid-answer
    provider = provider_for(wire)

    async def collect() -> list[Any]:
        return [item async for item in provider.stream(request())]

    with pytest.raises(ProviderError, match="finish reason") as info:
        run_sync(collect())
    assert info.value.retryable


def test_stream_tool_call_fragments_without_an_index_are_merged_into_one_call() -> None:
    from ai_agent_harness.core.types import ToolCall

    wire = Wire(
        _sse(
            _chunk({"tool_calls": [{"index": None, "id": "c1", "function": {"name": "echo", "arguments": ""}}]}),
            _chunk({"tool_calls": [{"index": None, "function": {"arguments": "{}"}}]}, finish="tool_calls"),
        )
    )
    provider = provider_for(wire)

    async def collect() -> list[Any]:
        return [item async for item in provider.stream(request())]

    final = run_sync(collect())[-1]
    calls = [b for b in final.message.content if isinstance(b, ToolCall)]
    assert [(c.id, c.name, c.arguments) for c in calls] == [("c1", "echo", {})]


# ------------------------------------------------------------------ Streaming tool calls and error kinds


def _fragment(index: int, call_id: str | None, name: str | None, arguments: str) -> dict[str, Any]:
    function: dict[str, Any] = {"arguments": arguments}
    if name is not None:
        function["name"] = name
    call: dict[str, Any] = {"index": index, "function": function}
    if call_id is not None:
        call.update(id=call_id, type="function")
    return {"tool_calls": [call]}


def _tool_call_chunks() -> list[dict[str, Any]]:
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "prompt_tokens_details": {"cached_tokens": 4},
    }
    return [
        _chunk({"role": "assistant", **_fragment(1, "call_b", "echo", '{"a')}),
        _chunk(_fragment(0, "call_a", "echo", '{"x": 1}')),
        _chunk(_fragment(1, None, None, '": 2}'), finish="tool_calls"),
        _chunk({}, usage=usage),
    ]


def test_stream_assembles_tool_calls_in_index_order_with_usage() -> None:
    wire = Wire(_sse(*_tool_call_chunks()))
    provider = provider_for(wire)

    async def collect() -> list[Any]:
        return [item async for item in provider.stream(request())]

    final = run_sync(collect())[-1]

    assert final.stop_reason is StopReason.TOOL_USE
    assert [(c.id, c.arguments) for c in final.message.tool_calls] == [("call_a", {"x": 1}), ("call_b", {"a": 2})]
    assert final.usage.input_tokens == 6 and final.usage.cache_read_tokens == 4
    assert wire.bodies[0]["stream"] is True and wire.bodies[0]["stream_options"] == {"include_usage": True}


def test_stream_mixes_missing_and_numeric_indexes_without_merging_calls() -> None:
    chunks = [
        _chunk({"tool_calls": [{"index": None, "id": "c1", "function": {"name": "echo", "arguments": "{}"}}]}),
        _chunk(
            {"tool_calls": [{"index": 1, "id": "c2", "function": {"name": "echo", "arguments": "{}"}}]},
            finish="tool_calls",
        ),
    ]
    provider = provider_for(Wire(_sse(*chunks)))

    async def collect() -> list[Any]:
        return [item async for item in provider.stream(request())]

    final = run_sync(collect())[-1]
    assert [c.id for c in final.message.tool_calls] == ["c1", "c2"]


def test_stream_http_error_becomes_a_provider_error() -> None:
    provider = provider_for(Wire(httpx.Response(429, json={"error": {"code": "rate_limit_exceeded"}})))

    async def collect() -> None:
        async for _ in provider.stream(request()):
            pass

    with pytest.raises(ProviderError) as info:
        run_sync(collect())
    assert info.value.retryable and "429" in str(info.value)


def test_the_error_kind_is_taken_from_the_body_when_it_looks_like_a_code() -> None:
    body = {"error": {"code": "context_length_exceeded", "message": "your prompt secret"}}
    provider = provider_for(Wire(httpx.Response(400, json=body)))
    with pytest.raises(ProviderError) as info:
        run_sync(provider.complete(request()))
    assert "(context_length_exceeded)" in str(info.value)
    assert "secret" not in str(info.value)


def test_the_error_kind_is_not_copied_unless_it_looks_like_a_code() -> None:
    provider = provider_for(Wire(httpx.Response(400, json={"error": {"code": "My Secret Token sk-ABC123"}})))
    with pytest.raises(ProviderError) as info:
        run_sync(provider.complete(request()))
    assert "sk-ABC123" not in str(info.value) and "(unknown)" in str(info.value)
    assert info.value.__cause__ is None and info.value.__context__ is None


def test_a_history_message_with_only_foreign_reasoning_becomes_the_foreign_note() -> None:
    from ai_agent_harness.providers.base import FOREIGN_THINKING_NOTE

    wire = Wire(ok("text"))
    provider = provider_for(wire)
    foreign = Message(role="assistant", content=(ThinkingBlock(text="t", provider_id="anthropic", provider_data={}),))

    complete(provider, ModelRequest(system="", messages=(Message.user("hi"), foreign, Message.user("go"))))

    assert wire.bodies[0]["messages"][1]["content"] == FOREIGN_THINKING_NOTE


# ------------------------------------------------------------------ Client setup, SDK timeout, odd responses


class _RecordedClient:
    """Stands in for `openai.AsyncOpenAI` to see how the adapter builds its client."""

    instances: ClassVar[list[_RecordedClient]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = False
        _RecordedClient.instances.append(self)

    async def close(self) -> None:
        self.closed = True


def test_the_adapter_builds_its_own_client_with_the_given_connection_and_no_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RecordedClient.instances.clear()
    monkeypatch.setattr(openai, "AsyncOpenAI", _RecordedClient)

    OpenAIProvider(
        MODEL,
        api_key=API_KEY,
        base_url="http://localhost:11434/v1",
        organization="org-1",
        default_headers={"X-Team": "a"},
    )

    (created,) = _RecordedClient.instances
    kwargs = created.kwargs
    assert kwargs["base_url"] == "http://localhost:11434/v1"
    assert kwargs["organization"] == "org-1" and kwargs["default_headers"] == {"X-Team": "a"}
    assert kwargs["max_retries"] == 0  # the router owns retries


def test_aclose_closes_a_client_the_adapter_created_but_not_one_it_was_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RecordedClient.instances.clear()
    monkeypatch.setattr(openai, "AsyncOpenAI", _RecordedClient)
    own = OpenAIProvider(MODEL, api_key=API_KEY)
    run_sync(own.aclose())
    assert _RecordedClient.instances[0].closed is True

    borrowed = _RecordedClient()
    given = OpenAIProvider(MODEL, client=cast("Any", borrowed))
    run_sync(given.aclose())
    assert borrowed.closed is False


def test_the_sdk_timeout_is_set_explicitly() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok("text")

    client = openai.AsyncOpenAI(
        api_key=API_KEY,
        max_retries=0,
        timeout=5.0,  # the client's own default: the adapter must still ask for its longer limit
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    complete(OpenAIProvider(MODEL, client=client))

    assert seen[0].extensions["timeout"]["read"] == 600.0


def test_a_response_without_choices_is_an_unexpected_result_error() -> None:
    provider = provider_for(Wire(ok("text", choices=[])))
    with pytest.raises(ProviderError, match="unexpected result"):
        complete(provider)


def test_tool_calls_that_are_not_functions_are_ignored() -> None:
    data = fixture("text")
    message = data["choices"][0]["message"]
    message["tool_calls"] = [{"id": "c1", "type": "custom", "custom": {"name": "x", "input": "y"}}]
    provider = provider_for(Wire(httpx.Response(200, json=data)))

    response = complete(provider)

    assert response.message.tool_calls == ()
