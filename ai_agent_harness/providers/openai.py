"""OpenAI provider adapter. Needs the optional SDK: `pip install "ai-agent-harness[openai]"`.

Implements the adapter contract of `providers/base.py` for the OpenAI Chat Completions API
(and compatible servers like Ollama, vLLM, OpenRouter, and Gemini-compat endpoints).
This module is the only place that imports the SDK; the core and `import ai_agent_harness` never need it.

Choices that follow from the contract and the API (ADR 0021):
- **No retries here.** The SDK client is created with `max_retries=0`; `ModelRouter` owns retry, backoff, and failover.
- **Errors carry only status and type/code.** Raw bodies and prompts are never copied into errors.
- **Thinking blocks** of other providers are stripped (`without_foreign_thinking`).
- **Tool errors**: On JSON parsing errors, returns `ToolCall(arguments={}, arguments_error=...)` for self-correction.
- **Capabilities**: `structured_output=True` sends `response_format: json_schema`. Set `structured_output=False` for
  servers that reject it (Ollama, vLLM, Gemini-compat): the runner then falls back to a tool call or a prompt.
  `max_tokens_param="max_completion_tokens"` is for models that reject `max_tokens` (o-series, GPT-5).
- **Usage**: `prompt_tokens` includes cached tokens, so `cached_tokens` is subtracted for `input_tokens`.
"""

from __future__ import annotations

import json
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
    api_error,
    assistant_message,
    check_not_empty,
    error_kind,
    tool_call,
    without_foreign_thinking,
)

try:
    import openai
    from openai.types import CompletionUsage
    from openai.types.chat import ChatCompletion
except ImportError as exc:
    raise ConfigError("The OpenAI adapter needs the SDK: pip install 'ai-agent-harness[openai]'") from exc

_STOP_REASONS: Final[dict[str, StopReason]] = {
    "stop": StopReason.END,
    "tool_calls": StopReason.TOOL_USE,
    "function_call": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
    "content_filter": StopReason.REFUSAL,
}
_ERROR_TYPE = re.compile(r"[a-z0-9_-]{1,64}")


class OpenAIProvider:
    """`ModelProvider` for OpenAI and OpenAI-compatible Chat Completions APIs."""

    def __init__(
        self,
        model: str,
        *,
        id: str = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        client: openai.AsyncOpenAI | None = None,
        vision: bool = True,
        structured_output: bool = True,
        max_tokens_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens",
        default_headers: dict[str, str] | None = None,
    ) -> None:
        check_not_empty("OpenAIProvider", model=model, id=id)
        if client is not None and (api_key is not None or base_url is not None or organization is not None):
            raise ConfigError("OpenAIProvider: pass either client or connection parameters, not both")
        if max_tokens_param not in ("max_tokens", "max_completion_tokens"):  # also for callers without a type checker
            raise ConfigError("OpenAIProvider: max_tokens_param must be 'max_tokens' or 'max_completion_tokens'")

        self._model = model
        self._id = id
        self._owns_client = client is None
        self._client = (
            client
            if client is not None
            else openai.AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                organization=organization,
                default_headers=default_headers,
                max_retries=0,
            )
        )
        self._max_tokens_param = max_tokens_param
        # Honest capabilities: many compatible servers (Ollama, vLLM, Gemini-compat) do not accept `json_schema`.
        self._capabilities = Capabilities(tools=True, vision=vision, structured_output=structured_output)

    @property
    def id(self) -> str:
        return self._id

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    async def aclose(self) -> None:
        """Closes HTTP connections of the client this adapter created."""
        if self._owns_client:
            await self._client.close()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        response: object = None
        failure: ProviderError | None = None
        try:
            parameters = self._parameters(request)
            response = cast("object", await self._client.chat.completions.create(**parameters, timeout=SDK_TIMEOUT_S))
        except openai.APIError as exc:
            failure = self._translate(exc)
        if failure is not None:
            # Raised outside the except block: the SDK exception echoes the response body (see the Anthropic adapter)
            raise failure
        if not (isinstance(response, ChatCompletion) and response.choices):
            raise ProviderError("OpenAI API returned an unexpected result", retryable=False, provider_id=self._id)
        return _to_response(response)

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamDelta | ModelResponse]:
        parameters = self._parameters(request)
        parameters["stream"] = True
        parameters["stream_options"] = {"include_usage": True}
        reasoning_parts: list[str] = []
        text_parts: list[str] = []
        tool_fragments: dict[int, dict[str, list[str]]] = {}  # index -> {"id" | "name" | "arguments": parts}
        finish_reason: str | None = None
        usage_obj: CompletionUsage | None = None
        model_name: str = self._model
        failure: ProviderError | None = None

        try:
            stream_obj = cast(
                "AsyncIterator[Any]",
                await self._client.chat.completions.create(**parameters, timeout=SDK_TIMEOUT_S),
            )
            async for chunk in stream_obj:
                model_name = str(getattr(chunk, "model", None) or model_name)
                usage_obj = getattr(chunk, "usage", None) or usage_obj
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                finish_reason = getattr(choices[0], "finish_reason", None) or finish_reason
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue
                reasoning = getattr(delta, "reasoning_content", None)
                if isinstance(reasoning, str) and reasoning:
                    reasoning_parts.append(reasoning)
                    yield StreamDelta(thinking=reasoning)
                content = getattr(delta, "content", None)
                if content:
                    text_parts.append(str(content))
                    yield StreamDelta(text=str(content))
                for fragment in getattr(delta, "tool_calls", None) or ():
                    entry = tool_fragments.setdefault(
                        getattr(fragment, "index", None) or 0, {"id": [], "name": [], "arguments": []}
                    )
                    function = getattr(fragment, "function", None)
                    for key, value in (
                        ("id", getattr(fragment, "id", None)),
                        ("name", getattr(function, "name", None)),
                        ("arguments", getattr(function, "arguments", None)),
                    ):
                        if value:
                            entry[key].append(str(value))
        except openai.APIError as exc:
            failure = self._translate(exc)
        if failure is not None:
            raise failure
        if finish_reason is None:  # the connection ended mid-answer: do not hand out a partial text as complete
            raise ProviderError("OpenAI stream ended without a finish reason", retryable=True, provider_id=self._id)

        tool_calls = [
            _tool_call(
                "".join(parts["id"]) or f"call_{index}",
                "".join(parts["name"]),
                "".join(parts["arguments"]),
            )
            for index, parts in sorted(tool_fragments.items())
        ]
        yield _assemble(
            reasoning="".join(reasoning_parts),
            text="".join(text_parts),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=_usage(usage_obj),
            model=model_name,
        )

    def _translate(self, exc: openai.APIError) -> ProviderError:
        if not isinstance(exc, openai.APIStatusError):
            return api_error("OpenAI", self._id, connection=isinstance(exc, openai.APIConnectionError))
        value: object = None
        body: object = exc.body
        if isinstance(body, dict):
            fields = cast("dict[str, object]", body)
            # The SDK hands over the inner error object; some compatible servers keep it under "error"
            nested = fields.get("error")
            detail = cast("dict[str, object]", nested) if isinstance(nested, dict) else fields
            value = detail.get("code") or detail.get("type")
        return api_error("OpenAI", self._id, status=exc.status_code, kind=error_kind(value, _ERROR_TYPE))

    def _parameters(self, request: ModelRequest) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})

        for m in without_foreign_thinking(request.messages, self._id):
            messages.extend(_to_api_messages(m))

        parameters: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if request.max_tokens:
            parameters[self._max_tokens_param] = request.max_tokens
        if request.temperature is not None:
            parameters["temperature"] = request.temperature
        if request.output_schema is not None and self._capabilities.structured_output:
            parameters["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "final_output",
                    "schema": request.output_schema,
                },
            }
        if request.tools:
            parameters["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in request.tools
            ]
        return parameters


def _to_api_messages(message: Message) -> list[dict[str, Any]]:
    if message.role == "assistant":
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in message.content:
            if isinstance(block, TextBlock) and block.text:
                text_parts.append(block.text)
            elif isinstance(block, ToolCall):
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {"name": block.name, "arguments": json.dumps(block.arguments)},
                    }
                )
        content: str | None = "".join(text_parts)
        if not content:
            content = None if tool_calls else EMPTY_RESPONSE_NOTE
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return [msg]

    api_msgs: list[dict[str, Any]] = []
    user_parts: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, ToolResult):
            api_msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": block.call_id,
                    "content": block.content,
                }
            )
        elif isinstance(block, TextBlock):
            if block.text:
                user_parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageBlock):
            user_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{block.media_type};base64,{block.data}",
                    },
                }
            )

    if user_parts:
        if len(user_parts) == 1 and user_parts[0]["type"] == "text":
            api_msgs.append({"role": "user", "content": user_parts[0]["text"]})
        else:
            api_msgs.append({"role": "user", "content": user_parts})
    elif not api_msgs:
        api_msgs.append({"role": "user", "content": EMPTY_RESPONSE_NOTE})

    return api_msgs


def _tool_call(call_id: str, name: str, raw_arguments: str) -> ToolCall:
    """A model tool call; unusable arguments become `arguments_error` so the model can correct itself."""
    if not raw_arguments:
        return ToolCall(id=call_id, name=name)
    try:
        parsed: object = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        return ToolCall(id=call_id, name=name, arguments_error=f"Invalid JSON: {exc.msg}")
    return tool_call(call_id, name, parsed)


def _assemble(
    *,
    reasoning: str,
    text: str,
    tool_calls: list[ToolCall],
    finish_reason: str | None,
    usage: Usage,
    model: str,
) -> ModelResponse:
    blocks: list[Block] = []
    if reasoning:
        blocks.append(ThinkingBlock(text=reasoning, provider_data={"kind": "reasoning"}))
    if text:
        blocks.append(TextBlock(text=text))
    blocks.extend(tool_calls)
    return ModelResponse(
        message=assistant_message(blocks),
        stop_reason=_STOP_REASONS.get(finish_reason or "", StopReason.END),
        usage=usage,
        model=model,
    )


def _to_response(completion: ChatCompletion) -> ModelResponse:
    choice = completion.choices[0]
    msg = choice.message
    reasoning = getattr(msg, "reasoning_content", None)
    return _assemble(
        reasoning=reasoning if isinstance(reasoning, str) else "",
        text=msg.content or "",
        tool_calls=[
            _tool_call(tc.id, tc.function.name, tc.function.arguments or "")
            for tc in msg.tool_calls or ()
            if tc.type == "function"
        ],
        finish_reason=getattr(choice, "finish_reason", None),
        usage=_usage(completion.usage),
        model=completion.model,
    )


def _usage(usage: CompletionUsage | None) -> Usage:
    if usage is None:
        return Usage(input_tokens=0, output_tokens=0)

    prompt_tokens = usage.prompt_tokens or 0
    completion_tokens = usage.completion_tokens or 0

    prompt_details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = getattr(prompt_details, "cached_tokens", 0) or 0
    input_tokens = max(0, prompt_tokens - cached_tokens)

    completion_details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) or 0

    return Usage(
        input_tokens=input_tokens,
        output_tokens=completion_tokens,
        cache_read_tokens=cached_tokens,
        reasoning_tokens=min(reasoning_tokens, completion_tokens),
    )
