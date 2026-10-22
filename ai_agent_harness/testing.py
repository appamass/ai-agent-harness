"""Test helpers for projects built on the harness.

`FakeProvider` replays scripted responses or errors — without calling a real
model. This lets you test the loop, guardrails, tools, and policies deterministically.
A plain `ModelResponse` streams its text word by word; `Streamed(...)` scripts the exact
deltas (and errors) of one streamed call. Do not use in production code.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from .core.types import Message, ModelRequest, ModelResponse, StopReason, TextBlock, ToolCall, Usage
from .providers.base import Capabilities, StreamDelta


class Streamed:
    """One scripted call that streams exactly these items in order; a scripted exception is raised where it stands."""

    def __init__(self, *items: StreamDelta | ModelResponse | BaseException) -> None:
        for item in items:
            if not isinstance(item, StreamDelta | ModelResponse | BaseException):
                raise TypeError(f"Streamed: unexpected item {type(item).__name__}")
        self.items = items


class FakeProvider:
    def __init__(
        self,
        script: Sequence[ModelResponse | BaseException | Streamed],
        *,
        id: str = "fake",
        capabilities: Capabilities | None = None,
    ) -> None:
        self._script: list[ModelResponse | BaseException | Streamed] = list(script)
        for item in self._script:
            if not isinstance(item, ModelResponse | BaseException | Streamed):
                raise TypeError(
                    f"FakeProvider: unexpected script item {type(item).__name__} (wrap streams in Streamed)"
                )
        self._id = id
        self._capabilities = capabilities or Capabilities(tools=True, vision=True)
        self.requests: list[ModelRequest] = []

    @property
    def id(self) -> str:
        return self._id

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    @property
    def remaining(self) -> int:
        return len(self._script)

    def _next_item(self, request: ModelRequest) -> ModelResponse | Streamed:
        self.requests.append(request)
        if not self._script:
            raise AssertionError(f"FakeProvider '{self._id}': script is empty, unexpected call")
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    @staticmethod
    def _events(item: ModelResponse | Streamed) -> Iterator[StreamDelta | ModelResponse]:
        """What a script item makes the provider emit; scripted exceptions are raised in order."""
        if isinstance(item, ModelResponse):
            yield from _word_deltas(item)
            yield item
            return
        for sub in item.items:
            if isinstance(sub, BaseException):
                raise sub
            yield sub

    async def complete(self, request: ModelRequest) -> ModelResponse:
        for event in self._events(self._next_item(request)):
            if isinstance(event, ModelResponse):
                return event
        raise AssertionError(f"FakeProvider '{self._id}': scripted sequence did not contain ModelResponse")

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamDelta | ModelResponse]:
        for event in self._events(self._next_item(request)):
            yield event


def _word_deltas(response: ModelResponse) -> Iterator[StreamDelta]:
    words = response.message.text.split(" ") if response.message.text else []
    for i, word in enumerate(words):
        yield StreamDelta(text=word if i == len(words) - 1 else word + " ")


def text_response(text: str, *, stop: StopReason = StopReason.END, tokens: int = 10) -> ModelResponse:
    return ModelResponse(
        message=Message(role="assistant", content=(TextBlock(text=text),)),
        stop_reason=stop,
        usage=Usage(input_tokens=tokens, output_tokens=tokens),
    )


def tool_response(
    *calls: tuple[str, dict[str, Any]], stop: StopReason = StopReason.TOOL_USE, text: str | None = None
) -> ModelResponse:
    """`tool_response(("read_file", {"path": "a.txt"}), ("ask_user", {...}))` — IDs are numbered sequentially."""
    blocks: list[TextBlock | ToolCall] = [TextBlock(text=text)] if text else []
    blocks += [ToolCall(id=f"call_{i}", name=name, arguments=args) for i, (name, args) in enumerate(calls, 1)]
    return ModelResponse(
        message=Message(role="assistant", content=tuple(blocks)),
        stop_reason=stop,
        usage=Usage(input_tokens=10, output_tokens=10),
    )
