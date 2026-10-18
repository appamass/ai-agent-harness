"""Test helpers for projects built on the harness.

`FakeProvider` replays scripted responses or errors — without calling a real
model. This lets you test the loop, guardrails, tools, and policies deterministically.
Do not use in production code.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .core.types import Message, ModelRequest, ModelResponse, StopReason, TextBlock, ToolCall, Usage
from .providers.base import Capabilities


class FakeProvider:
    def __init__(
        self,
        script: Sequence[ModelResponse | BaseException],
        *,
        id: str = "fake",
        capabilities: Capabilities | None = None,
    ) -> None:
        self._script = list(script)
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

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._script:
            raise AssertionError(f"FakeProvider '{self._id}': script is empty, unexpected call")
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


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
