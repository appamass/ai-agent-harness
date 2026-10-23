"""Scaffolding shared by the provider adapter tests (`test_anthropic_provider`, `test_openai_provider`).

The SDK-specific parts (the scripted `Wire` and its client factory) stay in each test file.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Protocol

from ai_agent_harness import ModelRequest, ModelResponse, StreamDelta
from ai_agent_harness.core.types import Message, ToolSpec
from ai_agent_harness.providers.base import ModelProvider

from .conftest import FIXTURES_DIR, run_sync

ECHO_SPEC = ToolSpec(name="echo", description="Returns 'echo'.", input_schema={"type": "object", "properties": {}})


class StreamingProvider(ModelProvider, Protocol):
    def stream(self, request: ModelRequest) -> AsyncIterator[StreamDelta | ModelResponse]: ...


def load_fixture(directory: str, name: str) -> dict[str, Any]:
    """Load `tests/fixtures/<directory>/<name>.json`."""
    data: dict[str, Any] = json.loads((FIXTURES_DIR / directory / f"{name}.json").read_text(encoding="utf-8"))
    return data


def request(*messages: Message, **kwargs: Any) -> ModelRequest:
    return ModelRequest(system="You are a test agent.", messages=messages or (Message.user("hi"),), **kwargs)


def complete(provider: ModelProvider, req: ModelRequest | None = None) -> Any:
    return run_sync(provider.complete(req or request()))


def stream_items(provider: StreamingProvider, req: ModelRequest | None = None) -> list[Any]:
    """Run `provider.stream` to the end and return every item (deltas first, the final response last)."""

    async def collect() -> list[Any]:
        return [item async for item in provider.stream(req or request())]

    return run_sync(collect())
