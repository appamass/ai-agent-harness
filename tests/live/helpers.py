"""Helpers for the live tests: a real provider (optionally recording responses) and a loop runner that cleans up."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import anthropic
import httpx2

from ai_agent_harness.providers.anthropic import AnthropicProvider

from ..conftest import run_sync

MODEL = os.environ.get("HARNESS_LIVE_MODEL", "claude-haiku-5-5")


class RecordingTransport(httpx2.AsyncHTTPTransport):
    """Saves successful responses (with the message ID replaced) when `HARNESS_RECORD_DIR` is set."""

    def __init__(self, directory: Path) -> None:
        super().__init__()
        directory.mkdir(parents=True, exist_ok=True)
        self._directory = directory

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        response = await super().handle_async_request(request)
        await response.aread()
        if response.status_code == 200:
            data: dict[str, Any] = json.loads(response.content)
            data["id"] = "msg_fixture"
            (self._directory / f"recorded-{uuid.uuid4().hex[:8]}.json").write_text(
                json.dumps(data, indent=2) + "\n", encoding="utf-8"
            )
        return response


OPEN_PROVIDERS: list[AnthropicProvider] = []
OPEN_CLIENTS: list[anthropic.AsyncAnthropic] = []  # clients we pass in (recording mode): the adapter does not own them


def track(provider: AnthropicProvider) -> AnthropicProvider:
    OPEN_PROVIDERS.append(provider)
    return provider


RUN_LIVE_CALLS: list[int] = []


def run_live[T](coro: Coroutine[Any, Any, T]) -> T:
    """Runs a coroutine and closes every provider and client it used inside the same loop (no leaked connections).

    Call it once per test: put all steps of a test into one async function.
    """
    RUN_LIVE_CALLS.append(1)

    async def main() -> T:
        try:
            return await coro
        finally:
            providers, OPEN_PROVIDERS[:] = list(OPEN_PROVIDERS), []
            clients, OPEN_CLIENTS[:] = list(OPEN_CLIENTS), []
            for provider in providers:
                await provider.aclose()
            for client in clients:
                await client.close()

    return run_sync(main())


def live_provider(**kwargs: Any) -> AnthropicProvider:
    record = os.environ.get("HARNESS_RECORD_DIR")
    if record:
        client = anthropic.AsyncAnthropic(
            max_retries=0, http_client=anthropic.DefaultAsyncHttpxClient(transport=RecordingTransport(Path(record)))
        )
        OPEN_CLIENTS.append(client)
        return track(AnthropicProvider(MODEL, client=client, **kwargs))
    return track(AnthropicProvider(MODEL, **kwargs))
