"""Live tests against the real OpenAI API. They cost money and are never part of a default run.

export OPENAI_API_KEY=...          # in your own shell, never in a file
pytest -m live -v                   # model: HARNESS_LIVE_OPENAI_MODEL (default gpt-4o-mini)
"""

from __future__ import annotations

import os

import pytest

from ai_agent_harness import (
    Agent,
    Autonomy,
    InMemoryCheckpointStore,
    ModelRouter,
    RetryPolicy,
    RunStatus,
    start,
)
from ai_agent_harness.core.types import Message, ModelRequest
from ai_agent_harness.providers.openai import OpenAIProvider

from ..conftest import Echo, run_sync

OPENAI_MODEL = os.environ.get("HARNESS_LIVE_OPENAI_MODEL", "gpt-4o-mini")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY is not set"),
]


def test_openai_live_text_completion() -> None:
    async def scenario() -> None:
        provider = OpenAIProvider(OPENAI_MODEL)
        try:
            req = ModelRequest(system="You are a helpful assistant.", messages=(Message.user("Say hello!"),))
            response = await provider.complete(req)
            assert response.message.text
            assert response.usage.input_tokens > 0
            assert response.usage.output_tokens > 0
        finally:
            await provider.aclose()  # in the loop that used the client

    run_sync(scenario())


def test_openai_live_tool_calling() -> None:
    async def scenario() -> None:
        provider = OpenAIProvider(OPENAI_MODEL)
        try:
            agent = Agent.create(
                ModelRouter(provider, retry=RetryPolicy(max_attempts=1)),
                "You are a test agent. Always call the echo tool when asked.",
                name="openai_live_agent",
                tools=[Echo()],
                autonomy=Autonomy.ACT_WITH_APPROVAL,
                checkpoints=InMemoryCheckpointStore(),
            )
            state = await start(agent, "Call echo please.")
            assert state.status is RunStatus.DONE
            assert [r for m in state.messages for r in m.tool_results], "the model never called the echo tool"
            state.check_protocol()
        finally:
            await provider.aclose()

    run_sync(scenario())
