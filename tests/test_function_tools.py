"""Function tools (@tool decorator) and Agent.create() shorthand."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ai_agent_harness import (
    Agent,
    Autonomy,
    InputRequest,
    ModelRequest,
    Risk,
    RunState,
    RunStatus,
    ToolError,
    resume,
    start,
    tool,
)
from ai_agent_harness.core import ConfigError
from ai_agent_harness.testing import FakeProvider, text_response, tool_response

from .conftest import results_of, run_sync


@dataclass
class TenantDeps:
    prefix: str


def test_sync_and_async_function_tools_with_and_without_state() -> None:
    @tool(risk=Risk.READ)
    def add(a: int, b: int = 10) -> str:
        """Adds two integers together."""
        return str(a + b)

    @tool(name="whoami", risk=Risk.READ, timeout_s=5.0)
    async def get_caller(label: str, state: RunState) -> str:
        """Returns the caller label scoped to the current tenant."""
        deps = state.deps_as(TenantDeps)
        return f"{deps.prefix}:{state.tenant_id}:{label}"

    @tool(risk=Risk.WRITE)
    def ask_clarification(topic: str) -> InputRequest:
        """Asks the user to clarify a topic."""
        return InputRequest(question=f"Clarify {topic}?")

    # `state` is hidden from the JSON schema exposed to the model
    assert set(get_caller.spec().input_schema["properties"]) == {"label"}

    provider = FakeProvider(
        [
            tool_response(("add", {"a": 5}), ("whoami", {"label": "admin"})),
            tool_response(("ask_clarification", {"topic": "budget"})),
            text_response("All done."),
        ]
    )
    agent = Agent.create(
        provider,
        "You are a concise assistant.",
        name="mini",
        tools=[add, get_caller, ask_clarification],
        autonomy=Autonomy.AUTONOMOUS,
    )
    waiting = run_sync(start(agent, "Run tools", tenant_id="t42", deps=TenantDeps(prefix="org")))
    assert waiting.status is RunStatus.WAITING_FOR_INPUT
    assert waiting.status_reason == "Clarify budget?"
    first_results = waiting.messages[2].tool_results
    assert [r.content for r in first_results] == ["15", "org:t42:admin"]

    done = run_sync(
        resume(agent, waiting, answer="1000 EUR", pending_id=waiting.pending_id, deps=TenantDeps(prefix="org"))
    )
    assert done.status is RunStatus.DONE
    assert done.last_text == "All done."


def test_agent_create_accepts_async_callable_directly() -> None:
    async def echo_model(request: ModelRequest) -> str:
        return f"Echo: {request.messages[-1].text}"

    agent = Agent.create(echo_model, "Echo everything.")
    state = run_sync(start(agent, "hello world"))
    assert state.status is RunStatus.DONE
    assert state.last_text == "Echo: hello world"


def test_function_tool_validation_errors_in_arguments_and_tool_error() -> None:
    @tool(risk=Risk.READ)
    def divide(a: int, b: int) -> str:
        """Divides a by b."""
        if b == 0:
            raise ToolError("Division by zero is not allowed")
        return str(a // b)

    provider = FakeProvider(
        [
            tool_response(("divide", {"a": "not_an_int", "b": 2})),
            tool_response(("divide", {"a": 10, "b": 0})),
            text_response("Handled errors."),
        ]
    )
    agent = Agent.create(provider, tools=[divide])
    state = run_sync(start(agent, "divide"))
    res = results_of(state)
    assert len(res) == 2 and all(r.is_error for r in res)
    assert "Invalid arguments" in res[0].content
    assert "Division by zero" in res[1].content


def test_invalid_function_tool_declarations_fail_at_definition_time() -> None:
    def untyped(x: int) -> str:
        """Has no type annotation at runtime."""
        return str(x)

    del untyped.__annotations__["x"]
    with pytest.raises(ConfigError, match="missing a type annotation"):
        tool(untyped, risk=Risk.READ)

    with pytest.raises(ConfigError, match="concrete JSON schema type"):

        @tool(risk=Risk.READ)
        def any_param(x: Any) -> str:
            """Uses unconstrained Any."""
            return str(x)

    with pytest.raises(ConfigError, match="description"):

        @tool(risk=Risk.READ)
        def no_doc(x: int) -> str:
            return str(x)

    with pytest.raises(ConfigError, match=r"\*args"):

        @tool(risk=Risk.READ)
        def varargs(*args: int) -> str:
            """Varargs are not allowed."""
            return str(args)

    bad_fn: Any = "not_a_function"
    with pytest.raises(ConfigError, match="callable"):
        tool(bad_fn, risk=Risk.READ)

    bad_provider: Any = 123
    with pytest.raises(ConfigError, match="provider"):
        Agent.create(bad_provider)
