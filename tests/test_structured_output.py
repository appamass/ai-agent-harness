"""Structured final output tests: Strategy 1 (native), Strategy 2 (tool call), Strategy 3 (prompt fallback)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.errors import PydanticInvalidForJsonSchema

from ai_agent_harness import (
    Agent,
    Autonomy,
    BaseTool,
    Capabilities,
    JsonFileCheckpointStore,
    ModelRouter,
    Risk,
    RunState,
    RunStatus,
    ToolRegistry,
    resume,
    start,
)
from ai_agent_harness.core.errors import ConfigError, InvalidStateError, ProviderError
from ai_agent_harness.core.hooks import CONTINUE, Decision, Hook, ToolInvocation
from ai_agent_harness.core.types import (
    FINAL_OUTPUT_TOOL_NAME,
    Message,
    ModelResponse,
    StopReason,
    ToolCall,
    ToolResult,
    Usage,
)
from ai_agent_harness.policy import Limits
from ai_agent_harness.testing import FakeProvider, text_response, tool_response

from .conftest import SendMail, assert_consistent, run_sync


class CityWeather(BaseModel):
    city: str
    temperature_c: float
    condition: str = Field(description="Weather condition e.g. sunny, rainy")


class LookupInput(BaseModel):
    city: str


class LookupTool(BaseTool[LookupInput]):
    name: ClassVar[str] = "lookup_city"
    description: ClassVar[str] = "Looks up city weather"
    input_model: ClassVar[type[BaseModel]] = LookupInput
    risk: ClassVar[Risk] = Risk.READ

    async def run(self, args: LookupInput, state: RunState) -> str:
        return f"Weather in {args.city}: 21.5C, Sunny"


# ------------------------------------------------------------------ Strategy 1: Native structured output


def test_native_structured_output_success() -> None:
    provider = FakeProvider(
        [text_response('{"city": "Berlin", "temperature_c": 19.0, "condition": "Sunny"}')],
        capabilities=Capabilities(tools=True, structured_output=True),
    )
    agent = Agent.create(provider)
    state = run_sync(start(agent, "Weather for Berlin", output_model=CityWeather))

    assert state.status is RunStatus.DONE
    assert state.output == {"city": "Berlin", "temperature_c": 19.0, "condition": "Sunny"}
    model = state.output_as(CityWeather)
    assert model == CityWeather(city="Berlin", temperature_c=19.0, condition="Sunny")
    assert len(provider.requests) == 1
    assert provider.requests[0].output_schema == CityWeather.model_json_schema()
    assert_consistent(state)


# ------------------------------------------------------------------ Strategy 2: Forced synthetic tool call


def test_tool_fallback_synthetic_call_success() -> None:
    provider = FakeProvider(
        [
            tool_response(
                (
                    FINAL_OUTPUT_TOOL_NAME,
                    {"city": "Munich", "temperature_c": 22.0, "condition": "Cloudy"},
                )
            )
        ],
        capabilities=Capabilities(tools=True, structured_output=False),
    )
    agent = Agent.create(provider)
    state = run_sync(start(agent, "Weather for Munich", output_model=CityWeather))

    assert state.status is RunStatus.DONE
    assert state.output == {"city": "Munich", "temperature_c": 22.0, "condition": "Cloudy"}
    assert state.output_as(CityWeather) == CityWeather(city="Munich", temperature_c=22.0, condition="Cloudy")
    assert len(provider.requests) == 1
    assert any(t.name == FINAL_OUTPUT_TOOL_NAME for t in provider.requests[0].tools)
    assert_consistent(state)


def test_tool_fallback_with_multi_turn_tool_use_first() -> None:
    provider = FakeProvider(
        [
            tool_response(("lookup_city", {"city": "Paris"})),
            tool_response(
                (
                    FINAL_OUTPUT_TOOL_NAME,
                    {"city": "Paris", "temperature_c": 21.5, "condition": "Sunny"},
                )
            ),
        ],
        capabilities=Capabilities(tools=True, structured_output=False),
    )
    agent = Agent.create(provider, tools=[LookupTool()])
    state = run_sync(start(agent, "Weather in Paris", output_model=CityWeather))

    assert state.status is RunStatus.DONE
    assert state.output == {"city": "Paris", "temperature_c": 21.5, "condition": "Sunny"}
    assert state.output_as(CityWeather) == CityWeather(city="Paris", temperature_c=21.5, condition="Sunny")
    assert len(provider.requests) == 2
    assert_consistent(state)


# ------------------------------------------------------------------ Strategy 3: Prompt injection fallback


def test_prompt_injection_fallback_markdown_json_parsing() -> None:
    markdown_response = """Here is the structured data you requested:
```json
{
  "city": "Hamburg",
  "temperature_c": 16.5,
  "condition": "Rainy"
}
```
Hope that helps!"""
    provider = FakeProvider(
        [text_response(markdown_response)],
        capabilities=Capabilities(tools=False, structured_output=False),
    )
    agent = Agent.create(provider)
    state = run_sync(start(agent, "Weather for Hamburg", output_model=CityWeather))

    assert state.status is RunStatus.DONE
    assert state.output == {"city": "Hamburg", "temperature_c": 16.5, "condition": "Rainy"}
    assert state.output_as(CityWeather) == CityWeather(city="Hamburg", temperature_c=16.5, condition="Rainy")
    assert len(provider.requests) == 1
    assert "schema" in provider.requests[0].system.lower()
    assert_consistent(state)


# ------------------------------------------------------------------ Self-correction on validation error


def test_self_correction_after_validation_error() -> None:
    # First response is missing required field 'temperature_c'
    # Second response corrects it
    provider = FakeProvider(
        [
            text_response('{"city": "Vienna", "condition": "Windy"}'),
            text_response('{"city": "Vienna", "temperature_c": 14.0, "condition": "Windy"}'),
        ],
        capabilities=Capabilities(tools=False, structured_output=False),
    )
    agent = Agent.create(provider)
    state = run_sync(start(agent, "Weather for Vienna", output_model=CityWeather))

    assert state.status is RunStatus.DONE
    assert state.output == {"city": "Vienna", "temperature_c": 14.0, "condition": "Windy"}
    assert state.output_as(CityWeather) == CityWeather(city="Vienna", temperature_c=14.0, condition="Windy")
    assert len(provider.requests) == 2
    # Verify rejection feedback was passed to the model
    last_user_msg = state.messages[-2].text
    assert "rejected" in last_user_msg.lower() or "validation" in last_user_msg.lower()
    assert_consistent(state)


def test_repeated_validation_errors_fail_gracefully() -> None:
    provider = FakeProvider(
        [
            text_response('{"city": "Rome"}'),
            text_response('{"city": "Rome"}'),
            text_response('{"city": "Rome"}'),
            text_response('{"city": "Rome"}'),
        ],
        capabilities=Capabilities(tools=False, structured_output=False),
    )
    agent = Agent.create(provider, limits=Limits(max_consecutive_retries=2))
    state = run_sync(start(agent, "Weather for Rome", output_model=CityWeather))

    assert state.status is RunStatus.FAILED
    assert "repeatedly" in (state.status_reason or "").lower()
    assert state.output is None
    with pytest.raises(InvalidStateError, match="no structured output"):
        state.output_as(CityWeather)
    assert_consistent(state)


# ------------------------------------------------------------------ Pause, Checkpoint & Resume with output_model


def test_checkpoint_and_resume_across_approval_preserves_output_model(tmp_path: Path) -> None:
    store = JsonFileCheckpointStore(tmp_path)
    provider = FakeProvider(
        [
            tool_response(("send_mail", {})),
            tool_response(
                (
                    FINAL_OUTPUT_TOOL_NAME,
                    {"city": "Zurich", "temperature_c": 17.0, "condition": "Clear"},
                )
            ),
        ],
        capabilities=Capabilities(tools=True, structured_output=False),
    )
    agent = Agent.create(
        provider,
        tools=[SendMail()],
        autonomy=Autonomy.ACT_WITH_APPROVAL,  # IRREVERSIBLE requires approval
        checkpoints=store,
    )

    paused = run_sync(start(agent, "Publish report for Zurich", output_model=CityWeather))
    assert paused.status is RunStatus.WAITING_FOR_APPROVAL
    assert paused.output_schema_digest is not None
    assert paused.output_model_name == "CityWeather"

    # Reload state from store to test persistence fidelity
    reloaded = run_sync(store.load(paused.tenant_id, paused.run_id))
    assert reloaded is not None
    assert reloaded.output_schema_digest is not None

    resumed = run_sync(resume(agent, reloaded, approved=True, pending_id=reloaded.pending_id, output_model=CityWeather))
    assert resumed.status is RunStatus.DONE
    assert resumed.output == {"city": "Zurich", "temperature_c": 17.0, "condition": "Clear"}
    assert resumed.output_as(CityWeather) == CityWeather(city="Zurich", temperature_c=17.0, condition="Clear")
    assert_consistent(resumed)


def test_tool_fallback_direct_json_text_accepted() -> None:
    provider = FakeProvider(
        [text_response('{"city": "Graz", "temperature_c": 18.0, "condition": "Sunny"}')],
        capabilities=Capabilities(tools=True, structured_output=False),
    )
    agent = Agent.create(provider)
    state = run_sync(start(agent, "Weather for Graz", output_model=CityWeather))

    assert state.status is RunStatus.DONE
    assert state.output == {"city": "Graz", "temperature_c": 18.0, "condition": "Sunny"}
    assert state.output_as(CityWeather) == CityWeather(city="Graz", temperature_c=18.0, condition="Sunny")
    assert_consistent(state)


def test_tool_fallback_non_json_text_triggers_hint_and_recovers() -> None:
    provider = FakeProvider(
        [
            text_response("Here is the weather: it is 20 degrees and sunny in Graz."),
            tool_response(
                (
                    FINAL_OUTPUT_TOOL_NAME,
                    {"city": "Graz", "temperature_c": 20.0, "condition": "Sunny"},
                )
            ),
        ],
        capabilities=Capabilities(tools=True, structured_output=False),
    )
    agent = Agent.create(provider)
    state = run_sync(start(agent, "Weather for Graz", output_model=CityWeather))

    assert state.status is RunStatus.DONE
    assert state.output == {"city": "Graz", "temperature_c": 20.0, "condition": "Sunny"}
    assert_consistent(state)
    # Check hint was sent in feedback
    assert f"'{FINAL_OUTPUT_TOOL_NAME}'" in state.messages[-3].text


def test_incompatible_model_validation_raises() -> None:
    class WrongModel(BaseModel):
        different_field: int

    provider = FakeProvider(
        [text_response('{"city": "Berlin", "temperature_c": 19.0, "condition": "Sunny"}')],
        capabilities=Capabilities(tools=True, structured_output=True),
    )
    agent = Agent.create(provider)
    state = run_sync(start(agent, "Weather for Berlin", output_model=CityWeather))

    assert state.status is RunStatus.DONE
    with pytest.raises(ValidationError):
        state.output_as(WrongModel)


# ------------------------------------------------------------------ Review fixes (ADR 0023): gate, fail-closed resume


def _call(name: str, args: dict[str, object], error: str | None = None) -> ModelResponse:
    return ModelResponse(
        message=Message(
            role="assistant", content=(ToolCall(id="call_1", name=name, arguments=args, arguments_error=error),)
        ),
        stop_reason=StopReason.TOOL_USE,
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def test_final_output_without_requested_schema_does_not_end_the_run() -> None:
    provider = FakeProvider([_call(FINAL_OUTPUT_TOOL_NAME, {"evil": 1}), text_response("done")])
    state = run_sync(start(Agent.create(provider), "hi"))

    assert state.status is RunStatus.DONE
    assert state.output is None  # the unknown-tool error was fed back, then the model answered in text
    assert "Unknown tool" in state.messages[2].tool_results[0].content
    assert_consistent(state)


def test_final_output_with_arguments_error_is_rejected() -> None:
    class Optional(BaseModel):
        x: int = 0

    provider = FakeProvider(
        [_call(FINAL_OUTPUT_TOOL_NAME, {}, "Malformed JSON"), _call(FINAL_OUTPUT_TOOL_NAME, {"x": 5})],
        capabilities=Capabilities(tools=True, structured_output=False),
    )
    state = run_sync(start(Agent.create(provider), "hi", output_model=Optional))

    assert state.status is RunStatus.DONE
    assert state.output == {"x": 5}  # not the empty first attempt
    assert state.messages[2].tool_results[0].is_error
    assert_consistent(state)


def test_final_output_goes_through_hooks_and_events() -> None:
    from ai_agent_harness.core.hooks import CONTINUE, Hook

    seen: list[str] = []

    class Spy(Hook):
        async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
            seen.append(invocation.tool_name)
            return CONTINUE

    provider = FakeProvider(
        [tool_response((FINAL_OUTPUT_TOOL_NAME, {"city": "Bern", "temperature_c": 1.0, "condition": "Snow"}))],
        capabilities=Capabilities(tools=True, structured_output=False),
    )
    state = run_sync(start(Agent.create(provider, hooks=[Spy()]), "hi", output_model=CityWeather))

    assert state.status is RunStatus.DONE
    assert seen == [FINAL_OUTPUT_TOOL_NAME]


def test_blocking_hook_applies_to_final_output() -> None:
    from ai_agent_harness.core.hooks import Decision, Hook

    class Deny(Hook):
        async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
            return Decision.block("no output allowed")

    bad = tool_response((FINAL_OUTPUT_TOOL_NAME, {"city": "Bern", "temperature_c": 1.0, "condition": "Snow"}))
    provider = FakeProvider(
        [bad] * 8,
        capabilities=Capabilities(tools=True, structured_output=False),
    )
    agent = Agent.create(provider, hooks=[Deny()], limits=Limits(max_consecutive_retries=2))
    state = run_sync(start(agent, "hi", output_model=CityWeather))

    assert state.status is RunStatus.FAILED  # retries exhausted, never DONE
    assert state.output is None
    assert_consistent(state)


def test_final_output_is_a_reserved_tool_name() -> None:
    class Clash(BaseTool[LookupInput]):
        name: ClassVar[str] = FINAL_OUTPUT_TOOL_NAME
        description: ClassVar[str] = "clash"
        input_model: ClassVar[type[BaseModel]] = LookupInput
        risk: ClassVar[Risk] = Risk.READ

        async def run(self, args: LookupInput, state: RunState) -> str:
            return "x"

    with pytest.raises(ConfigError, match="reserved"):
        Agent.create(FakeProvider([]), tools=[Clash()])


def test_resume_without_output_model_fails_closed() -> None:
    provider = FakeProvider([text_response('{"city": "Rome", "temperature_c": 25.0, "condition": "Hot"}')])
    agent = Agent.create(provider)
    state = run_sync(start(agent, "hi", output_model=CityWeather))
    assert state.status is RunStatus.DONE

    with pytest.raises(InvalidStateError, match="output_model"):
        run_sync(resume(agent, state, user_input="again"))


def test_resume_with_drifted_output_model_fails_closed() -> None:
    class Other(BaseModel):
        a: int

    provider = FakeProvider([text_response('{"city": "Rome", "temperature_c": 25.0, "condition": "Hot"}')])
    agent = Agent.create(provider)
    state = run_sync(start(agent, "hi", output_model=CityWeather))

    with pytest.raises(InvalidStateError, match="does not match"):
        run_sync(resume(agent, state, user_input="again", output_model=Other))


def test_resume_validates_against_the_model_not_the_output_itself() -> None:
    provider = FakeProvider(
        [
            text_response('{"city": "Rome", "temperature_c": 25.0, "condition": "Hot"}'),
            text_response('{"city": "Rome", "temperature_c": "NOT A NUMBER", "condition": "Hot"}'),
            text_response('{"city": "Rome", "temperature_c": 26.0, "condition": "Hot"}'),
        ]
    )
    agent = Agent.create(provider)
    first = run_sync(start(agent, "hi", output_model=CityWeather))
    second = run_sync(resume(agent, first, user_input="again", output_model=CityWeather))

    assert second.status is RunStatus.DONE
    assert second.output is not None
    assert second.output["temperature_c"] == 26.0  # the invalid middle answer was rejected, stale output dropped
    assert_consistent(second)


def test_mixed_provider_capabilities_use_the_common_tier() -> None:
    native = FakeProvider([], id="native", capabilities=Capabilities(tools=True, structured_output=True))
    plain = FakeProvider(
        [tool_response((FINAL_OUTPUT_TOOL_NAME, {"city": "Oslo", "temperature_c": -3.0, "condition": "Snow"}))],
        id="plain",
        capabilities=Capabilities(tools=True, structured_output=False),
    )
    # The first provider fails, the second answers: the request must be valid for both.
    native_failing = FakeProvider(
        [ProviderError("down", retryable=False, provider_id="native")],
        id="native",
        capabilities=Capabilities(tools=True, structured_output=True),
    )
    agent = Agent.create(ModelRouter(native_failing, [plain]))
    state = run_sync(start(agent, "hi", output_model=CityWeather))

    assert state.status is RunStatus.DONE
    assert native.requests == []
    assert native_failing.requests[0].output_schema is None
    assert any(t.name == FINAL_OUTPUT_TOOL_NAME for t in native_failing.requests[0].tools)


def test_invalid_final_output_counts_towards_retry_limit() -> None:
    bad = tool_response((FINAL_OUTPUT_TOOL_NAME, {"city": "Rome"}))
    provider = FakeProvider([bad] * 6, capabilities=Capabilities(tools=True, structured_output=False))
    agent = Agent.create(provider, limits=Limits(max_consecutive_retries=2))
    state = run_sync(start(agent, "hi", output_model=CityWeather))

    assert state.status is RunStatus.FAILED
    assert len(provider.requests) == 3
    assert_consistent(state)


def test_after_tool_block_withholds_final_output() -> None:
    from ai_agent_harness.core.hooks import CONTINUE, Decision, Hook

    class Dlp(Hook):
        def __init__(self) -> None:
            self.calls = 0

        async def after_tool(self, state: RunState, invocation: ToolInvocation, result: ToolResult) -> Decision:
            self.calls += 1
            return Decision.block("dlp") if self.calls == 1 else CONTINUE

    first = tool_response((FINAL_OUTPUT_TOOL_NAME, {"city": "Bern", "temperature_c": 1.0, "condition": "secret"}))
    second = tool_response((FINAL_OUTPUT_TOOL_NAME, {"city": "Bern", "temperature_c": 1.0, "condition": "Snow"}))
    provider = FakeProvider([first, second], capabilities=Capabilities(tools=True, structured_output=False))
    state = run_sync(start(Agent.create(provider, hooks=[Dlp()]), "hi", output_model=CityWeather))

    assert state.status is RunStatus.DONE
    assert state.output is not None
    assert state.output["condition"] == "Snow"  # the withheld first output was not accepted
    assert "withheld" in state.messages[2].tool_results[0].content.lower()
    assert_consistent(state)


def test_always_blocked_final_output_never_ends_done_and_leaves_no_output() -> None:
    from ai_agent_harness.core.hooks import Decision, Hook

    class Dlp(Hook):
        async def after_tool(self, state: RunState, invocation: ToolInvocation, result: ToolResult) -> Decision:
            return Decision.block("dlp")

    bad = tool_response((FINAL_OUTPUT_TOOL_NAME, {"city": "Bern", "temperature_c": 1.0, "condition": "Snow"}))
    provider = FakeProvider([bad] * 6, capabilities=Capabilities(tools=True, structured_output=False))
    agent = Agent.create(provider, hooks=[Dlp()], limits=Limits(max_consecutive_retries=2))
    state = run_sync(start(agent, "hi", output_model=CityWeather))

    assert state.status is RunStatus.FAILED
    assert state.output is None
    assert_consistent(state)


def test_output_models_with_free_form_fields_work_in_the_tool_tier() -> None:
    from typing import Any

    class Loose(BaseModel):
        title: str
        extra: dict[str, Any]
        anything: Any

    args = {"title": "t", "extra": {"a": 1}, "anything": [1, 2]}
    provider = FakeProvider(
        [tool_response((FINAL_OUTPUT_TOOL_NAME, args))], capabilities=Capabilities(tools=True, structured_output=False)
    )
    state = run_sync(start(Agent.create(provider), "hi", output_model=Loose))

    assert state.status is RunStatus.DONE
    assert state.output == args


class _NoSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    handler: Callable[[], None]


def test_a_failed_run_setup_leaves_the_callers_state_unchanged() -> None:
    agent = Agent.create(FakeProvider([text_response("first")]))
    state = run_sync(start(agent, "hi"))
    deps = object()

    with pytest.raises(PydanticInvalidForJsonSchema):  # the run is set up before anything is bound to the state
        run_sync(resume(agent, state, user_input="again", deps=deps, output_model=_NoSchema))

    assert state.output_schema_digest is None and state.output_model_name is None  # not locked to a model
    assert state.deps is None  # deps are not kept alive


def test_a_late_run_setup_failure_leaves_the_callers_state_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fails after the schema was built: binding must still come last."""
    agent = Agent.create(FakeProvider([text_response("first")], capabilities=_NO_NATIVE))
    state = run_sync(start(agent, "hi"))
    deps = object()

    def broken(*args: object, **kwargs: object) -> None:
        raise ConfigError("setup failed")

    monkeypatch.setattr(ToolRegistry, "extended", broken)  # the tool tier adds `final_output` while the run is set up
    with pytest.raises(ConfigError, match="setup failed"):
        run_sync(resume(agent, state, user_input="again", deps=deps, output_model=CityWeather))

    assert state.output_schema_digest is None and state.output_model_name is None
    assert state.deps is None


# ------------------------------------------------------------------ A withheld output must never finish a run

_NO_NATIVE = Capabilities(tools=True, structured_output=False)
_SECRET = {"city": "Bern", "temperature_c": 1.0, "condition": "secret"}
_SNOW = {"city": "Bern", "temperature_c": 1.0, "condition": "Snow"}


class _BlockFirstFinalOutput(Hook):
    """Withholds the result of the first `final_output` call (a DLP check) and lets later ones pass."""

    def __init__(self) -> None:
        self.seen = 0

    async def after_tool(self, state: RunState, invocation: ToolInvocation, result: ToolResult) -> Decision:
        if invocation.tool_name != FINAL_OUTPUT_TOOL_NAME:
            return CONTINUE
        self.seen += 1
        return Decision.block("dlp") if self.seen == 1 else CONTINUE


class _AskBeforeFinalOutput(_BlockFirstFinalOutput):
    async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
        if invocation.tool_name == FINAL_OUTPUT_TOOL_NAME:
            return Decision.require_approval("review the output")
        return CONTINUE


def test_a_withheld_final_output_cannot_finish_the_run_after_another_tool_call() -> None:
    provider = FakeProvider(
        [
            tool_response((FINAL_OUTPUT_TOOL_NAME, _SECRET)),
            tool_response(("lookup_city", {"city": "Bern"})),
            tool_response((FINAL_OUTPUT_TOOL_NAME, _SNOW)),
        ],
        capabilities=_NO_NATIVE,
    )
    agent = Agent.create(provider, tools=[LookupTool()], hooks=[_BlockFirstFinalOutput()])
    state = run_sync(start(agent, "hi", output_model=CityWeather))

    assert state.status is RunStatus.DONE
    assert state.output is not None and state.output["condition"] == "Snow"
    assert len(provider.requests) == 3  # the run did not end after the lookup, with the withheld data


def test_a_withheld_final_output_after_approval_is_not_accepted() -> None:
    provider = FakeProvider(
        [tool_response((FINAL_OUTPUT_TOOL_NAME, _SECRET)), tool_response((FINAL_OUTPUT_TOOL_NAME, _SNOW))],
        capabilities=_NO_NATIVE,
    )
    agent = Agent.create(provider, hooks=[_AskBeforeFinalOutput()])
    waiting = run_sync(start(agent, "hi", output_model=CityWeather))
    assert waiting.status is RunStatus.WAITING_FOR_APPROVAL

    again = run_sync(resume(agent, waiting, approved=True, pending_id=waiting.pending_id, output_model=CityWeather))

    assert again.status is RunStatus.WAITING_FOR_APPROVAL  # asked about the second attempt, not DONE with the secret
    assert again.output is None


def test_an_approved_final_output_ends_the_run_as_done() -> None:
    class AlwaysAsk(Hook):
        async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
            return Decision.require_approval("x") if invocation.tool_name == FINAL_OUTPUT_TOOL_NAME else CONTINUE

    provider = FakeProvider([tool_response((FINAL_OUTPUT_TOOL_NAME, _SNOW))], capabilities=_NO_NATIVE)
    agent = Agent.create(provider, hooks=[AlwaysAsk()])
    waiting = run_sync(start(agent, "hi", output_model=CityWeather))
    done = run_sync(resume(agent, waiting, approved=True, pending_id=waiting.pending_id, output_model=CityWeather))

    assert done.status is RunStatus.DONE and done.output == _SNOW and len(provider.requests) == 1


def test_final_output_is_allowed_under_an_allowlist() -> None:
    provider = FakeProvider([tool_response((FINAL_OUTPUT_TOOL_NAME, _SNOW))], capabilities=_NO_NATIVE)
    agent = Agent.create(provider, tools=[LookupTool()], allowed_tools=["lookup_city"])
    state = run_sync(start(agent, "hi", output_model=CityWeather))

    assert state.status is RunStatus.DONE and state.output == _SNOW


def _output_kept(state: RunState) -> bool:
    return state.output is not None


def test_structured_output_is_dropped_when_a_run_does_not_end_as_done() -> None:
    state = RunState(agent_name="a")
    state.set_output(CityWeather(city="x", temperature_c=1.0, condition="y"))
    assert _output_kept(state)

    state.set_status(RunStatus.FAILED, "boom")

    assert not _output_kept(state)


def test_the_output_schema_counts_towards_the_request_size() -> None:
    import json

    from ai_agent_harness.core.types import ModelRequest

    schema = {"type": "object", "properties": {"city": {"type": "string"}}}
    without = ModelRequest(system="s", messages=(Message.user("hi"),))
    with_schema = ModelRequest(system="s", messages=(Message.user("hi"),), output_schema=schema)

    assert with_schema.estimated_chars - without.estimated_chars == len(json.dumps(schema))
