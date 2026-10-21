"""Standalone usage of individual harness bricks without requiring a full Agent or RunState."""

from __future__ import annotations

from ai_agent_harness import (
    Autonomy,
    CallableProvider,
    Decision,
    Hook,
    HookPipeline,
    Message,
    ModelRequest,
    ModelResponse,
    ModelRouter,
    PermissionPolicy,
    ProtectedPaths,
    RetryPolicy,
    Risk,
    RunState,
    StopReason,
    TextBlock,
    ToolCall,
    ToolExecutor,
    ToolInvocation,
    ToolResult,
    tool,
)
from ai_agent_harness.core import ConfigError, ProviderError
from ai_agent_harness.core.hooks import Action
from ai_agent_harness.core.state import PendingCall

from .conftest import PathInput, no_sleep, run_sync


def test_hook_pipeline_runs_standalone_with_none_state() -> None:
    seen_agents: list[str] = []

    class Inspector(Hook):
        async def on_user_input(self, state: RunState, text: str) -> Decision:
            seen_agents.append(state.agent_name)
            return Decision.block("no") if text == "bad" else Decision()

        async def before_model(self, state: RunState, request: ModelRequest) -> Decision:
            seen_agents.append(state.agent_name)
            return Decision()

        async def after_model(self, state: RunState, response: ModelResponse) -> Decision:
            seen_agents.append(state.agent_name)
            return Decision()

        async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
            seen_agents.append(state.agent_name)
            return Decision()

        async def after_tool(self, state: RunState, invocation: ToolInvocation, result: ToolResult) -> Decision:
            seen_agents.append(state.agent_name)
            return Decision()

        async def on_run_end(self, state: RunState) -> None:
            seen_agents.append(state.agent_name)

    pipeline = HookPipeline([Inspector(), ProtectedPaths()])
    assert run_sync(pipeline.on_user_input(None, "hello")).action is Action.CONTINUE
    assert run_sync(pipeline.on_user_input(None, "bad")).action is Action.BLOCK

    req = ModelRequest(system="sys", messages=(Message.user("hi"),))
    resp = ModelResponse(
        message=Message(role="assistant", content=(TextBlock(text="ok"),)),
        stop_reason=StopReason.END,
    )
    assert run_sync(pipeline.before_model(None, req)).action is Action.CONTINUE
    assert run_sync(pipeline.after_model(None, resp)).action is Action.CONTINUE

    call = ToolCall(id="c1", name="write_file", arguments={"path": "tests/test_x.py"})
    inv = ToolInvocation(call=call, tool_name="write_file", risk=Risk.WRITE, args=PathInput(path="tests/test_x.py"))
    assert run_sync(pipeline.before_tool(None, inv)).action is Action.BLOCK

    ok_inv = ToolInvocation(call=call, tool_name="write_file", risk=Risk.WRITE, args=PathInput(path="src/app.py"))
    assert run_sync(pipeline.before_tool(None, ok_inv)).action is Action.CONTINUE
    res = ToolResult(call_id="c1", content="done")
    assert run_sync(pipeline.after_tool(None, ok_inv, res)).action is Action.CONTINUE
    run_sync(pipeline.on_run_end())
    assert all(name == "standalone" for name in seen_agents)


def test_tool_executor_runs_standalone_without_state_or_boilerplate() -> None:
    @tool(risk=Risk.READ)
    def double(n: int) -> str:
        """Doubles an integer."""
        return str(n * 2)

    @tool(risk=Risk.WRITE)
    def update_record(record_id: str) -> str:
        """Updates a record by ID."""
        return f"updated {record_id}"

    # Default policy is READ_ONLY: reads succeed, writes are denied
    gate = ToolExecutor([double, update_record])
    ok = run_sync(gate.execute(ToolCall(id="1", name="double", arguments={"n": 21})))
    assert isinstance(ok, ToolResult) and not ok.is_error and ok.content == "42"

    denied = run_sync(gate.execute(ToolCall(id="2", name="update_record", arguments={"record_id": "R1"})))
    assert isinstance(denied, ToolResult) and denied.is_error and "read_only" in denied.content

    # With SUGGEST autonomy: writes return a PendingCall requesting approval
    suggest_gate = ToolExecutor(
        [double, update_record],
        policy=PermissionPolicy(autonomy=Autonomy.SUGGEST),
        hooks=[ProtectedPaths()],
    )
    pending = run_sync(suggest_gate.execute(ToolCall(id="3", name="update_record", arguments={"record_id": "R1"})))
    assert isinstance(pending, PendingCall) and pending.kind == "approval"

    approved = run_sync(
        suggest_gate.execute(ToolCall(id="3", name="update_record", arguments={"record_id": "R1"}), approved=True)
    )
    assert isinstance(approved, ToolResult) and not approved.is_error and approved.content == "updated R1"


def test_callable_provider_and_standalone_router() -> None:
    attempts = 0

    async def flaky_primary(request: ModelRequest) -> str:
        nonlocal attempts
        attempts += 1
        raise ProviderError("temporary 503", retryable=True, provider_id="primary")

    async def backup_model(request: ModelRequest) -> str:
        return f"Answered: {request.messages[-1].text}"

    primary = CallableProvider(flaky_primary, id="primary")
    backup = CallableProvider(backup_model, id="backup", vision=True)
    assert primary.id == "primary" and not primary.capabilities.vision
    assert backup.id == "backup" and backup.capabilities.vision

    router = ModelRouter(
        primary,
        fallbacks=[backup],
        retry=RetryPolicy(max_attempts=2, base_delay_s=0.0, max_delay_s=0.0),
        sleep=no_sleep,
    )
    request = ModelRequest(system="Be concise.", messages=(Message.user("ping"),))
    response, winner = run_sync(router.complete(request))
    assert winner == "backup"
    assert attempts == 2
    assert response.message.text == "Answered: ping"
    assert response.stop_reason is StopReason.END


def test_callable_provider_validation() -> None:
    async def valid(request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            message=Message(role="assistant", content=(TextBlock(text="full"),)),
            stop_reason=StopReason.END,
        )

    provider = CallableProvider(valid, id="custom")
    req = ModelRequest(system="s", messages=(Message.user("hi"),))
    resp = run_sync(provider.complete(req))
    assert resp.message.text == "full"

    try:
        CallableProvider(valid, id="   ")
    except ConfigError:
        pass
    else:
        raise AssertionError("Expected ConfigError for empty provider id")
