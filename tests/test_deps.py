"""Runtime dependencies (`deps`): DB session, HTTP client, authenticated user token, etc.

They are passed on every `start`/`resume`, accessed by tools and hooks via `state.deps_as(Type)`,
and are **never** written to checkpoints (ADR 0012).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel

from ai_agent_harness import (
    Autonomy,
    BaseTool,
    Decision,
    Hook,
    HookPipeline,
    JsonFileCheckpointStore,
    Risk,
    RunState,
    RunStatus,
    ToolInvocation,
    resume,
    start,
)
from ai_agent_harness.core import InvalidStateError
from ai_agent_harness.testing import text_response, tool_response

from .conftest import EmptyInput, make_agent, run_sync


@dataclass
class Deps:
    user_token: str
    calls: list[str]


class WhoAmI(BaseTool[EmptyInput]):
    name: ClassVar[str] = "whoami"
    description: ClassVar[str] = "uses the authenticated user's token"
    input_model: ClassVar[type[BaseModel]] = EmptyInput
    risk: ClassVar[Risk] = Risk.WRITE

    async def run(self, args: EmptyInput, state: RunState) -> str:
        deps = state.deps_as(Deps)
        deps.calls.append(deps.user_token)
        return "authenticated"


def test_tools_receive_deps_and_checkpoints_never_contain_them(tmp_path: Path) -> None:
    secret = "token-secret-4711"
    deps = Deps(user_token=secret, calls=[])
    store = JsonFileCheckpointStore(tmp_path)
    agent, _, _ = make_agent(
        [tool_response(("whoami", {})), text_response("ok")],
        [WhoAmI()],
        autonomy=Autonomy.AUTONOMOUS,
        checkpoints=store,
    )
    state = run_sync(start(agent, "go", run_id="d1", deps=deps))
    assert state.status is RunStatus.DONE
    assert deps.calls == [secret]
    assert secret not in (tmp_path / "default" / "d1.json").read_text(encoding="utf-8")
    assert secret not in state.model_dump_json()


def test_deps_are_supplied_again_on_resume() -> None:
    """After an approval (possibly days later in a different process), fresh deps are passed."""
    agent, _, _ = make_agent(
        [tool_response(("whoami", {})), text_response("ok")], [WhoAmI()], autonomy=Autonomy.SUGGEST
    )
    waiting = run_sync(start(agent, "go", deps=Deps("old", [])))
    assert waiting.status is RunStatus.WAITING_FOR_APPROVAL
    restored = RunState.model_validate_json(waiting.model_dump_json())
    assert restored.deps is None  # nothing was serialized
    fresh = Deps("new", [])
    done = run_sync(resume(agent, restored, approved=True, deps=fresh, pending_id=restored.pending_id))
    assert done.status is RunStatus.DONE
    assert fresh.calls == ["new"]


def test_missing_deps_end_up_as_tool_error_not_as_crash() -> None:
    agent, provider, _ = make_agent(
        [tool_response(("whoami", {})), text_response("ok")], [WhoAmI()], autonomy=Autonomy.AUTONOMOUS
    )
    state = run_sync(start(agent, "go"))  # forgot to pass deps
    assert state.status is RunStatus.DONE
    result = provider.requests[1].messages[-1].tool_results[0]
    assert result.is_error


def test_hooks_can_use_deps() -> None:
    class OnlyAdmins(Hook):
        async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
            return Decision() if state.deps_as(Deps).user_token == "admin" else Decision.block("not an admin")

    agent, _, _ = make_agent(
        [tool_response(("whoami", {})), text_response("ok")],
        [WhoAmI()],
        autonomy=Autonomy.AUTONOMOUS,
        hooks=HookPipeline([OnlyAdmins()]),
    )
    user = Deps("guest", [])
    run_sync(start(agent, "go", deps=user))
    assert user.calls == []


def test_deps_as_checks_the_type() -> None:
    state = RunState(agent_name="a")
    with pytest.raises(InvalidStateError, match="no deps"):
        state.deps_as(Deps)
    state.bind_deps("wrong type")
    with pytest.raises(InvalidStateError, match="Deps"):
        state.deps_as(Deps)


def test_resume_without_deps_never_reuses_old_deps() -> None:
    """Resuming the same object without deps must not keep stale deps (e.g. an expired token) alive."""
    agent, _, _ = make_agent(
        [tool_response(("whoami", {})), text_response("ok")], [WhoAmI()], autonomy=Autonomy.SUGGEST
    )
    old = Deps("old", [])
    waiting = run_sync(start(agent, "go", deps=old))
    done = run_sync(resume(agent, waiting, approved=True, pending_id=waiting.pending_id))  # without deps
    assert done.status is RunStatus.DONE
    assert old.calls == []  # the tool did not receive the old deps


def test_deps_are_released_after_the_run_and_never_pickled() -> None:
    import pickle

    agent, _, _ = make_agent([text_response("ok")])
    state = run_sync(start(agent, "go", deps=Deps("token-secret-4711", [])))
    assert state.deps is None  # no DB session or token remains attached after the run
    state.bind_deps(Deps("token-secret-4711", []))
    assert b"token-secret-4711" not in pickle.dumps(state)
    assert pickle.loads(pickle.dumps(state)).deps is None


def test_deep_copy_never_copies_deps() -> None:
    """Copying state must not duplicate a DB session — nor fail on uncopyable deps."""
    import copy
    import threading

    state = RunState(agent_name="t")
    lock = threading.Lock()  # uncopyable, like many sessions and clients
    state.bind_deps(lock)
    clone = copy.deepcopy(state)
    assert clone.deps is None
    assert state.deps is lock
    assert state.model_copy(deep=True).deps is None
