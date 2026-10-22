"""Executor, tool definitions, and built-in tools."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from pydantic import BaseModel

from ai_agent_harness import (
    DEFAULT_DENIED_READS,
    Agent,
    AskUser,
    Autonomy,
    BaseTool,
    Decision,
    Hook,
    HookPipeline,
    ReadFile,
    Risk,
    RunState,
    RunStatus,
    SyncTool,
    ToolCall,
    ToolError,
    ToolExecutor,
    ToolRegistry,
    start,
    tool,
)
from ai_agent_harness.core import ConfigError, ToolInvocation, ToolResult
from ai_agent_harness.testing import FakeProvider, text_response, tool_response
from ai_agent_harness.tools.builtin import ReadFileInput

from .conftest import Echo, EmptyInput, PathInput, WriteFile, make_agent, results_of, run_sync


class BlockingSync(SyncTool[EmptyInput]):
    name: ClassVar[str] = "blocking"
    description: ClassVar[str] = "blocks"
    input_model: ClassVar[type[BaseModel]] = EmptyInput
    risk: ClassVar[Risk] = Risk.READ
    timeout_s: ClassVar[float] = 0.1

    def run_sync(self, args: EmptyInput, state: RunState) -> str:
        time.sleep(0.5)
        return "too late"


def test_sync_tool_timeout_is_enforced() -> None:
    """Regression K4: blocking code previously bypassed the timeout."""
    agent, _, _ = make_agent([tool_response(("blocking", {})), text_response("ok")], [BlockingSync()])

    async def timed() -> tuple[float, RunState]:
        began = time.monotonic()
        state = await start(agent, "go")
        return time.monotonic() - began, state

    # Measured inside the run: asyncio.run waits on shutdown for the (uninterruptible) worker thread.
    duration, state = run_sync(timed())
    assert duration < 0.4
    assert "timeout" in results_of(state)[0].content


def test_internal_tool_errors_do_not_leak_details() -> None:
    class Leaky(BaseTool[EmptyInput]):
        name: ClassVar[str] = "leaky"
        description: ClassVar[str] = "raises"
        input_model: ClassVar[type[BaseModel]] = EmptyInput
        risk: ClassVar[Risk] = Risk.READ

        async def run(self, args: EmptyInput, state: RunState) -> str:
            raise RuntimeError("password=hunter2 at /etc/secret")

    agent, _, _ = make_agent([tool_response(("leaky", {})), text_response("ok")], [Leaky()])
    content = results_of(run_sync(start(agent, "go")))[0].content
    assert "hunter2" not in content and "/etc" not in content
    assert "RuntimeError" in content


def test_tool_error_message_reaches_model() -> None:
    class Friendly(BaseTool[EmptyInput]):
        name: ClassVar[str] = "friendly"
        description: ClassVar[str] = "raises ToolError"
        input_model: ClassVar[type[BaseModel]] = EmptyInput
        risk: ClassVar[Risk] = Risk.READ

        async def run(self, args: EmptyInput, state: RunState) -> str:
            raise ToolError("Customer 42 does not exist")

    agent, _, _ = make_agent([tool_response(("friendly", {})), text_response("ok")], [Friendly()])
    assert "Customer 42 does not exist" in results_of(run_sync(start(agent, "go")))[0].content


def test_non_string_output_is_rejected() -> None:
    class Bad(BaseTool[EmptyInput]):
        name: ClassVar[str] = "bad"
        description: ClassVar[str] = "returns dict"
        input_model: ClassVar[type[BaseModel]] = EmptyInput
        risk: ClassVar[Risk] = Risk.READ

        async def run(self, args: EmptyInput, state: RunState) -> Any:
            return {"x": 1}

    agent, _, _ = make_agent([tool_response(("bad", {})), text_response("ok")], [Bad()])
    assert results_of(run_sync(start(agent, "go")))[0].is_error


def test_long_output_is_truncated() -> None:
    class Big(BaseTool[EmptyInput]):
        name: ClassVar[str] = "big"
        description: ClassVar[str] = "large text"
        input_model: ClassVar[type[BaseModel]] = EmptyInput
        risk: ClassVar[Risk] = Risk.READ

        async def run(self, args: EmptyInput, state: RunState) -> str:
            return "x" * 5000

    agent, _, _ = make_agent([tool_response(("big", {})), text_response("ok")], [Big()], max_result_chars=1000)
    content = results_of(run_sync(start(agent, "go")))[0].content
    assert len(content) < 1100 and "truncated" in content


def test_after_tool_block_withholds_result() -> None:
    """Regression H1: previously a block decision in after_tool was ignored."""

    class NoEcho(Hook):
        async def after_tool(self, state: RunState, invocation: ToolInvocation, result: ToolResult) -> Decision:
            return Decision.block("contains secret")

    agent, _, _ = make_agent(
        [tool_response(("echo", {})), text_response("ok")], [Echo()], hooks=HookPipeline([NoEcho()])
    )
    result = results_of(run_sync(start(agent, "go")))[0]
    assert result.is_error and "withheld" in result.content and "echo" not in result.content.split(":")[-1]


def test_before_tool_can_require_approval_with_validated_args() -> None:
    seen: list[BaseModel] = []

    class BigWrites(Hook):
        async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
            seen.append(invocation.args)
            return Decision.require_approval("large write operation")

    write = WriteFile()
    agent, _, _ = make_agent(
        [tool_response(("write_file", {"path": "a"}))],
        [write],
        autonomy=Autonomy.AUTONOMOUS,
        hooks=HookPipeline([BigWrites()]),
    )
    state = run_sync(start(agent, "go"))
    assert state.status is RunStatus.WAITING_FOR_APPROVAL
    assert "large write operation" in (state.status_reason or "")
    assert isinstance(seen[0], PathInput)
    assert seen[0].path == "a"
    assert write.written == []


def test_policy_is_checked_before_hooks() -> None:
    calls: list[str] = []

    class Spy(Hook):
        async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
            calls.append(invocation.tool_name)
            return Decision()

    agent, _, _ = make_agent(
        [tool_response(("write_file", {"path": "a"})), text_response("ok")], [WriteFile()], hooks=HookPipeline([Spy()])
    )
    run_sync(start(agent, "go"))
    assert calls == []


def test_allowlist_restricts_tools() -> None:
    agent, provider, _ = make_agent(
        [tool_response(("echo", {})), text_response("ok")], [Echo(), AskUser()], allowed_tools=frozenset({"ask_user"})
    )
    state = run_sync(start(agent, "go"))
    assert [t.name for t in provider.requests[0].tools] == ["ask_user"]
    assert results_of(state)[0].is_error


def test_allowlist_with_unknown_tool_is_config_error() -> None:
    with pytest.raises(ConfigError):
        make_agent([], [Echo()], allowed_tools=frozenset({"ehco"}))


class TestToolDefinitions:
    def test_bad_name_is_rejected_at_registration(self) -> None:
        class BadName(Echo):
            name: ClassVar[str] = "has spaces"

        with pytest.raises(ConfigError):
            ToolRegistry([BadName()])

    def test_missing_description_is_rejected(self) -> None:
        class NoDescription(Echo):
            description: ClassVar[str] = " "

        with pytest.raises(ConfigError):
            ToolRegistry([NoDescription()])

    def test_invalid_timeout_is_rejected(self) -> None:
        class NoTimeout(Echo):
            timeout_s: ClassVar[float] = 0

        with pytest.raises(ConfigError):
            ToolRegistry([NoTimeout()])

    def test_duplicate_name_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            ToolRegistry([Echo(), Echo()])

    def test_bare_base_model_as_input_is_rejected(self) -> None:
        """`BaseModel` itself cannot validate — it would only fail at runtime as `failed`."""

        class BareInput(Echo):
            input_model: ClassVar[type[BaseModel]] = BaseModel

        with pytest.raises(ConfigError, match="subclass"):
            ToolRegistry([BareInput()])

    def test_untyped_field_in_input_model_is_rejected(self) -> None:
        """Fields with unconstrained `Any`/`object` (including in dicts or tuples) produce a typeless JSON schema."""

        class UntypedInput(BaseModel):
            payload: Any

        class UntypedDictInput(BaseModel):
            mapping: dict[str, Any]

        class UntypedTool(Echo):
            input_model: ClassVar[type[BaseModel]] = UntypedInput

        class UntypedDictTool(Echo):
            input_model: ClassVar[type[BaseModel]] = UntypedDictInput

        with pytest.raises(ConfigError, match="payload"):
            ToolRegistry([UntypedTool()])
        with pytest.raises(ConfigError, match="mapping"):
            ToolRegistry([UntypedDictTool()])

    def test_non_json_schema_field_in_input_model_is_rejected(self) -> None:
        """An `input_model` that cannot produce a valid JSON schema fails immediately on registration."""
        from pydantic import ConfigDict

        class ArbitraryInput(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            lock: asyncio.Lock

        class ArbitraryTool(Echo):
            input_model: ClassVar[type[BaseModel]] = ArbitraryInput

        with pytest.raises(ConfigError, match="JSON schema"):
            ToolRegistry([ArbitraryTool()])


class TestReadFile:
    @pytest.fixture
    def root(self, tmp_path: Path) -> Path:
        (tmp_path / "project").mkdir()
        (tmp_path / "project" / "ok.txt").write_text("hello", encoding="utf-8")
        (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
        return tmp_path / "project"

    def read(self, root: Path, path: str, **kwargs: Any) -> str:
        output = asyncio.run(ReadFile(root, **kwargs).run(ReadFileInput(path=path), RunState(agent_name="t")))
        assert isinstance(output, str)
        return output

    def test_reads_file(self, root: Path) -> None:
        assert self.read(root, "ok.txt") == "hello"

    @pytest.mark.parametrize("path", ["../secret.txt", "/etc/passwd", "sub/../../secret.txt"])
    def test_traversal_is_blocked(self, root: Path, path: str) -> None:
        with pytest.raises(ToolError, match=r"outside|not an existing"):
            self.read(root, path)

    def test_symlink_escape_is_blocked(self, root: Path) -> None:
        (root / "link.txt").symlink_to(root.parent / "secret.txt")
        with pytest.raises(ToolError, match="outside"):
            self.read(root, "link.txt")

    def test_size_limit(self, root: Path) -> None:
        with pytest.raises(ToolError, match="exceeds"):
            self.read(root, "ok.txt", max_bytes=2)

    def test_binary_file(self, root: Path) -> None:
        (root / "bin.dat").write_bytes(b"\xff\xfe\x00")
        with pytest.raises(ToolError, match="UTF-8"):
            self.read(root, "bin.dat")

    def test_missing_root_is_config_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError):
            ReadFile(tmp_path / "does-not-exist")

    @pytest.mark.parametrize(
        "path",
        [
            ".env",
            ".env.local",
            "envs/production.env",
            "secrets/token.txt",
            "config/server.pem",
            "deploy.KEY",
            ".git/config",
            ".GIT/HEAD",
            "x/../.env",
        ],
    )
    def test_sensitive_files_are_denied_by_default(self, root: Path, path: str) -> None:
        """ReadFile is Risk.READ and allowed even at read_only, so it must not leak secrets by default."""
        target = root / path.replace("x/../", "")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("secret", encoding="utf-8")
        with pytest.raises(ToolError, match="protected"):
            self.read(root, path)

    def test_symlink_to_a_denied_file_is_blocked(self, root: Path) -> None:
        (root / ".env").write_text("secret", encoding="utf-8")
        (root / "harmless.txt").symlink_to(root / ".env")
        with pytest.raises(ToolError, match="protected"):
            self.read(root, "harmless.txt")

    @pytest.mark.parametrize(("link", "real"), [(".env", "envs/production.env"), ("secrets", "vault")])
    def test_denied_name_that_is_a_symlink_to_an_allowed_name_is_blocked(
        self, root: Path, link: str, real: str
    ) -> None:
        """Common layout: `.env -> envs/production.env`. The requested name must be checked, not only the target."""
        target = root / real
        target.parent.mkdir(parents=True, exist_ok=True)
        if link == "secrets":
            target.mkdir()
            (target / "tok").write_text("secret", encoding="utf-8")
        else:
            target.write_text("SECRET=1", encoding="utf-8")
        (root / link).symlink_to(target)
        requested = ["secrets/tok"] if link == "secrets" else [".env", "./.env", ".ENV"]
        for path in requested:
            with pytest.raises(ToolError, match="protected"):
                self.read(root, path)

    def test_similar_names_are_not_denied(self, root: Path) -> None:
        (root / "environment.md").write_text("docs", encoding="utf-8")
        (root / "my_secrets_guide.txt").write_text("guide", encoding="utf-8")
        assert self.read(root, "environment.md") == "docs"
        assert self.read(root, "my_secrets_guide.txt") == "guide"

    def test_deny_list_can_be_overridden(self, root: Path) -> None:
        (root / ".env").write_text("A=1", encoding="utf-8")
        (root / "data.sqlite").write_text("db", encoding="utf-8")
        assert self.read(root, ".env", denied=()) == "A=1"
        with pytest.raises(ToolError, match="protected"):
            self.read(root, "data.sqlite", denied=(*DEFAULT_DENIED_READS, "*.sqlite"))

    @pytest.mark.parametrize("pattern", ["", "secrets/", "a/b", "  ", " .env", ".env ", None, 3])
    def test_invalid_deny_pattern_is_config_error(self, root: Path, pattern: Any) -> None:
        with pytest.raises(ConfigError):
            ReadFile(root, denied=(pattern,))

    def test_single_string_as_deny_list_is_config_error(self, root: Path) -> None:
        with pytest.raises(ConfigError):
            ReadFile(root, denied=".env")


def test_hanging_sync_tools_never_starve_the_checkpoint_store(tmp_path: Path) -> None:
    """Finding 2.5: hanging SyncTools use a dedicated thread pool, not the event loop's default executor.

    Here the default pool is shrunk to a single thread. If SyncTools ran in it, the first hanging
    call would occupy that thread and checkpoint saving would block forever.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from ai_agent_harness import JsonFileCheckpointStore, RunStatus, SyncTool, start

    release = threading.Event()

    class Hangs(SyncTool[EmptyInput]):
        name: ClassVar[str] = "hangs"
        description: ClassVar[str] = "hangs (e.g. socket without timeout)"
        input_model: ClassVar[type[BaseModel]] = EmptyInput
        risk: ClassVar[Risk] = Risk.READ
        timeout_s: ClassVar[float] = 0.05

        def run_sync(self, args: EmptyInput, state: RunState) -> str:
            release.wait(10)
            return "late"

    agent, _, _ = make_agent(
        [tool_response(("hangs", {}), ("hangs", {}), ("hangs", {})), text_response("ok")],
        [Hangs()],
        checkpoints=JsonFileCheckpointStore(tmp_path),
    )

    async def main() -> RunStatus:
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
        state = await asyncio.wait_for(start(agent, "go"), timeout=5)
        return state.status

    try:
        assert asyncio.run(main()) is RunStatus.DONE
    finally:
        release.set()


def test_sync_tool_sees_context_variables_of_the_caller() -> None:
    """Tracing context (e.g. OpenTelemetry) must propagate into the SyncTool worker thread."""
    import contextvars

    from ai_agent_harness import SyncTool

    trace: contextvars.ContextVar[str] = contextvars.ContextVar("trace", default="none")
    seen: list[str] = []

    class Traced(SyncTool[EmptyInput]):
        name: ClassVar[str] = "traced"
        description: ClassVar[str] = "reads tracing context"
        input_model: ClassVar[type[BaseModel]] = EmptyInput
        risk: ClassVar[Risk] = Risk.READ

        def run_sync(self, args: EmptyInput, state: RunState) -> str:
            seen.append(trace.get())
            return "ok"

    agent, _, _ = make_agent([tool_response(("traced", {})), text_response("ok")], [Traced()])

    async def main() -> None:
        trace.set("trace-1")
        await start(agent, "go")

    asyncio.run(main())
    assert seen == ["trace-1"]


def test_smart_truncation_head_and_tail_preserves_error_summary() -> None:
    """Smart truncation (head_and_tail) preserves 60% head and 40% tail so traceback/errors are visible."""
    prefix = "START OF LOGS: " + ("abcdefghij" * 50)  # 515 chars
    suffix = "FINAL ERROR: ZeroDivisionError: integer division by zero\nExit code 1"
    full_output = f"{prefix}\n... MIDDLE NOISE ...\n{suffix}"

    @tool(risk=Risk.READ)
    def run_job() -> str:
        """Runs a job producing large output."""
        return full_output

    gate = ToolExecutor([run_job], max_result_chars=200, truncation_mode="head_and_tail")
    outcome = run_sync(gate.execute(ToolCall(id="c1", name="run_job", arguments={})))
    assert isinstance(outcome, ToolResult) and not outcome.is_error
    assert outcome.content.startswith("START OF LOGS:")
    assert outcome.content.endswith("integer division by zero\nExit code 1")
    assert "[truncated," in outcome.content
    assert "chars omitted" in outcome.content


def test_smart_truncation_head_only_mode() -> None:
    """When truncation_mode is 'head', only the beginning of the output is retained."""
    prefix = "HEAD ONLY DATA: " + ("1234567890" * 30)
    suffix = "TAIL DATA THAT SHOULD BE DROPPED"
    full_output = f"{prefix}\n{suffix}"

    @tool(risk=Risk.READ)
    def fetch_stream() -> str:
        """Fetches a stream."""
        return full_output

    gate = ToolExecutor([fetch_stream], max_result_chars=150, truncation_mode="head")
    outcome = run_sync(gate.execute(ToolCall(id="c2", name="fetch_stream", arguments={})))
    assert isinstance(outcome, ToolResult) and not outcome.is_error
    assert outcome.content.startswith("HEAD ONLY DATA:")
    assert suffix not in outcome.content
    assert "[truncated," in outcome.content


def test_per_tool_max_result_chars_and_truncation_mode_override() -> None:
    """Individual tools can override max_result_chars and truncation_mode."""

    @tool(risk=Risk.READ, max_result_chars=80, truncation_mode="head")
    def custom_tool() -> str:
        """Tool with strict per-tool truncation."""
        return "A" * 200 + "END_TAIL"

    gate = ToolExecutor([custom_tool], max_result_chars=10_000, truncation_mode="head_and_tail")
    outcome = run_sync(gate.execute(ToolCall(id="c3", name="custom_tool", arguments={})))
    assert isinstance(outcome, ToolResult)
    assert outcome.content.startswith("A" * 80)
    assert "END_TAIL" not in outcome.content
    assert "[truncated, 128 chars omitted]" in outcome.content


def test_tool_definition_validation_for_truncation() -> None:
    """Invalid truncation settings are rejected at tool definition time."""

    class BadTool1(BaseTool[EmptyInput]):
        name: ClassVar[str] = "bad1"
        description: ClassVar[str] = "bad"
        input_model: ClassVar[type[BaseModel]] = EmptyInput
        risk: ClassVar[Risk] = Risk.READ
        max_result_chars: ClassVar[int | None] = 0

        async def run(self, args: EmptyInput, state: RunState) -> str:
            return "ok"

    with pytest.raises(ConfigError, match="max_result_chars"):
        BadTool1.validate_definition()

    class BadTool2(BaseTool[EmptyInput]):
        name: ClassVar[str] = "bad2"
        description: ClassVar[str] = "bad"
        input_model: ClassVar[type[BaseModel]] = EmptyInput
        risk: ClassVar[Risk] = Risk.READ
        truncation_mode: ClassVar[Any] = "invalid_mode"

        async def run(self, args: EmptyInput, state: RunState) -> str:
            return "ok"

    with pytest.raises(ConfigError, match="truncation_mode"):
        BadTool2.validate_definition()


def _same_name_tool(**settings: Any) -> BaseTool[Any]:
    @tool(name="same", risk=Risk.READ, **settings)
    def same() -> str:
        """Same."""
        return "ok"

    return same


def test_tool_digest_changes_with_the_truncation_settings_alone() -> None:
    """Identical tools (same name and body) must differ in the digest when only a truncation setting differs."""
    base = _same_name_tool().digest()
    assert _same_name_tool(max_result_chars=500).digest() != base
    assert _same_name_tool(truncation_mode="head").digest() != base
    assert _same_name_tool(max_result_chars=500).digest() != _same_name_tool(max_result_chars=501).digest()


def test_digest_of_a_plain_tool_matches_the_documented_formula() -> None:
    """Pins stay valid across releases: SHA-256 over this canonical JSON (UTF-8, no ASCII escapes)."""

    def expected(tool_: BaseTool[Any], description: str) -> str:
        canonical = json.dumps(
            {
                "name": "same",
                "description": description,
                "input_schema": tool_.input_model.model_json_schema(),
                "risk": "read",
                "timeout_s": 30.0,
                "untrusted_output": False,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    plain = _same_name_tool()
    assert plain.digest() == expected(plain, "Same.")

    @tool(name="same", risk=Risk.READ)
    def umlaut() -> str:
        """Grüße."""
        return "ok"

    assert umlaut.digest() == expected(umlaut, "Grüße.")


def test_agent_truncation_mode_reaches_the_gate_of_a_run() -> None:
    @tool(risk=Risk.READ)
    def big() -> str:
        """Returns a long text."""
        return "HEAD" + "x" * 5000 + "TAIL"

    provider = FakeProvider([tool_response(("big", {})), text_response("ok")])
    agent = Agent.create(provider, tools=[big], max_result_chars=200, truncation_mode="head")
    state = run_sync(start(agent, "go"))

    content = state.messages[2].tool_results[0].content
    assert content.startswith("HEAD") and "TAIL" not in content


def test_agent_rejects_an_invalid_truncation_mode() -> None:
    with pytest.raises(ConfigError, match="truncation_mode"):
        Agent.create(FakeProvider([]), truncation_mode=cast("Any", "tail"))


def test_per_tool_limit_can_lower_but_never_raise_the_agent_limit() -> None:
    @tool(risk=Risk.READ, max_result_chars=1_000_000)
    def greedy() -> str:
        """A tool that asks for a larger limit than the agent allows."""
        return "x" * 5_000

    gate = ToolExecutor([greedy], max_result_chars=300, truncation_mode="head")
    outcome = run_sync(gate.execute(ToolCall(id="c1", name="greedy", arguments={})))

    assert isinstance(outcome, ToolResult)
    assert outcome.content.startswith("x" * 300 + "\n")
    assert "4700 chars omitted" in outcome.content


def test_executor_rejects_invalid_truncation_mode() -> None:
    with pytest.raises(ConfigError, match="truncation_mode"):
        ToolExecutor([], truncation_mode=cast("Any", "tail"))


def test_tool_error_text_is_truncated_and_framed_for_untrusted_tools() -> None:
    from ai_agent_harness.core import ToolError

    @tool(risk=Risk.READ, untrusted_output=True)
    def noisy() -> str:
        """Fails with a huge message."""
        raise ToolError("E" * 50_000)

    gate = ToolExecutor([noisy], max_result_chars=300)
    outcome = run_sync(gate.execute(ToolCall(id="c9", name="noisy", arguments={})))

    assert isinstance(outcome, ToolResult)
    assert outcome.is_error
    assert outcome.content.startswith("<untrusted")
    assert len(outcome.content) < 1_000
