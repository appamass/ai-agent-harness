"""Tool base classes.

A tool declares its name, description, parameters (Pydantic model), risk level, and
timeout. The executor and policy derive everything else from this declaration. Declarations
are validated on registration — an invalid tool fails at startup, not during the first
production run.

Two variants:
- `BaseTool`: `async def run(...)`. Must NEVER block the event loop
  (no `time.sleep`, no synchronous I/O), otherwise timeouts cannot take effect.
- `SyncTool`: `def run_sync(...)` for blocking code. Runs in a worker thread
  so timeouts and concurrent runs are not blocked. Note: Python cannot forcibly
  kill threads — after a timeout, the thread continues running in the background.
  For hard isolation boundaries, run such code in a subprocess or sandbox.

Return value: text for the model, or `InputRequest` to pause the run and ask the
user a clarifying question. Raise `ToolError` for error messages intended for the model.

Tips so any LLM uses tools reliably: write descriptions as if explaining to a new
colleague when to use the tool and when not to; keep parameter schemas flat; prefer
fewer well-described tools over many.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import inspect
import json
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar, get_type_hints, overload

from pydantic import BaseModel, ConfigDict, create_model

from ..core.errors import ConfigError
from ..core.state import RunState
from ..core.types import ID_PATTERN, Risk, ToolSpec

InputT = TypeVar("InputT", bound=BaseModel)
_NAME = re.compile(ID_PATTERN)
_SCHEMA_TYPE_KEYS = frozenset({"type", "anyOf", "oneOf", "allOf", "$ref", "enum", "const"})


def _has_concrete_schema_type(raw: Any) -> bool:
    """Recursively verifies that a JSON schema node and its children have concrete types."""
    prop: dict[str, Any] = raw
    if not isinstance(raw, dict) or not (_SCHEMA_TYPE_KEYS & prop.keys()):
        return False
    items = prop.get("items")
    if isinstance(items, dict) and not _has_concrete_schema_type(prop.get("items")):
        return False
    prefix_raw = prop.get("prefixItems")
    entries: list[Any] = prefix_raw or []
    if isinstance(prefix_raw, list) and not all(_has_concrete_schema_type(item) for item in entries):
        return False
    additional = prop.get("additionalProperties")
    if additional is True or (
        isinstance(additional, dict) and not _has_concrete_schema_type(prop.get("additionalProperties"))
    ):
        return False
    nested_props: dict[str, dict[str, Any]] = prop.get("properties") or {}
    return all(_has_concrete_schema_type(sub) for sub in nested_props.values())


@dataclass(frozen=True)
class InputRequest:
    """Return value from a tool to pause the run for a clarifying question.

    When resumed, the user's answer becomes the result of this tool call.
    """

    question: str


ToolOutput = str | InputRequest


class BaseTool(ABC, Generic[InputT]):
    name: ClassVar[str]
    description: ClassVar[str]
    input_model: ClassVar[type[BaseModel]]
    risk: ClassVar[Risk]
    timeout_s: ClassVar[float] = 30.0
    untrusted_output: ClassVar[bool] = False

    @abstractmethod
    async def run(self, args: InputT, state: RunState) -> ToolOutput: ...

    @classmethod
    def spec(cls) -> ToolSpec:
        return ToolSpec(name=cls.name, description=cls.description, input_schema=cls.input_model.model_json_schema())

    @classmethod
    def digest(cls) -> str:
        """SHA-256 fingerprint (64 hex chars) of everything that decides what the tool is and may do.

        Covers name, description, input schema, risk, timeout, and `untrusted_output`. Pin it with
        `ToolRegistry(expected_digests=...)` to notice a changed tool.
        """
        canonical = json.dumps(
            {
                "name": cls.name,
                "description": cls.description,
                "input_schema": cls.input_model.model_json_schema(),
                "risk": cls.risk.value,
                "timeout_s": float(cls.timeout_s),
                "untrusted_output": cls.untrusted_output,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def validate_definition(cls) -> None:
        """Validates the tool declaration. Called by the registry upon registration."""
        where = f"Tool class {cls.__name__}"
        if inspect.isabstract(cls):
            raise ConfigError(f"{where} is abstract (missing run or run_sync?)")
        name = getattr(cls, "name", None)
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ConfigError(f"{where}: 'name' must match {ID_PATTERN}, got {name!r}")
        description = getattr(cls, "description", None)
        if not isinstance(description, str) or not description.strip():
            raise ConfigError(f"{where}: 'description' is missing")
        input_model = getattr(cls, "input_model", None)
        if not (isinstance(input_model, type) and issubclass(input_model, BaseModel)):
            raise ConfigError(f"{where}: 'input_model' must be a Pydantic model")
        if input_model is BaseModel:
            raise ConfigError(f"{where}: 'input_model' must be a subclass of BaseModel (even without fields)")
        try:
            schema = input_model.model_json_schema()
            json.dumps(schema)
        except Exception as exc:
            raise ConfigError(f"{where}: 'input_model' cannot be represented as a JSON schema: {exc}") from exc
        if schema.get("type") != "object":
            raise ConfigError(f"{where}: 'input_model' must produce a JSON schema of type 'object'")
        defs: dict[str, dict[str, Any]] = schema.get("$defs") or {}
        for node in (schema, *defs.values()):
            props: dict[str, dict[str, Any]] = node.get("properties") or {}
            for field_name, prop in props.items():
                if not _has_concrete_schema_type(prop):
                    raise ConfigError(
                        f"{where}: field '{field_name}' in 'input_model' lacks a concrete JSON schema type "
                        "(do not use unconstrained Any/object)"
                    )
        if not isinstance(getattr(cls, "risk", None), Risk):
            raise ConfigError(f"{where}: 'risk' must be a Risk value")
        timeout = cls.timeout_s
        if isinstance(timeout, bool) or not isinstance(timeout, int | float) or timeout <= 0:
            raise ConfigError(f"{where}: 'timeout_s' must be > 0")
        if not isinstance(cls.untrusted_output, bool):
            raise ConfigError(f"{where}: 'untrusted_output' must be a bool")


SYNC_TOOL_THREADS = 16
"""Size of the dedicated thread pool for SyncTools. Adjust before the first SyncTool call."""

_sync_pool: list[ThreadPoolExecutor] = []  # at most one, created lazily
_sync_pool_lock = threading.Lock()


def _sync_executor() -> ThreadPoolExecutor:
    with _sync_pool_lock:
        if not _sync_pool:
            _sync_pool.append(ThreadPoolExecutor(max_workers=SYNC_TOOL_THREADS, thread_name_prefix="harness-sync-tool"))
        return _sync_pool[0]


class SyncTool(BaseTool[InputT]):
    """For blocking code. `run_sync` executes in a worker thread.

    SyncTools have a **dedicated, bounded thread pool** (`SYNC_TOOL_THREADS`). Python cannot forcibly
    terminate threads: a tool that hangs past its timeout occupies its worker thread until it finishes.
    In a dedicated pool, this can never starve the default event-loop executor used by the checkpoint
    store and other components. If the pool is full, new calls queue up and hit their timeout — the run continues.
    """

    @abstractmethod
    def run_sync(self, args: InputT, state: RunState) -> ToolOutput: ...

    async def run(self, args: InputT, state: RunState) -> ToolOutput:
        context = contextvars.copy_context()  # like asyncio.to_thread: propagate contextvars such as tracing context
        loop = asyncio.get_running_loop()

        def call() -> ToolOutput:
            return context.run(self.run_sync, args, state)

        return await loop.run_in_executor(_sync_executor(), call)


ToolFn = Callable[..., Any]


@overload
def tool(
    fn: ToolFn,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    risk: Risk,
    timeout_s: float = 30.0,
    untrusted_output: bool = False,
) -> BaseTool[Any]: ...


@overload
def tool(
    fn: None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    risk: Risk,
    timeout_s: float = 30.0,
    untrusted_output: bool = False,
) -> Callable[[ToolFn], BaseTool[Any]]: ...


def tool(
    fn: ToolFn | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    risk: Risk,
    timeout_s: float = 30.0,
    untrusted_output: bool = False,
) -> BaseTool[Any] | Callable[[ToolFn], BaseTool[Any]]:
    """Turns a typed Python function (sync or async) into a validated `BaseTool` or `SyncTool`.

    Parameters are derived from the function's type annotations into a strict Pydantic model
    (`extra="forbid"`). If the function declares a `state: RunState` parameter, the active
    `RunState` is passed automatically and hidden from the model's JSON schema.
    """

    def decorate(target: ToolFn) -> BaseTool[Any]:
        return _build_function_tool(
            target,
            name=name,
            description=description,
            risk=risk,
            timeout_s=timeout_s,
            untrusted_output=untrusted_output,
        )

    return decorate(fn) if fn is not None else decorate


def _build_function_tool(
    fn: ToolFn,
    *,
    name: str | None,
    description: str | None,
    risk: Risk,
    timeout_s: float,
    untrusted_output: bool,
) -> BaseTool[Any]:
    if not callable(fn):
        raise ConfigError("@tool requires a callable function")
    tool_name = name if name is not None else getattr(fn, "__name__", "")
    tool_desc = (description if description is not None else inspect.getdoc(fn)) or ""
    tool_risk = risk
    tool_timeout = timeout_s
    tool_untrusted = untrusted_output

    try:
        hints = get_type_hints(fn, localns={"RunState": RunState})
    except Exception as exc:
        raise ConfigError(f"Tool function '{tool_name}': failed to resolve type annotations: {exc}") from exc

    sig = inspect.signature(fn)
    fields: dict[str, Any] = {}
    field_names: list[str] = []
    passes_state = False
    for param_name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ConfigError(f"Tool function '{tool_name}': *args and **kwargs are not allowed")
        if param_name == "state" and hints.get("state", RunState) is RunState:
            passes_state = True
            continue
        if param_name not in hints:
            raise ConfigError(f"Tool function '{tool_name}': parameter '{param_name}' is missing a type annotation")
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param_name] = (hints[param_name], default)
        field_names.append(param_name)

    model_cls: type[BaseModel] = create_model(
        f"{tool_name}_Input",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )
    names_tuple = tuple(field_names)

    if inspect.iscoroutinefunction(fn):

        class _AsyncFunctionTool(BaseTool[BaseModel]):
            name: ClassVar[str] = tool_name
            description: ClassVar[str] = tool_desc
            input_model: ClassVar[type[BaseModel]] = model_cls
            risk: ClassVar[Risk] = tool_risk
            timeout_s: ClassVar[float] = tool_timeout
            untrusted_output: ClassVar[bool] = tool_untrusted

            async def run(self, args: BaseModel, state: RunState) -> ToolOutput:
                kwargs: dict[str, Any] = {k: getattr(args, k) for k in names_tuple}
                if passes_state:
                    kwargs["state"] = state
                result: ToolOutput = await fn(**kwargs)
                return result

        _AsyncFunctionTool.validate_definition()
        return _AsyncFunctionTool()

    class _SyncFunctionTool(SyncTool[BaseModel]):
        name: ClassVar[str] = tool_name
        description: ClassVar[str] = tool_desc
        input_model: ClassVar[type[BaseModel]] = model_cls
        risk: ClassVar[Risk] = tool_risk
        timeout_s: ClassVar[float] = tool_timeout
        untrusted_output: ClassVar[bool] = tool_untrusted

        def run_sync(self, args: BaseModel, state: RunState) -> ToolOutput:
            kwargs: dict[str, Any] = {k: getattr(args, k) for k in names_tuple}
            if passes_state:
                kwargs["state"] = state
            result: ToolOutput = fn(**kwargs)
            return result

    _SyncFunctionTool.validate_definition()
    return _SyncFunctionTool()
