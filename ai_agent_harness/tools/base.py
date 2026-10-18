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
import inspect
import json
import re
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel

from ..core.errors import ConfigError
from ..core.types import ID_PATTERN, Risk, ToolSpec

if TYPE_CHECKING:
    from ..core.state import RunState

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

    @abstractmethod
    async def run(self, args: InputT, state: RunState) -> ToolOutput: ...

    @classmethod
    def spec(cls) -> ToolSpec:
        return ToolSpec(name=cls.name, description=cls.description, input_schema=cls.input_model.model_json_schema())

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
