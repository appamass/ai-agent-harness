"""Built-in guardrails (implemented as hooks). Custom guardrails follow the same pattern:
a class that inherits from `Hook` and overrides only the required method.
"""

from __future__ import annotations

import inspect
import posixpath
import re
from collections.abc import Awaitable, Callable, Iterable, Iterator
from fnmatch import fnmatchcase
from typing import Any

from .core.hooks import CONTINUE, Decision, Hook, ToolInvocation
from .core.state import RunState
from .core.types import ModelRequest, Risk

DEFAULT_PROTECTED: tuple[str, ...] = (
    "tests",
    "tests/*",
    "test/*",
    "conftest.py",
    ".github/*",
    ".gitlab-ci.yml",
    ".pre-commit-config.yaml",
    "evals/*",
    "*thresholds*",
    "pyproject.toml",
    "setup.cfg",
    "pytest.ini",
    "tox.ini",
)

KillCheck = Callable[[RunState], bool | Awaitable[bool]]


class KillSwitch(Hook):
    """Emergency stop: halts runs before the next model call, before every tool, and on every input.

    `check(state)` returns True when the kill switch is active — synchronously or async, e.g., a
    feature flag, a database row, or a file. Via `state.tenant_id`, it can be toggled per tenant.
    Fail-closed: if `check` raises (flag service unreachable) or returns a non-bool, the run blocks too.

    An already running tool is not interrupted; the run stops immediately afterward.
    If the kill switch hits an already approved action, it is answered as blocked: the approval
    is consumed, and after turning the switch off the model must request approval again.
    """

    def __init__(self, check: KillCheck, reason: str = "Kill switch is active") -> None:
        if not callable(check):
            raise TypeError("KillSwitch requires a check function")
        self._check = check
        self._reason = reason

    async def _decide(self, state: RunState) -> Decision:
        value = self._check(state)
        if inspect.isawaitable(value):
            value = await value
        if not isinstance(value, bool):
            raise TypeError(f"KillSwitch check must return bool, got {type(value).__name__}")
        return Decision.block(self._reason) if value else CONTINUE

    async def on_user_input(self, state: RunState, text: str) -> Decision:
        return await self._decide(state)

    async def before_model(self, state: RunState, request: ModelRequest) -> Decision:
        return await self._decide(state)

    async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
        return await self._decide(state)


_PATH_KEY = re.compile(
    r"^(paths?|files?|filename|filenames|target|destination|dest|dst|src|source|dir|directory)$|_(path|paths|file|files|dir)$"
)


class ProtectedPaths(Hook):
    """Reward-hacking protection: write tools must not touch protected paths.

    This prevents an agent from modifying the evaluators it is measured against (tests, CI, thresholds).
    For write tools (risk WRITE/IRREVERSIBLE), it checks — including inside lists and nested objects —
    - all arguments whose field name looks like a path (`path`, `files`, `target_path`, ...)
      or appears in `extra_keys`, and
    - **every** string value without whitespace, regardless of field name (`location`, `save_to`, `uri`, ...).
    Specify paths containing spaces under custom field names via `extra_keys`.

    Comparison is hardened against bypasses: case-insensitivity (important on macOS/Windows),
    `./`, `..`, backslashes, and absolute paths (every path suffix is checked, so `/repo/tests/x.py`
    matches `tests/*`).

    Boundary: shell-like tools accepting free-form command strings cannot be secured this way.
    Run such tools only inside a sandbox with read-only mounts.
    """

    def __init__(self, patterns: Iterable[str] = DEFAULT_PROTECTED, *, extra_keys: Iterable[str] = ()) -> None:
        self._patterns = tuple(p.casefold().strip("/") for p in patterns)
        if not self._patterns:
            raise ValueError("ProtectedPaths requires at least one pattern")
        self._extra_keys = frozenset(k.casefold() for k in extra_keys)

    async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
        if invocation.risk is Risk.READ:
            return CONTINUE
        for value in _path_values(invocation.args.model_dump(), self._extra_keys):
            if self.is_protected(value):
                return Decision.block(f"'{value}' is protected and must not be modified by the agent")
        return CONTINUE

    def is_protected(self, raw: str) -> bool:
        normalized = posixpath.normpath(raw.replace("\\", "/")).casefold()
        segments = [s for s in normalized.split("/") if s not in ("", ".")]
        for start in range(len(segments)):
            candidate = "/".join(segments[start:])
            if any(fnmatchcase(candidate, pattern) for pattern in self._patterns):
                return True
        return False


def _path_values(data: Any, extra_keys: frozenset[str], key: str | None = None, *, top: bool = True) -> Iterator[str]:
    """All values that could be a path: always under path-like field names, otherwise any string without whitespace.

    Free text (email body, comment) almost always contains whitespace and remains allowed. A single
    token like `pyproject.toml` under any field name is checked — fail-closed, better to block once
    too often than allow a modified test file. Keys of nested dictionaries also count when they look
    like paths (`files={"tests/x.py": "..."}`: containing `/`, `\\`, or `.`) — plain names like `tests`
    and top-level tool model field names do not.
    """
    if isinstance(data, dict):
        mapping: dict[Any, Any] = data  # pyright: ignore[reportUnknownVariableType] – JSON-like data
        for k, v in mapping.items():
            if not top and any(sep in str(k) for sep in ("/", "\\", ".")):
                yield str(k)  # keys like "tests/x.py" or "conftest.py", not plain names like "tests"
            yield from _path_values(v, extra_keys, str(k), top=False)
    elif isinstance(data, list | tuple | set):
        items: list[Any] | tuple[Any, ...] | set[Any] = data  # pyright: ignore[reportUnknownVariableType]
        for item in items:
            yield from _path_values(item, extra_keys, key, top=False)
    elif isinstance(data, str):
        named = key is not None and (_PATH_KEY.search(key.casefold()) is not None or key.casefold() in extra_keys)
        if named or (data and not any(c.isspace() for c in data)):
            yield data
