"""Built-in example tools demonstrating both patterns: async `BaseTool` and `SyncTool`."""

from __future__ import annotations

import os
from collections.abc import Iterable
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..core.errors import ConfigError, ToolError
from ..core.types import Risk
from .base import BaseTool, InputRequest, SyncTool, ToolOutput

if TYPE_CHECKING:
    from ..core.state import RunState


class AskUserInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000, description="A specific, concise question for the user.")


class AskUser(BaseTool[AskUserInput]):
    """Clarifying question on ambiguity. Pauses the run until the user responds."""

    name: ClassVar[str] = "ask_user"
    description: ClassVar[str] = (
        "Asks the user a clarifying question. Only use when the task cannot be reasonably solved "
        "without the answer — never to confirm obvious steps."
    )
    input_model: ClassVar[type[BaseModel]] = AskUserInput
    risk: ClassVar[Risk] = Risk.READ
    timeout_s: ClassVar[float] = 5.0

    async def run(self, args: AskUserInput, state: RunState) -> ToolOutput:
        return InputRequest(question=args.question)


DEFAULT_DENIED_READS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.env",
    ".envrc",
    "secrets",
    ".git",
    ".ssh",
    ".aws",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "id_rsa",
    "id_ed25519",
    "credentials.json",
)
"""Name patterns that `ReadFile` refuses by default. A pattern matches any single part of the path
(directory or file name, case-insensitive), so `secrets` also blocks `secrets/token.txt`."""


class ReadFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        min_length=1, max_length=1024, description="Relative path inside the project directory, e.g. 'src/main.py'."
    )


class ReadFile(SyncTool[ReadFileInput]):
    """Reads text files within a fixed root directory. The boundary is enforced in code, not in the prompt.

    It is `Risk.READ` and therefore allowed even at `read_only`. So it refuses secrets by default
    (`DEFAULT_DENIED_READS`). Pass `denied=` to replace the list, e.g. `(*DEFAULT_DENIED_READS, "*.sqlite")`,
    or `denied=()` to allow everything under the root.
    """

    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = "Reads a UTF-8 text file from the project directory and returns its content."
    input_model: ClassVar[type[BaseModel]] = ReadFileInput
    risk: ClassVar[Risk] = Risk.READ
    timeout_s: ClassVar[float] = 10.0

    def __init__(
        self, root: str | Path, *, max_bytes: int = 1_000_000, denied: Iterable[str] = DEFAULT_DENIED_READS
    ) -> None:
        self._root = Path(root).resolve(strict=False)
        if not self._root.is_dir():
            raise ConfigError(f"ReadFile: root directory does not exist: {self._root}")
        if max_bytes < 1:
            raise ConfigError("ReadFile: max_bytes must be > 0")
        self._max_bytes = max_bytes
        if isinstance(denied, str):
            raise ConfigError("ReadFile: denied must be a collection of patterns, not a single string")
        patterns = tuple(denied)
        if any(not _is_name_pattern(pattern) for pattern in patterns):
            raise ConfigError(
                "ReadFile: denied patterns must be non-empty names without path separators or outer spaces"
            )
        self._denied = tuple(pattern.casefold() for pattern in patterns)

    def run_sync(self, args: ReadFileInput, state: RunState) -> ToolOutput:
        target = (self._root / args.path).resolve(strict=False)  # also resolves symlinks
        if not target.is_relative_to(self._root):
            raise ToolError(f"'{args.path}' is outside the project directory.")
        # Both the requested and the resolved path are checked: `harmless.txt -> .env` and the alias
        # `.env -> vault/prod` are both blocked. The target `vault/prod` itself stays readable by its own name.
        requested = Path(os.path.normpath(self._root / args.path))
        if self._is_denied(target) or self._is_denied(requested):
            raise ToolError(f"'{args.path}' is protected and cannot be read.")
        if not target.is_file():
            raise ToolError(f"'{args.path}' is not an existing file.")
        if target.stat().st_size > self._max_bytes:
            raise ToolError(f"'{args.path}' exceeds the maximum size of {self._max_bytes} bytes.")
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"'{args.path}' is not a valid UTF-8 text file.") from exc

    def _is_denied(self, path: Path) -> bool:
        if not path.is_relative_to(self._root):
            return False  # outside paths are rejected by the root check on the resolved path
        parts = [part.casefold() for part in path.relative_to(self._root).parts]
        return any(fnmatchcase(part, pattern) for part in parts for pattern in self._denied)


def _is_name_pattern(pattern: object) -> bool:
    return (
        isinstance(pattern, str)
        and bool(pattern)
        and pattern == pattern.strip()
        and "/" not in pattern
        and "\\" not in pattern
    )
