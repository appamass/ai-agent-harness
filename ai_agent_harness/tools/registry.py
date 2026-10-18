"""Tool registry: tracks available tools and validates every tool definition on registration."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from ..core.errors import ConfigError
from .base import BaseTool


class ToolRegistry:
    def __init__(self, tools: Iterable[BaseTool[Any]] = ()) -> None:
        self._tools: dict[str, BaseTool[Any]] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: BaseTool[Any]) -> None:
        if not isinstance(tool, BaseTool):
            raise ConfigError(f"{tool!r} is not a BaseTool")
        type(tool).validate_definition()
        if tool.name in self._tools:
            raise ConfigError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool[Any] | None:
        return self._tools.get(name)

    def names(self) -> frozenset[str]:
        return frozenset(self._tools)

    def __iter__(self) -> Iterator[BaseTool[Any]]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)
