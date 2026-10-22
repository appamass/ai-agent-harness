"""Tool registry: tracks available tools and validates every tool definition on registration."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from ..core.errors import ConfigError
from .base import BaseTool


class ToolRegistry:
    """Holds the tools of an agent.

    `expected_digests` pins tools by `BaseTool.digest()`. A pin that matches no registered tool, or a tool
    whose digest differs, is a `ConfigError`. Pins are checked when a tool is registered and again on
    `verify()`. Tools that have no pin are not checked. Pins need their tools in the same constructor call.
    """

    def __init__(
        self,
        tools: Iterable[BaseTool[Any]] = (),
        *,
        expected_digests: Mapping[str, str] | None = None,
    ) -> None:
        self._tools: dict[str, BaseTool[Any]] = {}
        self._expected_digests: dict[str, str] = dict(expected_digests) if expected_digests is not None else {}
        for tool in tools:
            self.register(tool)
        unknown = sorted(set(self._expected_digests) - set(self._tools))
        if unknown:
            raise ConfigError(f"Pinned digests for tools that are not registered: {', '.join(unknown)}")

    def register(self, tool: BaseTool[Any]) -> None:
        if not isinstance(tool, BaseTool):
            raise ConfigError(f"{tool!r} is not a BaseTool")
        type(tool).validate_definition()
        if tool.name in self._tools:
            raise ConfigError(f"Tool '{tool.name}' is already registered")
        self._check_pin(tool)
        self._tools[tool.name] = tool

    def extended(self, tool: BaseTool[Any]) -> ToolRegistry:
        """A copy with one more tool. Only the new tool is validated; the others already were."""
        copy = ToolRegistry()
        copy._tools = dict(self._tools)
        copy._expected_digests = dict(self._expected_digests)
        copy.register(tool)
        return copy

    def digests(self) -> dict[str, str]:
        """Returns `{tool_name: digest}` for all registered tools, as they are now."""
        return {name: tool.digest() for name, tool in self._tools.items()}

    def verify(self) -> None:
        """Checks all pinned tools against their pins again. Raises `ConfigError` on any difference.

        Call it when tool classes could have changed after startup. The executor does not call it.
        """
        for tool in self._tools.values():
            self._check_pin(tool)

    def _check_pin(self, tool: BaseTool[Any]) -> None:
        expected = self._expected_digests.get(tool.name)
        if expected is not None and tool.digest() != expected:
            raise ConfigError(f"Tool '{tool.name}' failed the integrity check: its definition differs from the pin")

    def get(self, name: str) -> BaseTool[Any] | None:
        return self._tools.get(name)

    def names(self) -> frozenset[str]:
        return frozenset(self._tools)

    def __iter__(self) -> Iterator[BaseTool[Any]]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)
