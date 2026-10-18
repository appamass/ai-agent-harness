"""Context assembly: turns state and tools into a `ModelRequest` for each step.

Order inside the system prompt: fixed prompt first, then sources (memory, RAG,
skills, etc.). Stable parts go first so provider prompt caching stays effective.
Additional sources implement `ContextSource` and are passed to the builder.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .core.errors import ConfigError
from .core.state import RunState
from .core.types import ModelRequest, ToolSpec


class ContextSource(Protocol):
    """Returns a section for the system prompt or None when nothing applies."""

    async def render(self, state: RunState) -> str | None: ...


@dataclass(frozen=True)
class ContextBuilder:
    system_prompt: str
    sources: Sequence[ContextSource] = field(default_factory=tuple)
    max_tokens: int = 4096
    temperature: float | None = None

    def __post_init__(self) -> None:
        if not self.system_prompt.strip():
            raise ConfigError("ContextBuilder: system_prompt must not be empty")
        if self.max_tokens < 1:
            raise ConfigError("ContextBuilder: max_tokens must be > 0")

    async def build(self, state: RunState, tools: Sequence[ToolSpec]) -> ModelRequest:
        sections = [self.system_prompt]
        for source in self.sources:
            text = await source.render(state)
            if text:
                sections.append(text)
        return ModelRequest(
            system="\n\n".join(sections),
            messages=tuple(state.messages),
            tools=tuple(tools),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
