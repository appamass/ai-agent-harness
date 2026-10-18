"""Agent: a validated composition of building blocks. No custom logic, no separate codebase."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..context import ContextBuilder
from ..core.errors import ConfigError
from ..core.events import EventBus
from ..core.hooks import HookPipeline
from ..persistence import CheckpointStore
from ..policy import Autonomy, Limits, PermissionPolicy
from ..providers.router import ModelRouter
from ..tools.registry import ToolRegistry


@dataclass(frozen=True)
class Agent:
    name: str
    router: ModelRouter
    context: ContextBuilder
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    autonomy: Autonomy = Autonomy.READ_ONLY  # safe default
    allowed_tools: frozenset[str] | None = None  # None = all registered tools
    hooks: HookPipeline = field(default_factory=HookPipeline)
    events: EventBus = field(default_factory=EventBus)
    checkpoints: CheckpointStore | None = None  # None = no persistence
    limits: Limits = field(default_factory=Limits)
    max_result_chars: int = 20_000

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ConfigError("Agent.name must not be empty")
        if self.allowed_tools is not None:
            unknown = self.allowed_tools - self.tools.names()
            if unknown:
                raise ConfigError(f"allowed_tools contains unregistered tools: {sorted(unknown)}")
        if self.max_result_chars < 100:
            raise ConfigError("max_result_chars must be >= 100")

    @property
    def policy(self) -> PermissionPolicy:
        return PermissionPolicy(autonomy=self.autonomy, allowed_tools=self.allowed_tools)
