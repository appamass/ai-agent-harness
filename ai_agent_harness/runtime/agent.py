"""Agent: a validated composition of building blocks. No custom logic, no separate codebase."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..context import ContextBuilder
from ..core.errors import ConfigError
from ..core.events import EventBus
from ..core.hooks import Hook, HookPipeline
from ..persistence import CheckpointStore
from ..policy import Autonomy, Limits, PermissionPolicy
from ..providers.base import CallableProvider, ModelFn, ModelProvider
from ..providers.router import ModelRouter
from ..tools.base import BaseTool
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

    @classmethod
    def create(
        cls,
        provider: ModelRouter | ModelProvider | ModelFn,
        system_prompt: ContextBuilder | str = "You are a helpful assistant.",
        *,
        name: str = "agent",
        tools: ToolRegistry | Iterable[BaseTool[Any]] = (),
        autonomy: Autonomy = Autonomy.READ_ONLY,
        allowed_tools: Iterable[str] | None = None,
        hooks: HookPipeline | Sequence[Hook] = (),
        events: EventBus | None = None,
        checkpoints: CheckpointStore | None = None,
        limits: Limits | None = None,
        max_result_chars: int = 20_000,
    ) -> Agent:
        """Shorthand constructor that normalizes a provider, prompt string, tool list, and hook list."""
        if isinstance(provider, ModelRouter):
            router = provider
        elif isinstance(provider, ModelProvider):
            router = ModelRouter(provider)
        elif callable(provider):
            router = ModelRouter(CallableProvider(provider))
        else:
            raise ConfigError("Agent.create: provider must be a ModelRouter, ModelProvider, or async callable")
        context = system_prompt if isinstance(system_prompt, ContextBuilder) else ContextBuilder(system_prompt)
        registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        pipeline = hooks if isinstance(hooks, HookPipeline) else HookPipeline(hooks)
        return cls(
            name=name,
            router=router,
            context=context,
            tools=registry,
            autonomy=autonomy,
            allowed_tools=frozenset(allowed_tools) if allowed_tools is not None else None,
            hooks=pipeline,
            events=events if events is not None else EventBus(),
            checkpoints=checkpoints,
            limits=limits if limits is not None else Limits(),
            max_result_chars=max_result_chars,
        )
