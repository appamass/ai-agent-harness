"""Core: types, state, events, hooks, and errors.

Dependencies point strictly inward: `core` imports nothing from other
packages of the harness and no third-party library except Pydantic.
"""

from .errors import (
    ConcurrencyError,
    ConfigError,
    HarnessError,
    InvalidStateError,
    ProtocolError,
    ProviderError,
    SchemaVersionError,
    ToolError,
)
from .events import Emit, Event, EventBus, EventType, Listener
from .hooks import CONTINUE, Action, Decision, Hook, HookPipeline, ToolInvocation
from .state import MIGRATIONS, SCHEMA_VERSION, PendingBatch, PendingCall, RunState, RunStatus, restore_state
from .types import (
    ID_PATTERN,
    ImageBlock,
    Message,
    ModelRequest,
    ModelResponse,
    Risk,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)

__all__ = [
    "CONTINUE",
    "ID_PATTERN",
    "MIGRATIONS",
    "SCHEMA_VERSION",
    "Action",
    "ConcurrencyError",
    "ConfigError",
    "Decision",
    "Emit",
    "Event",
    "EventBus",
    "EventType",
    "HarnessError",
    "Hook",
    "HookPipeline",
    "ImageBlock",
    "InvalidStateError",
    "Listener",
    "Message",
    "ModelRequest",
    "ModelResponse",
    "PendingBatch",
    "PendingCall",
    "ProtocolError",
    "ProviderError",
    "Risk",
    "RunState",
    "RunStatus",
    "SchemaVersionError",
    "StopReason",
    "TextBlock",
    "ThinkingBlock",
    "ToolCall",
    "ToolError",
    "ToolInvocation",
    "ToolResult",
    "ToolSpec",
    "Usage",
    "restore_state",
]
