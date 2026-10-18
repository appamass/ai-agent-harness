"""Event bus: a single source for tracing, cost tracking, UI progress, and audit logs.

The core only emits events. Listeners (logging, OpenTelemetry, WebSockets) are
registered from outside. A failing listener never aborts a run, but is always
logged — errors are never swallowed silently.

Listeners run sequentially and block the run while executing. Slow work
(network calls, database writes) belongs in a background queue, not directly
inside the listener.

**Events contain metadata only** (type, IDs, status, tool name, counters, retryability),
never content: no user prompts, tool arguments, approval previews, hook reasons,
model responses, or provider error messages (which often echo the input).
This allows events to flow safely into logs, tracing, and monitoring.
Content lives exclusively in `RunState` (and therefore in the — ideally encrypted —
checkpoint). In addition, a validator restricts event data to scalar types and short strings.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

log = logging.getLogger("ai_agent_harness.events")


class EventType(StrEnum):
    RUN_STARTED = "run_started"
    RUN_RESUMED = "run_resumed"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    PROVIDER_RETRY = "provider_retry"
    PROVIDER_FAILOVER = "provider_failover"
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    TOOL_BLOCKED = "tool_blocked"
    APPROVAL_REQUESTED = "approval_requested"
    INPUT_REQUESTED = "input_requested"
    CHECKPOINT_SAVED = "checkpoint_saved"
    RUN_FINISHED = "run_finished"


class Event(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: EventType
    run_id: str
    tenant_id: str
    step: int
    data: dict[str, Any] = Field(default_factory=dict)
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("data")
    @classmethod
    def _metadata_only(cls, data: dict[str, Any]) -> dict[str, Any]:
        return {key: _scalar(value, depth=0) for key, value in data.items()}


MAX_EVENT_TEXT = 200


def _scalar(value: Any, depth: int) -> Any:
    """Allow only scalar values (and one level of counter dicts); truncate long strings."""
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value if len(value) <= MAX_EVENT_TEXT else value[:MAX_EVENT_TEXT] + "…"
    if isinstance(value, dict):
        if depth > 0:
            raise ValueError("Event data may be nested at most one level deep")
        nested = cast("dict[object, object]", value)
        return {str(k): _scalar(v, depth=1) for k, v in nested.items()}
    raise ValueError(f"Event data may only contain scalar values, got {type(value).__name__}")


Listener = Callable[[Event], Awaitable[None]]
Emit = Callable[[EventType, dict[str, Any]], Awaitable[None]]
"""Lean emit callback passed to components outside the runner."""


class EventBus:
    def __init__(self) -> None:
        self._listeners: list[Listener] = []

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    async def emit(self, event: Event) -> None:
        for listener in tuple(self._listeners):
            try:
                await listener(event)
            except Exception:
                log.exception("Event listener %r failed on %s", listener, event.type)
