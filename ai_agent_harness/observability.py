"""Observability: listeners for the event bus.

`log_listener` writes every event as structured log output. For OpenTelemetry,
cost tracking, or a UI, register custom listeners following the same pattern:
`agent.events.subscribe(my_listener)`.
"""

from __future__ import annotations

import logging

from .core.events import Event

log = logging.getLogger("ai_agent_harness.run")


async def log_listener(event: Event) -> None:
    log.info(
        "%s run=%s tenant=%s step=%d %s",
        event.type,
        event.run_id,
        event.tenant_id,
        event.step,
        event.data,
        extra={"harness_event": event.model_dump(mode="json")},
    )
