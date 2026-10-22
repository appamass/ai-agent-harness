"""RunState: the complete, serializable state of a run.

The state guards the tool-call protocol required by all model providers:

1. Messages strictly alternate between `user` and `assistant`, starting with `user`.
2. Every tool call in an assistant message receives exactly one result —
   in the immediately following user message.
3. Tool results only exist for tool calls that actually occurred.

All mutations to `messages` go through the methods of this class, which verify the
protocol and raise `ProtocolError` on violations. No execution path in the runner
can produce a message history that a real provider API would reject.

**Schema versioning:** Saved states carry `schema_version`. They are loaded
exclusively via `restore_state`, which upgrades older versions through `MIGRATIONS`
and rejects newer ones. Rule for schema changes (ADR 0010): increment `SCHEMA_VERSION`,
add a migration from the previous version, and keep a fixture of the old format in
`tests/fixtures/`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, Self, TypeVar, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .errors import InvalidStateError, ProtocolError, SchemaVersionError
from .types import ID_PATTERN, Block, Message, TextBlock, ToolCall, ToolResult, Usage

CONTINUE_PROMPT = (
    "Your response was truncated due to the length limit. Continue from the exact point where you stopped."
)
"""User prompt sent by the runner to continue a truncated response (`StopReason.MAX_TOKENS`)."""

SCHEMA_VERSION = 3
"""Current version of the serialized state. Increment on every schema change (ADR 0010).

History: 1 — initial version. 2 — `batch_done`, cache/reasoning tokens in `usage`,
`ThinkingBlock.provider_id`, `ToolCall.arguments_error` (ADR 0012). 3 — `output_schema`,
`output_model_name`, `output` for structured final output (ADR 0023).
"""

Migration = Callable[[dict[str, Any]], dict[str, Any]]


def _v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """v2 only adds fields with default values — a v1 state is valid without structural changes.

    The migration is still explicit: it documents the step and keeps the migration chain unbroken.
    """
    return data


def _v2_to_v3(data: dict[str, Any]) -> dict[str, Any]:
    """v3 adds `output_schema`, `output_model_name`, and `output` with None defaults (ADR 0023)."""
    return data


MIGRATIONS: dict[int, Migration] = {1: _v1_to_v2, 2: _v2_to_v3}
"""Migrations from version n to n+1, keyed by n."""


class RunStatus(StrEnum):
    RUNNING = "running"
    WAITING_FOR_INPUT = "waiting_for_input"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    DONE = "done"
    REFUSED = "refused"
    BLOCKED = "blocked"
    LIMIT_REACHED = "limit_reached"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_waiting(self) -> bool:
        return self in (RunStatus.WAITING_FOR_INPUT, RunStatus.WAITING_FOR_APPROVAL)


class PendingCall(BaseModel):
    """A tool call waiting for a human (input answer or approval)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["input", "approval"]
    call: ToolCall
    prompt: str = Field(description="Question for the user or preview of the action awaiting approval")


class PendingBatch(BaseModel):
    """A paused tool batch: already completed results, the waiting call, and the remaining calls."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    done: tuple[ToolResult, ...] = ()
    waiting: PendingCall
    remaining: tuple[ToolCall, ...] = ()


DepsT = TypeVar("DepsT")
OutputT = TypeVar("OutputT", bound=BaseModel)


def _now() -> datetime:
    return datetime.now(UTC)


class RunState(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    run_id: str = Field(default_factory=lambda: uuid4().hex, pattern=ID_PATTERN)
    tenant_id: str = Field(default="default", pattern=ID_PATTERN)
    agent_name: str = Field(min_length=1)
    revision: int = Field(default=0, ge=0, description="Optimistic concurrency lock for checkpoints")
    status: RunStatus = RunStatus.RUNNING
    status_reason: str | None = None
    messages: list[Message] = Field(default_factory=list[Message])
    pending: PendingBatch | None = None
    batch_done: tuple[ToolResult, ...] = Field(
        default=(),
        description="Completed results of the current tool batch — survive a crash (write-ahead)",
    )
    output_schema: dict[str, Any] | None = Field(
        default=None, description="JSON schema for structured final output (ADR 0023)"
    )
    output_model_name: str | None = Field(default=None, description="Class name of required output_model (ADR 0023)")
    output: dict[str, Any] | None = Field(
        default=None, description="Validated structured final output as JSON-safe dictionary (ADR 0023)"
    )
    step: int = Field(default=0, ge=0, description="Number of model calls across the entire run")
    consecutive_retries: int = Field(default=0, ge=0)
    usage: Usage = Field(default_factory=Usage)
    elapsed_s: float = Field(default=0.0, ge=0.0, description="Active execution time across all invocations")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    _deps: object = PrivateAttr(default=None)  # never serialized: neither in checkpoints nor in events
    _output_object: BaseModel | None = PrivateAttr(default=None)  # in-memory cached model instance

    # ------------------------------------------------------- Runtime dependencies

    @property
    def deps(self) -> object:
        """Runtime dependencies passed to `start`/`resume` (DB session, HTTP clients, user token).

        They are **never saved** and must be passed again on every `start` and `resume`.
        Read with type checking inside tools and hooks via `state.deps_as(MyDeps)`.
        """
        return self._deps

    def deps_as(self, kind: type[DepsT]) -> DepsT:
        """Return `deps` typed as `kind`, or raise `InvalidStateError` if missing or of the wrong type."""
        deps = self._deps
        if deps is None:
            raise InvalidStateError("Run has no deps: pass deps=... to start()/resume()")
        if not isinstance(deps, kind):
            raise InvalidStateError(f"deps is {type(deps).__name__}, expected {kind.__name__}")
        return deps

    def bind_deps(self, deps: object) -> None:
        """Bind `deps` for the current execution. Called by `start`/`resume`, reset to `None` afterwards."""
        self._deps = deps

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        """A deep copy never copies `deps` (avoids duplicating DB sessions or failing on locks)."""
        deps, self._deps = self._deps, None
        try:
            return super().__deepcopy__(memo)
        finally:
            self._deps = deps

    def __getstate__(self) -> dict[Any, Any]:
        """Pickle (e.g. task queues) never includes `deps` — they may hold tokens and live sessions."""
        data = super().__getstate__()
        private = dict(data.get("__pydantic_private__") or {})
        private["_deps"] = None
        return {**data, "__pydantic_private__": private}

    # ------------------------------------------------------------------ Reading

    @property
    def last_message(self) -> Message | None:
        return self.messages[-1] if self.messages else None

    @property
    def last_text(self) -> str:
        """Latest model response — seamlessly reassembled across length continuations."""
        msgs = self.messages
        index = next((i for i in range(len(msgs) - 1, -1, -1) if msgs[i].role == "assistant" and msgs[i].text), None)
        if index is None:
            return ""
        parts = [msgs[index].text]
        continuation = (TextBlock(text=CONTINUE_PROMPT),)
        while index >= 2 and msgs[index - 1].content == continuation and msgs[index - 2].text:
            index -= 2
            parts.append(msgs[index].text)
        return "".join(reversed(parts))  # continuation resumes at the exact cutoff point

    def output_as(self, model: type[OutputT]) -> OutputT:
        """Returns the structured final output validated as `model`.

        Raises `InvalidStateError` if the run produced no structured output.
        """
        if self.output is None:
            raise InvalidStateError(f"Run has no structured output (status: {self.status})")
        return model.model_validate(self.output)

    def clear_output(self) -> None:
        """Drops the structured output of an earlier turn before the run continues."""
        self.output = None
        self._output_object = None

    @property
    def output_object(self) -> BaseModel | None:
        """The in-memory validated Pydantic object from this execution, if available."""
        return self._output_object

    def set_output(self, output_object: BaseModel) -> None:
        """Stores a validated structured output: as a JSON-safe dict (for checkpoints) and as the instance."""
        self.output = output_object.model_dump(mode="json")
        self._output_object = output_object
        self._touch()

    @property
    def pending_id(self) -> str | None:
        """Identifies the current pause; `None` when the run is not waiting (ADR 0014).

        Show it to the human together with `status_reason` and pass it back to `resume(..., pending_id=...)`.
        `resume` rejects any other value, so a stale dialog can never approve or answer a later call.
        Never read it from a freshly loaded state to handle a click: it must come from what the human saw.
        Unique per pause: tenant, run ID, and creation time separate runs (also a reused `run_id`); messages
        only grow, the waiting call is in the last assistant message, call IDs are unique within a message,
        and the kind separates an approval from a later question of the same call.
        Derived from the state, never stored.
        """
        if self.pending is None:
            return None
        waiting = self.pending.waiting
        created = self.created_at.isoformat()
        key = f"{self.tenant_id}|{self.run_id}|{created}|{len(self.messages) - 1}|{waiting.kind}|{waiting.call.id}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def open_tool_calls(self) -> tuple[ToolCall, ...]:
        """Tool calls from the last assistant message that do not yet have a result."""
        last = self.last_message
        if last is None or last.role != "assistant":
            return ()
        return last.tool_calls

    # ------------------------------------------------------------------ Writing

    def add_user_text(self, text: str) -> None:
        """Append user text. Merge into the last message if it is already a user message."""
        if self.open_tool_calls():
            raise ProtocolError("Open tool calls must be answered before new user text is added")
        block = TextBlock(text=text)
        last = self.last_message
        if last is not None and last.role == "user":
            self.messages[-1] = Message(role="user", content=(*last.content, block))
        else:
            self.messages.append(Message(role="user", content=(block,)))
        self._touch()

    def add_assistant(self, message: Message) -> None:
        if message.role != "assistant":
            raise ProtocolError("add_assistant expects an assistant message")
        last = self.last_message
        if last is None or last.role != "user":
            raise ProtocolError("An assistant message must follow a user message")
        self.messages.append(message)
        self._touch()

    def add_tool_results(self, results: Sequence[ToolResult], text: str | None = None) -> None:
        """Answer ALL open tool calls in a single user message (preserving the order of the calls)."""
        calls = self.open_tool_calls()
        if not calls:
            raise ProtocolError("There are no open tool calls")
        by_id = {r.call_id: r for r in results}
        if set(by_id) != {c.id for c in calls} or len(by_id) != len(results):
            raise ProtocolError("Every open tool call requires exactly one result, and no others")
        blocks: list[Block] = [by_id[c.id] for c in calls]
        if text:
            blocks.append(TextBlock(text=text))
        self.messages.append(Message(role="user", content=tuple(blocks)))
        self._touch()

    def close_open_tool_calls(self, reason: str, known: Sequence[ToolResult] = (), text: str | None = None) -> None:
        """Answer open calls: keep known results and fill the rest as errors with `reason`. `text` rides along."""
        calls = self.open_tool_calls()
        if not calls:
            return
        by_id = {r.call_id: r for r in known}
        results = [by_id.get(c.id) or ToolResult(call_id=c.id, content=reason, is_error=True) for c in calls]
        self.add_tool_results(results, text=text)

    def set_status(self, status: RunStatus, reason: str | None = None) -> None:
        if status is not RunStatus.DONE:
            self.clear_output()  # a structured output belongs to a run that ended as done, never to anything else
        self.status = status
        self.status_reason = reason
        self._touch()

    # ------------------------------------------------------------------ Checking

    def check_protocol(self) -> None:
        """Verify the entire message history and state consistency. Used for loaded checkpoints and tests."""
        for i, msg in enumerate(self.messages):
            expected = "user" if i % 2 == 0 else "assistant"
            if msg.role != expected:
                raise ProtocolError(f"Message {i}: role '{msg.role}', expected '{expected}'")
            if msg.role == "user":
                prev_calls = {c.id for c in self.messages[i - 1].tool_calls} if i > 0 else set[str]()
                got = [r.call_id for r in msg.tool_results]
                if set(got) != prev_calls:
                    raise ProtocolError(f"Message {i}: tool results do not match the preceding tool calls")
        open_calls = self.open_tool_calls()
        if open_calls and not (self.status == RunStatus.RUNNING or self.status.is_waiting):
            raise ProtocolError(f"Finished run ({self.status}) has unanswered tool calls")
        if self.status.is_waiting != (self.pending is not None):
            raise ProtocolError("Waiting status and pending batch do not match")
        if self.batch_done:
            done_ids = [r.call_id for r in self.batch_done]
            if self.pending is not None or len(set(done_ids)) != len(done_ids):
                raise ProtocolError("Completed results of the tool batch are inconsistent")
            if not set(done_ids) <= {c.id for c in open_calls}:
                raise ProtocolError("Completed results do not belong to open tool calls")
        if self.pending is not None:
            batch = self.pending
            batch_ids = [r.call_id for r in batch.done] + [batch.waiting.call.id] + [c.id for c in batch.remaining]
            if sorted(batch_ids) != sorted(c.id for c in open_calls):
                raise ProtocolError("Paused tool batch does not match open tool calls")
            expected_kind = "input" if self.status == RunStatus.WAITING_FOR_INPUT else "approval"
            if batch.waiting.kind != expected_kind:
                raise ProtocolError("Pending call kind does not match waiting status")

    def _touch(self) -> None:
        self.updated_at = _now()


def restore_state(raw: str | bytes, *, tenant_id: str | None = None, run_id: str | None = None) -> RunState:
    """Only way to load saved states: check version, migrate, validate, verify ownership and protocol.

    Stores pass the requested `tenant_id` and `run_id`: a checkpoint stored under a foreign key
    (copied file, swapped row) is rejected.
    """
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise SchemaVersionError("Saved state is not a JSON object")
    fields = cast("dict[str, Any]", data)
    version = fields.get("schema_version", 1)  # states before schema versioning = version 1
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise SchemaVersionError(f"Invalid schema version: {version!r}")
    if version > SCHEMA_VERSION:
        raise SchemaVersionError(
            f"State has schema version {version}, but this code only supports up to {SCHEMA_VERSION}. "
            "Rolling back to an older template version cannot read newer states."
        )
    while version < SCHEMA_VERSION:
        migrate = MIGRATIONS.get(version)
        if migrate is None:
            raise SchemaVersionError(f"No migration from schema version {version} to {version + 1}")
        migrated: object = migrate(dict(fields))
        if not isinstance(migrated, dict):
            raise SchemaVersionError(f"Migration from schema version {version} did not return a JSON object")
        fields = cast("dict[str, Any]", migrated)
        version += 1
        fields["schema_version"] = version
    state = RunState.model_validate(fields)
    for label, expected, actual in (("tenant", tenant_id, state.tenant_id), ("run", run_id, state.run_id)):
        if expected is not None and expected != actual:
            raise InvalidStateError(f"Saved state belongs to {label} '{actual}', requested '{expected}'")
    state.check_protocol()
    return state
