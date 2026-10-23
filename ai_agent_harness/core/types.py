"""Canonical message format.

Every provider adapter translates only into and out of this format. The agent
loop, guardrails, tools, and tests work exclusively with these types.

All types are immutable (`frozen`) and reject unknown fields (`extra="forbid"`)
so typos in adapters fail immediately.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
"""Allowed format for tool names, run IDs, and tenant IDs (safe for filenames and APIs)."""
ID_RE = re.compile(ID_PATTERN)
_NOT_ID_CHARS = re.compile(r"[^A-Za-z0-9_-]")


def canonical_digest(payload: dict[str, Any]) -> str:
    """SHA-256 (64 hex chars) over the canonical JSON form of `payload`: stable across key order and platforms."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sanitize_id(value: str) -> str:
    """Replaces every character an ID may not contain with `_` (the length is not changed)."""
    return _NOT_ID_CHARS.sub("_", value)


def is_positive_number(value: object, *, integer: bool = False) -> bool:
    """True for a finite number > 0 (an `int` when `integer`). Bools are ints in Python and are rejected."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    if isinstance(value, float):  # ints are always finite (and `math.isfinite` would overflow on huge ones)
        return not integer and math.isfinite(value) and value > 0
    return value > 0


TruncationMode = Literal["head_and_tail", "head"]
"""How an oversized tool result is cut: keep head and tail, or only the head."""

DEFAULT_MAX_RESULT_CHARS = 20_000
DEFAULT_TRUNCATION_MODE: TruncationMode = "head_and_tail"

CHARS_PER_TOKEN = 3
"""Characters per token when a size in characters becomes a token estimate (ADR 0026).
Deliberately low (real text is about 3.5 to 4 characters per token), so the estimate errs on the side of caution."""

FINAL_OUTPUT_TOOL_NAME = "final_output"
"""Reserved tool name the runner offers for structured final output (ADR 0023). Agents cannot register it."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Risk(StrEnum):
    """Risk level of a tool action — the basis for policy and human approvals."""

    READ = "read"  # read-only, no side effects
    WRITE = "write"  # modifies state, reversible
    IRREVERSIBLE = "irreversible"  # send, delete, pay, publish, merge


class StopReason(StrEnum):
    """Why the model stopped generating. Adapters map provider values onto this enum."""

    END = "end"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"


class TextBlock(_Frozen):
    type: Literal["text"] = "text"
    text: str


class ImageBlock(_Frozen):
    type: Literal["image"] = "image"
    media_type: Literal["image/png", "image/jpeg", "image/gif", "image/webp"]
    data: str = Field(min_length=1, description="Base64-encoded image data")


class ThinkingBlock(_Frozen):
    """Reasoning output from a model.

    Some providers require these blocks to be echoed back unchanged on subsequent
    turns. `provider_data` carries provider-specific fields (such as cryptographic
    signatures) opaquely through the harness.
    """

    type: Literal["thinking"] = "thinking"
    text: str = ""
    provider_data: dict[str, Any] = Field(default_factory=dict)
    provider_id: str | None = Field(
        default=None,
        description="Provider that produced this block — stamped by the runner. Other providers never receive it.",
    )


MAX_ARGUMENTS_ERROR = 500


class ToolCall(_Frozen):
    type: Literal["tool_call"] = "tool_call"
    id: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=64)
    arguments: dict[str, Any] = Field(default_factory=dict)
    arguments_error: str | None = Field(
        default=None,
        description="Adapter: arguments were not valid JSON (parser error message, without raw input text)",
    )

    @field_validator("arguments_error", mode="before")
    @classmethod
    def _short_error(cls, value: object) -> object:
        # A verbose parser error must never cause the response itself to fail validation: truncate instead of rejecting
        if isinstance(value, str) and len(value) > MAX_ARGUMENTS_ERROR:
            return value[: MAX_ARGUMENTS_ERROR - 1] + "…"
        return value

    @model_validator(mode="after")
    def _error_excludes_arguments(self) -> Self:
        if self.arguments_error is not None and self.arguments:
            raise ValueError("ToolCall with arguments_error must not have arguments")
        return self


class ToolResult(_Frozen):
    type: Literal["tool_result"] = "tool_result"
    call_id: str = Field(min_length=1, max_length=256)
    content: str
    is_error: bool = False


Block = Annotated[
    TextBlock | ImageBlock | ThinkingBlock | ToolCall | ToolResult,
    Field(discriminator="type"),
]

_ASSISTANT_ONLY = (ThinkingBlock, ToolCall)
_USER_ONLY = (ImageBlock, ToolResult)


class Message(_Frozen):
    role: Literal["user", "assistant"]
    content: tuple[Block, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_blocks(self) -> Self:
        forbidden = _USER_ONLY if self.role == "assistant" else _ASSISTANT_ONLY
        for block in self.content:
            if isinstance(block, forbidden):
                raise ValueError(  # noqa: TRY004 – Pydantic validators report errors via ValueError
                    f"Block '{block.type}' is not allowed in a {self.role} message"
                )
        call_ids = [b.id for b in self.content if isinstance(b, ToolCall)]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("Tool call IDs within a message must be unique")
        result_ids = [b.call_id for b in self.content if isinstance(b, ToolResult)]
        if len(result_ids) != len(set(result_ids)):
            raise ValueError("Only one tool result per tool call is allowed")
        return self

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role="user", content=(TextBlock(text=text),))

    @property
    def text(self) -> str:
        """All text blocks joined by newlines (never concatenated without separator)."""
        return "\n".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(b for b in self.content if isinstance(b, ToolCall))

    @property
    def tool_results(self) -> tuple[ToolResult, ...]:
        return tuple(b for b in self.content if isinstance(b, ToolResult))

    @property
    def has_images(self) -> bool:
        return any(isinstance(b, ImageBlock) for b in self.content)


class Usage(_Frozen):
    """Token usage. Fixed semantics across all adapters (cost is computed by a listener with a price table):

    - `input_tokens`: input tokens **excluding** cache hits and cache writes
    - `cache_read_tokens` / `cache_write_tokens`: tokens read from / written to the prompt cache
    - `output_tokens`: total output tokens **including** reasoning tokens
    - `reasoning_tokens`: reasoning subset of `output_tokens`
    """

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    cache_write_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _reasoning_within_output(self) -> Self:
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError("reasoning_tokens is a subset of output_tokens and cannot exceed it")
        return self

    @property
    def total(self) -> int:
        """All processed tokens — basis for `Limits.max_total_tokens`."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens + self.output_tokens

    @property
    def prompt_tokens(self) -> int:
        """All input tokens of one call: fresh input plus cache reads and cache writes (ADR 0026)."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


class ToolSpec(_Frozen):
    """What the model sees of a tool."""

    name: str = Field(pattern=ID_PATTERN)
    description: str = Field(min_length=1)
    input_schema: dict[str, Any]


class ModelRequest(_Frozen):
    """Canonical request to a model."""

    system: str
    messages: tuple[Message, ...] = Field(min_length=1)
    tools: tuple[ToolSpec, ...] = ()
    output_schema: dict[str, Any] | None = None
    max_tokens: int = Field(default=4096, gt=0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)

    @property
    def has_images(self) -> bool:
        return any(m.has_images for m in self.messages)

    @property
    def estimated_chars(self) -> int:
        """Rough context size in characters: system prompt, text, reasoning, tool calls, tool results, and specs.

        Provider data on thinking blocks (e.g. encrypted reasoning) is included. Image data is not
        counted (providers bill images by pixel dimensions, not base64 string length).
        Basis for `Limits.max_context_chars` — an estimate, not a tokenizer. `CHARS_PER_TOKEN` turns it into tokens.
        """
        size = len(self.system) + sum(tool_spec_chars(spec) for spec in self.tools)
        if self.output_schema is not None:
            size += json_chars(self.output_schema)
        return size + sum(message_chars(message) for message in self.messages)


def json_chars(data: dict[str, Any]) -> int:
    """Length of opaque provider data as JSON — with a fallback so the estimate never fails."""
    try:
        return len(json.dumps(data, default=repr, ensure_ascii=False))
    except (TypeError, ValueError):  # e.g. non-string dictionary keys
        return len(repr(data))


def tool_spec_chars(spec: ToolSpec) -> int:
    """Size of one tool spec in characters, as counted by `ModelRequest.estimated_chars`."""
    return len(spec.model_dump_json())


def message_chars(message: Message) -> int:
    """Size of one message in characters, as counted by `ModelRequest.estimated_chars`."""
    size = 0
    for block in message.content:
        if isinstance(block, TextBlock):
            size += len(block.text)
        elif isinstance(block, ThinkingBlock):
            size += len(block.text) + (json_chars(block.provider_data) if block.provider_data else 0)
        elif isinstance(block, ToolResult):
            size += len(block.content)
        elif isinstance(block, ToolCall):
            size += len(block.model_dump_json())
    return size


class ModelResponse(_Frozen):
    """Canonical response from a model."""

    message: Message
    stop_reason: StopReason
    usage: Usage = Field(default_factory=Usage)
    model: str | None = Field(default=None, description="Actual model served (for logs and evals)")

    @model_validator(mode="after")
    def _assistant_only(self) -> Self:
        if self.message.role != "assistant":
            raise ValueError("ModelResponse.message must have role 'assistant'")
        return self
