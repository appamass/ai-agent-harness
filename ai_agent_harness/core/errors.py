"""Error types.

Principle: Errors that the model can fix on its own (invalid arguments,
unknown tool) are not raised as exceptions, but returned as
`ToolResult(is_error=True)`. Exceptions are reserved for system failures
and programming errors.
"""

from __future__ import annotations


class HarnessError(Exception):
    """Base class for all harness errors."""


class ConfigError(HarnessError):
    """Invalid configuration (agent, tool, limits). Fails at startup, not mid-run."""


class ProtocolError(HarnessError):
    """The message history would violate the tool-call protocol."""


class InvalidStateError(HarnessError):
    """Action does not match the run status (e.g. resuming without answering a pending question)."""


class SchemaVersionError(HarnessError):
    """A saved checkpoint has a schema version that this code cannot process."""


class ConcurrencyError(HarnessError):
    """The checkpoint was modified by another process in the meantime."""


class ProviderError(HarnessError):
    """Error from a model provider. `retryable` determines whether the call is retried."""

    def __init__(self, message: str, *, retryable: bool, provider_id: str | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.provider_id = provider_id


class ToolError(HarnessError):
    """Raised by tools when the message is intended for the model and safe to expose.

    All other exceptions from tools are logged with full tracebacks, but reported
    to the model by their type name only — sensitive details (paths, secrets) stay in the log.
    """
