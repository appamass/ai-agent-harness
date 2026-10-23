"""Policy: autonomy levels, tool allowlist, and limits — the hard boundaries of the system.

Autonomy × Risk:

|                     | READ  | WRITE    | IRREVERSIBLE |
|---------------------|-------|----------|--------------|
| read_only           | allow | deny     | deny         |
| suggest             | allow | approval | deny         |
| act_with_approval   | allow | approval | approval     |
| autonomous          | allow | allow    | approval     |

Irreversible actions never run without a human: denied below `act_with_approval`, approval above.
Tools with `deny` are never even offered to the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from enum import StrEnum

from .core.errors import ConfigError
from .core.state import RunState
from .core.types import Risk, is_positive_number


class Autonomy(StrEnum):
    READ_ONLY = "read_only"
    SUGGEST = "suggest"
    ACT_WITH_APPROVAL = "act_with_approval"
    AUTONOMOUS = "autonomous"


class Verdict(StrEnum):
    ALLOW = "allow"
    NEEDS_APPROVAL = "needs_approval"
    DENY = "deny"


_A, _N, _D = Verdict.ALLOW, Verdict.NEEDS_APPROVAL, Verdict.DENY
_MATRIX: dict[Autonomy, dict[Risk, Verdict]] = {
    Autonomy.READ_ONLY: {Risk.READ: _A, Risk.WRITE: _D, Risk.IRREVERSIBLE: _D},
    Autonomy.SUGGEST: {Risk.READ: _A, Risk.WRITE: _N, Risk.IRREVERSIBLE: _D},
    Autonomy.ACT_WITH_APPROVAL: {Risk.READ: _A, Risk.WRITE: _N, Risk.IRREVERSIBLE: _N},
    Autonomy.AUTONOMOUS: {Risk.READ: _A, Risk.WRITE: _A, Risk.IRREVERSIBLE: _N},
}
if {(a, r) for a in _MATRIX for r in _MATRIX[a]} != {(a, r) for a in Autonomy for r in Risk}:
    raise RuntimeError("Policy matrix is incomplete")  # safeguard when adding new enum values


@dataclass(frozen=True)
class PermissionPolicy:
    autonomy: Autonomy = Autonomy.READ_ONLY
    allowed_tools: frozenset[str] | None = None
    """None = all registered tools; otherwise only these (allowlist)."""

    def check(self, tool_name: str, risk: Risk) -> Verdict:
        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            return Verdict.DENY
        return _MATRIX[self.autonomy][risk]


_OPTIONAL_FIELDS = frozenset({"max_context_tokens"})  # fields of `Limits` that may be `None` (= no limit)


@dataclass(frozen=True)
class Limits:
    """Apply to the entire run across all resumes.

    Resuming a run does not reset them. To continue after `limit_reached`,
    the caller must explicitly increase the agent's limits.
    """

    max_steps: int = 30
    max_total_tokens: int = 500_000
    max_seconds: float = 600.0
    max_consecutive_retries: int = 3
    max_context_chars: int = 400_000
    """Upper bound on the estimated size of each model request (`ModelRequest.estimated_chars`).
    If exceeded, the run ends with `limit_reached` instead of failing at the provider's context
    window. Without a tokenizer, `CHARS_PER_TOKEN` (3) characters count as one token; tune to fit the
    smallest model in the router."""
    max_input_chars: int = 50_000
    """Upper bound on each human input (`user_input`, `answer`). Longer inputs raise `ValueError`."""
    max_context_tokens: int | None = None
    """Optional upper bound on the estimated tokens of each model request. The estimate starts from the prompt
    tokens the provider reported for the last call and adds the new characters (ADR 0026). `None` = no limit.
    `max_context_chars` still applies."""

    def __post_init__(self) -> None:
        for item in fields(self):
            value: object = getattr(self, item.name)
            if value is None and item.name in _OPTIONAL_FIELDS:
                continue
            # NaN compares false with everything: `max_seconds=nan` would silently disable the time limit
            invalid_type = isinstance(value, bool) or not isinstance(value, int | float)
            if invalid_type or (isinstance(value, float) and not math.isfinite(value)):
                raise ConfigError(f"Limits.{item.name} must be a finite number, got {value!r}")
        positive = (
            self.max_steps,
            self.max_total_tokens,
            self.max_seconds,
            self.max_context_chars,
            self.max_input_chars,
        )
        if any(value <= 0 for value in positive):
            raise ConfigError("Limits must be positive")
        if self.max_consecutive_retries < 0:
            raise ConfigError("max_consecutive_retries must not be negative")
        if self.max_context_tokens is not None and not is_positive_number(self.max_context_tokens, integer=True):
            raise ConfigError(f"Limits.max_context_tokens must be a positive integer, got {self.max_context_tokens!r}")

    def exceeded(self, state: RunState, elapsed_s: float) -> str | None:
        if state.step >= self.max_steps:
            return f"Step limit reached ({self.max_steps})"
        if state.usage.total >= self.max_total_tokens:
            return f"Token budget reached ({self.max_total_tokens})"
        if elapsed_s >= self.max_seconds:
            return f"Time limit reached ({self.max_seconds:g} s)"
        return None
