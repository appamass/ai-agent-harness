"""Providers: protocol, capabilities, and router. Adapters for Anthropic and OpenAI-compatible
APIs are optional modules here (`providers/anthropic.py`, `providers/openai.py`).
Other providers belong in your project."""

from .base import CallableProvider, Capabilities, ModelFn, ModelProvider, StreamDelta, without_foreign_thinking
from .router import ModelRouter, RetryPolicy

__all__ = [
    "CallableProvider",
    "Capabilities",
    "ModelFn",
    "ModelProvider",
    "ModelRouter",
    "RetryPolicy",
    "StreamDelta",
    "without_foreign_thinking",
]
