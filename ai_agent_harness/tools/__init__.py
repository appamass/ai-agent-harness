"""Tools: base classes, registry, executor (the single execution gate), and built-in tools."""

from .base import BaseTool, InputRequest, SyncTool, ToolOutput, tool
from .builtin import DEFAULT_DENIED_READS, AskUser, ReadFile
from .executor import ToolExecutor, ToolOutcome
from .mcp import (
    InMemoryMCPTransport,
    MCPClient,
    MCPServerConfig,
    MCPToolDefinition,
    MCPToolset,
    MCPTransport,
    MockMCPServer,
    StdioMCPTransport,
    load_mcp_tools,
    mcp_tool_digest,
)
from .registry import ToolRegistry

__all__ = [
    "DEFAULT_DENIED_READS",
    "AskUser",
    "BaseTool",
    "InMemoryMCPTransport",
    "InputRequest",
    "MCPClient",
    "MCPServerConfig",
    "MCPToolDefinition",
    "MCPToolset",
    "MCPTransport",
    "MockMCPServer",
    "ReadFile",
    "StdioMCPTransport",
    "SyncTool",
    "ToolExecutor",
    "ToolOutcome",
    "ToolOutput",
    "ToolRegistry",
    "load_mcp_tools",
    "mcp_tool_digest",
    "tool",
]
