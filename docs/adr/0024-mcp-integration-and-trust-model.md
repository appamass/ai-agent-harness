# 0024 – MCP Integration and Trust Model

- **Status:** Accepted (maintainer, 2026-10-22)
- **Date:** 2026-10-22
- **Affects:** `ai_agent_harness/tools/mcp.py`, `tools/executor.py`, `tools/__init__.py`, `ai_agent_harness/__init__.py`

## Context

MCP servers bring many tools but also risk: a tool with no declared risk can do irreversible things; a server can change tool descriptions later (rug pull); tool output and errors are untrusted text; and calls must not bypass the gate.

## Decision

1. **Real tools through the gate.** Each MCP tool becomes a `BaseTool` run only by `ToolExecutor`: write-ahead, timeout, policy, hooks, approval.
2. **Risk is local and mandatory.** Every tool must be in `MCPServerConfig.tool_risks`; an unclassified tool makes `load_mcp_tools` raise `ConfigError`. Servers cannot declare their own risk.
3. **Pinned definitions.** `tool_digests` maps names to `mcp_tool_digest(MCPToolDefinition)`: SHA-256 over name, description, and the full input schema exactly as the server sent them (not the lossy local model). A changed, unpinned, or missing pinned tool raises `ConfigError`. The check runs at load time; after `notifications/tools/list_changed` every call fails closed until the tools are reloaded.
4. **Allowlist is opt-in:** `load_mcp_tools(..., allowed_servers={…})`; without it any server may be loaded.
5. **Untrusted by default.** Output and error text are truncated and wrapped in `<untrusted>` frames (`untrusted_output=True`).
6. **Standard library only.** `StdioMCPTransport` (JSON-RPC lines over a subprocess) accepts messages up to 16 MiB, fails all pending and later requests when the server exits or breaks the protocol, answers or rejects server requests without ever treating them as our responses, drains stderr, gives the server only `PATH`, `HOME`, locale, and temp-dir variables (more through `MCPServerConfig.env`), and bounds startup by `connect_timeout_s`. `tools/list` follows `nextCursor` (up to 50 pages). `InMemoryMCPTransport` and `MockMCPServer` are for tests.
7. **Lifecycle.** `load_mcp_tools` closes the connection on every failure and returns an `MCPToolset` (a list with `aclose()` and `async with`).
8. The local Pydantic model of the arguments is a coarse pre-check only; the server validates. The model sees the server's real input schema (`spec()` returns it verbatim), and the concrete-type check is skipped (`strict_schema = False`), so required arrays, objects, `anyOf`, and enums load. Omitted optional parameters are not sent. Server tool names with characters tool names cannot have (`github.search`) are mapped (`github_search`) and called by their original name.

## Rejected

- Letting servers declare their risk: they cannot be trusted to.
- Digest over the local model: it drops `enum`, `pattern`, bounds, and nested schemas, so changes go unnoticed.
- An external MCP SDK: new dependency for a small protocol subset.

## Consequences

No HTTP/SSE transport, resources, or prompts. Only tested against a stub stdio server and an in-memory server (`tests/test_mcp.py`).
