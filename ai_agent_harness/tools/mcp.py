"""Model Context Protocol (MCP) integration with mandatory risk classification.

Trust model for external MCP servers (ADR 0024):
1. Mandatory local risk classification: Every MCP tool must be explicitly mapped to
   a Risk in configuration. Unclassified tools fail closed at startup.
2. Rug-pull prevention: Optional tool digests (`mcp_tool_digest`) pin the raw name, description, and input
   schema as the server sent them. Mutated tools fail closed at load time. A server that announces a changed
   tool list later (`notifications/tools/list_changed`) makes further calls fail closed until the tools are reloaded.
3. Server allowlist: opt-in. Pass `allowed_servers` to `load_mcp_tools` to restrict which servers may be loaded.
4. Untrusted by default: All MCP tool outputs have untrusted_output=True and are
   wrapped in `<untrusted>` frames by ToolExecutor to neutralize prompt injections.
5. Invariant preservation: MCP tools execute exclusively through ToolExecutor (the
   8-step gate), obeying PermissionPolicy (Autonomy x Risk), hooks, and write-ahead.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, create_model

from .._version import __version__
from ..core.errors import ConfigError, ToolError
from ..core.state import RunState
from ..core.types import ID_PATTERN, Risk, ToolSpec, TruncationMode
from .base import BaseTool, canonical_digest, check_truncation_mode

log = logging.getLogger("ai_agent_harness.mcp")

MCPHandler = Callable[[dict[str, Any]], Awaitable[str] | str]


class MCPTransport(Protocol):
    """Protocol for MCP JSON-RPC 2.0 communication."""

    async def connect(self) -> None: ...

    async def send_request(self, method: str, params: dict[str, Any] | None = None) -> Any: ...

    async def send_notification(self, method: str, params: dict[str, Any] | None = None) -> None: ...

    def set_notification_handler(self, handler: Callable[[str], None]) -> None:
        """Registers a callback for server notifications (called with the notification method)."""
        ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True)
class MCPToolDefinition:
    """Raw tool definition as returned by an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict[str, Any])


class MockMCPServer:
    """Deterministic in-memory mock MCP server for testing tools, errors, and rug-pulls."""

    def __init__(self, name: str = "mock-server") -> None:
        self.name = name
        self.tools: dict[str, tuple[MCPToolDefinition, MCPHandler]] = {}
        self.simulate_timeout: bool = False
        self.simulate_protocol_error: bool = False
        self._listeners: list[Callable[[str], None]] = []

    def add_listener(self, listener: Callable[[str], None]) -> None:
        """Registers a callback for server notifications (used by `InMemoryMCPTransport`)."""
        self._listeners.append(listener)

    def notify_tools_changed(self) -> None:
        """Simulates `notifications/tools/list_changed` towards every connected client."""
        for listener in list(self._listeners):
            listener("notifications/tools/list_changed")

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: MCPHandler,
    ) -> None:
        self.tools[name] = (MCPToolDefinition(name=name, description=description, input_schema=input_schema), handler)

    async def handle_request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        if method == "tools/call" and self.simulate_timeout:
            await asyncio.sleep(60.0)
        if self.simulate_protocol_error:
            raise ToolError("Simulated MCP protocol error")

        if method == "initialize":
            return {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": self.name, "version": "1.0.0"},
            }
        if method == "tools/list":
            return {
                "tools": [
                    {
                        "name": tool_def.name,
                        "description": tool_def.description,
                        "inputSchema": tool_def.input_schema,
                    }
                    for tool_def, _ in self.tools.values()
                ]
            }
        if method == "tools/call":
            p = params or {}
            tool_name = str(p.get("name", ""))
            args_raw = p.get("arguments", {})
            args = cast(dict[str, Any], args_raw) if isinstance(args_raw, dict) else {}
            if tool_name not in self.tools:
                return {
                    "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                    "isError": True,
                }
            _, handler = self.tools[tool_name]
            try:
                res = handler(args)
                text = await res if asyncio.iscoroutine(res) else str(res)
                return {
                    "content": [{"type": "text", "text": text}],
                    "isError": False,
                }
            except (RuntimeError, ValueError, TypeError, KeyError, OSError, ToolError) as exc:
                log.debug("Mock MCP server tool execution failed: %s", exc)
                return {
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                }
        raise ToolError(f"Unsupported MCP method: {method}")

    async def handle_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        if method == "notifications/initialized":
            return
        log.debug("Mock MCP server received notification: %s", method)


class InMemoryMCPTransport:
    """In-memory transport directly connecting an MCPClient to a MockMCPServer."""

    def __init__(self, server: MockMCPServer) -> None:
        self.server = server
        self.connected = False
        self.closed = False

    async def connect(self) -> None:
        self.connected = True

    async def send_request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        if not self.connected or self.closed:
            raise ToolError("InMemoryMCPTransport is not connected or already closed")
        return await self.server.handle_request(method, params)

    async def send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        if not self.connected or self.closed:
            raise ToolError("InMemoryMCPTransport is not connected or already closed")
        await self.server.handle_notification(method, params)

    def set_notification_handler(self, handler: Callable[[str], None]) -> None:
        self.server.add_listener(handler)

    async def aclose(self) -> None:
        self.closed = True


STOP_TIMEOUT_S = 5.0
"""How long a stdio server may take to exit after `terminate()` before it is killed."""

MAX_TOOL_PAGES = 50
"""`tools/list` pages followed before giving up (a server that never ends its cursor chain)."""

DEFAULT_MAX_MESSAGE_BYTES = 16 * 1024 * 1024
"""Largest single JSON-RPC message the stdio transport accepts (one line)."""

_INHERITED_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT", "USERPROFILE")
"""Environment variables a stdio server inherits. Everything else (API keys, tokens) must be passed explicitly."""


def _server_env(extra: Mapping[str, str] | None) -> dict[str, str]:
    env = {key: os.environ[key] for key in _INHERITED_ENV if key in os.environ}
    env.update(extra or {})
    return env


class StdioMCPTransport:
    """Subprocess stdio transport communicating via newline-delimited JSON-RPC 2.0 messages.

    The server gets a minimal environment (`PATH`, `HOME`, locale, temp dirs); pass anything else, such as
    credentials the server needs, through `env`. Messages up to `max_message_bytes` are accepted. When the
    connection breaks (exit, oversized or unreadable output), every pending and later request fails with `ToolError`.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> None:
        if not command:
            raise ConfigError("StdioMCPTransport requires a non-empty command sequence")
        if max_message_bytes < 1024:
            raise ConfigError("StdioMCPTransport: max_message_bytes must be >= 1024")
        self._command = tuple(command)
        self._env = _server_env(env)
        self._max_message_bytes = max_message_bytes
        self._proc: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._tasks: set[asyncio.Task[None]] = set()  # reader, stderr drain, and replies to server requests
        self._on_notification: Callable[[str], None] | None = None
        self._broken: ToolError | None = None

    def set_notification_handler(self, handler: Callable[[str], None]) -> None:
        self._on_notification = handler

    async def connect(self) -> None:
        if self._proc is not None:
            return
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self._command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._env,
                limit=self._max_message_bytes,
            )
        except Exception as exc:
            raise ToolError(f"Failed to spawn MCP subprocess {self._command[0]}: {exc}") from exc
        self._spawn(self._read_loop())
        self._spawn(self._drain_stderr())

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _drain_stderr(self) -> None:
        """Reads stderr so a chatty server never blocks on a full pipe. Content is dropped, never logged."""
        assert self._proc is not None
        assert self._proc.stderr is not None
        total = 0
        with contextlib.suppress(Exception):
            while chunk := await self._proc.stderr.read(4096):
                total += len(chunk)
        log.debug("MCP server wrote %d bytes to stderr", total)

    def _fail_pending(self, error: ToolError) -> None:
        pending, self._pending = self._pending, {}
        for fut in pending.values():
            if not fut.done():
                fut.set_exception(error)

    async def _read_loop(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        error = ToolError("MCP server closed the connection")
        try:
            while True:
                try:
                    line = await self._proc.stdout.readline()
                except ValueError:  # a line longer than max_message_bytes: the stream cannot be resynchronized
                    error = ToolError(f"MCP server sent a message larger than {self._max_message_bytes} bytes")
                    return
                if not line:
                    return
                try:
                    payload_raw: object = json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    log.warning("MCP stdio reader skipped an unreadable line")
                    continue
                if isinstance(payload_raw, dict):
                    try:
                        self._dispatch(cast(dict[str, Any], payload_raw))
                    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError) as exc:
                        # a single odd message must not take the connection down
                        log.warning("MCP stdio reader skipped a message (%s)", type(exc).__name__)
        except asyncio.CancelledError:
            error = ToolError("MCP transport closed")
            raise
        finally:
            self._broken = error
            self._fail_pending(error)

    def _dispatch(self, payload: dict[str, Any]) -> None:
        method = payload.get("method")
        msg_id = payload.get("id")
        if isinstance(method, str):  # a request or notification FROM the server, never a response to ours
            if msg_id is None:
                if self._on_notification is not None:
                    self._on_notification(method)
            else:
                reply: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id}
                if method == "ping":
                    reply["result"] = {}
                else:
                    reply["error"] = {"code": -32601, "message": "Method not supported by this client"}
                self._reply(reply)
            return
        if not isinstance(msg_id, int) or isinstance(msg_id, bool):
            return
        fut = self._pending.pop(msg_id, None)
        if fut is None or fut.done():
            return
        if "error" in payload:
            err = payload["error"]
            message = str(cast(dict[str, Any], err).get("message", err)) if isinstance(err, dict) else str(err)
            fut.set_exception(ToolError(message))
        else:
            fut.set_result(payload.get("result"))

    def _reply(self, message: dict[str, Any]) -> None:
        """Answers a server request without blocking the reader (a server that stops reading must not stall us)."""

        async def send() -> None:
            with contextlib.suppress(ToolError):
                await self._write(message)

        self._spawn(send())

    async def _write(self, message: dict[str, Any]) -> None:
        if self._broken is not None:
            raise self._broken
        if self._proc is None or self._proc.stdin is None:
            raise ToolError("StdioMCPTransport is not connected")
        self._proc.stdin.write(json.dumps(message).encode("utf-8") + b"\n")
        try:
            await self._proc.stdin.drain()
        except OSError as exc:
            raise ToolError("MCP server is not reading its input") from exc

    async def send_request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        if self._broken is not None:
            raise self._broken
        self._request_id += 1
        req_id = self._request_id
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        try:
            await self._write({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}})
            return await fut
        finally:
            self._pending.pop(req_id, None)  # timed out, cancelled, or answered

    async def send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        await self._write({"jsonrpc": "2.0", "method": method, "params": params or {}})

    async def aclose(self) -> None:
        error = ToolError("MCP transport closed")
        self._broken = self._broken or error
        self._fail_pending(error)
        if self._proc is not None:
            try:
                await self._stop_process(self._proc)
            except OSError as exc:  # includes ProcessLookupError: the server is gone already
                log.debug("MCP subprocess shutdown error: %s", exc)
        # The readers end by themselves once the process closed its pipes; cancel only what does not
        tasks = list(self._tasks)
        if tasks:
            _, stuck = await asyncio.wait(tasks, timeout=STOP_TIMEOUT_S)
            for task in stuck:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    async def _stop_process(proc: asyncio.subprocess.Process) -> None:
        if proc.stdin:
            proc.stdin.close()
        if proc.returncode is not None:
            return
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=STOP_TIMEOUT_S)
        except TimeoutError:
            proc.kill()
            await proc.wait()


def _parse_tool(item: object) -> MCPToolDefinition | None:
    """One entry of `tools/list`; entries without a name are ignored."""
    if not isinstance(item, dict) or "name" not in item:
        return None
    raw = cast(dict[str, Any], item)
    schema = raw.get("inputSchema")
    return MCPToolDefinition(
        name=str(raw["name"]),
        description=str(raw.get("description", "")),
        input_schema=cast(dict[str, Any], schema) if isinstance(schema, dict) else {},
    )


class MCPClient:
    """Client for querying and invoking tools on an MCP server."""

    def __init__(self, transport: MCPTransport) -> None:
        self._transport = transport
        self._initialized = False
        self._stale = False
        transport.set_notification_handler(self._on_notification)

    @property
    def stale(self) -> bool:
        """True once the server announced a changed tool list: reload the tools before trusting them again."""
        return self._stale

    def _on_notification(self, method: str) -> None:
        if method == "notifications/tools/list_changed":
            self._stale = True

    async def connect(self) -> None:
        await self._transport.connect()
        init_res = await self._transport.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "ai-agent-harness", "version": __version__},
            },
        )
        if not isinstance(init_res, dict):
            raise ToolError(f"Invalid initialize response from MCP server: {init_res!r}")
        await self._transport.send_notification("notifications/initialized", {})
        self._initialized = True

    async def list_tools(self) -> list[MCPToolDefinition]:
        if not self._initialized:
            raise ToolError("MCP client must be connected before calling list_tools")
        tools: list[MCPToolDefinition] = []
        cursor: str | None = None
        for _ in range(MAX_TOOL_PAGES):
            res_raw = await self._transport.send_request("tools/list", {"cursor": cursor} if cursor else {})
            if not isinstance(res_raw, dict):
                raise ToolError(f"Invalid tools/list response from MCP server: {res_raw!r}")
            res = cast(dict[str, Any], res_raw)
            raw_tools_obj = res.get("tools")
            if not isinstance(raw_tools_obj, list):
                raise ToolError(f"Invalid tools/list response from MCP server: {res!r}")
            tools.extend(
                definition
                for item in cast(list[object], raw_tools_obj)
                if (definition := _parse_tool(item)) is not None
            )
            next_cursor = res.get("nextCursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                return tools
            cursor = next_cursor
        raise ToolError(f"MCP server lists more than {MAX_TOOL_PAGES} pages of tools")

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if not self._initialized:
            raise ToolError("MCP client must be connected before calling call_tool")
        if self._stale:
            raise ToolError(
                "The MCP server announced a changed tool list: reload the tools with load_mcp_tools before calling it"
            )
        res_raw = await self._transport.send_request("tools/call", {"name": name, "arguments": arguments})
        if not isinstance(res_raw, dict):
            raise ToolError(f"Invalid response from MCP server for tool '{name}': {res_raw!r}")
        res = cast(dict[str, Any], res_raw)

        is_error = bool(res.get("isError", False))
        content_obj = res.get("content", [])
        text_parts: list[str] = []
        if isinstance(content_obj, list):
            content_list = cast(list[object], content_obj)
            for block_obj in content_list:
                if isinstance(block_obj, dict):
                    block = cast(dict[str, Any], block_obj)
                    if block.get("type") == "text":
                        text_parts.append(str(block.get("text", "")))
        result_text = "\n".join(text_parts) if text_parts else json.dumps(content_obj)

        if is_error:
            raise ToolError(result_text or f"Tool '{name}' failed")
        return result_text

    async def aclose(self) -> None:
        await self._transport.aclose()


_JSON_TYPES: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list[Any],
    "object": dict[str, Any],
}


def json_schema_to_pydantic_model(model_name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Converts a JSON Schema object definition into a Pydantic model: a coarse pre-check, the server validates."""
    properties_obj = schema.get("properties")
    properties = cast(dict[str, Any], properties_obj) if isinstance(properties_obj, dict) else {}
    required_obj = schema.get("required")
    required: set[str] = set(cast(list[str], required_obj)) if isinstance(required_obj, list) else set()
    fields: dict[str, Any] = {}
    for prop_name_raw, raw_prop in properties.items():
        prop_name = str(prop_name_raw)
        prop_schema = cast(dict[str, Any], raw_prop) if isinstance(raw_prop, dict) else {}
        py_type: Any = _JSON_TYPES.get(str(prop_schema.get("type", "")), Any)
        description = prop_schema.get("description")
        is_required = prop_name in required
        default: Any = ... if is_required else prop_schema.get("default")
        fields[prop_name] = (
            py_type if is_required else py_type | None,
            Field(default, description=str(description) if description is not None else None),
        )
    return create_model(model_name, __config__=ConfigDict(extra="forbid"), **fields)


@dataclass(frozen=True)
class MCPServerConfig:
    """Security and connection configuration for an MCP server."""

    server_name: str
    command: Sequence[str] | None = None
    transport: MCPTransport | None = None
    tool_risks: dict[str, Risk] = field(default_factory=dict[str, Risk])
    """Mandatory: tool_name -> Risk. Any unclassified tool fails closed at startup."""
    tool_digests: dict[str, str] | None = None
    """Optional SHA-256 tool digest pins. Any mutated tool fails closed at startup."""
    timeout_s: float = 30.0
    untrusted_output: bool = True
    """Untrusted by default: wraps tool output in an <untrusted> frame."""
    name_prefix: str | None = None
    max_result_chars: int | None = None
    truncation_mode: TruncationMode | None = None
    connect_timeout_s: float = 30.0
    """Upper bound for starting the server, `initialize`, and `tools/list`."""
    env: Mapping[str, str] | None = None
    """Extra environment for a `command` server. It only inherits `PATH`, `HOME`, locale, and temp dirs otherwise."""
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES
    """Largest accepted message of a `command` server."""

    def __post_init__(self) -> None:
        if not re.fullmatch(ID_PATTERN, self.server_name):
            raise ConfigError(f"MCPServerConfig: server_name must match {ID_PATTERN}, got {self.server_name!r}")
        if (self.command is None) == (self.transport is None):
            raise ConfigError("MCPServerConfig: provide either 'command' or 'transport', not both")
        if self.timeout_s <= 0:
            raise ConfigError("MCPServerConfig: timeout_s must be > 0")
        if self.truncation_mode is not None:
            check_truncation_mode(self.truncation_mode, "MCPServerConfig")
        if self.connect_timeout_s <= 0:
            raise ConfigError("MCPServerConfig: connect_timeout_s must be > 0")
        if self.env is not None and self.command is None:
            raise ConfigError("MCPServerConfig: 'env' only applies to a 'command' server")


def _build_mcp_tool(
    client: MCPClient,
    config: MCPServerConfig,
    definition: MCPToolDefinition,
    *,
    exposed_name: str,
    input_model: type[BaseModel],
    risk: Risk,
) -> BaseTool[BaseModel]:
    # Class bodies read these names before they assign their own attributes of the same name
    server_tool_name, model, tool_risk, raw_schema = definition.name, input_model, risk, definition.input_schema

    class _MCPToolInstance(BaseTool[BaseModel]):
        name: ClassVar[str] = exposed_name
        description: ClassVar[str] = definition.description or f"MCP tool {server_tool_name}"
        input_model: ClassVar[type[BaseModel]] = model
        risk: ClassVar[Risk] = tool_risk
        timeout_s: ClassVar[float] = config.timeout_s
        untrusted_output: ClassVar[bool] = config.untrusted_output
        strict_schema: ClassVar[bool] = False  # the server's schema is arbitrary; the server validates
        max_result_chars: ClassVar[int | None] = config.max_result_chars
        truncation_mode: ClassVar[TruncationMode | None] = config.truncation_mode

        @classmethod
        def spec(cls) -> ToolSpec:
            # The model sees the server's real schema (enums, bounds, nesting); the local model is only a pre-check
            return ToolSpec(name=cls.name, description=cls.description, input_schema=raw_schema)

        async def run(self, args: BaseModel, state: RunState) -> str:
            # Only what the model set: omitted optional parameters must not arrive at the server as invented nulls
            return await client.call_tool(server_tool_name, args.model_dump(mode="json", exclude_unset=True))

    _MCPToolInstance.validate_definition()
    return _MCPToolInstance()


def mcp_tool_digest(definition: MCPToolDefinition) -> str:
    """SHA-256 fingerprint (64 hex chars) of a tool exactly as the server announced it.

    Covers name, description, and the complete input schema. Pin it in `MCPServerConfig.tool_digests`.
    The local Pydantic model is only a coarse pre-check, so pins must not depend on it.
    """
    return canonical_digest(
        {"name": definition.name, "description": definition.description, "input_schema": definition.input_schema}
    )


class MCPToolset(list[BaseTool[BaseModel]]):
    """The tools of one MCP server. Behaves like a list; `aclose()` stops the server connection."""

    def __init__(self, client: MCPClient, tools: Sequence[BaseTool[BaseModel]]) -> None:
        super().__init__(tools)
        self._client = client

    @property
    def stale(self) -> bool:
        """True once the server announced a changed tool list (calls then fail closed until reloaded)."""
        return self._client.stale

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> MCPToolset:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()


async def load_mcp_tools(
    config: MCPServerConfig,
    *,
    allowed_servers: frozenset[str] | set[str] | None = None,
) -> MCPToolset:
    """Connects to an MCP server, verifies security invariants, and returns the wrapped tools.

    Security guarantees:
    - Server allowlist when `allowed_servers` is given (opt-in: without it every server may be loaded)
    - Fail-closed mandatory local risk classification (every tool must be in tool_risks)
    - Fail-closed rug-pull prevention: `mcp_tool_digest` of every tool must match `tool_digests` when pinned
    - Untrusted output wrapping enabled by default
    - Bounded startup (`connect_timeout_s`); the connection is closed again if loading fails

    Close the result (`await tools.aclose()` or `async with`) when the tools are no longer needed.
    """
    if allowed_servers is not None and config.server_name not in allowed_servers:
        raise ConfigError(f"MCP server '{config.server_name}' is not in allowed_servers: {sorted(allowed_servers)}")

    transport: MCPTransport
    if config.transport is not None:
        transport = config.transport
    else:
        assert config.command is not None
        transport = StdioMCPTransport(config.command, env=config.env, max_message_bytes=config.max_message_bytes)

    client = MCPClient(transport)
    try:
        try:
            async with asyncio.timeout(config.connect_timeout_s):
                await client.connect()
                raw_tools = await client.list_tools()
        except TimeoutError:
            raise ToolError(
                f"MCP server '{config.server_name}' did not answer within {config.connect_timeout_s:g} s"
            ) from None
        return MCPToolset(client, _wrap_tools(config, client, raw_tools))
    except BaseException:
        with contextlib.suppress(Exception):
            await client.aclose()
        raise


def _wrap_tools(
    config: MCPServerConfig, client: MCPClient, raw_tools: Sequence[MCPToolDefinition]
) -> list[BaseTool[BaseModel]]:
    tools: list[BaseTool[BaseModel]] = []
    exposed: set[str] = set()
    for tool_def in raw_tools:
        # 1. Mandatory risk classification check
        if tool_def.name not in config.tool_risks:
            raise ConfigError(
                f"MCP server '{config.server_name}' exposed tool '{tool_def.name}' which has no Risk classification "
                f"in tool_risks. All MCP tools must be explicitly classified (fail-closed)."
            )
        tool_risk = config.tool_risks[tool_def.name]

        # 2. Rug-pull digest verification of the raw definition, if pinned
        if config.tool_digests is not None:
            if tool_def.name not in config.tool_digests:
                raise ConfigError(
                    f"MCP tool '{tool_def.name}' from server '{config.server_name}' is not in pinned tool_digests."
                )
            expected_digest = config.tool_digests[tool_def.name]
            actual_digest = mcp_tool_digest(tool_def)
            if actual_digest != expected_digest:
                raise ConfigError(
                    f"MCP tool '{tool_def.name}' from server '{config.server_name}' failed hash verification: "
                    f"expected {expected_digest}, got {actual_digest}. Description or schema was mutated."
                )

        # 3. Derive exposed name and input model, construct the BaseTool subclass
        # Server names may contain characters tool names cannot (`github.search`): map them, call by the original
        exposed_name = f"{config.name_prefix or ''}{re.sub(r'[^A-Za-z0-9_-]', '_', tool_def.name)}"
        model_name = f"MCP_{config.server_name}_{tool_def.name}_Input"
        if exposed_name in exposed:
            raise ConfigError(f"MCP server '{config.server_name}': two tools map to the name '{exposed_name}'")
        exposed.add(exposed_name)
        tools.append(
            _build_mcp_tool(
                client,
                config,
                tool_def,
                exposed_name=exposed_name,
                input_model=json_schema_to_pydantic_model(model_name, tool_def.input_schema),
                risk=tool_risk,
            )
        )

    if config.tool_digests is not None:
        missing = sorted(set(config.tool_digests) - {t.name for t in raw_tools})
        if missing:
            raise ConfigError(f"Pinned MCP tools missing on server '{config.server_name}': {', '.join(missing)}")
    return tools
