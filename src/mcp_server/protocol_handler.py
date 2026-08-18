"""MCP SDK adapter backed by the shared :class:`ToolRegistry`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mcp import types
from mcp.server.lowlevel import Server

from src.mcp_server.tool import ToolContext, ToolRegistry, register_default_tools
from src.observability.logger import get_logger


class JSONRPCErrorCodes:
    """Standard JSON-RPC 2.0 error codes retained for protocol consumers."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


@dataclass
class ProtocolHandler:
    """Thin MCP adapter; the registry owns tool state and execution."""

    server_name: str
    server_version: str
    registry: ToolRegistry = field(default_factory=ToolRegistry)

    def __post_init__(self) -> None:
        self._logger = get_logger(log_level="INFO")

    def get_tool_schemas(self) -> list[types.Tool]:
        """Delegate MCP ``tools/list`` to the single registry."""
        return self.registry.list_mcp_tools()

    async def execute_tool(
        self, name: str, arguments: dict[str, Any] | None
    ) -> types.CallToolResult:
        """Delegate MCP ``tools/call`` to the single registry."""
        return await self.registry.exec(name, arguments, ToolContext())

    def get_capabilities(self) -> dict[str, Any]:
        """Return MCP capabilities declared by this server."""
        return {"tools": {}}


def create_mcp_server(
    server_name: str,
    server_version: str,
    protocol_handler: ProtocolHandler | None = None,
    register_tools: bool = True,
) -> Server:
    """Create an MCP SDK server wired only to a ``ToolRegistry``."""
    if protocol_handler is None:
        protocol_handler = ProtocolHandler(
            server_name=server_name,
            server_version=server_version,
        )
    if register_tools:
        register_default_tools(protocol_handler.registry)

    server = Server(server_name)

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return protocol_handler.get_tool_schemas()

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict[str, Any]
    ) -> types.CallToolResult:
        return await protocol_handler.execute_tool(name, arguments)

    server._protocol_handler = protocol_handler  # type: ignore[attr-defined]
    return server


def get_protocol_handler(server: Server) -> ProtocolHandler:
    """Return the ProtocolHandler attached by :func:`create_mcp_server`."""
    return server._protocol_handler  # type: ignore[attr-defined]
