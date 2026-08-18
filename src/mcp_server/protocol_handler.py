"""MCP SDK adapter backed by the shared :class:`ToolRegistry`.

Per `docs/design/tool_manager_design.md` §5, this class is a pure MCP SDK
adaptation layer: it holds no second tool dict, imports no concrete tool
classes, and only delegates to the registry.

| MCP protocol  | Delegation                                    |
|---------------|-----------------------------------------------|
| ``tools/list``   | ``registry.list_mcp_tools()``                 |
| ``tools/call``   | ``await registry.exec(name, arguments)``      |
| ``prompts/list`` | ``registry.list_prompt()`` → MCP ``Prompt``   |
| ``prompts/get``  | ``registry.get_prompt(name)`` → system message|
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

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

    # ── tools ──────────────────────────────────────────────────────────

    def get_tool_schemas(self) -> list[types.Tool]:
        """Delegate MCP ``tools/list`` to the single registry."""
        return self.registry.list_mcp_tools()

    async def execute_tool(
        self, name: str, arguments: dict[str, Any] | None
    ) -> types.CallToolResult:
        """Delegate MCP ``tools/call`` to the single registry."""
        return await self.registry.exec(name, arguments, ToolContext())

    # ── prompts ────────────────────────────────────────────────────────

    def list_prompts(self) -> list[types.Prompt]:
        """Delegate MCP ``prompts/list`` to the single registry.

        Returns a MCP ``Prompt`` for every registered tool (including tools
        that have not configured a ``prompt`` body yet).
        """
        return [
            types.Prompt(name=info.name, description=info.description or "")
            for info in self.registry.list_prompt()
        ]

    def get_prompt(self, name: str) -> types.GetPromptResult:
        """Delegate MCP ``prompts/get`` to the single registry.

        The tool prompt text is returned as a **user** message so the MCP
        host can feed it into the model context. (The MCP SDK 1.x
        ``PromptMessage.role`` only accepts ``user``/``assistant``; the
        design intent of a "system-style" instruction is preserved.)

        Args:
            name: Tool/prompt name.

        Raises:
            ValueError: If the prompt (or its ``prompt`` body) is not found.
        """
        info = self.registry.get_prompt(name)
        if info is None or not info.prompt:
            raise ValueError(f"Prompt '{name}' not found")
        return types.GetPromptResult(
            description=info.description,
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=info.prompt),
                )
            ],
        )

    def get_capabilities(self) -> dict[str, Any]:
        """Return MCP capabilities declared by this server."""
        return {"tools": {}, "prompts": {}}


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

    @server.list_prompts()
    async def handle_list_prompts() -> list[types.Prompt]:
        return protocol_handler.list_prompts()

    @server.get_prompt()
    async def handle_get_prompt(
        name: str, arguments: dict[str, str] | None
    ) -> types.GetPromptResult:
        del arguments  # tool prompts are static; no parameterized arguments yet
        return protocol_handler.get_prompt(name)

    server._protocol_handler = protocol_handler  # type: ignore[attr-defined]
    return server


def get_protocol_handler(server: Server) -> ProtocolHandler:
    """Return the ProtocolHandler attached by :func:`create_mcp_server`."""
    # The low-level Server stores the handler on a private attribute.
    return cast(ProtocolHandler, server._protocol_handler)  # type: ignore[attr-defined]
