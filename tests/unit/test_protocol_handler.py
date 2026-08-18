"""Unit tests for the MCP adapter after the ToolRegistry migration."""

from __future__ import annotations

from typing import Any

import pytest
from mcp import types

from src.mcp_server.protocol_handler import (
    JSONRPCErrorCodes,
    ProtocolHandler,
    create_mcp_server,
    get_protocol_handler,
)
from src.mcp_server.tool import BaseTool, ToolContext, ToolRegistry, ToolResult


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo a text value."
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        return ToolResult(
            content=[types.TextContent(type="text", text=arguments["text"])],
            metadata={"request_id": context.request_id},
        )


@pytest.fixture
def registry() -> ToolRegistry:
    result = ToolRegistry()
    result.register(EchoTool())
    return result


@pytest.fixture
def protocol_handler(registry: ToolRegistry) -> ProtocolHandler:
    return ProtocolHandler("test-server", "1.0.0", registry=registry)


class TestProtocolHandler:
    def test_list_delegates_to_registry(self, protocol_handler: ProtocolHandler) -> None:
        tools = protocol_handler.get_tool_schemas()

        assert [tool.name for tool in tools] == ["echo"]
        assert tools[0].inputSchema == EchoTool.input_schema

    @pytest.mark.asyncio
    async def test_call_delegates_to_registry(self, protocol_handler: ProtocolHandler) -> None:
        result = await protocol_handler.execute_tool("echo", {"text": "hello"})

        assert result.isError is False
        assert result.content[0].text == "hello"

    @pytest.mark.asyncio
    async def test_call_unknown_tool_is_normalized(self, protocol_handler: ProtocolHandler) -> None:
        result = await protocol_handler.execute_tool("missing", {})

        assert result.isError is True
        assert result.content[0].text == "Tool 'missing' not found"


class TestCreateMCPServer:
    def test_attaches_the_supplied_handler(self, protocol_handler: ProtocolHandler) -> None:
        server = create_mcp_server(
            "test-server", "1.0.0", protocol_handler=protocol_handler, register_tools=False
        )

        assert get_protocol_handler(server) is protocol_handler
        assert get_protocol_handler(server).registry is protocol_handler.registry


class TestJSONRPCErrorCodes:
    def test_error_codes_are_correct(self) -> None:
        assert JSONRPCErrorCodes.PARSE_ERROR == -32700
        assert JSONRPCErrorCodes.INVALID_REQUEST == -32600
        assert JSONRPCErrorCodes.METHOD_NOT_FOUND == -32601
        assert JSONRPCErrorCodes.INVALID_PARAMS == -32602
        assert JSONRPCErrorCodes.INTERNAL_ERROR == -32603
