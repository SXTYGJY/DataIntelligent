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
from src.mcp_server.tool import BaseTool, ToolRegistry, ToolResult

ECHO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


def make_echo_tool() -> BaseTool:
    return BaseTool(
        name="echo",
        input_schema=ECHO_SCHEMA,
        description="Echo a text value.",
        prompt="Call this tool with a text value to echo it back.",
    )


async def echo_impl(text: str) -> ToolResult:
    return ToolResult(
        content=[types.TextContent(type="text", text=text)],
    )


@pytest.fixture
def registry() -> ToolRegistry:
    result = ToolRegistry()
    result.register(make_echo_tool(), echo_impl)
    return result


@pytest.fixture
def protocol_handler(registry: ToolRegistry) -> ProtocolHandler:
    return ProtocolHandler("test-server", "1.0.0", registry=registry)


class TestProtocolHandler:
    def test_list_delegates_to_registry(self, protocol_handler: ProtocolHandler) -> None:
        tools = protocol_handler.get_tool_schemas()

        assert [tool.name for tool in tools] == ["echo"]
        assert tools[0].inputSchema == ECHO_SCHEMA

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

    def test_list_prompts_delegates_to_registry(self, protocol_handler: ProtocolHandler) -> None:
        prompts = protocol_handler.list_prompts()

        assert [prompt.name for prompt in prompts] == ["echo"]
        assert prompts[0].description == "Echo a text value."

    def test_get_prompt_returns_user_message(self, protocol_handler: ProtocolHandler) -> None:
        result = protocol_handler.get_prompt("echo")

        assert result.description == "Echo a text value."
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert "echo" in result.messages[0].content.text

    def test_get_prompt_unknown_raises(self, protocol_handler: ProtocolHandler) -> None:
        with pytest.raises(ValueError, match="not found"):
            protocol_handler.get_prompt("missing")


class TestCreateMCPServer:
    def test_attaches_the_supplied_handler(self, protocol_handler: ProtocolHandler) -> None:
        server = create_mcp_server(
            "test-server", "1.0.0", protocol_handler=protocol_handler, register_tools=False
        )

        assert get_protocol_handler(server) is protocol_handler
        assert get_protocol_handler(server).registry is protocol_handler.registry

    def test_registers_prompts_handlers(self, protocol_handler: ProtocolHandler) -> None:
        server = create_mcp_server(
            "test-server", "1.0.0", protocol_handler=protocol_handler, register_tools=False
        )

        from mcp import types as mcp_types

        assert mcp_types.ListPromptsRequest in server.request_handlers
        assert mcp_types.GetPromptRequest in server.request_handlers


class TestJSONRPCErrorCodes:
    def test_error_codes_are_correct(self) -> None:
        assert JSONRPCErrorCodes.PARSE_ERROR == -32700
        assert JSONRPCErrorCodes.INVALID_REQUEST == -32600
        assert JSONRPCErrorCodes.METHOD_NOT_FOUND == -32601
        assert JSONRPCErrorCodes.INVALID_PARAMS == -32602
        assert JSONRPCErrorCodes.INTERNAL_ERROR == -32603
