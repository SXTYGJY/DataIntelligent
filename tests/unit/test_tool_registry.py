"""Contract tests for BaseTool and the unified ToolRegistry."""

from __future__ import annotations

from typing import Any

import pytest
from mcp import types

from src.mcp_server.tool import (
    BaseTool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    register_default_tools,
)


class GreetingTool(BaseTool):
    name = "greeting"
    description = "Return a greeting for a name."
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        return ToolResult(
            content=[types.TextContent(type="text", text=f"Hello {arguments['name']}")],
            metadata={"request_id": context.request_id},
        )


class FailingTool(GreetingTool):
    name = "failing"
    description = "Raise an unexpected exception."

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        raise RuntimeError("secret backend failure")


class BusinessErrorTool(GreetingTool):
    name = "business_error"
    description = "Return a handled business error."

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        return ToolResult(
            content=[types.TextContent(type="text", text="Document is unavailable")],
            is_error=True,
        )


@pytest.fixture
def registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(GreetingTool())
    return registry


class TestToolRegistryRegistration:
    def test_register_get_list_and_mcp_definition(self, registry: ToolRegistry) -> None:
        assert registry.get("greeting").name == "greeting"
        assert [tool.name for tool in registry.list()] == ["greeting"]

        definitions = registry.list_mcp_tools()
        assert definitions[0].name == "greeting"
        assert definitions[0].inputSchema == GreetingTool.input_schema

    def test_duplicate_name_is_rejected(self, registry: ToolRegistry) -> None:
        with pytest.raises(ValueError, match="already registered"):
            registry.register(GreetingTool())

    def test_non_base_tool_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="BaseTool"):
            ToolRegistry().register(object())  # type: ignore[arg-type]

    def test_unregister_returns_removed_tool(self, registry: ToolRegistry) -> None:
        removed = registry.unregister("greeting")

        assert removed.name == "greeting"
        assert registry.get("greeting") is None

    def test_default_tools_are_all_registered_as_base_tools(self) -> None:
        registry = ToolRegistry()
        register_default_tools(registry)

        assert [tool.name for tool in registry.list()] == [
            "query_knowledge_hub",
            "list_collections",
            "get_document_summary",
        ]
        assert all(isinstance(tool, BaseTool) for tool in registry.list())


class TestToolRegistryExec:
    @pytest.mark.asyncio
    async def test_exec_returns_successful_mcp_result(self, registry: ToolRegistry) -> None:
        result = await registry.exec(
            "greeting", {"name": "Ada"}, ToolContext(request_id="request-1")
        )

        assert result.isError is False
        assert result.content[0].text == "Hello Ada"

    @pytest.mark.asyncio
    async def test_exec_rejects_missing_required_argument(self, registry: ToolRegistry) -> None:
        result = await registry.exec("greeting", {})

        assert result.isError is True
        assert "missing required field 'name'" in result.content[0].text

    @pytest.mark.asyncio
    async def test_exec_rejects_wrong_argument_type(self, registry: ToolRegistry) -> None:
        result = await registry.exec("greeting", {"name": 42})

        assert result.isError is True
        assert "must be a string" in result.content[0].text

    @pytest.mark.asyncio
    async def test_exec_rejects_unexpected_argument(self, registry: ToolRegistry) -> None:
        result = await registry.exec("greeting", {"name": "Ada", "extra": True})

        assert result.isError is True
        assert "unexpected field 'extra'" in result.content[0].text

    @pytest.mark.asyncio
    async def test_exec_normalizes_unknown_tool(self, registry: ToolRegistry) -> None:
        result = await registry.exec("unknown", {})

        assert result.isError is True
        assert result.content[0].text == "Tool 'unknown' not found"

    @pytest.mark.asyncio
    async def test_exec_preserves_handled_business_error(self, registry: ToolRegistry) -> None:
        registry.register(BusinessErrorTool())

        result = await registry.exec("business_error", {"name": "Ada"})

        assert result.isError is True
        assert result.content[0].text == "Document is unavailable"

    @pytest.mark.asyncio
    async def test_exec_hides_unexpected_exception_details(self, registry: ToolRegistry) -> None:
        registry.register(FailingTool())

        result = await registry.exec("failing", {"name": "Ada"})

        assert result.isError is True
        assert result.content[0].text == "Internal error while executing 'failing'"
        assert "secret" not in result.content[0].text
