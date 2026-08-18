"""Contract tests for BaseTool descriptors and the unified ToolRegistry."""

from __future__ import annotations

from typing import Any

import pytest
from mcp import types

from src.mcp_server.tool import (
    BaseTool,
    PromptInfo,
    ToolContext,
    ToolMetadata,
    ToolRegistry,
    ToolResult,
    register_default_tools,
)

GREETING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
    "additionalProperties": False,
}


def make_greeting_tool() -> BaseTool:
    return BaseTool(
        name="greeting",
        input_schema=GREETING_SCHEMA,
        description="Return a greeting for a name.",
        prompt="Call this tool with a name to get a greeting.",
    )


async def greeting_impl(name: str) -> ToolResult:
    return ToolResult(
        content=[types.TextContent(type="text", text=f"Hello {name}")],
    )


async def failing_impl(name: str) -> ToolResult:
    del name
    raise RuntimeError("secret backend failure")


async def business_error_impl(name: str) -> ToolResult:
    del name
    return ToolResult(
        content=[types.TextContent(type="text", text="Document is unavailable")],
        is_error=True,
    )


@pytest.fixture
def registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(make_greeting_tool(), greeting_impl)
    return registry


class TestBaseTool:
    def test_to_mcp_definition(self) -> None:
        tool = make_greeting_tool()

        definition = tool.to_mcp_definition()

        assert isinstance(definition, types.Tool)
        assert definition.name == "greeting"
        assert definition.description == "Return a greeting for a name."
        assert definition.inputSchema == GREETING_SCHEMA

    def test_id_defaults_to_name(self) -> None:
        assert make_greeting_tool().id == "greeting"

    def test_empty_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            BaseTool(name="", input_schema={"type": "object"})

    def test_non_object_schema_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="JSON Schema object"):
            BaseTool(name="bad", input_schema={"type": "string"})


class TestToolMetadata:
    def test_from_dict_keeps_known_fields(self) -> None:
        metadata = ToolMetadata.from_dict(
            {"user_id": "u1", "class_major": "knowledge", "unknown": "ignored"}
        )

        assert metadata.user_id == "u1"
        assert metadata.class_major == "knowledge"
        assert metadata.access_level is None
        assert metadata.class_major_desc is None

    def test_from_dict_empty(self) -> None:
        assert ToolMetadata.from_dict({}) == ToolMetadata()


class TestToolRegistryRegistration:
    def test_register_get_list_and_mcp_definition(self, registry: ToolRegistry) -> None:
        assert registry.get("greeting").name == "greeting"
        assert [tool.name for tool in registry.list_tools()] == ["greeting"]

        definitions = registry.list_mcp_tools()
        assert definitions[0].name == "greeting"
        assert definitions[0].inputSchema == GREETING_SCHEMA

    def test_duplicate_name_is_rejected(self, registry: ToolRegistry) -> None:
        with pytest.raises(ValueError, match="already registered"):
            registry.register(make_greeting_tool(), greeting_impl)

    def test_non_base_tool_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="BaseTool"):
            ToolRegistry().register(object(), greeting_impl)  # type: ignore[arg-type]

    def test_non_callable_implementation_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            ToolRegistry().register(make_greeting_tool(), "not-callable")  # type: ignore[arg-type]

    def test_unregister_returns_removed_descriptor(self, registry: ToolRegistry) -> None:
        removed = registry.unregister("greeting")

        assert isinstance(removed, BaseTool)
        assert removed.name == "greeting"
        assert registry.get("greeting") is None

    def test_unregister_unknown_raises(self, registry: ToolRegistry) -> None:
        with pytest.raises(KeyError, match="not registered"):
            registry.unregister("missing")

    def test_default_tools_are_all_registered_as_base_tools(self) -> None:
        registry = ToolRegistry()
        register_default_tools(registry)

        assert [tool.name for tool in registry.list_tools()] == [
            "query_knowledge_hub",
            "list_collections",
            "get_document_summary",
        ]
        assert all(isinstance(tool, BaseTool) for tool in registry.list_tools())


class TestToolRegistryPrompts:
    def test_list_prompt_returns_all_tools(self, registry: ToolRegistry) -> None:
        prompts = registry.list_prompt()

        assert isinstance(prompts, list)
        assert all(isinstance(p, PromptInfo) for p in prompts)
        assert prompts[0].name == "greeting"
        assert prompts[0].description == "Return a greeting for a name."
        assert prompts[0].prompt is not None

    def test_get_prompt_returns_single(self, registry: ToolRegistry) -> None:
        info = registry.get_prompt("greeting")

        assert info is not None
        assert info.name == "greeting"
        assert "greeting" in (info.prompt or "")

    def test_get_prompt_unknown_returns_none(self, registry: ToolRegistry) -> None:
        assert registry.get_prompt("missing") is None


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
        registry.register(
            BaseTool(
                name="business_error",
                input_schema=GREETING_SCHEMA,
                description="Return a handled business error.",
            ),
            business_error_impl,
        )

        result = await registry.exec("business_error", {"name": "Ada"})

        assert result.isError is True
        assert result.content[0].text == "Document is unavailable"

    @pytest.mark.asyncio
    async def test_exec_hides_unexpected_exception_details(self, registry: ToolRegistry) -> None:
        registry.register(
            BaseTool(
                name="failing",
                input_schema=GREETING_SCHEMA,
                description="Raise an unexpected exception.",
            ),
            failing_impl,
        )

        result = await registry.exec("failing", {"name": "Ada"})

        assert result.isError is True
        assert result.content[0].text == "Internal error while executing 'failing'"
        assert "secret" not in result.content[0].text

    @pytest.mark.asyncio
    async def test_exec_returns_structured_content(self, registry: ToolRegistry) -> None:
        async def structured_impl(name: str) -> ToolResult:
            return ToolResult(
                content=[types.TextContent(type="text", text=f"Hello {name}")],
                structured_content={"name": name},
            )

        registry.register(
            BaseTool(
                name="structured",
                input_schema=GREETING_SCHEMA,
                description="Return structured content.",
            ),
            structured_impl,
        )

        result = await registry.exec("structured", {"name": "Ada"})

        assert result.isError is False
        assert result.structuredContent == {"name": "Ada"}

    @pytest.mark.asyncio
    async def test_exec_rejects_non_tool_result(self, registry: ToolRegistry) -> None:
        async def bad_impl(name: str) -> str:  # type: ignore[return-value]
            del name
            return "not a ToolResult"

        registry.register(
            BaseTool(
                name="bad_result",
                input_schema=GREETING_SCHEMA,
                description="Return a non-ToolResult.",
            ),
            bad_impl,
        )

        result = await registry.exec("bad_result", {"name": "Ada"})

        assert result.isError is True
        assert "must return ToolResult" in result.content[0].text
