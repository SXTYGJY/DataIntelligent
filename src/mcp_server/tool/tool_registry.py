"""Single registration, discovery, and execution boundary for MCP tools."""

from __future__ import annotations

import builtins
import logging
from collections import OrderedDict
from typing import Any

from mcp import types

from src.mcp_server.tool.base import BaseTool, CallbackTool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Own the complete set of implemented, callable MCP tools."""

    def __init__(self) -> None:
        self._tools: OrderedDict[str, BaseTool] = OrderedDict()

    def register(self, tool: BaseTool) -> None:
        """Register one tool, rejecting invalid or duplicate definitions."""
        self._validate_tool(tool)
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        logger.info("Registered MCP tool: %s", tool.name)

    def unregister(self, name: str) -> BaseTool:
        """Remove a tool; intended for controlled shutdown and tests only."""
        try:
            return self._tools.pop(name)
        except KeyError as exc:
            raise KeyError(f"Tool '{name}' is not registered") from exc

    def get(self, name: str) -> BaseTool | None:
        """Return a registered tool, or ``None`` when it does not exist."""
        return self._tools.get(name)

    def list(self) -> builtins.list[BaseTool]:
        """Return a snapshot of tools in registration order."""
        return list(self._tools.values())

    def list_mcp_tools(self) -> builtins.list[types.Tool]:
        """Return the sole source of truth for MCP ``tools/list``."""
        return [tool.to_mcp_definition() for tool in self.list()]

    async def exec(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        context: ToolContext | None = None,
    ) -> types.CallToolResult:
        """Execute a named tool and normalize all client-visible outcomes."""
        tool = self.get(name)
        if tool is None:
            return self._error(f"Tool '{name}' not found")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return self._error("Invalid parameters: arguments must be an object")

        validation_error = self._validate_arguments(tool.input_schema, arguments)
        if validation_error:
            return self._error(f"Invalid parameters: {validation_error}")

        try:
            result = await tool.execute(arguments, context or ToolContext())
            if not isinstance(result, ToolResult):
                raise TypeError("BaseTool.execute must return ToolResult")
            response_kwargs: dict[str, Any] = {
                "content": result.content,
                "isError": result.is_error,
            }
            model_fields = getattr(types.CallToolResult, "model_fields", {})
            if result.structured_content is not None and "structuredContent" in model_fields:
                response_kwargs["structuredContent"] = result.structured_content
            return types.CallToolResult(**response_kwargs)
        except (TypeError, ValueError) as exc:
            logger.info("Invalid parameters for tool %s: %s", name, exc)
            return self._error(f"Invalid parameters: {exc}")
        except Exception:
            logger.exception("Internal error executing tool %s", name)
            return self._error(f"Internal error while executing '{name}'")

    @staticmethod
    def _error(message: str) -> types.CallToolResult:
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=message)],
            isError=True,
        )

    @staticmethod
    def _validate_tool(tool: BaseTool) -> None:
        if not isinstance(tool, BaseTool):
            raise TypeError("Only BaseTool instances can be registered")
        if not isinstance(getattr(tool, "name", None), str) or not tool.name:
            raise ValueError("Tool name must be a non-empty string")
        if not isinstance(getattr(tool, "description", None), str) or not tool.description:
            raise ValueError("Tool description must be a non-empty string")
        schema = getattr(tool, "input_schema", None)
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ValueError("Tool input_schema must be a JSON Schema object")

    @staticmethod
    def _validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> str | None:
        required = schema.get("required", [])
        for key in required:
            if key not in arguments:
                return f"missing required field '{key}'"

        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unexpected = set(arguments) - set(properties)
            if unexpected:
                return f"unexpected field '{sorted(unexpected)[0]}'"

        expected_types = {"string": str, "integer": int, "number": (int, float), "boolean": bool}
        for key, value in arguments.items():
            definition = properties.get(key)
            if not definition or "type" not in definition:
                continue
            expected = expected_types.get(definition["type"])
            if expected and (not isinstance(value, expected) or isinstance(value, bool) and definition["type"] != "boolean"):
                return f"field '{key}' must be a {definition['type']}"
        return None


def register_default_tools(registry: ToolRegistry) -> None:
    """Register existing public tools through BaseTool migration adapters."""
    from src.mcp_server.tool.get_document_summary import (
        TOOL_DESCRIPTION as SUMMARY_DESCRIPTION,
    )
    from src.mcp_server.tool.get_document_summary import (
        TOOL_INPUT_SCHEMA as SUMMARY_SCHEMA,
    )
    from src.mcp_server.tool.get_document_summary import (
        TOOL_NAME as SUMMARY_NAME,
    )
    from src.mcp_server.tool.get_document_summary import (
        GetDocumentSummaryTool,
    )
    from src.mcp_server.tool.list_collections import (
        TOOL_DESCRIPTION as COLLECTIONS_DESCRIPTION,
    )
    from src.mcp_server.tool.list_collections import (
        TOOL_INPUT_SCHEMA as COLLECTIONS_SCHEMA,
    )
    from src.mcp_server.tool.list_collections import (
        TOOL_NAME as COLLECTIONS_NAME,
    )
    from src.mcp_server.tool.list_collections import (
        ListCollectionsTool,
    )
    from src.mcp_server.tool.query_knowledge_hub import (
        TOOL_DESCRIPTION as QUERY_DESCRIPTION,
    )
    from src.mcp_server.tool.query_knowledge_hub import (
        TOOL_INPUT_SCHEMA as QUERY_SCHEMA,
    )
    from src.mcp_server.tool.query_knowledge_hub import (
        TOOL_NAME as QUERY_NAME,
    )
    from src.mcp_server.tool.query_knowledge_hub import (
        query_knowledge_hub_handler,
    )

    collections_tool = ListCollectionsTool()
    summary_tool = GetDocumentSummaryTool()

    async def collections_callback(**arguments: Any) -> types.CallToolResult:
        return await collections_tool.execute(**arguments)

    async def summary_callback(**arguments: Any) -> types.CallToolResult:
        return await summary_tool.execute(**arguments)

    registry.register(
        CallbackTool(
            name=QUERY_NAME,
            description=QUERY_DESCRIPTION,
            input_schema=QUERY_SCHEMA,
            callback=query_knowledge_hub_handler,
        )
    )
    registry.register(
        CallbackTool(
            name=COLLECTIONS_NAME,
            description=COLLECTIONS_DESCRIPTION,
            input_schema=COLLECTIONS_SCHEMA,
            callback=collections_callback,
        )
    )
    registry.register(
        CallbackTool(
            name=SUMMARY_NAME,
            description=SUMMARY_DESCRIPTION,
            input_schema=SUMMARY_SCHEMA,
            callback=summary_callback,
        )
    )
