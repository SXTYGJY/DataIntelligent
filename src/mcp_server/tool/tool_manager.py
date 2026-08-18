"""Single registration, discovery, prompt, and execution boundary for MCP tools.

Per `docs/design/tool_manager_design.md`:

- :class:`ToolRegistry` is the only boundary for tool registration,
  discovery (``list_tools``), prompt exposure (``list_prompt`` /
  ``get_prompt``) and execution (``exec``).
- ``exec`` is the only execution entry point; the :class:`_Executor` class
  below is its internal implementation and is **not** exported.
- MCP ``tools/list`` / ``tools/call`` / ``prompts/list`` / ``prompts/get``
  are thin ProtocolHandler delegations to this registry.
"""

from __future__ import annotations

import builtins
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from mcp import types

from src.mcp_server.tool.base import BaseTool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class PromptInfo:
    """Tool prompt summary backing MCP ``prompts/list`` and ``prompts/get``."""

    name: str
    description: str | None = None
    prompt: str | None = None


@dataclass
class ToolRegistration:
    """One registration record: declarative descriptor + callable implementation."""

    tool: BaseTool
    implementation: Callable[..., Awaitable[ToolResult]]


class ToolRegistry:
    """Own the complete set of implemented, callable MCP tools."""

    def __init__(self) -> None:
        self._tools: OrderedDict[str, ToolRegistration] = OrderedDict()

    def register(
        self,
        tool: BaseTool,
        implementation: Callable[..., Awaitable[ToolResult]],
    ) -> None:
        """Register a tool descriptor plus its implementation.

        Args:
            tool: Declarative tool descriptor.
            implementation: Async callable returning :class:`ToolResult`.

        Raises:
            TypeError: If ``tool`` or ``implementation`` has the wrong type.
            ValueError: If the tool name is already registered.
        """
        if not isinstance(tool, BaseTool):
            raise TypeError("Only BaseTool instances can be registered")
        if not callable(implementation):
            raise TypeError("Tool implementation must be callable")
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = ToolRegistration(tool=tool, implementation=implementation)
        logger.info("Registered MCP tool: %s", tool.name)

    def unregister(self, name: str) -> BaseTool:
        """Remove a tool; intended for controlled shutdown and tests only.

        Raises:
            KeyError: If the tool is not registered.
        """
        try:
            return self._tools.pop(name).tool
        except KeyError as exc:
            raise KeyError(f"Tool '{name}' is not registered") from exc

    def get(self, name: str) -> BaseTool | None:
        """Return a registered tool descriptor, or ``None`` when missing."""
        registration = self._tools.get(name)
        return registration.tool if registration is not None else None

    def list_tools(self) -> builtins.list[BaseTool]:
        """Return tool descriptors in registration order (``tools/list`` source)."""
        return [registration.tool for registration in self._tools.values()]

    def list_mcp_tools(self) -> builtins.list[types.Tool]:
        """Return the MCP-ready ``tools/list`` definitions."""
        return [tool.to_mcp_definition() for tool in self.list_tools()]

    def list_prompt(self) -> builtins.list[PromptInfo]:
        """Return prompt summaries for every registered tool (``prompts/list`` source)."""
        return [
            PromptInfo(
                name=tool.name,
                description=tool.description,
                prompt=tool.prompt,
            )
            for tool in self.list_tools()
        ]

    def get_prompt(self, name: str) -> PromptInfo | None:
        """Return a single tool prompt, or ``None`` when missing/unconfigured."""
        tool = self.get(name)
        if tool is None:
            return None
        return PromptInfo(
            name=tool.name,
            description=tool.description,
            prompt=tool.prompt,
        )

    async def exec(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        context: ToolContext | None = None,
    ) -> types.CallToolResult:
        """Execute a named tool and normalize all client-visible outcomes.

        This is the **only** execution entry point; it delegates to the
        module-internal :class:`_Executor`.
        """
        return await _Executor(self).run(name, arguments, context)


class _Executor:
    """Internal execution engine used by :meth:`ToolRegistry.exec`.

    Not exported: lookup → argument validation → implementation call →
    normalization to :class:`types.CallToolResult`.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def run(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        context: ToolContext | None,
    ) -> types.CallToolResult:
        registration = self._registry._tools.get(name)
        if registration is None:
            return self._error(f"Tool '{name}' not found")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return self._error("Invalid parameters: arguments must be an object")

        validation_error = self._validate_arguments(registration.tool.input_schema, arguments)
        if validation_error:
            return self._error(f"Invalid parameters: {validation_error}")

        try:
            result = await registration.implementation(**arguments)
            if not isinstance(result, ToolResult):
                raise TypeError("Tool implementation must return ToolResult")
            return self._to_call_tool_result(result)
        except (TypeError, ValueError) as exc:
            logger.info("Invalid parameters for tool %s: %s", name, exc)
            return self._error(f"Invalid parameters: {exc}")
        except Exception:
            logger.exception("Internal error executing tool %s", name)
            return self._error(f"Internal error while executing '{name}'")

    @staticmethod
    def _to_call_tool_result(result: ToolResult) -> types.CallToolResult:
        response_kwargs: dict[str, Any] = {
            "content": result.content,
            "isError": result.is_error,
        }
        model_fields = getattr(types.CallToolResult, "model_fields", {})
        if result.structured_content is not None and "structuredContent" in model_fields:
            response_kwargs["structuredContent"] = result.structured_content
        return types.CallToolResult(**response_kwargs)

    @staticmethod
    def _error(message: str) -> types.CallToolResult:
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=message)],
            isError=True,
        )

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

        expected_types: dict[str, Any] = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
        }
        for key, value in arguments.items():
            definition = properties.get(key)
            if not definition or "type" not in definition:
                continue
            expected = expected_types.get(definition["type"])
            if expected and (
                not isinstance(value, expected)
                or isinstance(value, bool)
                and definition["type"] != "boolean"
            ):
                return f"field '{key}' must be a {definition['type']}"
        return None


def register_default_tools(registry: ToolRegistry) -> None:
    """Register the published tools as descriptors + implementations.

    The three existing tool classes are kept as-is; their ``execute``
    methods are registered as implementations (unified ``ToolResult``
    contract). New tools register through :meth:`ToolRegistry.register`.
    """
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
        TOOL_PROMPT as SUMMARY_PROMPT,
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
        TOOL_PROMPT as COLLECTIONS_PROMPT,
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
        TOOL_PROMPT as QUERY_PROMPT,
    )
    from src.mcp_server.tool.query_knowledge_hub import (
        get_tool_instance,
    )

    registry.register(
        BaseTool(
            name=QUERY_NAME,
            input_schema=QUERY_SCHEMA,
            description=QUERY_DESCRIPTION,
            prompt=QUERY_PROMPT,
        ),
        get_tool_instance().execute,
    )
    registry.register(
        BaseTool(
            name=COLLECTIONS_NAME,
            input_schema=COLLECTIONS_SCHEMA,
            description=COLLECTIONS_DESCRIPTION,
            prompt=COLLECTIONS_PROMPT,
        ),
        ListCollectionsTool().execute,
    )
    registry.register(
        BaseTool(
            name=SUMMARY_NAME,
            input_schema=SUMMARY_SCHEMA,
            description=SUMMARY_DESCRIPTION,
            prompt=SUMMARY_PROMPT,
        ),
        GetDocumentSummaryTool().execute,
    )
