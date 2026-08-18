"""Common contracts for every MCP tool exposed by DataIntelligent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from mcp import types


@dataclass(frozen=True)
class ToolContext:
    """Request-scoped values made available to a tool execution."""

    request_id: str | None = None
    settings: Any = None
    trace: Any = None


@dataclass
class ToolResult:
    """Protocol-independent result returned by :class:`BaseTool`."""

    content: list[Any]
    is_error: bool = False
    structured_content: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mcp(cls, result: types.CallToolResult) -> ToolResult:
        """Adapt an existing MCP SDK result during the migration period."""
        return cls(
            content=list(result.content),
            is_error=bool(result.isError),
            structured_content=getattr(result, "structuredContent", None),
        )


class BaseTool(ABC):
    """Minimal contract required for every callable MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    annotations: dict[str, Any] | None = None

    def to_mcp_definition(self) -> types.Tool:
        """Build the SDK model returned by ``tools/list``."""
        return types.Tool(
            name=self.name,
            description=self.description,
            inputSchema=self.input_schema,
        )

    @abstractmethod
    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        """Execute this tool with validated arguments."""


class CallbackTool(BaseTool):
    """Adapter for an existing async MCP handler.

    This is intentionally a migration adapter. New tools should implement
    :class:`BaseTool` directly and return :class:`ToolResult`.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        callback: Callable[..., Awaitable[types.CallToolResult]],
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self._callback = callback

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        del context  # Legacy handlers do not yet accept request context.
        return ToolResult.from_mcp(await self._callback(**arguments))
