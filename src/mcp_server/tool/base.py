"""Common contracts for every MCP tool exposed by DataIntelligent.

This module defines the *declarative* tool contract:

- :class:`BaseTool`  : declarative tool descriptor (name / schema / prompt / metadata)
- :class:`ToolMetadata`: governance metadata carried by a tool
- :class:`ToolResult`: protocol-independent result returned by tool implementations
- :class:`ToolContext`: request-scoped values passed through :meth:`ToolRegistry.exec`

Tools hold **no execution logic** here; execution is delegated through
``ToolRegistry.exec`` to the implementation registered alongside each
descriptor (see ``tool_manager.py``).
"""

from __future__ import annotations

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
    """Protocol-independent result returned by a tool implementation.

    Attributes:
        content: Content blocks; the first item must be a ``TextContent`` block.
        is_error: Whether this is a handled business error.
        structured_content: Optional structured payload exposed as MCP
            ``CallToolResult.structuredContent``.
        metadata: Additional response metadata (request_id, trace, ...).
    """

    content: list[Any]
    is_error: bool = False
    structured_content: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolMetadata:
    """Tool governance metadata (lean version).

    Reserved fields for future progressive loading (documented, not yet
    implemented): ``tenant_id`` / ``class_minor`` / ``class_minor_desc`` /
    ``software_version`` / ``character``.
    """

    user_id: str | None = None
    access_level: str | None = None
    class_major: str | None = None
    class_major_desc: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolMetadata:
        """Build metadata from a partial dict, ignoring unknown keys."""
        fields = {
            f.name: data[f.name]
            for f in cls.__dataclass_fields__.values()
            if f.name in data
        }
        return cls(**fields)


@dataclass
class BaseTool:
    """Declarative tool descriptor shared by every MCP tool.

    This is intentionally **not** an ABC: it carries no execution logic.
    A tool is described here and its implementation is registered alongside
    it in :class:`ToolRegistry` (``tool_manager.py``).

    Attributes:
        name: MCP tool name (required, validated at construction).
        input_schema: JSON Schema describing the tool arguments.
        id: Optional stable tool id (defaults to ``name`` when absent).
        show_name: Optional human-friendly display name.
        description: One-line summary shown in ``tools/list``.
        prompt: Detailed usage guidance returned by ``prompts/get``.
        metadata: Optional governance metadata.
    """

    name: str
    input_schema: dict[str, Any]
    id: str | None = None
    show_name: str | None = None
    description: str | None = None
    prompt: str | None = None
    metadata: ToolMetadata | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Tool name must be a non-empty string")
        if not isinstance(self.input_schema, dict) or self.input_schema.get("type") != "object":
            raise ValueError("Tool input_schema must be a JSON Schema object")
        if self.id is None:
            self.id = self.name

    def to_mcp_definition(self) -> types.Tool:
        """Generate the MCP ``tools/list`` definition.

        Returns:
            MCP Tool model (name / description / inputSchema).
        """
        return types.Tool(
            name=self.name,
            description=self.description or "",
            inputSchema=self.input_schema,
        )
