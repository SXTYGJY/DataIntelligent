"""Shared MCP tool contracts and the server tool registry."""

from src.mcp_server.tool.base import BaseTool, ToolContext, ToolMetadata, ToolResult
from src.mcp_server.tool.tool_manager import (
    PromptInfo,
    ToolRegistration,
    ToolRegistry,
    register_default_tools,
)

__all__ = [
    "BaseTool",
    "PromptInfo",
    "ToolContext",
    "ToolMetadata",
    "ToolRegistration",
    "ToolRegistry",
    "ToolResult",
    "register_default_tools",
]
