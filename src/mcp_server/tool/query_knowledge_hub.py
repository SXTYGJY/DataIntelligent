"""MCP Tool: query_knowledge_hub (thin protocol adapter).

The retrieval pipeline (hybrid search, rerank, response building, tracing)
lives in :class:`src.core.service.query.KnowledgeQueryService`.  This module
only declares the MCP tool contract and maps service results to the protocol.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.core.response.response_builder import MCPToolResponse
from src.core.service import KnowledgeQueryService, QueryKnowledgeHubConfig
from src.mcp_server.tool.base import ToolResult

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


# Tool metadata
TOOL_NAME = "query_knowledge_hub"
TOOL_DESCRIPTION = """Search the knowledge base for relevant documents.

This tool uses hybrid search (semantic + keyword) to find the most relevant
documents matching your query. Results include source citations for reference.

Parameters:
- query: Your search question or keywords
- top_k: Maximum number of results (default: 5)
- collection: Limit search to a specific document collection
"""

TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query or question to find relevant documents for.",
        },
        "top_k": {
            "type": "integer",
            "description": "Maximum number of results to return.",
            "default": 5,
            "minimum": 1,
            "maximum": 20,
        },
        "collection": {
            "type": "string",
            "description": "Optional collection name to limit the search scope.",
        },
    },
    "required": ["query"],
}

# Prompt exposed via MCP prompts/list + prompts/get (placeholder wording)
TOOL_PROMPT = """You are searching the knowledge hub for relevant documents.

Given a user question, call this tool with:
- query: the search question or keywords (required)
- top_k: how many results to return (optional, default 5, max 20)
- collection: restrict the search to one collection (optional)

The result contains a Markdown answer with citation markers ([1], [2], ...),
optionally followed by image blocks. When citations are present, prefer the
structuredContent.citations array for exact source/page/score data and render
the Markdown text to the user.
"""


class QueryKnowledgeHubTool:
    """MCP adapter for the knowledge query service.

    Validates the MCP input contract and maps :class:`MCPToolResponse` to the
    unified :class:`ToolResult`; all business orchestration is delegated to
    :class:`KnowledgeQueryService`.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        config: QueryKnowledgeHubConfig | None = None,
        service: KnowledgeQueryService | None = None,
    ) -> None:
        """Initialize QueryKnowledgeHubTool.

        Args:
            settings: Application settings. If None, loaded from default path.
            config: Query configuration. If None, uses service defaults.
            service: Optional pre-built KnowledgeQueryService (DI for tests).
        """
        self._settings = settings
        self._config = config
        self._service = service

    @property
    def service(self) -> KnowledgeQueryService:
        """Get (or lazily build) the underlying business service."""
        if self._service is None:
            self._service = KnowledgeQueryService(
                settings=self._settings,
                config=self._config,
            )
        return self._service

    async def execute(
        self,
        query: str,
        top_k: int | None = None,
        collection: str | None = None,
    ) -> ToolResult:
        """Execute the query_knowledge_hub tool.

        Args:
            query: Search query string.
            top_k: Maximum results to return.
            collection: Target collection name.

        Returns:
            ToolResult with formatted content, optional image blocks, and
            citations in ``structured_content``.

        Raises:
            ValueError: If query is empty or invalid.
        """
        # Input-contract validation (MCP concern)
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        logger.info(
            f"Executing query_knowledge_hub: query='{query[:50]}...', "
            f"top_k={top_k}, collection={collection}"
        )

        response = await self.service.query(
            query=query,
            top_k=top_k,
            collection=collection,
        )
        return self._to_tool_result(response)

    @staticmethod
    def _to_tool_result(response: MCPToolResponse) -> ToolResult:
        """Convert a built MCPToolResponse into the unified ToolResult contract.

        The human-readable Markdown (plus optional image blocks) stays in
        ``content``; citations move to ``structured_content`` so MCP clients
        can parse them as ``structuredContent`` (design decision D9).
        """
        return ToolResult(
            content=response.to_mcp_content(),
            structured_content={
                "citations": [citation.to_dict() for citation in response.citations],
                "metadata": response.metadata,
                "isEmpty": response.is_empty,
            },
            is_error=response.is_empty and "error" in response.metadata,
        )


# Module-level tool instance (lazy-initialized; heavy deps load on first use)
_tool_instance: QueryKnowledgeHubTool | None = None


def get_tool_instance(settings: Settings | None = None) -> QueryKnowledgeHubTool:
    """Get or create the module-level :class:`QueryKnowledgeHubTool` instance.

    Args:
        settings: Optional settings to use for initialization.

    Returns:
        QueryKnowledgeHubTool instance.
    """
    global _tool_instance
    if _tool_instance is None:
        _tool_instance = QueryKnowledgeHubTool(settings=settings)
    return _tool_instance
