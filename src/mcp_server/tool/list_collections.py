"""MCP Tool: list_collections (thin protocol adapter).

Collection enumeration and formatting live in
:class:`src.core.service.collections.CollectionService`.  This module only
declares the MCP tool contract and maps service results to the protocol.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from mcp import types

from src.core.service import (
    CollectionService,
    ListCollectionsConfig,
    format_collections_response,
)
from src.mcp_server.tool.base import ToolResult

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


# Tool metadata
TOOL_NAME = "list_collections"
TOOL_DESCRIPTION = """List all available document collections in the knowledge base.

Returns information about each collection including:
- Collection name
- Document count (if include_stats=true)
- Collection metadata

Use this tool to discover available collections before querying.
"""

TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "include_stats": {
            "type": "boolean",
            "description": "Whether to include statistics (document count) for each collection.",
            "default": True,
        },
    },
    "required": [],
}


# Prompt exposed via MCP prompts/list + prompts/get (placeholder wording)
TOOL_PROMPT = """Discover the collections available in the knowledge hub.

Call this tool with no arguments, or with:
- include_stats: whether to include the document/chunk count per collection
  (optional, default true)

Use the returned list to pick a collection name before querying
query_knowledge_hub or fetching a document summary.
"""


class ListCollectionsTool:
    """MCP adapter for the collection listing service.

    All business logic (ChromaDB access, enumeration, statistics) is delegated
    to :class:`CollectionService`; this class only maps results to
    :class:`ToolResult`.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        config: ListCollectionsConfig | None = None,
        service: CollectionService | None = None,
    ) -> None:
        """Initialize ListCollectionsTool.

        Args:
            settings: Application settings. If None, loaded from default path.
            config: Service configuration. If None, derived from settings.
            service: Optional pre-built CollectionService (DI for tests).
        """
        self._settings = settings
        self._config = config
        self._service = service

    @property
    def service(self) -> CollectionService:
        """Get (or lazily build) the underlying business service."""
        if self._service is None:
            self._service = CollectionService(
                settings=self._settings,
                config=self._config,
            )
        return self._service

    async def execute(
        self,
        include_stats: bool = True,
    ) -> ToolResult:
        """Execute the list_collections tool.

        Args:
            include_stats: Whether to include statistics for each collection.

        Returns:
            ToolResult with a formatted collection list and structured
            ``collections`` data in ``structured_content``.
        """
        logger.info(f"Executing list_collections (include_stats={include_stats})")

        try:
            # Run blocking ChromaDB I/O in a thread to avoid blocking
            # the async event loop / MCP stdio transport
            collections = await asyncio.to_thread(
                self.service.list_collections, include_stats=include_stats,
            )
            response_text = format_collections_response(collections)

            return ToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=response_text,
                    )
                ],
                structured_content={
                    "collections": [coll.to_dict() for coll in collections],
                    "count": len(collections),
                },
            )

        except Exception as e:
            logger.exception("Error executing list_collections")
            return ToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Error listing collections: {str(e)}",
                    )
                ],
                is_error=True,
            )
