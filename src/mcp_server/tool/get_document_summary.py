"""MCP Tool: get_document_summary (thin protocol adapter).

Document metadata / summary retrieval lives in
:class:`src.core.service.document_summary.DocumentSummaryService`.  This
module only declares the MCP tool contract and maps service results to the
protocol.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from mcp import types

from src.core.service import (
    DocumentNotFoundError,
    DocumentSummaryService,
    GetDocumentSummaryConfig,
    format_document_error,
    format_document_summary,
)
from src.mcp_server.tool.base import ToolResult

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


# Tool metadata
TOOL_NAME = "get_document_summary"
TOOL_DESCRIPTION = """Get summary and metadata for a specific document.

Returns structured information about a document including:
- Title (extracted or inferred from content)
- Summary (first chunk preview or metadata summary)
- Tags (document-level tags/categories)
- Source path
- Chunk count

Use this tool after list_collections to get details about specific documents.
"""

TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "doc_id": {
            "type": "string",
            "description": "The document ID to retrieve summary for. Can be full doc_id (e.g., 'doc_abc123') or the hash portion.",
        },
        "collection": {
            "type": "string",
            "description": "Collection name to search in. If not specified, searches the default collection.",
        },
    },
    "required": ["doc_id"],
}


# Prompt exposed via MCP prompts/list + prompts/get (placeholder wording)
TOOL_PROMPT = """Fetch the summary and metadata of a single knowledge document.

Call this tool with:
- doc_id: the document id (required; full id or hash portion)
- collection: which collection to search (optional, defaults to the default collection)

Use this after list_collections to get details about a specific document.
"""


class GetDocumentSummaryTool:
    """MCP adapter for the document summary service.

    All business logic (ChromaDB access, chunk discovery, title/summary/tags
    extraction) is delegated to :class:`DocumentSummaryService`; this class
    only maps results to :class:`ToolResult`.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        config: GetDocumentSummaryConfig | None = None,
        service: DocumentSummaryService | None = None,
    ) -> None:
        """Initialize GetDocumentSummaryTool.

        Args:
            settings: Application settings. If None, loaded from default path.
            config: Service configuration. If None, derived from settings.
            service: Optional pre-built DocumentSummaryService (DI for tests).
        """
        self._settings = settings
        self._config = config
        self._service = service

    @property
    def service(self) -> DocumentSummaryService:
        """Get (or lazily build) the underlying business service."""
        if self._service is None:
            self._service = DocumentSummaryService(
                settings=self._settings,
                config=self._config,
            )
        return self._service

    async def execute(
        self,
        doc_id: str,
        collection: str | None = None,
    ) -> ToolResult:
        """Execute the get_document_summary tool.

        Args:
            doc_id: Document ID to retrieve summary for.
            collection: Optional collection name.

        Returns:
            ToolResult with a formatted summary and structured document data
            in ``structured_content``, or a business-error result.
        """
        logger.info(
            f"Executing get_document_summary (doc_id={doc_id}, collection={collection})"
        )

        try:
            # Run blocking ChromaDB I/O in a thread to avoid blocking
            # the async event loop / MCP stdio transport
            summary = await asyncio.to_thread(
                self.service.get_document_summary, doc_id, collection,
            )
            response_text = format_document_summary(summary)

            return ToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=response_text,
                    )
                ],
                structured_content=summary.to_dict(),
            )

        except DocumentNotFoundError as e:
            logger.warning(f"Document not found: {e}")
            return ToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=format_document_error(e),
                    )
                ],
                is_error=True,
            )

        except Exception as e:
            logger.exception("Error executing get_document_summary")
            return ToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=format_document_error(e),
                    )
                ],
                is_error=True,
            )
