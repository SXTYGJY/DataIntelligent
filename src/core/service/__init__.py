"""Knowledge business service layer.

Business logic lives here, **outside** the MCP server package, so that
``src.mcp_server`` only contains protocol adaptation (tool descriptors,
argument mapping, transport wiring).

Modules:
- ``collections``      : vector-store collection listing
- ``document_summary`` : document metadata / summary retrieval
- ``query``            : hybrid search + rerank + response building
- ``preload``          : heavy-dependency preloading for worker threads
"""

from src.core.service.collections import (
    CollectionInfo,
    CollectionService,
    ListCollectionsConfig,
    format_collections_response,
)
from src.core.service.document_summary import (
    DocumentNotFoundError,
    DocumentSummary,
    DocumentSummaryService,
    GetDocumentSummaryConfig,
    format_document_error,
    format_document_summary,
)
from src.core.service.query import (
    KnowledgeQueryService,
    QueryKnowledgeHubConfig,
)

__all__ = [
    "CollectionInfo",
    "CollectionService",
    "DocumentNotFoundError",
    "DocumentSummary",
    "DocumentSummaryService",
    "GetDocumentSummaryConfig",
    "KnowledgeQueryService",
    "ListCollectionsConfig",
    "QueryKnowledgeHubConfig",
    "format_collections_response",
    "format_document_error",
    "format_document_summary",
]
