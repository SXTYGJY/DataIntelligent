"""Unit tests for the slimmed MCP tool adapters.

After the service-layer refactor, the tool modules in ``src.mcp_server.tool``
are thin protocol adapters: they declare the MCP contract (name / schema /
description / prompt) and map service results to :class:`ToolResult`.  All
business logic is tested against ``src.core.service`` (see
``test_service_collections.py`` / ``test_service_document_summary.py``).
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from src.core.response.response_builder import MCPToolResponse
from src.core.service import (
    CollectionInfo,
    DocumentSummary,
)
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
    QueryKnowledgeHubTool,
)

SAMPLE_COLLECTIONS = [
    CollectionInfo(name="knowledge_hub", count=150, metadata={"hnsw:space": "cosine"}),
    CollectionInfo(name="documents", count=75, metadata={"description": "Main docs"}),
]


class TestToolMetadata:
    """All three published tools keep their MCP contract constants."""

    def test_query_knowledge_hub_metadata(self) -> None:
        assert QUERY_NAME == "query_knowledge_hub"
        assert QUERY_DESCRIPTION
        assert QUERY_SCHEMA["type"] == "object"
        assert "query" in QUERY_SCHEMA["required"]

    def test_list_collections_metadata(self) -> None:
        assert COLLECTIONS_NAME == "list_collections"
        assert COLLECTIONS_DESCRIPTION
        assert COLLECTIONS_SCHEMA["type"] == "object"
        assert COLLECTIONS_SCHEMA["required"] == []

    def test_get_document_summary_metadata(self) -> None:
        assert SUMMARY_NAME == "get_document_summary"
        assert SUMMARY_DESCRIPTION
        assert SUMMARY_SCHEMA["type"] == "object"
        assert "doc_id" in SUMMARY_SCHEMA["required"]


class TestListCollectionsToolAdapter:
    """list_collections delegates to CollectionService and maps ToolResult."""

    @pytest.mark.asyncio
    async def test_execute_success(self) -> None:
        tool = ListCollectionsTool()
        with patch.object(
            tool.service, "list_collections", return_value=SAMPLE_COLLECTIONS
        ):
            result = await tool.execute(include_stats=True)

        assert result.is_error is False
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert "Available Collections (2 total)" in result.content[0].text
        assert result.structured_content == {
            "collections": [c.to_dict() for c in SAMPLE_COLLECTIONS],
            "count": 2,
        }

    @pytest.mark.asyncio
    async def test_execute_empty_result(self) -> None:
        tool = ListCollectionsTool()
        with patch.object(tool.service, "list_collections", return_value=[]):
            result = await tool.execute()

        assert result.is_error is False
        assert "No collections found" in result.content[0].text

    @pytest.mark.asyncio
    async def test_execute_error(self) -> None:
        tool = ListCollectionsTool()
        with patch.object(
            tool.service,
            "list_collections",
            side_effect=RuntimeError("boom"),
        ):
            result = await tool.execute()

        assert result.is_error is True
        assert "Error listing collections" in result.content[0].text

    @pytest.mark.asyncio
    async def test_execute_passes_include_stats(self) -> None:
        tool = ListCollectionsTool()
        mock_list = Mock(return_value=[])
        with patch.object(tool.service, "list_collections", mock_list):
            await tool.execute(include_stats=False)

        mock_list.assert_called_once_with(include_stats=False)


class TestGetDocumentSummaryToolAdapter:
    """get_document_summary delegates to DocumentSummaryService."""

    def _make_tool(self) -> GetDocumentSummaryTool:
        return GetDocumentSummaryTool()

    @pytest.mark.asyncio
    async def test_execute_success(self) -> None:
        tool = self._make_tool()
        summary = DocumentSummary(
            doc_id="doc_abc123",
            title="Test Document Title",
            summary="A summary.",
            tags=["PDF"],
            chunk_count=3,
        )
        with patch.object(tool.service, "get_document_summary", return_value=summary):
            result = await tool.execute(doc_id="doc_abc123")

        assert result.is_error is False
        assert result.content[0].type == "text"
        assert "Test Document Title" in result.content[0].text
        assert result.structured_content is not None
        assert result.structured_content["doc_id"] == "doc_abc123"

    @pytest.mark.asyncio
    async def test_execute_document_not_found(self) -> None:
        tool = self._make_tool()
        from src.core.service.document_summary import DocumentNotFoundError

        with patch.object(
            tool.service,
            "get_document_summary",
            side_effect=DocumentNotFoundError("missing", "knowledge_hub"),
        ):
            result = await tool.execute(doc_id="missing")

        assert result.is_error is True
        assert "Not Found" in result.content[0].text

    @pytest.mark.asyncio
    async def test_execute_error(self) -> None:
        tool = self._make_tool()
        with patch.object(
            tool.service,
            "get_document_summary",
            side_effect=RuntimeError("boom"),
        ):
            result = await tool.execute(doc_id="doc")

        assert result.is_error is True
        assert "An error occurred" in result.content[0].text


class TestQueryKnowledgeHubToolAdapter:
    """query_knowledge_hub delegates to KnowledgeQueryService."""

    def _make_tool(self) -> QueryKnowledgeHubTool:
        return QueryKnowledgeHubTool()

    @pytest.mark.asyncio
    async def test_empty_query_raises_value_error(self) -> None:
        tool = self._make_tool()
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await tool.execute(query="   ")

    @pytest.mark.asyncio
    async def test_execute_success_maps_structured_content(self) -> None:
        tool = self._make_tool()
        response = MCPToolResponse(
            content="Answer with [1]",
            citations=[],
            metadata={"query": "q", "collection": "default"},
            is_empty=False,
        )
        with patch.object(tool.service, "query", return_value=response) as mock_query:
            result = await tool.execute(query="q", top_k=5)

        mock_query.assert_called_once_with(query="q", top_k=5, collection=None)
        assert result.is_error is False
        assert result.content == response.to_mcp_content()
        assert result.structured_content == {
            "citations": [],
            "metadata": {"query": "q", "collection": "default"},
            "isEmpty": False,
        }

    @pytest.mark.asyncio
    async def test_execute_maps_error_response(self) -> None:
        tool = self._make_tool()
        response = MCPToolResponse(
            content="## 查询失败\n\n...",
            citations=[],
            metadata={"query": "q", "error": "boom"},
            is_empty=True,
        )
        with patch.object(tool.service, "query", return_value=response):
            result = await tool.execute(query="q")

        assert result.is_error is True
