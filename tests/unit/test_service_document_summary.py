"""Unit tests for the document summary business service.

This module tests :class:`DocumentSummaryService` and the ``format_document_*``
helpers in ``src.core.service.document_summary``.
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest

from src.core.service.document_summary import (
    DocumentNotFoundError,
    DocumentSummary,
    DocumentSummaryService,
    GetDocumentSummaryConfig,
    format_document_error,
    format_document_summary,
)

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_settings() -> Mock:
    """Create mock settings object."""
    settings = Mock()
    settings.vector_store = Mock()
    settings.vector_store.persist_directory = "data/db/chroma"
    settings.vector_store.collection_name = "test_collection"
    return settings


@pytest.fixture
def mock_config() -> GetDocumentSummaryConfig:
    """Create test configuration."""
    return GetDocumentSummaryConfig(
        persist_directory="./test_data/chroma",
        default_collection="test_collection",
        summary_max_length=200,
    )


@pytest.fixture
def service_with_mock_settings(mock_settings: Mock) -> DocumentSummaryService:
    """Create DocumentSummaryService with mock settings."""
    return DocumentSummaryService(settings=mock_settings)


@pytest.fixture
def service_with_config(mock_config: GetDocumentSummaryConfig) -> DocumentSummaryService:
    """Create DocumentSummaryService with explicit config."""
    return DocumentSummaryService(config=mock_config)


@pytest.fixture
def sample_chunks() -> list[dict[str, Any]]:
    """Create sample chunk data for testing."""
    return [
        {
            'id': 'doc_abc123_0000_hash1',
            'text': '# Introduction\n\nThis is the first paragraph of the document.',
            'metadata': {
                'source_ref': 'doc_abc123',
                'source_path': '/docs/test.pdf',
                'chunk_index': 0,
                'title': 'Test Document Title',
                'doc_type': 'pdf',
            }
        },
        {
            'id': 'doc_abc123_0001_hash2',
            'text': 'This is the second chunk with more content.',
            'metadata': {
                'source_ref': 'doc_abc123',
                'source_path': '/docs/test.pdf',
                'chunk_index': 1,
                'doc_type': 'pdf',
            }
        },
        {
            'id': 'doc_abc123_0002_hash3',
            'text': 'Final chunk with conclusion.',
            'metadata': {
                'source_ref': 'doc_abc123',
                'source_path': '/docs/test.pdf',
                'chunk_index': 2,
                'doc_type': 'pdf',
            }
        },
    ]


@pytest.fixture
def mock_collection(sample_chunks: list[dict[str, Any]]) -> Mock:
    """Create mock ChromaDB collection."""
    collection = Mock()

    # Setup get() method for source_ref search
    def mock_get(where=None, include=None):
        if where and where.get('source_ref') == 'doc_abc123':
            return {
                'ids': [c['id'] for c in sample_chunks],
                'documents': [c['text'] for c in sample_chunks],
                'metadatas': [c['metadata'] for c in sample_chunks],
            }
        return {'ids': [], 'documents': [], 'metadatas': []}

    collection.get = Mock(side_effect=mock_get)
    return collection


# =============================================================================
# Test: DocumentSummary Dataclass
# =============================================================================

class TestDocumentSummary:
    """Tests for DocumentSummary dataclass."""

    def test_document_summary_creation(self):
        """Test basic DocumentSummary creation."""
        summary = DocumentSummary(
            doc_id="doc_123",
            title="Test Title",
            summary="This is a summary.",
        )
        assert summary.doc_id == "doc_123"
        assert summary.title == "Test Title"
        assert summary.summary == "This is a summary."

    def test_document_summary_defaults(self):
        """Test DocumentSummary default values."""
        summary = DocumentSummary(
            doc_id="doc_123",
            title="Title",
            summary="Summary",
        )
        assert summary.tags == []
        assert summary.source_path is None
        assert summary.chunk_count == 0
        assert summary.metadata == {}

    def test_document_summary_with_all_fields(self):
        """Test DocumentSummary with all fields populated."""
        summary = DocumentSummary(
            doc_id="doc_abc123",
            title="Full Document",
            summary="Complete summary.",
            tags=["python", "testing"],
            source_path="/docs/test.pdf",
            chunk_count=5,
            metadata={"author": "Test Author"},
        )
        assert summary.doc_id == "doc_abc123"
        assert summary.tags == ["python", "testing"]
        assert summary.source_path == "/docs/test.pdf"
        assert summary.chunk_count == 5
        assert summary.metadata["author"] == "Test Author"

    def test_document_summary_to_dict(self):
        """Test DocumentSummary.to_dict() method."""
        summary = DocumentSummary(
            doc_id="doc_123",
            title="Test",
            summary="Summary",
            tags=["tag1"],
            source_path="/path/file.pdf",
            chunk_count=3,
            metadata={"key": "value"},
        )
        result = summary.to_dict()

        assert result["doc_id"] == "doc_123"
        assert result["title"] == "Test"
        assert result["summary"] == "Summary"
        assert result["tags"] == ["tag1"]
        assert result["source_path"] == "/path/file.pdf"
        assert result["chunk_count"] == 3
        assert result["metadata"] == {"key": "value"}


# =============================================================================
# Test: DocumentNotFoundError
# =============================================================================

class TestDocumentNotFoundError:
    """Tests for DocumentNotFoundError exception."""

    def test_error_with_doc_id_only(self):
        """Test error message with just doc_id."""
        error = DocumentNotFoundError("doc_123")
        assert "doc_123" in str(error)
        assert "not found" in str(error)

    def test_error_with_collection(self):
        """Test error message with doc_id and collection."""
        error = DocumentNotFoundError("doc_123", "my_collection")
        assert "doc_123" in str(error)
        assert "my_collection" in str(error)

    def test_error_attributes(self):
        """Test error has correct attributes."""
        error = DocumentNotFoundError("doc_xyz", "test_coll")
        assert error.doc_id == "doc_xyz"
        assert error.collection == "test_coll"


# =============================================================================
# Test: GetDocumentSummaryConfig
# =============================================================================

class TestGetDocumentSummaryConfig:
    """Tests for GetDocumentSummaryConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = GetDocumentSummaryConfig()
        assert config.persist_directory == "data/db/chroma"
        assert config.default_collection == "knowledge_hub"
        assert config.summary_max_length == 500

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = GetDocumentSummaryConfig(
            persist_directory="/custom/path",
            default_collection="custom_collection",
            summary_max_length=200,
        )
        assert config.persist_directory == "/custom/path"
        assert config.default_collection == "custom_collection"
        assert config.summary_max_length == 200


# =============================================================================
# Test: DocumentSummaryService Initialization
# =============================================================================

class TestDocumentSummaryServiceInit:
    """Tests for DocumentSummaryService initialization."""

    def test_init_with_settings(self, mock_settings: Mock):
        """Test initialization with settings."""
        service = DocumentSummaryService(settings=mock_settings)
        assert service._settings is mock_settings
        assert service._config is None

    def test_init_with_config(self, mock_config: GetDocumentSummaryConfig):
        """Test initialization with explicit config."""
        service = DocumentSummaryService(config=mock_config)
        assert service._config is mock_config
        assert service._settings is None

    def test_init_with_both(self, mock_settings: Mock, mock_config: GetDocumentSummaryConfig):
        """Test initialization with both settings and config."""
        service = DocumentSummaryService(settings=mock_settings, config=mock_config)
        assert service._settings is mock_settings
        assert service._config is mock_config

    def test_init_with_defaults(self):
        """Test initialization with no parameters."""
        service = DocumentSummaryService()
        assert service._settings is None
        assert service._config is None


# =============================================================================
# Test: Settings Property
# =============================================================================

class TestSettingsProperty:
    """Tests for the settings property lazy loading."""

    def test_settings_returns_provided_settings(self, mock_settings: Mock):
        """Test that provided settings are returned."""
        service = DocumentSummaryService(settings=mock_settings)
        assert service.settings is mock_settings

    @patch("src.core.settings.load_settings")
    def test_settings_lazy_loads_when_none(self, mock_load: Mock):
        """Test that settings are loaded lazily when not provided."""
        mock_loaded = Mock()
        mock_load.return_value = mock_loaded

        service = DocumentSummaryService()
        result = service.settings

        mock_load.assert_called_once()
        assert result is mock_loaded


# =============================================================================
# Test: Config Property
# =============================================================================

class TestConfigProperty:
    """Tests for the config property."""

    def test_config_returns_provided_config(self, mock_config: GetDocumentSummaryConfig):
        """Test that provided config is returned."""
        service = DocumentSummaryService(config=mock_config)
        assert service.config is mock_config

    def test_config_derived_from_settings(self, mock_settings: Mock):
        """Test that config is derived from settings when not provided."""
        mock_settings.vector_store.persist_directory = "/settings/path"
        mock_settings.vector_store.collection_name = "settings_collection"

        service = DocumentSummaryService(settings=mock_settings)
        config = service.config

        assert config.persist_directory == "/settings/path"
        assert config.default_collection == "settings_collection"

    def test_config_uses_defaults_on_missing_attributes(self):
        """Test that config uses defaults when settings attributes are missing."""
        settings = Mock()
        settings.vector_store = None  # Will cause AttributeError

        service = DocumentSummaryService(settings=settings)
        config = service.config

        assert config.persist_directory == "data/db/chroma"
        assert config.default_collection == "knowledge_hub"


# =============================================================================
# Test: Title Extraction
# =============================================================================

class TestTitleExtraction:
    """Tests for _extract_title method."""

    def test_title_from_metadata(self, service_with_config: DocumentSummaryService):
        """Test title extraction from metadata."""
        metadata = {"title": "Explicit Title"}
        result = service_with_config._extract_title(metadata, "")
        assert result == "Explicit Title"

    def test_title_from_markdown_heading(self, service_with_config: DocumentSummaryService):
        """Test title extraction from markdown heading."""
        metadata = {}
        text = "# Document Title\n\nContent here"
        result = service_with_config._extract_title(metadata, text)
        assert result == "Document Title"

    def test_title_from_source_path(self, service_with_config: DocumentSummaryService):
        """Test title extraction from source_path."""
        metadata = {"source_path": "/docs/my_test_document.pdf"}
        result = service_with_config._extract_title(metadata, "")
        assert result == "My Test Document"

    def test_title_from_source_key(self, service_with_config: DocumentSummaryService):
        """Test title extraction from 'source' key."""
        metadata = {"source": "/docs/another-document.pdf"}
        result = service_with_config._extract_title(metadata, "")
        assert result == "Another Document"

    def test_title_default_untitled(self, service_with_config: DocumentSummaryService):
        """Test default title when nothing available."""
        result = service_with_config._extract_title({}, "")
        assert result == "Untitled Document"

    def test_title_priority_metadata_over_heading(self, service_with_config: DocumentSummaryService):
        """Test that metadata title has priority over markdown heading."""
        metadata = {"title": "Metadata Title"}
        text = "# Markdown Title\n\nContent"
        result = service_with_config._extract_title(metadata, text)
        assert result == "Metadata Title"


# =============================================================================
# Test: Summary Extraction
# =============================================================================

class TestSummaryExtraction:
    """Tests for _extract_summary method."""

    def test_summary_from_metadata(self, service_with_config: DocumentSummaryService):
        """Test summary extraction from metadata."""
        chunks = [{'metadata': {'summary': 'Explicit summary'}, 'text': 'Content'}]
        result = service_with_config._extract_summary(chunks)
        assert result == "Explicit summary"

    def test_summary_from_first_chunk_text(self, service_with_config: DocumentSummaryService):
        """Test summary extraction from first chunk content."""
        chunks = [{'metadata': {}, 'text': 'This is the document content.'}]
        result = service_with_config._extract_summary(chunks)
        assert "This is the document content" in result

    def test_summary_skips_headers(self, service_with_config: DocumentSummaryService):
        """Test that summary skips markdown headers."""
        chunks = [{'metadata': {}, 'text': '# Title\n\n## Section\n\nActual content here.'}]
        result = service_with_config._extract_summary(chunks)
        assert "Actual content here" in result
        assert "# Title" not in result

    def test_summary_truncation(self, service_with_config: DocumentSummaryService):
        """Test summary is truncated to max length."""
        long_text = "A" * 1000
        chunks = [{'metadata': {}, 'text': long_text}]
        result = service_with_config._extract_summary(chunks)
        assert len(result) <= service_with_config.config.summary_max_length
        assert result.endswith("...")

    def test_summary_empty_chunks(self, service_with_config: DocumentSummaryService):
        """Test summary with empty chunks list."""
        result = service_with_config._extract_summary([])
        assert result == "No summary available."

    def test_summary_no_content(self, service_with_config: DocumentSummaryService):
        """Test summary when chunk has no text."""
        chunks = [{'metadata': {}, 'text': ''}]
        result = service_with_config._extract_summary(chunks)
        assert "No" in result or "available" in result


# =============================================================================
# Test: Tags Extraction
# =============================================================================

class TestTagsExtraction:
    """Tests for _extract_tags method."""

    def test_tags_from_list(self, service_with_config: DocumentSummaryService):
        """Test tags extraction from list."""
        metadata = {'tags': ['python', 'testing', 'docs']}
        result = service_with_config._extract_tags(metadata)
        assert 'python' in result
        assert 'testing' in result
        assert 'docs' in result

    def test_tags_from_comma_string(self, service_with_config: DocumentSummaryService):
        """Test tags extraction from comma-separated string."""
        metadata = {'tags': 'python, testing, docs'}
        result = service_with_config._extract_tags(metadata)
        assert 'python' in result
        assert 'testing' in result
        assert 'docs' in result

    def test_tags_includes_doc_type(self, service_with_config: DocumentSummaryService):
        """Test that doc_type is added as a tag."""
        metadata = {'doc_type': 'pdf'}
        result = service_with_config._extract_tags(metadata)
        assert 'PDF' in result

    def test_tags_no_duplicate_doc_type(self, service_with_config: DocumentSummaryService):
        """Test that doc_type is not duplicated if already in tags."""
        metadata = {'tags': ['PDF', 'other'], 'doc_type': 'pdf'}
        result = service_with_config._extract_tags(metadata)
        assert result.count('PDF') == 1

    def test_tags_empty_metadata(self, service_with_config: DocumentSummaryService):
        """Test tags extraction with no tag-related metadata."""
        result = service_with_config._extract_tags({})
        assert result == []


# =============================================================================
# Test: Metadata Filtering
# =============================================================================

class TestMetadataFiltering:
    """Tests for _filter_metadata method."""

    def test_filter_removes_internal_fields(self, service_with_config: DocumentSummaryService):
        """Test that internal fields are removed."""
        metadata = {
            'source_ref': 'doc_123',
            'chunk_index': 0,
            'start_offset': 0,
            'end_offset': 100,
            'author': 'Test Author',
        }
        result = service_with_config._filter_metadata(metadata)

        assert 'source_ref' not in result
        assert 'chunk_index' not in result
        assert 'author' in result

    def test_filter_removes_underscore_prefix(self, service_with_config: DocumentSummaryService):
        """Test that underscore-prefixed fields are removed."""
        metadata = {
            '_placeholder': 'true',
            '_internal': 'value',
            'public_field': 'value',
        }
        result = service_with_config._filter_metadata(metadata)

        assert '_placeholder' not in result
        assert '_internal' not in result
        assert 'public_field' in result

    def test_filter_keeps_user_fields(self, service_with_config: DocumentSummaryService):
        """Test that user-relevant fields are kept."""
        metadata = {
            'author': 'John Doe',
            'created_date': '2025-01-01',
            'page_count': 10,
        }
        result = service_with_config._filter_metadata(metadata)

        assert result['author'] == 'John Doe'
        assert result['created_date'] == '2025-01-01'
        assert result['page_count'] == 10


# =============================================================================
# Test: ChromaDB Integration
# =============================================================================

class TestChromaDBIntegration:
    """Tests for ChromaDB client and collection methods."""

    def test_get_chroma_client_success(self, service_with_config: DocumentSummaryService):
        """Test successful ChromaDB client creation with mocked import."""
        mock_client = Mock()
        mock_chromadb = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client

        with patch.dict('sys.modules', {'chromadb': mock_chromadb, 'chromadb.config': Mock()}):
            # Reset client cache
            service_with_config._chroma_client = None
            result = service_with_config._get_chroma_client()

        assert result is mock_client
        mock_chromadb.PersistentClient.assert_called_once()

    def test_get_chroma_client_import_error(self, service_with_config: DocumentSummaryService):
        """Test ImportError when chromadb not installed."""
        # Reset cached client
        service_with_config._chroma_client = None

        with patch.dict('sys.modules', {'chromadb': None}):
            with pytest.raises(ImportError) as exc_info:
                service_with_config._get_chroma_client()

            assert "chromadb" in str(exc_info.value)

    def test_get_collection_success(self, service_with_config: DocumentSummaryService):
        """Test successful collection retrieval."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_collection.return_value = mock_collection

        # Mock _get_chroma_client to return our mock client
        service_with_config._get_chroma_client = Mock(return_value=mock_client)

        result = service_with_config._get_collection("test_collection")

        assert result is mock_collection
        mock_client.get_collection.assert_called_once_with(name="test_collection")

    def test_get_collection_not_found(self, service_with_config: DocumentSummaryService):
        """Test error when collection doesn't exist."""
        mock_client = Mock()
        mock_client.get_collection.side_effect = Exception("Collection not found")

        # Mock _get_chroma_client to return our mock client
        service_with_config._get_chroma_client = Mock(return_value=mock_client)

        with pytest.raises(ValueError) as exc_info:
            service_with_config._get_collection("nonexistent")

        assert "nonexistent" in str(exc_info.value)


# =============================================================================
# Test: Document Chunk Finding
# =============================================================================

class TestFindDocumentChunks:
    """Tests for _find_document_chunks method."""

    def test_find_chunks_by_source_ref(
        self,
        service_with_config: DocumentSummaryService,
        sample_chunks: list[dict[str, Any]],
    ):
        """Test finding chunks by source_ref metadata."""
        mock_collection = Mock()
        mock_collection.get.return_value = {
            'ids': [c['id'] for c in sample_chunks],
            'documents': [c['text'] for c in sample_chunks],
            'metadatas': [c['metadata'] for c in sample_chunks],
        }

        # Mock the _get_collection method
        service_with_config._get_collection = Mock(return_value=mock_collection)

        result = service_with_config._find_document_chunks("doc_abc123")

        assert len(result) == 3
        mock_collection.get.assert_called()

    def test_find_chunks_by_id_prefix(
        self,
        service_with_config: DocumentSummaryService,
        sample_chunks: list[dict[str, Any]],
    ):
        """Test finding chunks by ID prefix when source_ref search fails."""
        mock_collection = Mock()

        # First call (source_ref search) returns empty
        # Second call (get all) returns all chunks
        mock_collection.get.side_effect = [
            {'ids': [], 'documents': [], 'metadatas': []},
            {
                'ids': [c['id'] for c in sample_chunks],
                'documents': [c['text'] for c in sample_chunks],
                'metadatas': [c['metadata'] for c in sample_chunks],
            },
        ]

        # Mock the _get_collection method
        service_with_config._get_collection = Mock(return_value=mock_collection)

        result = service_with_config._find_document_chunks("doc_abc123")

        assert len(result) == 3

    def test_find_chunks_not_found(
        self,
        service_with_config: DocumentSummaryService,
    ):
        """Test empty result when no chunks found."""
        mock_collection = Mock()
        mock_collection.get.return_value = {'ids': [], 'documents': [], 'metadatas': []}

        # Mock the _get_collection method
        service_with_config._get_collection = Mock(return_value=mock_collection)

        result = service_with_config._find_document_chunks("nonexistent_doc")

        assert result == []


# =============================================================================
# Test: get_document_summary Method
# =============================================================================

class TestGetDocumentSummary:
    """Tests for get_document_summary method."""

    def test_get_summary_success(
        self,
        service_with_config: DocumentSummaryService,
        sample_chunks: list[dict[str, Any]],
    ):
        """Test successful document summary retrieval."""
        mock_collection = Mock()
        mock_collection.get.return_value = {
            'ids': [c['id'] for c in sample_chunks],
            'documents': [c['text'] for c in sample_chunks],
            'metadatas': [c['metadata'] for c in sample_chunks],
        }

        # Mock the _get_collection method
        service_with_config._get_collection = Mock(return_value=mock_collection)

        result = service_with_config.get_document_summary("doc_abc123")

        assert isinstance(result, DocumentSummary)
        assert result.doc_id == "doc_abc123"
        assert result.title == "Test Document Title"
        assert result.chunk_count == 3
        assert result.source_path == "/docs/test.pdf"
        assert "PDF" in result.tags

    def test_get_summary_not_found(
        self,
        service_with_config: DocumentSummaryService,
    ):
        """Test DocumentNotFoundError when document doesn't exist."""
        mock_collection = Mock()
        mock_collection.get.return_value = {'ids': [], 'documents': [], 'metadatas': []}

        # Mock the _get_collection method
        service_with_config._get_collection = Mock(return_value=mock_collection)

        with pytest.raises(DocumentNotFoundError) as exc_info:
            service_with_config.get_document_summary("nonexistent_doc")

        assert exc_info.value.doc_id == "nonexistent_doc"

    def test_get_summary_chunks_sorted_by_index(
        self,
        service_with_config: DocumentSummaryService,
    ):
        """Test that chunks are sorted by chunk_index."""
        # Provide chunks in wrong order
        unsorted_chunks = [
            {'id': 'c2', 'text': 'Second', 'metadata': {'chunk_index': 1, 'source_path': '/doc.pdf'}},
            {'id': 'c0', 'text': '# Title\nFirst', 'metadata': {'chunk_index': 0, 'source_path': '/doc.pdf'}},
            {'id': 'c1', 'text': 'Middle', 'metadata': {'chunk_index': 2, 'source_path': '/doc.pdf'}},
        ]

        mock_collection = Mock()
        mock_collection.get.return_value = {
            'ids': [c['id'] for c in unsorted_chunks],
            'documents': [c['text'] for c in unsorted_chunks],
            'metadatas': [c['metadata'] for c in unsorted_chunks],
        }

        # Mock the _get_collection method
        service_with_config._get_collection = Mock(return_value=mock_collection)

        result = service_with_config.get_document_summary("doc_test")

        # Title should be extracted from first chunk (chunk_index=0)
        assert result.title == "Title"


# =============================================================================
# Test: Response Formatting
# =============================================================================

class TestFormatResponse:
    """Tests for format_document_summary."""

    def test_format_response_includes_title(self, service_with_config: DocumentSummaryService):
        """Test formatted response includes title."""
        summary = DocumentSummary(
            doc_id="doc_123",
            title="Test Title",
            summary="Test summary",
        )
        result = format_document_summary(summary)

        assert "Test Title" in result

    def test_format_response_includes_doc_id(self, service_with_config: DocumentSummaryService):
        """Test formatted response includes doc_id."""
        summary = DocumentSummary(
            doc_id="doc_abc123",
            title="Title",
            summary="Summary",
        )
        result = format_document_summary(summary)

        assert "doc_abc123" in result

    def test_format_response_includes_tags(self, service_with_config: DocumentSummaryService):
        """Test formatted response includes tags."""
        summary = DocumentSummary(
            doc_id="doc_123",
            title="Title",
            summary="Summary",
            tags=["python", "testing"],
        )
        result = format_document_summary(summary)

        assert "python" in result
        assert "testing" in result

    def test_format_response_includes_metadata(self, service_with_config: DocumentSummaryService):
        """Test formatted response includes additional metadata."""
        summary = DocumentSummary(
            doc_id="doc_123",
            title="Title",
            summary="Summary",
            metadata={"author": "John Doe"},
        )
        result = format_document_summary(summary)

        assert "author" in result
        assert "John Doe" in result


# =============================================================================
# Test: Error Formatting
# =============================================================================

class TestFormatError:
    """Tests for format_document_error."""

    def test_format_document_not_found_error(self, service_with_config: DocumentSummaryService):
        """Test formatting DocumentNotFoundError."""
        error = DocumentNotFoundError("doc_123", "test_collection")
        result = format_document_error(error)

        assert "Not Found" in result
        assert "doc_123" in result

    def test_format_value_error(self, service_with_config: DocumentSummaryService):
        """Test formatting ValueError."""
        error = ValueError("Invalid parameter")
        result = format_document_error(error)

        assert "Invalid" in result

    def test_format_generic_error(self, service_with_config: DocumentSummaryService):
        """Test formatting generic exception."""
        error = RuntimeError("Something went wrong")
        result = format_document_error(error)

        assert "Error" in result
        assert "Something went wrong" in result
