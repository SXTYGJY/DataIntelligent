"""Unit tests for the collection listing business service.

This module tests :class:`CollectionService` and :func:`format_collections_response`
in ``src.core.service.collections``.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.core.service.collections import (
    CollectionInfo,
    CollectionService,
    ListCollectionsConfig,
    format_collections_response,
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
    return settings


@pytest.fixture
def mock_config() -> ListCollectionsConfig:
    """Create test configuration."""
    return ListCollectionsConfig(
        persist_directory="./test_data/chroma",
        include_stats_default=True,
    )


@pytest.fixture
def service_with_mock_settings(mock_settings: Mock) -> CollectionService:
    """Create CollectionService with mock settings."""
    return CollectionService(settings=mock_settings)


@pytest.fixture
def service_with_config(mock_config: ListCollectionsConfig) -> CollectionService:
    """Create CollectionService with explicit config."""
    return CollectionService(config=mock_config)


@pytest.fixture
def sample_collections() -> list[CollectionInfo]:
    """Create sample collection info list."""
    return [
        CollectionInfo(
            name="knowledge_hub",
            count=150,
            metadata={"hnsw:space": "cosine"},
        ),
        CollectionInfo(
            name="documents",
            count=75,
            metadata={"description": "Main documents"},
        ),
        CollectionInfo(
            name="test_collection",
            count=0,
            metadata=None,
        ),
    ]


# =============================================================================
# CollectionInfo Tests
# =============================================================================

class TestCollectionInfo:
    """Tests for CollectionInfo dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic CollectionInfo creation."""
        info = CollectionInfo(name="test_coll")

        assert info.name == "test_coll"
        assert info.count is None
        assert info.metadata is None

    def test_creation_with_all_fields(self) -> None:
        """Test CollectionInfo creation with all fields."""
        info = CollectionInfo(
            name="documents",
            count=100,
            metadata={"type": "pdf"},
        )

        assert info.name == "documents"
        assert info.count == 100
        assert info.metadata == {"type": "pdf"}

    def test_to_dict_minimal(self) -> None:
        """Test to_dict with minimal fields."""
        info = CollectionInfo(name="test")
        result = info.to_dict()

        assert result == {"name": "test"}
        assert "count" not in result
        assert "metadata" not in result

    def test_to_dict_with_count(self) -> None:
        """Test to_dict with count."""
        info = CollectionInfo(name="test", count=50)
        result = info.to_dict()

        assert result == {"name": "test", "count": 50}

    def test_to_dict_full(self) -> None:
        """Test to_dict with all fields."""
        info = CollectionInfo(
            name="docs",
            count=25,
            metadata={"source": "local"},
        )
        result = info.to_dict()

        assert result == {
            "name": "docs",
            "count": 25,
            "metadata": {"source": "local"},
        }

    def test_to_dict_empty_metadata(self) -> None:
        """Test to_dict with empty metadata dict."""
        info = CollectionInfo(name="test", metadata={})
        result = info.to_dict()

        # Empty metadata should not be included
        assert result == {"name": "test"}


# =============================================================================
# ListCollectionsConfig Tests
# =============================================================================

class TestListCollectionsConfig:
    """Tests for ListCollectionsConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ListCollectionsConfig()

        assert config.persist_directory == "data/db/chroma"
        assert config.include_stats_default is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ListCollectionsConfig(
            persist_directory="/custom/path",
            include_stats_default=False,
        )

        assert config.persist_directory == "/custom/path"
        assert config.include_stats_default is False


# =============================================================================
# CollectionService Initialization Tests
# =============================================================================

class TestCollectionServiceInit:
    """Tests for CollectionService initialization."""

    def test_init_with_settings(self, mock_settings: Mock) -> None:
        """Test initialization with settings."""
        service = CollectionService(settings=mock_settings)

        assert service._settings == mock_settings
        assert service._config is None

    def test_init_with_config(self, mock_config: ListCollectionsConfig) -> None:
        """Test initialization with explicit config."""
        service = CollectionService(config=mock_config)

        assert service._settings is None
        assert service._config == mock_config

    def test_init_no_args(self) -> None:
        """Test initialization without arguments."""
        service = CollectionService()

        assert service._settings is None
        assert service._config is None

    def test_settings_lazy_load(self) -> None:
        """Test that settings are loaded lazily."""
        service = CollectionService()

        with patch('src.core.settings.load_settings') as mock_load:
            mock_settings = Mock()
            mock_load.return_value = mock_settings

            # Access settings property
            result = service.settings

            mock_load.assert_called_once()
            assert result == mock_settings

    def test_config_derived_from_settings(self, mock_settings: Mock) -> None:
        """Test that config is derived from settings."""
        service = CollectionService(settings=mock_settings)

        config = service.config

        assert config.persist_directory == "data/db/chroma"

    def test_config_fallback_no_vector_store(self) -> None:
        """Test config fallback when vector_store config missing."""
        settings = Mock(spec=[])  # No vector_store attribute
        service = CollectionService(settings=settings)

        config = service.config

        assert config.persist_directory == "data/db/chroma"


# =============================================================================
# CollectionService ChromaDB Client Tests
# =============================================================================

class TestCollectionServiceChromaClient:
    """Tests for ChromaDB client management."""

    def test_get_chroma_client_chromadb_not_installed(
        self,
        service_with_config: CollectionService
    ) -> None:
        """Test error when chromadb is not installed."""
        with patch.dict('sys.modules', {'chromadb': None}):
            with patch('builtins.__import__', side_effect=ImportError("No chromadb")):
                with pytest.raises(ImportError) as exc_info:
                    service_with_config._get_chroma_client()

                assert "chromadb package is required" in str(exc_info.value)

    @patch('chromadb.PersistentClient')
    @patch('chromadb.config.Settings')
    def test_get_chroma_client_success(
        self,
        mock_chroma_settings: Mock,
        mock_client_class: Mock,
        service_with_config: CollectionService,
        tmp_path: Path,
    ) -> None:
        """Test successful ChromaDB client creation."""
        # Update config to use temp path
        service_with_config._config = ListCollectionsConfig(
            persist_directory=str(tmp_path / "chroma")
        )

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        result = service_with_config._get_chroma_client()

        assert result == mock_client
        mock_client_class.assert_called_once()

    @patch('chromadb.PersistentClient')
    @patch('chromadb.config.Settings')
    def test_get_chroma_client_creates_directory(
        self,
        mock_chroma_settings: Mock,
        mock_client_class: Mock,
        service_with_config: CollectionService,
        tmp_path: Path,
    ) -> None:
        """Test that missing directory is created."""
        new_path = tmp_path / "new_chroma_dir"
        service_with_config._config = ListCollectionsConfig(
            persist_directory=str(new_path)
        )

        mock_client_class.return_value = Mock()

        service_with_config._get_chroma_client()

        assert new_path.exists()

    @patch('chromadb.PersistentClient')
    @patch('chromadb.config.Settings')
    def test_get_chroma_client_init_failure(
        self,
        mock_chroma_settings: Mock,
        mock_client_class: Mock,
        service_with_config: CollectionService,
        tmp_path: Path,
    ) -> None:
        """Test error handling when client init fails."""
        service_with_config._config = ListCollectionsConfig(
            persist_directory=str(tmp_path)
        )

        mock_client_class.side_effect = Exception("Connection failed")

        with pytest.raises(RuntimeError) as exc_info:
            service_with_config._get_chroma_client()

        assert "Failed to initialize ChromaDB client" in str(exc_info.value)


# =============================================================================
# CollectionService list_collections Method Tests
# =============================================================================

class TestListCollectionsMethod:
    """Tests for list_collections method."""

    def test_list_collections_empty(
        self,
        service_with_config: CollectionService
    ) -> None:
        """Test listing when no collections exist."""
        mock_client = Mock()
        mock_client.list_collections.return_value = []

        with patch.object(service_with_config, '_get_chroma_client', return_value=mock_client):
            result = service_with_config.list_collections()

        assert result == []

    def test_list_collections_with_stats(
        self,
        service_with_config: CollectionService
    ) -> None:
        """Test listing collections with statistics."""
        mock_coll1 = Mock()
        mock_coll1.name = "collection1"
        mock_coll1.metadata = {"type": "docs"}
        mock_coll1.count.return_value = 100

        mock_coll2 = Mock()
        mock_coll2.name = "collection2"
        mock_coll2.metadata = {}
        mock_coll2.count.return_value = 50

        mock_client = Mock()
        mock_client.list_collections.return_value = [mock_coll1, mock_coll2]

        with patch.object(service_with_config, '_get_chroma_client', return_value=mock_client):
            result = service_with_config.list_collections(include_stats=True)

        assert len(result) == 2
        assert result[0].name == "collection1"
        assert result[0].count == 100
        assert result[0].metadata == {"type": "docs"}
        assert result[1].name == "collection2"
        assert result[1].count == 50

    def test_list_collections_without_stats(
        self,
        service_with_config: CollectionService
    ) -> None:
        """Test listing collections without statistics."""
        mock_coll = Mock()
        mock_coll.name = "test"
        mock_coll.metadata = None

        mock_client = Mock()
        mock_client.list_collections.return_value = [mock_coll]

        with patch.object(service_with_config, '_get_chroma_client', return_value=mock_client):
            result = service_with_config.list_collections(include_stats=False)

        assert len(result) == 1
        assert result[0].name == "test"
        assert result[0].count is None  # Not fetched
        mock_coll.count.assert_not_called()

    def test_list_collections_count_error_graceful(
        self,
        service_with_config: CollectionService
    ) -> None:
        """Test graceful handling when count() fails."""
        mock_coll = Mock()
        mock_coll.name = "problematic"
        mock_coll.metadata = {}
        mock_coll.count.side_effect = Exception("Count failed")

        mock_client = Mock()
        mock_client.list_collections.return_value = [mock_coll]

        with patch.object(service_with_config, '_get_chroma_client', return_value=mock_client):
            result = service_with_config.list_collections(include_stats=True)

        # Should still return collection, but with None count
        assert len(result) == 1
        assert result[0].name == "problematic"
        assert result[0].count is None

    def test_list_collections_client_error(
        self,
        service_with_config: CollectionService
    ) -> None:
        """Test error handling when client fails."""
        with patch.object(
            service_with_config,
            '_get_chroma_client',
            side_effect=RuntimeError("Client error")
        ):
            result = service_with_config.list_collections()

        assert result == []

    def test_list_collections_list_error(
        self,
        service_with_config: CollectionService
    ) -> None:
        """Test error handling when list_collections fails."""
        mock_client = Mock()
        mock_client.list_collections.side_effect = Exception("List failed")

        with patch.object(service_with_config, '_get_chroma_client', return_value=mock_client):
            result = service_with_config.list_collections()

        assert result == []


# =============================================================================
# CollectionService response formatting
# =============================================================================

class TestFormatResponse:
    """Tests for format_collections_response."""

    def test_format_empty_collections(
        self,
        service_with_config: CollectionService
    ) -> None:
        """Test formatting empty collection list."""
        result = format_collections_response([])

        assert result == "No collections found in the knowledge base."

    def test_format_single_collection(
        self,
        service_with_config: CollectionService
    ) -> None:
        """Test formatting single collection."""
        collections = [CollectionInfo(name="docs", count=50)]
        result = format_collections_response(collections)

        assert "## Available Collections (1 total)" in result
        assert "1. **docs** - 50 documents" in result

    def test_format_multiple_collections(
        self,
        service_with_config: CollectionService,
        sample_collections: list[CollectionInfo]
    ) -> None:
        """Test formatting multiple collections."""
        result = format_collections_response(sample_collections)

        assert "## Available Collections (3 total)" in result
        assert "1. **knowledge_hub** - 150 documents" in result
        assert "2. **documents** - 75 documents" in result
        assert "3. **test_collection** - 0 documents" in result

    def test_format_with_metadata(
        self,
        service_with_config: CollectionService
    ) -> None:
        """Test formatting with user metadata."""
        collections = [
            CollectionInfo(
                name="research",
                count=30,
                metadata={"category": "papers", "year": 2024},
            )
        ]
        result = format_collections_response(collections)

        assert "**research**" in result
        assert "30 documents" in result
        assert "category=papers" in result
        assert "year=2024" in result

    def test_format_filters_internal_metadata(
        self,
        service_with_config: CollectionService
    ) -> None:
        """Test that internal metadata is filtered out."""
        collections = [
            CollectionInfo(
                name="test",
                count=10,
                metadata={
                    "hnsw:space": "cosine",
                    "_internal": "value",
                    "user_field": "visible",
                },
            )
        ]
        result = format_collections_response(collections)

        # Internal metadata should be filtered
        assert "hnsw:space" not in result
        assert "_internal" not in result
        # User metadata should be visible
        assert "user_field=visible" in result

    def test_format_without_count(
        self,
        service_with_config: CollectionService
    ) -> None:
        """Test formatting when count is None."""
        collections = [CollectionInfo(name="no_count")]
        result = format_collections_response(collections)

        assert "1. **no_count**" in result
        assert "documents" not in result  # No count shown


# =============================================================================
# CollectionService execute Method Tests
