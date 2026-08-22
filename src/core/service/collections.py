"""Collection listing business service.

Owns vector-store (ChromaDB) access, collection enumeration and result
formatting.  The MCP tool layer only adapts these results to the protocol.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.core.settings import resolve_path

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class CollectionInfo:
    """Information about a single collection.

    Attributes:
        name: Collection name
        count: Number of documents/chunks in the collection (optional)
        metadata: Collection metadata dictionary
    """

    name: str
    count: int | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result: dict[str, Any] = {"name": self.name}
        if self.count is not None:
            result["count"] = self.count
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class ListCollectionsConfig:
    """Configuration for collection listing.

    Attributes:
        persist_directory: Path to ChromaDB storage directory
        include_stats_default: Default value for include_stats parameter
    """

    persist_directory: str = "data/db/chroma"
    include_stats_default: bool = True


class CollectionService:
    """Business service that lists knowledge base collections.

    Design Principles:
    - Config-Driven: Paths from settings.yaml
    - Error Resilience: Graceful handling of missing directories
    - Observable: Logging for debugging
    """

    def __init__(
        self,
        settings: Settings | None = None,
        config: ListCollectionsConfig | None = None,
    ) -> None:
        """Initialize CollectionService.

        Args:
            settings: Application settings. If None, loaded from default path.
            config: Service configuration. If None, derived from settings.
        """
        self._settings = settings
        self._config = config

    @property
    def settings(self) -> Settings:
        """Get settings, loading if necessary."""
        if self._settings is None:
            from src.core.settings import load_settings

            self._settings = load_settings()
        return self._settings

    @property
    def config(self) -> ListCollectionsConfig:
        """Get configuration, deriving from settings if necessary."""
        if self._config is None:
            try:
                persist_dir = getattr(
                    self.settings.vector_store,
                    "persist_directory",
                    "data/db/chroma",
                )
            except AttributeError:
                persist_dir = "data/db/chroma"

            self._config = ListCollectionsConfig(persist_directory=persist_dir)
        return self._config

    def _get_chroma_client(self) -> Any:
        """Get or create ChromaDB client.

        Returns:
            ChromaDB PersistentClient instance.

        Raises:
            ImportError: If chromadb is not installed.
            RuntimeError: If client creation fails.
        """
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError(
                "chromadb package is required for list_collections. "
                "Install it with: pip install chromadb"
            )

        persist_path = resolve_path(self.config.persist_directory)

        if not persist_path.exists():
            logger.warning(f"ChromaDB directory does not exist: {persist_path}")
            # Return client anyway - it will just have no collections
            persist_path.mkdir(parents=True, exist_ok=True)

        try:
            client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            return client
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize ChromaDB client at '{persist_path}': {e}"
            ) from e

    def list_collections(self, include_stats: bool = True) -> list[CollectionInfo]:
        """List all available collections.

        Args:
            include_stats: Whether to include document counts.

        Returns:
            List of CollectionInfo objects.
        """
        try:
            client = self._get_chroma_client()
        except (ImportError, RuntimeError) as e:
            logger.error(f"Failed to get ChromaDB client: {e}")
            return []

        collections_info: list[CollectionInfo] = []

        try:
            # Get all collections from ChromaDB
            collections = client.list_collections()

            for collection in collections:
                info = CollectionInfo(
                    name=collection.name,
                    metadata=collection.metadata,
                )

                if include_stats:
                    try:
                        info.count = collection.count()
                    except Exception as e:
                        logger.warning(
                            f"Failed to get count for collection '{collection.name}': {e}"
                        )
                        info.count = None

                collections_info.append(info)

        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

        logger.info(f"Found {len(collections_info)} collections")
        return collections_info


def format_collections_response(collections: list[CollectionInfo]) -> str:
    """Format collections list as a readable string.

    Args:
        collections: List of CollectionInfo objects.

    Returns:
        Formatted string suitable for an MCP response.
    """
    if not collections:
        return "No collections found in the knowledge base."

    lines = [f"## Available Collections ({len(collections)} total)\n"]

    for i, coll in enumerate(collections, 1):
        line = f"{i}. **{coll.name}**"

        if coll.count is not None:
            line += f" - {coll.count} documents"

        if coll.metadata:
            # Filter out internal metadata
            user_metadata = {
                k: v
                for k, v in coll.metadata.items()
                if not k.startswith("_") and not k.startswith("hnsw:")
            }
            if user_metadata:
                meta_str = ", ".join(f"{k}={v}" for k, v in user_metadata.items())
                line += f" ({meta_str})"

        lines.append(line)

    return "\n".join(lines)
