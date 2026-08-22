"""Heavy-dependency preloading for worker threads.

MCP tool execution offloads blocking work to ``asyncio.to_thread`` worker
threads.  If a worker thread tries to ``import chromadb`` (which transitively
pulls in onnxruntime, numpy, sqlite3 C extensions …) while another thread is
still importing, Python's global import lock can deadlock — in particular
with the stdin-reader thread of the stdio transport.

Pre-importing these modules in the main thread *before* any I/O threads are
spun up avoids the deadlock: subsequent ``import`` statements in worker
threads simply hit ``sys.modules`` and return immediately.

This is a business-layer concern (which heavy modules are needed), not an
MCP protocol concern, so it lives here rather than in ``src.mcp_server``.
"""

from __future__ import annotations


def preload_heavy_dependencies() -> None:
    """Eagerly import heavy third-party and internal modules.

    Safe to call more than once; imports are idempotent.
    """
    # chromadb is the heaviest culprit (onnxruntime, numpy, …)
    try:
        import chromadb  # noqa: F401
        import chromadb.config  # noqa: F401
    except ImportError:
        pass  # optional at install time

    # Internal modules that tools lazy-import inside asyncio.to_thread
    try:
        import src.core.query_engine.dense_retriever  # noqa: F401
        import src.core.query_engine.hybrid_search  # noqa: F401
        import src.core.query_engine.query_processor  # noqa: F401
        import src.core.query_engine.reranker  # noqa: F401
        import src.core.query_engine.sparse_retriever  # noqa: F401
        import src.ingestion.storage.bm25_indexer  # noqa: F401
        import src.libs.embedding.embedding_factory  # noqa: F401
        import src.libs.vector_store.vector_store_factory  # noqa: F401
    except ImportError:
        pass
