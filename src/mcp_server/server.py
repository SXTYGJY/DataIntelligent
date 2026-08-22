"""MCP Server entry point using official MCP SDK.

This module implements the MCP server using the official Python MCP SDK.
The default transport is stdio (stdout only carries protocol messages, all
logs go to stderr); an optional Streamable HTTP transport is available via
``--transport http`` (see :mod:`src.mcp_server.http_server`).
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

from src.mcp_server.protocol_handler import create_mcp_server
from src.observability.logger import get_logger

if TYPE_CHECKING:
    pass


SERVER_NAME = "modular-rag-mcp-server"
SERVER_VERSION = "0.1.0"


def _redirect_all_loggers_to_stderr() -> None:
    """Redirect all root logger handlers to stderr.

    MCP stdio transport reserves stdout for JSON-RPC messages.
    Any logging to stdout corrupts the protocol stream.
    """
    import logging as _logging

    root = _logging.getLogger()
    stderr_handler = _logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(
        _logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    # Replace any existing stream handlers that might point to stdout
    for handler in root.handlers[:]:
        if isinstance(handler, _logging.StreamHandler) and not isinstance(
            handler, _logging.FileHandler
        ):
            root.removeHandler(handler)
    root.addHandler(stderr_handler)


def _preload_heavy_imports() -> None:
    """Preload heavy business dependencies before worker threads start.

    MCP tool execution offloads blocking work to ``asyncio.to_thread`` worker
    threads; importing heavy libraries there can deadlock on Python's import
    lock.  The list of business modules is owned by the business layer
    (:func:`src.core.service.preload.preload_heavy_dependencies`), not by
    the MCP server.
    """
    from src.core.service.preload import preload_heavy_dependencies

    preload_heavy_dependencies()


def create_server():
    """Create the shared low-level MCP server.

    The same server (and therefore the same :class:`ToolRegistry`) is used by
    both the stdio and the Streamable HTTP transports.
    """
    return create_mcp_server(SERVER_NAME, SERVER_VERSION)


async def run_stdio_server_async() -> int:
    """Run MCP server over stdio asynchronously.

    Returns:
        Exit code.
    """
    # Import here to avoid import errors if mcp not installed
    import mcp.server.stdio

    # Ensure ALL logging goes to stderr (stdout is reserved for JSON-RPC)
    _redirect_all_loggers_to_stderr()

    # Pre-load heavy deps in main thread to prevent import-lock deadlocks
    # when tool handlers later call asyncio.to_thread().
    _preload_heavy_imports()

    logger = get_logger(log_level="INFO")
    logger.info("Starting MCP server (stdio transport) with official SDK.")

    # Create server with protocol handler (shared with HTTP transport)
    server = create_server()

    # Run with stdio transport
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

    logger.info("MCP server shutting down.")
    return 0


def run_stdio_server() -> int:
    """Run MCP server over stdio (synchronous wrapper).

    Returns:
        Exit code.
    """
    return asyncio.run(run_stdio_server_async())


def main() -> int:
    """Entry point for the MCP server (default: stdio transport).

    Kept as the default for backward compatibility with MCP clients that
    spawn the server as a subprocess (``python main.py`` / ``mcp-server``).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Modular RAG MCP Server (stdio default, optional Streamable HTTP)."
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport to serve on (default: stdio).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address for --transport http (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=57666,
        help="Bind port for --transport http (default: 57666).",
    )
    args = parser.parse_args()

    if args.transport == "http":
        from src.mcp_server.http_server import run_http_server

        return run_http_server(host=args.host, port=args.port)
    return run_stdio_server()


if __name__ == "__main__":
    sys.exit(main())
