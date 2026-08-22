"""MCP Server over Streamable HTTP (uvicorn runner).

Run with::

    uv run mcp-server-http --host 127.0.0.1 --port 57666
    uv run python -m src.mcp_server.server --transport http
"""

from __future__ import annotations

import argparse
import sys

import uvicorn

from src.mcp_server.http_app import create_http_app
from src.mcp_server.protocol_handler import create_mcp_server
from src.mcp_server.server import (
    SERVER_NAME,
    SERVER_VERSION,
    _preload_heavy_imports,
    _redirect_all_loggers_to_stderr,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 57666


def create_server():
    """Create the shared low-level MCP server (same registry as stdio)."""
    return create_mcp_server(SERVER_NAME, SERVER_VERSION)


def run_http_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> int:
    """Run the MCP server over Streamable HTTP.

    Args:
        host: Bind address (default ``127.0.0.1``).
        port: Bind port (default ``57666``).

    Returns:
        Exit code.
    """
    _redirect_all_loggers_to_stderr()
    _preload_heavy_imports()
    app = create_http_app(create_server())
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


def main() -> int:
    """Entry point for the ``mcp-server-http`` console script."""
    parser = argparse.ArgumentParser(
        description="MCP Server entry point (Streamable HTTP transport)."
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help=f"Bind address (default: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"Bind port (default: {DEFAULT_PORT})"
    )
    args = parser.parse_args()
    return run_http_server(host=args.host, port=args.port)


if __name__ == "__main__":
    sys.exit(main())
