"""Modular RAG MCP Server - Main Entry Point.

This entry point forwards to the real MCP stdio server in
``src.mcp_server.server`` (also exposed as the ``mcp-server``
console script).
"""

import sys

from src.mcp_server.server import main as mcp_server_main


def main() -> int:
    """Run the MCP server over stdio transport."""
    return mcp_server_main()


if __name__ == "__main__":
    sys.exit(main())
