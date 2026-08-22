"""Streamable HTTP transport wiring for the MCP server.

Wraps a low-level ``mcp.Server`` into a Starlette ASGI application that
exposes the MCP Streamable HTTP protocol (SSE responses, stateless
sessions) at a configurable mount path (default ``/mcp``).

Note: a :class:`starlette.routing.Route` is used instead of
``starlette.routing.Mount`` so the canonical MCP endpoint ``/mcp`` (no
trailing slash) is served directly. A ``Mount`` would 307-redirect
``/mcp`` to ``/mcp/``, which some MCP clients do not follow.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

DEFAULT_MOUNT_PATH = "/mcp"
#: Methods the Streamable HTTP endpoint responds to (POST is the core one;
#: GET/DELETE are used by stateful session reconnection/termination).
HTTP_METHODS = ["POST", "GET", "DELETE"]


class _StreamableHTTPHandler:
    """Adapter exposing a session manager as a :class:`Route` endpoint.

    Starlette treats a ``Route`` endpoint that is *not* a plain function as
    an ASGI app, so this instance wrapper lets us reuse the session-manager
    callable directly.
    """

    def __init__(self, session_manager: StreamableHTTPSessionManager) -> None:
        self._session_manager = session_manager

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Forward a single ASGI request to the session manager."""
        await self._session_manager.handle_request(scope, receive, send)


def create_http_app(
    server: Server,
    *,
    json_response: bool = False,
    stateless: bool = True,
    mount_path: str = DEFAULT_MOUNT_PATH,
) -> Starlette:
    """Build the Streamable HTTP ASGI application.

    Args:
        server: Low-level MCP server built by ``create_mcp_server``.
        json_response: ``True`` responds with plain JSON, ``False`` uses SSE
            streams (the MCP Streamable HTTP default).
        stateless: ``True`` creates a fresh transport per HTTP request with no
            server-side session tracking (suits stateless request/response
            tools). ``False`` tracks sessions via ``Mcp-Session-Id`` and
            requires an event store for resumability.
        mount_path: URL path where the MCP endpoint is exposed.

    Returns:
        A ready-to-serve Starlette application.
    """
    session_manager = StreamableHTTPSessionManager(
        app=server,
        event_store=None,
        json_response=json_response,
        stateless=stateless,
    )

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        """ASGI lifespan: start/stop the session manager task group."""
        async with session_manager.run():
            yield

    return Starlette(
        routes=[
            Route(
                mount_path,
                endpoint=_StreamableHTTPHandler(session_manager),
                methods=HTTP_METHODS,
            )
        ],
        lifespan=lifespan,
    )
