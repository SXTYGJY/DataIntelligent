"""Integration tests for the MCP Streamable HTTP transport (ASGI-level).

Exercises the Starlette application produced by ``create_http_app`` with
the ``TestClient`` (which runs the ASGI lifespan). Covers:

- JSON response mode (``json_response=True``) protocol lifecycle
- SSE response mode (default) and Accept-header enforcement
- stateless session behavior (no ``Mcp-Session-Id`` required)
- the same tool surface as the stdio tests: initialize → tools/list →
  tools/call → prompts/list
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from starlette.testclient import TestClient

from src.mcp_server.http_app import create_http_app
from src.mcp_server.http_server import create_server

CONTENT_TYPE = "application/json"
ACCEPT_JSON = "application/json"
ACCEPT_SSE = "application/json, text/event-stream"

INIT_REQUEST: dict[str, Any] = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "clientInfo": {"name": "http-pytest-client", "version": "1.0.0"},
        "capabilities": {},
    },
}

TOOLS_LIST_REQUEST: dict[str, Any] = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list",
    "params": {},
}

TOOLS_CALL_REQUEST: dict[str, Any] = {
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
        "name": "list_collections",
        "arguments": {"include_stats": False},
    },
}

PROMPTS_LIST_REQUEST: dict[str, Any] = {
    "jsonrpc": "2.0",
    "id": 4,
    "method": "prompts/list",
    "params": {},
}


def _post(
    client: TestClient,
    payload: dict[str, Any],
    accept: str = ACCEPT_JSON,
    headers: dict[str, str] | None = None,
):
    """POST a single JSON-RPC message to the mounted endpoint."""
    request_headers = {"Accept": accept, "Content-Type": CONTENT_TYPE}
    if headers:
        request_headers.update(headers)
    return client.post("/mcp", json=payload, headers=request_headers)


def _sse_data_payloads(text: str) -> list[dict[str, Any]]:
    """Extract and parse ``data:`` lines from an SSE response body."""
    payloads: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("data:"):
            payloads.append(json.loads(stripped[len("data:") :].strip()))
    return payloads


@pytest.fixture()
def json_client():
    """TestClient over a JSON-response-mode HTTP app (lifespan active)."""
    app = create_http_app(create_server(), json_response=True)
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def sse_client():
    """TestClient over the default SSE-mode HTTP app (lifespan active)."""
    app = create_http_app(create_server())
    with TestClient(app) as client:
        yield client


class TestJsonResponseMode:
    """Protocol lifecycle over plain-JSON responses."""

    def test_initialize_returns_protocol_info(self, json_client: TestClient) -> None:
        response = _post(json_client, INIT_REQUEST)

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        assert body.get("jsonrpc") == "2.0"
        result = body["result"]
        assert result["protocolVersion"] == "2025-06-18"
        assert result["serverInfo"]["name"] == "modular-rag-mcp-server"

    def test_tools_list_returns_registered_tools(self, json_client: TestClient) -> None:
        # stateless mode: tools/list works without a prior initialize/session
        response = _post(json_client, TOOLS_LIST_REQUEST)

        assert response.status_code == 200
        tools = response.json()["result"]["tools"]
        names = {tool["name"] for tool in tools}
        assert names == {
            "query_knowledge_hub",
            "list_collections",
            "get_document_summary",
        }

    def test_tools_call_returns_result(self, json_client: TestClient) -> None:
        response = _post(json_client, TOOLS_CALL_REQUEST)

        assert response.status_code == 200
        result = response.json()["result"]
        assert result["isError"] is False
        assert len(result["content"]) >= 1

    def test_unknown_tool_is_error(self, json_client: TestClient) -> None:
        request = {
            "jsonrpc": "2.0",
            "id": 99,
            "method": "tools/call",
            "params": {"name": "missing_tool", "arguments": {}},
        }
        response = _post(json_client, request)

        assert response.status_code == 200
        result = response.json()["result"]
        assert result["isError"] is True
        assert "missing_tool" in result["content"][0]["text"]

    def test_prompts_list_returns_prompts(self, json_client: TestClient) -> None:
        response = _post(json_client, PROMPTS_LIST_REQUEST)

        assert response.status_code == 200
        prompts = response.json()["result"]["prompts"]
        names = {prompt["name"] for prompt in prompts}
        assert "query_knowledge_hub" in names
        assert "list_collections" in names
        assert "get_document_summary" in names


class TestSSEResponseMode:
    """Default (SSE) response mode and header enforcement."""

    def test_initialize_streams_sse_events(self, sse_client: TestClient) -> None:
        response = _post(sse_client, INIT_REQUEST, accept=ACCEPT_SSE)

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        payloads = _sse_data_payloads(response.text)
        assert payloads, "SSE response contained no data events"
        result = payloads[0]["result"]
        assert result["serverInfo"]["name"] == "modular-rag-mcp-server"

    def test_tools_list_streams_sse_events(self, sse_client: TestClient) -> None:
        response = _post(sse_client, TOOLS_LIST_REQUEST, accept=ACCEPT_SSE)

        assert response.status_code == 200
        payloads = _sse_data_payloads(response.text)
        tools = payloads[0]["result"]["tools"]
        assert {tool["name"] for tool in tools} == {
            "query_knowledge_hub",
            "list_collections",
            "get_document_summary",
        }

    def test_sse_mode_rejects_json_only_accept(self, sse_client: TestClient) -> None:
        response = _post(sse_client, TOOLS_LIST_REQUEST, accept=ACCEPT_JSON)

        assert response.status_code == 406


class TestStatelessBehavior:
    """Stateless sessions: no Mcp-Session-Id is required or returned."""

    def test_no_session_id_required(self, json_client: TestClient) -> None:
        response = _post(json_client, TOOLS_LIST_REQUEST)

        assert response.status_code == 200
        assert "mcp-session-id" not in {
            key.lower() for key in response.headers.keys()
        }

    def test_consecutive_requests_share_registry(
        self, json_client: TestClient
    ) -> None:
        first = _post(json_client, TOOLS_LIST_REQUEST).json()
        second = _post(json_client, TOOLS_LIST_REQUEST).json()

        assert first["result"]["tools"] == second["result"]["tools"]
