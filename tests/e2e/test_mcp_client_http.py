"""E2E test: real MCP server over Streamable HTTP on a real TCP port.

Launches ``python -m src.mcp_server.server --transport http`` as a
subprocess, waits for the port to accept connections, then drives the
JSON-RPC lifecycle with ``httpx``:

    initialize → tools/list → tools/call → prompts/list

The server runs in the default SSE response mode, so every response is
parsed from ``text/event-stream`` bodies.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
BASE_MOUNT = "/mcp"
ACCEPT_SSE = "application/json, text/event-stream"
CONTENT_TYPE = "application/json"


def _free_port() -> int:
    """Ask the OS for a currently-free localhost port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_until_ready(port: int, timeout: float = 45.0) -> None:
    """Poll until the HTTP server accepts connections or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"HTTP server did not become ready on port {port}")


def _start_http_server(port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "src.mcp_server.server",
            "--transport",
            "http",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PROJECT_ROOT),
        env=env,
    )


def _post(
    client: httpx.Client, url: str, payload: dict[str, Any]
) -> httpx.Response:
    return client.post(
        url,
        json=payload,
        headers={"Accept": ACCEPT_SSE, "Content-Type": CONTENT_TYPE},
    )


def _sse_data_payloads(text: str) -> list[dict[str, Any]]:
    """Extract and parse ``data:`` lines from an SSE response body."""
    payloads: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("data:"):
            payloads.append(json.loads(stripped[len("data:") :].strip()))
    return payloads


@pytest.fixture()
def http_server_url() -> str:
    """Yield ``http://127.0.0.1:<port>`` for a fresh server subprocess."""
    port = _free_port()
    proc = _start_http_server(port)
    try:
        _wait_until_ready(port)
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@pytest.mark.e2e
def test_http_real_port_full_session(http_server_url: str) -> None:
    """Full lifecycle against the real server: init → list → call → prompts."""
    url = http_server_url + BASE_MOUNT
    with httpx.Client(timeout=20.0) as client:
        init_response = _post(
            client,
            url,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "clientInfo": {"name": "http-e2e-client", "version": "1.0.0"},
                    "capabilities": {},
                },
            },
        )
        assert init_response.status_code == 200
        assert init_response.headers["content-type"].startswith("text/event-stream")
        init_payloads = _sse_data_payloads(init_response.text)
        assert init_payloads, "initialize produced no SSE data"
        server_info = init_payloads[0]["result"]["serverInfo"]
        assert server_info["name"] == "modular-rag-mcp-server"

        list_response = _post(
            client, url, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        )
        assert list_response.status_code == 200
        tools = _sse_data_payloads(list_response.text)[0]["result"]["tools"]
        assert {tool["name"] for tool in tools} == {
            "query_knowledge_hub",
            "list_collections",
            "get_document_summary",
        }

        call_response = _post(
            client,
            url,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "list_collections",
                    "arguments": {"include_stats": False},
                },
            },
        )
        assert call_response.status_code == 200
        call_result = _sse_data_payloads(call_response.text)[0]["result"]
        assert call_result["isError"] is False
        assert len(call_result["content"]) >= 1

        prompts_response = _post(
            client, url, {"jsonrpc": "2.0", "id": 4, "method": "prompts/list", "params": {}}
        )
        assert prompts_response.status_code == 200
        prompts = _sse_data_payloads(prompts_response.text)[0]["result"]["prompts"]
        assert "query_knowledge_hub" in {prompt["name"] for prompt in prompts}


@pytest.mark.e2e
def test_http_real_port_unknown_tool(http_server_url: str) -> None:
    """Error normalization is preserved over HTTP."""
    url = http_server_url + BASE_MOUNT
    with httpx.Client(timeout=20.0) as client:
        response = _post(
            client,
            url,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "does_not_exist", "arguments": {}},
            },
        )
        assert response.status_code == 200
        result = _sse_data_payloads(response.text)[0]["result"]
        assert result["isError"] is True
        assert "does_not_exist" in result["content"][0]["text"]
