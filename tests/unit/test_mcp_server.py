from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mem_zero import mcp_server
from mem_zero.mcp_server import _format_record, mcp_router, set_engine


class TestFormatRecord:
    def test_converts_epoch_to_iso(self) -> None:
        record = {
            "id": "abc",
            "text": "hello",
            "created_at": 1779494400.0,
            "updated_at": 1779494400.0,
        }
        result = _format_record(record)
        expected = datetime.fromtimestamp(1779494400.0, tz=timezone.utc).isoformat()
        assert result["created_at"] == expected
        assert result["updated_at"] == expected

    def test_preserves_other_fields(self) -> None:
        record = {
            "id": "abc",
            "text": "hello",
            "created_at": 1779494400.0,
            "updated_at": 1779494400.0,
            "user_id": "john",
            "score": 0.95,
        }
        result = _format_record(record)
        assert result["id"] == "abc"
        assert result["text"] == "hello"
        assert result["user_id"] == "john"
        assert result["score"] == 0.95

    def test_handles_zero_timestamp(self) -> None:
        record = {"created_at": 0, "updated_at": 0}
        result = _format_record(record)
        assert result["created_at"] == 0
        assert result["updated_at"] == 0

    def test_handles_missing_timestamps(self) -> None:
        record = {"id": "abc", "text": "hello"}
        result = _format_record(record)
        assert "created_at" not in result

    def test_output_is_parseable_iso(self) -> None:
        ts = datetime(2026, 5, 20, 12, 30, 0, tzinfo=timezone.utc).timestamp()
        record = {"created_at": ts, "updated_at": ts}
        result = _format_record(record)
        parsed = datetime.fromisoformat(result["created_at"])
        assert parsed.year == 2026
        assert parsed.month == 5
        assert parsed.day == 20
        assert parsed.hour == 12
        assert parsed.tzinfo == timezone.utc

    def test_handles_int_timestamps(self) -> None:
        record = {"created_at": 1779494400, "updated_at": 1779494400}
        result = _format_record(record)
        assert "2026" in result["created_at"]


# ── ASGI bridge round-trips ──────────────────────────────────────────────────
# These drive the real transport in-process: router mounted on a bare FastAPI
# app, JSON-RPC tools/call POSTs, engine mocked via set_engine.

_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


@pytest.fixture
def mock_engine() -> AsyncMock:
    engine = AsyncMock()
    engine.add.return_value = ["id-1", "id-2"]
    engine.search.return_value = []
    engine.list_all.return_value = []
    engine.delete.return_value = True
    engine.delete_all.return_value = 3
    engine.count_memories.return_value = 3
    return engine


@pytest.fixture
def client(mock_engine: AsyncMock):
    app = FastAPI()
    app.include_router(mcp_router, prefix="/mcp")
    old = mcp_server._engine
    set_engine(mock_engine)
    yield TestClient(app, raise_server_exceptions=False)
    mcp_server._engine = old


def _call_tool(client: TestClient, path: str, name: str, arguments: dict):
    return client.post(
        path,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
        headers=_HEADERS,
    )


class TestMcpClientLifecycle:
    """Pins the full handshake real clients perform (Claude Code, Grok Build,
    Cursor): initialize -> notifications/initialized -> tools/list ->
    tools/call. The bare-tools/call tests below don't exercise this, and an
    mcp-library upgrade could break the handshake while they stay green."""

    def test_full_client_handshake(
        self, client: TestClient, mock_engine: AsyncMock
    ) -> None:
        path = "/mcp/my-project/http/john"
        # 1. initialize
        resp = client.post(
            path,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "grok-build", "version": "1.0"},
                },
            },
            headers=_HEADERS,
        )
        assert resp.status_code == 200
        init = resp.json()["result"]
        assert init["serverInfo"]["name"] == "mem-zero"
        assert "tools" in init["capabilities"]

        # 2. notifications/initialized — a notification, accepted with 202
        resp = client.post(
            path,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers=_HEADERS,
        )
        assert resp.status_code == 202

        # 3. tools/list — all five tools advertised
        resp = client.post(
            path,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            headers=_HEADERS,
        )
        names = {t["name"] for t in resp.json()["result"]["tools"]}
        assert names == {
            "add_memories",
            "search_memory",
            "list_memories",
            "delete_memories",
            "delete_all_memories",
        }

        # 4. tools/call still works after the handshake
        resp = _call_tool(client, path, "add_memories", {"text": "hello"})
        assert resp.json()["result"].get("isError") is not True
        mock_engine.add.assert_awaited_once_with("my-project", "john", ["hello"])


class TestMcpBridge:
    def test_add_memories_roundtrip(
        self, client: TestClient, mock_engine: AsyncMock
    ) -> None:
        resp = _call_tool(
            client, "/mcp/my-project/http/john", "add_memories", {"text": "hello"}
        )
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert result.get("isError") is not True
        payload = json.loads(result["content"][0]["text"])
        assert payload == {"stored": 2, "ids": ["id-1", "id-2"]}
        # The URL path params flowed through the context vars into the engine.
        mock_engine.add.assert_awaited_once_with("my-project", "john", ["hello"])

    def test_sequential_requests_use_own_context(
        self, client: TestClient, mock_engine: AsyncMock
    ) -> None:
        _call_tool(client, "/mcp/proj-a/http/alice", "add_memories", {"text": "one"})
        _call_tool(client, "/mcp/proj-b/http/bob", "add_memories", {"text": "two"})
        calls = mock_engine.add.await_args_list
        assert calls[0].args[:2] == ("proj-a", "alice")
        assert calls[1].args[:2] == ("proj-b", "bob")

    def test_invalid_slug_400(self, client: TestClient) -> None:
        resp = _call_tool(
            client, "/mcp/Bad Slug/http/john", "add_memories", {"text": "x"}
        )
        assert resp.status_code == 400

    def test_invalid_user_400(self, client: TestClient) -> None:
        resp = _call_tool(
            client, "/mcp/my-project/http/john!!", "add_memories", {"text": "x"}
        )
        assert resp.status_code == 400

    def test_get_is_405(self, client: TestClient) -> None:
        # The stateless server must reject the SSE GET stream, which would
        # otherwise hang and buffer forever.
        resp = client.get("/mcp/my-project/http/john", headers=_HEADERS)
        assert resp.status_code == 405

    def test_out_of_range_top_k_rejected_before_engine(
        self, client: TestClient, mock_engine: AsyncMock
    ) -> None:
        resp = _call_tool(
            client,
            "/mcp/my-project/http/john",
            "search_memory",
            {"query": "q", "top_k": 500},
        )
        assert resp.json()["result"]["isError"] is True
        mock_engine.search.assert_not_awaited()

    def test_engine_error_becomes_tool_error(
        self, client: TestClient, mock_engine: AsyncMock
    ) -> None:
        mock_engine.add.side_effect = RuntimeError("qdrant down")
        resp = _call_tool(
            client, "/mcp/my-project/http/john", "add_memories", {"text": "x"}
        )
        assert resp.json()["result"]["isError"] is True

    def test_delete_memories_validates_all_ids_first(
        self, client: TestClient, mock_engine: AsyncMock
    ) -> None:
        good = "550e8400-e29b-41d4-a716-446655440000"
        resp = _call_tool(
            client,
            "/mcp/my-project/http/john",
            "delete_memories",
            {"memory_ids": [good, "not-a-uuid"]},
        )
        assert resp.json()["result"]["isError"] is True
        mock_engine.delete.assert_not_awaited()


class TestMcpToolHardening:
    def test_delete_all_without_confirm_is_error_naming_project_and_count(
        self, client: TestClient, mock_engine: AsyncMock
    ) -> None:
        # The most destructive tool used to be a single unguarded call.
        resp = _call_tool(client, "/mcp/my-project/http/john", "delete_all_memories", {})
        result = resp.json()["result"]
        assert result["isError"] is True
        text = result["content"][0]["text"]
        assert "my-project" in text and "3" in text and "confirm=true" in text
        mock_engine.delete_all.assert_not_awaited()

    def test_delete_all_with_confirm_deletes(
        self, client: TestClient, mock_engine: AsyncMock
    ) -> None:
        resp = _call_tool(
            client, "/mcp/my-project/http/john", "delete_all_memories", {"confirm": True}
        )
        result = resp.json()["result"]
        assert result.get("isError") is not True
        assert json.loads(result["content"][0]["text"]) == {"deleted": 3}
        mock_engine.delete_all.assert_awaited_once_with("my-project")

    def test_list_memories_returns_total_and_truncated(
        self, client: TestClient, mock_engine: AsyncMock
    ) -> None:
        # Used to silently cap at 50 while claiming "List all".
        from mem_zero.models import MemoryRecord
        mock_engine.list_all.return_value = [
            MemoryRecord(id="a", text="t", user_id="u", created_at=0, updated_at=0)
        ]
        mock_engine.count_memories.return_value = 42
        resp = _call_tool(
            client, "/mcp/my-project/http/john", "list_memories", {"limit": 1}
        )
        payload = json.loads(resp.json()["result"]["content"][0]["text"])
        assert payload["total"] == 42
        assert payload["truncated"] is True
        assert len(payload["memories"]) == 1
        mock_engine.list_all.assert_awaited_once_with("my-project", limit=1)

    def test_list_memories_limit_out_of_range_is_error(
        self, client: TestClient, mock_engine: AsyncMock
    ) -> None:
        resp = _call_tool(
            client, "/mcp/my-project/http/john", "list_memories", {"limit": 5000}
        )
        assert resp.json()["result"]["isError"] is True
        mock_engine.list_all.assert_not_awaited()

    def test_search_query_too_long_is_error(
        self, client: TestClient, mock_engine: AsyncMock
    ) -> None:
        resp = _call_tool(
            client, "/mcp/my-project/http/john", "search_memory",
            {"query": "x" * 2001},
        )
        assert resp.json()["result"]["isError"] is True
        mock_engine.search.assert_not_awaited()

    def test_delete_memories_counts_only_real_deletes(
        self, client: TestClient, mock_engine: AsyncMock
    ) -> None:
        # engine.delete now returns False for ids that never existed.
        mock_engine.delete.side_effect = [True, False]
        good1 = "550e8400-e29b-41d4-a716-446655440000"
        good2 = "660e8400-e29b-41d4-a716-446655440000"
        resp = _call_tool(
            client, "/mcp/my-project/http/john", "delete_memories",
            {"memory_ids": [good1, good2]},
        )
        assert json.loads(resp.json()["result"]["content"][0]["text"]) == {"deleted": 1}

    def test_no_response_started_yields_502(
        self, client: TestClient, mock_engine: AsyncMock
    ) -> None:
        # If the transport never emits http.response.start the old code
        # returned the initialised 200 with an empty body — "success".
        async def silent(*a, **kw):
            return None

        with patch(
            "mem_zero.mcp_server.StreamableHTTPServerTransport.handle_request",
            new=silent,
        ):
            resp = _call_tool(
                client, "/mcp/my-project/http/john", "add_memories", {"text": "x"}
            )
        assert resp.status_code == 502
        assert resp.json()["error"]["code"] == -32603

    def test_tool_descriptions_are_instructive(self, client: TestClient) -> None:
        # Descriptions are the model's only decision signal; pin the key facts.
        resp = client.post(
            "/mcp/my-project/http/john",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers=_HEADERS,
        )
        by_name = {t["name"]: t["description"] for t in resp.json()["result"]["tools"]}
        assert "may legitimately be 0" in by_name["add_memories"]
        assert "Semantic" in by_name["search_memory"]
        assert "truncated" in by_name["list_memories"]
        assert "IRREVERSIBLE" in by_name["delete_all_memories"]
        assert "confirm=true" in by_name["delete_all_memories"]
