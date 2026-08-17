"""Tests for the FastAPI server endpoints.

These tests require the server module to load, which triggers module-level
backend initialization. We mock create_backend before import to avoid needing
llama_cpp/fastembed.
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_engine() -> AsyncMock:
    engine = AsyncMock()
    engine.health_check.return_value = True
    engine.list_projects.return_value = []
    engine.list_all.return_value = []
    engine.list_page.return_value = ([], None)
    engine.search.return_value = []
    engine.add.return_value = ["abc-123"]
    engine.delete.return_value = True
    engine.delete_all.return_value = 5
    engine.delete_project.return_value = True
    engine.reembed_all.return_value = 10
    engine.cleanup_text.return_value = {"cleaned": 2, "split_into_multiple": 1, "skipped": 8}
    engine.consolidate.return_value = {"clusters": 3, "memories_removed": 6, "memories_created": 3}
    engine.close = AsyncMock()
    return engine


@pytest.fixture
def mock_backend() -> MagicMock:
    backend = MagicMock()
    backend.health_ping = AsyncMock(return_value=True)
    backend.embedding_dimensions = 768
    backend.close = AsyncMock()
    return backend


@pytest.fixture
def client(mock_engine: AsyncMock, mock_backend: MagicMock):
    from fastapi.testclient import TestClient

    mock_stats = MagicMock()
    mock_stats.snapshot = MagicMock(return_value={"uptime_seconds": 100})
    mock_stats.start_flush_loop = AsyncMock()
    mock_stats.shutdown = AsyncMock()

    for mod_name in list(sys.modules):
        if mod_name.startswith("mem_zero.server"):
            del sys.modules[mod_name]

    with (
        patch("mem_zero.backends.create_backend", return_value=mock_backend),
        patch("mem_zero.stats.DiagnosticStats", return_value=mock_stats),
        patch("mem_zero.memory_engine.MemoryEngine", return_value=mock_engine),
    ):
        if "mem_zero.server" in sys.modules:
            del sys.modules["mem_zero.server"]
        import mem_zero.server as server_mod
        server_mod.engine = mock_engine
        server_mod.backend = mock_backend
        server_mod.stats = mock_stats
        yield TestClient(server_mod.app, raise_server_exceptions=False)

    for mod_name in list(sys.modules):
        if mod_name == "mem_zero.server":
            del sys.modules[mod_name]


class TestHealthEndpoint:
    def test_healthy(self, client, mock_engine: AsyncMock, mock_backend: MagicMock) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_degraded_qdrant(self, client, mock_engine: AsyncMock) -> None:
        mock_engine.health_check.side_effect = Exception("connection refused")
        resp = client.get("/health")
        assert resp.status_code == 503

    def test_degraded_llm(self, client, mock_backend: MagicMock) -> None:
        mock_backend.health_ping.return_value = False
        resp = client.get("/health")
        assert resp.status_code == 503


class TestProjectsEndpoint:
    def test_list_empty(self, client) -> None:
        resp = client.get("/api/v1/projects")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_with_projects(self, client, mock_engine: AsyncMock) -> None:
        from mem_zero.models import ProjectInfo
        mock_engine.list_projects.return_value = [
            ProjectInfo(slug="alpha", collection="mem-zero_alpha", memory_count=10),
            ProjectInfo(slug="beta", collection="mem-zero_beta", memory_count=5),
        ]
        resp = client.get("/api/v1/projects")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["slug"] == "alpha"
        assert data[0]["memory_count"] == 10


class TestMemoriesEndpoint:
    def test_get_memories(self, client, mock_engine: AsyncMock) -> None:
        from mem_zero.models import MemoryRecord
        mock_engine.list_page.return_value = ([
            MemoryRecord(
                id="abc", text="test memory", user_id="john",
                created_at=1700000000, updated_at=1700000000,
            )
        ], None)
        resp = client.get("/api/v1/projects/my-project/memories")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["text"] == "test memory"
        assert "X-Next-Offset" not in resp.headers  # last page

    def test_get_memories_with_limit(self, client, mock_engine: AsyncMock) -> None:
        resp = client.get("/api/v1/projects/my-project/memories?limit=5")
        assert resp.status_code == 200
        mock_engine.list_page.assert_called_with("my-project", limit=5, offset=None)

    def test_get_memories_next_offset_header(self, client, mock_engine: AsyncMock) -> None:
        # The cursor rides in a header so the body stays a plain list.
        nxt = "550e8400-e29b-41d4-a716-446655440000"
        mock_engine.list_page.return_value = ([], nxt)
        resp = client.get("/api/v1/projects/my-project/memories")
        assert resp.headers["X-Next-Offset"] == nxt

    def test_get_memories_forwards_offset(self, client, mock_engine: AsyncMock) -> None:
        cur = "550e8400-e29b-41d4-a716-446655440000"
        client.get(f"/api/v1/projects/my-project/memories?offset={cur}")
        mock_engine.list_page.assert_called_with("my-project", limit=50, offset=cur)

    def test_get_memories_bad_offset_400(self, client) -> None:
        resp = client.get("/api/v1/projects/my-project/memories?offset=not-a-uuid")
        assert resp.status_code == 400

    def test_create_memory(self, client, mock_engine: AsyncMock) -> None:
        resp = client.post(
            "/api/v1/projects/my-project/memories",
            json={"text": "I prefer dark mode"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["stored"] == 1
        assert "ids" in data

    def test_delete_all_memories(self, client, mock_engine: AsyncMock) -> None:
        resp = client.delete("/api/v1/projects/my-project/memories")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 5

    def test_delete_single_memory(self, client, mock_engine: AsyncMock) -> None:
        resp = client.delete(
            "/api/v1/projects/my-project/memories/550e8400-e29b-41d4-a716-446655440000"
        )
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True


class TestSearchEndpoint:
    def test_search_empty(self, client) -> None:
        resp = client.post(
            "/api/v1/projects/my-project/search",
            json={"query": "dark mode"},
        )
        assert resp.status_code == 200
        assert resp.json() == []

    def test_search_with_results(self, client, mock_engine: AsyncMock) -> None:
        from mem_zero.models import MemoryRecord
        mock_engine.search.return_value = [
            MemoryRecord(
                id="abc", text="User prefers dark mode", user_id="john",
                created_at=1700000000, updated_at=1700000000, score=0.92,
            )
        ]
        resp = client.post(
            "/api/v1/projects/my-project/search",
            json={"query": "dark mode", "top_k": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["score"] == 0.92

    def test_search_validates_top_k(self, client) -> None:
        resp = client.post(
            "/api/v1/projects/my-project/search",
            json={"query": "test", "top_k": 0},
        )
        assert resp.status_code == 422


class TestSlugValidation:
    def test_rejects_invalid_slug(self, client) -> None:
        resp = client.get("/api/v1/projects/Invalid Slug!/memories")
        assert resp.status_code == 400

    def test_rejects_uppercase_slug(self, client) -> None:
        resp = client.get("/api/v1/projects/MyProject/memories")
        assert resp.status_code == 400

    def test_accepts_valid_slug(self, client) -> None:
        resp = client.get("/api/v1/projects/my-project/memories")
        assert resp.status_code == 200


class TestMaintenanceEndpoints:
    def test_reembed(self, client, mock_engine: AsyncMock) -> None:
        resp = client.post("/api/v1/projects/my-project/reembed")
        assert resp.status_code == 200
        assert resp.json()["reembedded"] == 10

    def test_cleanup(self, client, mock_engine: AsyncMock) -> None:
        resp = client.post("/api/v1/projects/my-project/cleanup")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cleaned"] == 2
        assert data["split_into_multiple"] == 1

    def test_consolidate(self, client, mock_engine: AsyncMock) -> None:
        resp = client.post("/api/v1/projects/my-project/consolidate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["clusters"] == 3

    def test_consolidate_dry_run(self, client, mock_engine: AsyncMock) -> None:
        mock_engine.consolidate.return_value = {
            "clusters": 2, "previews": [{"count": 2, "texts": ["a", "b"]}],
        }
        resp = client.post("/api/v1/projects/my-project/consolidate?dry_run=true")
        assert resp.status_code == 200

    def test_delete_project(self, client, mock_engine: AsyncMock) -> None:
        resp = client.delete("/api/v1/projects/my-project")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    def test_delete_nonexistent_project(self, client, mock_engine: AsyncMock) -> None:
        mock_engine.delete_project.return_value = False
        resp = client.delete("/api/v1/projects/ghost")
        assert resp.status_code == 404



class TestErrorLadder:
    def test_delete_invalid_id_is_400(self, client, mock_engine: AsyncMock) -> None:
        # Used to escape as a raw 500 (no try around engine.delete).
        mock_engine.delete.side_effect = ValueError("Invalid memory ID: 'nope'")
        resp = client.delete("/api/v1/projects/my-project/memories/nope")
        assert resp.status_code == 400

    def test_delete_missing_is_404(self, client, mock_engine: AsyncMock) -> None:
        mock_engine.delete.return_value = False
        resp = client.delete(
            "/api/v1/projects/my-project/memories/550e8400-e29b-41d4-a716-446655440000"
        )
        assert resp.status_code == 404

    def test_internal_error_detail_is_generic(self, client, mock_engine: AsyncMock) -> None:
        # detail=str(exc) used to leak Qdrant URLs / backend bodies to clients.
        mock_engine.add.side_effect = RuntimeError("secret internal path /qdrant/xyz")
        resp = client.post(
            "/api/v1/projects/my-project/memories", json={"text": "hello world"}
        )
        assert resp.status_code == 500
        assert "secret" not in resp.text
        assert "xyz" not in resp.text

    def test_backend_unavailable_is_503(self, client, mock_engine: AsyncMock) -> None:
        from mem_zero.memory_engine import LLMError
        mock_engine.add.side_effect = LLMError("ollama down")
        resp = client.post(
            "/api/v1/projects/my-project/memories", json={"text": "hello world"}
        )
        assert resp.status_code == 503

    def test_search_dimension_mismatch_is_409(self, client, mock_engine: AsyncMock) -> None:
        from mem_zero.memory_engine import DimensionMismatchError
        mock_engine.search.side_effect = DimensionMismatchError("run reembed")
        resp = client.post(
            "/api/v1/projects/my-project/search", json={"query": "q"}
        )
        assert resp.status_code == 409
        assert "reembed" in resp.text

    def test_reembed_in_progress_is_409(self, client, mock_engine: AsyncMock) -> None:
        from mem_zero.memory_engine import MaintenanceInProgressError
        mock_engine.reembed_all.side_effect = MaintenanceInProgressError("busy")
        resp = client.post("/api/v1/projects/my-project/reembed")
        assert resp.status_code == 409

    def test_consolidate_too_large_is_413(self, client, mock_engine: AsyncMock) -> None:
        from mem_zero.memory_engine import ConsolidationTooLargeError
        mock_engine.consolidate.side_effect = ConsolidationTooLargeError("too big")
        resp = client.post("/api/v1/projects/my-project/consolidate")
        assert resp.status_code == 413


class TestInputBounds:
    def test_search_query_too_long_422(self, client) -> None:
        resp = client.post(
            "/api/v1/projects/my-project/search", json={"query": "x" * 2001}
        )
        assert resp.status_code == 422

    def test_search_empty_query_422(self, client) -> None:
        resp = client.post("/api/v1/projects/my-project/search", json={"query": ""})
        assert resp.status_code == 422

    def test_metadata_too_many_keys_422(self, client) -> None:
        meta = {f"k{i}": "v" for i in range(33)}
        resp = client.post(
            "/api/v1/projects/my-project/memories",
            json={"text": "hello world", "metadata": meta},
        )
        assert resp.status_code == 422

    def test_metadata_nested_dict_422(self, client) -> None:
        resp = client.post(
            "/api/v1/projects/my-project/memories",
            json={"text": "hello world", "metadata": {"deep": {"a": 1}}},
        )
        assert resp.status_code == 422

    def test_metadata_flat_scalars_ok(self, client, mock_engine: AsyncMock) -> None:
        resp = client.post(
            "/api/v1/projects/my-project/memories",
            json={"text": "hello world", "metadata": {"tag": "x", "n": 1, "ok": True,
                                                       "list": ["a", 2]}},
        )
        assert resp.status_code == 200

    def test_invalid_user_id_422(self, client) -> None:
        # REST used to accept anything, unbounded; MCP enforced a rule.
        resp = client.post(
            "/api/v1/projects/my-project/memories?user_id=bad%20user!",
            json={"text": "hello world"},
        )
        assert resp.status_code == 422

    def test_valid_user_id_ok(self, client, mock_engine: AsyncMock) -> None:
        resp = client.post(
            "/api/v1/projects/my-project/memories?user_id=john_doe-1",
            json={"text": "hello world"},
        )
        assert resp.status_code == 200
        assert mock_engine.add.call_args.args[1] == "john_doe-1"
