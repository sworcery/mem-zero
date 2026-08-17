"""Auth middleware tests.

server.py builds config/engine/middleware at import time, so each fixture
re-imports the module with the auth env vars set and the heavy pieces patched.
"""
from __future__ import annotations

import base64
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_engine() -> AsyncMock:
    engine = AsyncMock()
    engine.health_check.return_value = True
    engine.list_projects.return_value = []
    engine.list_all.return_value = []
    engine.close = AsyncMock()
    return engine


@pytest.fixture
def mock_backend() -> MagicMock:
    backend = MagicMock()
    backend.health_ping = AsyncMock(return_value=True)
    backend.embedding_dimensions = 768
    backend.close = AsyncMock()
    return backend


def _make_client(mock_engine, mock_backend, monkeypatch, env: dict[str, str]):
    from fastapi.testclient import TestClient

    for key, value in env.items():
        monkeypatch.setenv(key, value)

    mock_stats = MagicMock()
    mock_stats.snapshot = MagicMock(return_value={"uptime_seconds": 1})
    mock_stats.start_flush_loop = AsyncMock()
    mock_stats.shutdown = AsyncMock()

    sys.modules.pop("mem_zero.server", None)
    with (
        patch("mem_zero.backends.create_backend", return_value=mock_backend),
        patch("mem_zero.stats.DiagnosticStats", return_value=mock_stats),
        patch("mem_zero.memory_engine.MemoryEngine", return_value=mock_engine),
    ):
        import mem_zero.server as server_mod

        server_mod.engine = mock_engine
        server_mod.backend = mock_backend
        server_mod.stats = mock_stats
        client = TestClient(server_mod.app, raise_server_exceptions=False)
    return client


@pytest.fixture
def api_client(mock_engine, mock_backend, monkeypatch):
    yield _make_client(mock_engine, mock_backend, monkeypatch, {"API_KEY": "sekrit"})
    sys.modules.pop("mem_zero.server", None)


@pytest.fixture
def dash_client(mock_engine, mock_backend, monkeypatch):
    yield _make_client(
        mock_engine,
        mock_backend,
        monkeypatch,
        {"API_KEY": "sekrit", "DASHBOARD_USER": "admin", "DASHBOARD_PASS": "hunter2"},
    )
    sys.modules.pop("mem_zero.server", None)


class TestAPIKeyMiddleware:
    def test_api_requires_key(self, api_client) -> None:
        resp = api_client.get("/api/v1/projects")
        assert resp.status_code == 401
        assert resp.headers["WWW-Authenticate"].startswith("Bearer")

    def test_bearer_token_accepted(self, api_client) -> None:
        resp = api_client.get(
            "/api/v1/projects", headers={"Authorization": "Bearer sekrit"}
        )
        assert resp.status_code == 200

    def test_wrong_bearer_rejected(self, api_client) -> None:
        resp = api_client.get(
            "/api/v1/projects", headers={"Authorization": "Bearer wrong"}
        )
        assert resp.status_code == 401

    def test_query_param_accepted(self, api_client) -> None:
        # The dashboard's export links use ?api_key=.
        resp = api_client.get("/api/v1/projects?api_key=sekrit")
        assert resp.status_code == 200

    def test_health_is_open(self, api_client) -> None:
        # Unraid's health probe runs unauthenticated.
        resp = api_client.get("/health")
        assert resp.status_code == 200

    def test_mcp_requires_key(self, api_client) -> None:
        resp = api_client.post("/mcp/my-project/http/john", json={})
        assert resp.status_code == 401


class TestDashboardAuth:
    @staticmethod
    def _basic(user: str, pw: str) -> dict[str, str]:
        token = base64.b64encode(f"{user}:{pw}".encode()).decode()
        return {"Authorization": f"Basic {token}"}

    def test_dashboard_requires_basic_auth(self, dash_client) -> None:
        resp = dash_client.get("/nonexistent-page")
        assert resp.status_code == 401
        assert resp.headers["WWW-Authenticate"].startswith("Basic")

    def test_dashboard_basic_auth_accepted(self, dash_client) -> None:
        resp = dash_client.get("/health", headers=self._basic("admin", "hunter2"))
        assert resp.status_code == 200

    def test_basic_auth_satisfies_api_key_middleware(self, dash_client) -> None:
        # The dashboard JS calls /api/ with Basic auth and no Bearer token;
        # DashboardAuthMiddleware marks the request authenticated and
        # APIKeyMiddleware honors it.
        resp = dash_client.get(
            "/api/v1/projects", headers=self._basic("admin", "hunter2")
        )
        assert resp.status_code == 200

    def test_wrong_basic_auth_on_api_still_401(self, dash_client) -> None:
        resp = dash_client.get(
            "/api/v1/projects", headers=self._basic("admin", "wrong")
        )
        assert resp.status_code == 401

    def test_garbage_basic_header_rejected(self, dash_client) -> None:
        resp = dash_client.get(
            "/api/v1/projects", headers={"Authorization": "Basic !!!notbase64"}
        )
        assert resp.status_code == 401

    def test_health_open_without_any_auth(self, dash_client) -> None:
        resp = dash_client.get("/health")
        assert resp.status_code == 200


@pytest.fixture
def dash_only_client(mock_engine, mock_backend, monkeypatch):
    # The previously-untested combination: dashboard creds, NO api key.
    monkeypatch.delenv("API_KEY", raising=False)
    yield _make_client(
        mock_engine,
        mock_backend,
        monkeypatch,
        {"DASHBOARD_USER": "admin", "DASHBOARD_PASS": "hunter2"},
    )
    sys.modules.pop("mem_zero.server", None)


class TestDashboardOnlyConfig:
    @staticmethod
    def _basic(user: str, pw: str) -> dict[str, str]:
        token = base64.b64encode(f"{user}:{pw}".encode()).decode()
        return {"Authorization": f"Basic {token}"}

    def test_api_requires_basic_auth_when_no_api_key(self, dash_only_client) -> None:
        # The bug: DashboardAuthMiddleware fell through unauthenticated /api
        # requests, and with no APIKeyMiddleware behind it the whole API and
        # MCP were open behind a dashboard that LOOKED password-protected.
        resp = dash_only_client.get("/api/v1/projects")
        assert resp.status_code == 401
        assert resp.headers["WWW-Authenticate"].startswith("Basic")

    def test_mcp_requires_basic_auth_when_no_api_key(self, dash_only_client) -> None:
        resp = dash_only_client.post("/mcp/my-project/http/john", json={})
        assert resp.status_code == 401

    def test_basic_auth_opens_api(self, dash_only_client) -> None:
        resp = dash_only_client.get(
            "/api/v1/projects", headers=self._basic("admin", "hunter2")
        )
        assert resp.status_code == 200

    def test_health_still_open(self, dash_only_client) -> None:
        assert dash_only_client.get("/health").status_code == 200


class TestAlwaysOpenExactMatch:
    def test_health_prefix_variants_are_not_treated_as_open(self, api_client) -> None:
        # _ALWAYS_OPEN used startswith, so any path beginning "/health" was
        # exempted from the auth middlewares. Only the exact "/health" is open
        # now; a prefix variant is just an ordinary (unrouted -> 404) path and
        # is no longer special-cased.
        from mem_zero import server as server_mod
        assert "/health" in server_mod._ALWAYS_OPEN
        assert api_client.get("/health").status_code == 200
        assert api_client.get("/healthz").status_code == 404
        # And a gated path is still gated regardless.
        assert api_client.get("/api/v1/projects").status_code == 401


class TestHalfDashboardCreds:
    def test_only_user_set_warns_and_disables_dashboard_auth(
        self, mock_engine, mock_backend, monkeypatch, caplog
    ) -> None:
        import logging
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.delenv("DASHBOARD_PASS", raising=False)
        with caplog.at_level(logging.WARNING, logger="mem_zero.server"):
            client = _make_client(mock_engine, mock_backend, monkeypatch,
                                  {"DASHBOARD_USER": "admin"})
        try:
            assert any("Only one of DASHBOARD_USER" in r.message for r in caplog.records)
            # Dashboard is not protected (as documented) — but loudly.
            assert client.get("/api/v1/projects").status_code == 200
        finally:
            sys.modules.pop("mem_zero.server", None)
