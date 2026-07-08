from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from mem_zero.cli import build_parser, cmd_export, cmd_health, cmd_projects


@pytest.fixture
def mock_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    return resp


def _mock_client_ctx(mock_response: MagicMock):
    """Create a patched _client context that returns mock_response for get/post."""
    http = MagicMock()
    http.get.return_value = mock_response
    http.post.return_value = mock_response
    http.delete.return_value = mock_response
    ctx = patch("mem_zero.cli._client")
    mock = ctx.start()
    mock.return_value.__enter__ = MagicMock(return_value=http)
    mock.return_value.__exit__ = MagicMock(return_value=False)
    return ctx, mock


class TestBuildParser:
    def test_has_all_subcommands(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["projects"])
        assert args.command == "projects"

    def test_health_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["health"])
        assert args.command == "health"

    def test_list_requires_project(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["list"])

    def test_search_requires_project_and_query(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["search"])

    def test_add_requires_project_and_text(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["add"])

    def test_export_requires_project(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["export"])

    def test_import_requires_file(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["import"])

    def test_default_url(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["health"])
        assert args.url == "http://localhost:8765"

    def test_custom_url(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--url", "http://192.168.1.10:8765", "health"])
        assert args.url == "http://192.168.1.10:8765"

    def test_api_key_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--api-key", "secret123", "health"])
        assert args.api_key == "secret123"

    def test_json_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--json", "health"])
        assert args.json is True

    def test_search_top_k(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["search", "myproj", "test query", "--top-k", "5"])
        assert args.top_k == 5

    def test_add_user_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["add", "myproj", "some text", "--user", "john"])
        assert args.user == "john"

    def test_export_output_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["export", "myproj", "-o", "backup.json"])
        assert args.output == "backup.json"

    def test_stats_project_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["stats", "--project", "myproj"])
        assert args.project == "myproj"


class TestCmdHealth:
    def test_healthy_server(
        self, mock_response: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        mock_response.json.return_value = {
            "status": "ok",
            "version": "0.1.38",
            "services": {"qdrant": True, "llm": True},
        }
        ctx, _ = _mock_client_ctx(mock_response)
        try:
            parser = build_parser()
            args = parser.parse_args(["health"])
            result = cmd_health(args)
        finally:
            ctx.stop()
        assert result == 0
        captured = capsys.readouterr()
        assert "ok" in captured.out
        assert "0.1.38" in captured.out


class TestCmdProjects:
    def test_empty_projects(
        self, mock_response: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        mock_response.json.return_value = []
        ctx, _ = _mock_client_ctx(mock_response)
        try:
            parser = build_parser()
            args = parser.parse_args(["projects"])
            result = cmd_projects(args)
        finally:
            ctx.stop()
        assert result == 0
        captured = capsys.readouterr()
        assert "No projects" in captured.out

    def test_projects_listed(
        self, mock_response: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        mock_response.json.return_value = [
            {"slug": "alpha", "memory_count": 10, "last_updated": 1700000000},
            {"slug": "beta", "memory_count": 5, "last_updated": None},
        ]
        ctx, _ = _mock_client_ctx(mock_response)
        try:
            parser = build_parser()
            args = parser.parse_args(["projects"])
            result = cmd_projects(args)
        finally:
            ctx.stop()
        assert result == 0
        captured = capsys.readouterr()
        assert "alpha" in captured.out
        assert "beta" in captured.out


class TestCmdExport:
    def test_export_to_file(self, mock_response: MagicMock, tmp_path) -> None:
        mock_response.json.return_value = [
            {
                "id": "abc", "text": "test memory", "user_id": "john",
                "created_at": 1700000000, "updated_at": 1700000000, "metadata": {},
            },
        ]
        output_file = str(tmp_path / "export.json")
        ctx, _ = _mock_client_ctx(mock_response)
        try:
            parser = build_parser()
            args = parser.parse_args(["export", "myproj", "-o", output_file])
            result = cmd_export(args)
        finally:
            ctx.stop()

        assert result == 0
        with open(output_file) as f:
            data = json.load(f)
        assert data["project"] == "myproj"
        assert data["count"] == 1
        assert data["memories"][0]["text"] == "test memory"

    def test_export_json_structure(self, mock_response: MagicMock, tmp_path) -> None:
        mock_response.json.return_value = [
            {
                "id": "1", "text": "fact one", "user_id": "u1",
                "created_at": 100, "updated_at": 100, "metadata": {},
            },
            {
                "id": "2", "text": "fact two", "user_id": "u1",
                "created_at": 200, "updated_at": 200, "metadata": {"source": "test"},
            },
        ]
        output_file = str(tmp_path / "export.json")
        ctx, _ = _mock_client_ctx(mock_response)
        try:
            parser = build_parser()
            args = parser.parse_args(["export", "myproj", "-o", output_file])
            cmd_export(args)
        finally:
            ctx.stop()

        with open(output_file) as f:
            data = json.load(f)
        assert "exported_at" in data
        assert data["count"] == 2
