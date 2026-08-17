from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from mem_zero.cli import build_parser, cmd_export, cmd_health, cmd_projects, main


@pytest.fixture
def mock_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.headers = {}
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


class TestMainErrorHandling:
    def _run(self, argv: list[str], http: MagicMock) -> int:
        ctx = patch("mem_zero.cli._client")
        mock = ctx.start()
        mock.return_value.__enter__ = MagicMock(return_value=http)
        mock.return_value.__exit__ = MagicMock(return_value=False)
        try:
            with patch("sys.argv", argv):
                return main()
        finally:
            ctx.stop()

    def test_timeout_returns_exit_1(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        http = MagicMock()
        http.get.side_effect = httpx.ReadTimeout("slow")
        rc = self._run(["mem-zero", "health"], http)
        assert rc == 1
        assert "timed out" in capsys.readouterr().err

    def test_invalid_json_returns_exit_1(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("not json")
        http = MagicMock()
        http.get.return_value = resp
        rc = self._run(["mem-zero", "health"], http)
        assert rc == 1
        assert "invalid response" in capsys.readouterr().err

    def test_missing_field_returns_exit_1(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Version-skewed server renames a field the renderer indexes directly.
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = [{"name": "wrong-shape"}]
        resp.raise_for_status = MagicMock()
        http = MagicMock()
        http.get.return_value = resp
        rc = self._run(["mem-zero", "projects"], http)
        assert rc == 1
        assert "unexpected response" in capsys.readouterr().err


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
        args = parser.parse_args(["--url", "http://192.0.2.10:8765", "health"])
        assert args.url == "http://192.0.2.10:8765"

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


def _resp(body: object, headers: dict[str, str] | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = body
    resp.headers = headers or {}
    return resp


def _run_main(argv: list[str], http: MagicMock) -> int:
    ctx = patch("mem_zero.cli._client")
    mock = ctx.start()
    mock.return_value.__enter__ = MagicMock(return_value=http)
    mock.return_value.__exit__ = MagicMock(return_value=False)
    try:
        with patch("sys.argv", argv):
            return main()
    finally:
        ctx.stop()


class TestExportPagination:
    def test_follows_next_offset(self, tmp_path) -> None:
        page1 = [{"id": "a", "text": "one"}, {"id": "b", "text": "two"}]
        page2 = [{"id": "c", "text": "three"}]
        http = MagicMock()
        http.get.side_effect = [
            _resp(page1, {"X-Next-Offset": "b"}),
            _resp(page2),
        ]
        out = tmp_path / "export.json"
        rc = _run_main(["mem-zero-cli", "export", "myproj", "-o", str(out)], http)
        assert rc == 0
        assert http.get.call_count == 2
        second_params = http.get.call_args_list[1].kwargs["params"]
        assert second_params["offset"] == "b"
        assert second_params["limit"] == 1000
        data = json.loads(out.read_text())
        assert [m["id"] for m in data["memories"]] == ["a", "b", "c"]
        assert data["count"] == 3

    def test_repeated_offset_breaks_loop(self, tmp_path) -> None:
        http = MagicMock()
        http.get.return_value = _resp([{"id": "a", "text": "x"}], {"X-Next-Offset": "a"})
        out = tmp_path / "export.json"
        rc = _run_main(["mem-zero-cli", "export", "myproj", "-o", str(out)], http)
        assert rc == 0
        assert http.get.call_count == 2

    def test_atomic_on_failure(self, tmp_path) -> None:
        out = tmp_path / "export.json"
        out.write_text('{"previous": true}')
        http = MagicMock()
        http.get.return_value = _resp([{"id": "a", "text": "x"}])
        with patch("mem_zero.cli.json.dump", side_effect=OSError("disk full")):
            rc = _run_main(["mem-zero-cli", "export", "myproj", "-o", str(out)], http)
        assert rc == 1
        assert json.loads(out.read_text()) == {"previous": True}
        assert [p.name for p in tmp_path.iterdir()] == ["export.json"]


class TestImportValidation:
    def test_rejects_top_level_list(
        self, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / "in.json"
        f.write_text(json.dumps([{"text": "hello"}]))
        http = MagicMock()
        rc = _run_main(["mem-zero-cli", "import", str(f), "--project", "p"], http)
        assert rc == 1
        assert "invalid import file" in capsys.readouterr().err
        http.post.assert_not_called()

    def test_rejects_memories_without_text(
        self, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / "in.json"
        f.write_text(json.dumps({"project": "p", "memories": ["just a string", {"text": "ok"}]}))
        http = MagicMock()
        rc = _run_main(["mem-zero-cli", "import", str(f)], http)
        assert rc == 1
        assert "invalid import file" in capsys.readouterr().err
        http.post.assert_not_called()

    def test_reports_stored_sum(
        self, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / "in.json"
        f.write_text(json.dumps({"project": "p", "memories": [{"text": "a"}, {"text": "b"}]}))
        http = MagicMock()
        http.post.side_effect = [_resp({"stored": 2, "ids": ["1", "2"]}), _resp({"stored": 0})]
        rc = _run_main(["mem-zero-cli", "import", str(f)], http)
        assert rc == 0
        assert "Imported 2 inputs, stored 2 facts." in capsys.readouterr().out

    def test_json_output(self, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
        f = tmp_path / "in.json"
        f.write_text(json.dumps({"project": "p", "memories": [{"text": "a"}]}))
        http = MagicMock()
        http.post.return_value = _resp({"stored": 1, "ids": ["1"]})
        rc = _run_main(["mem-zero-cli", "--json", "import", str(f)], http)
        assert rc == 0
        assert json.loads(capsys.readouterr().out) == {"inputs": 1, "stored": 1}


class TestClientAuth:
    def test_env_var_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mem_zero.cli import _client

        monkeypatch.setenv("MEM_ZERO_API_KEY", "from-env")
        args = build_parser().parse_args(["health"])
        with _client(args) as client:
            assert client.headers["Authorization"] == "Bearer from-env"

    def test_flag_beats_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mem_zero.cli import _client

        monkeypatch.setenv("MEM_ZERO_API_KEY", "from-env")
        args = build_parser().parse_args(["--api-key", "from-flag", "health"])
        with _client(args) as client:
            assert client.headers["Authorization"] == "Bearer from-flag"

    def test_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mem_zero.cli import _client

        monkeypatch.delenv("MEM_ZERO_API_KEY", raising=False)
        args = build_parser().parse_args(["health"])
        with _client(args) as client:
            assert client.timeout.read == 120.0
            assert client.timeout.connect == 10.0
            assert "Authorization" not in client.headers


class TestSlugAndPaths:
    def test_prog_name(self) -> None:
        assert build_parser().prog == "mem-zero-cli"

    def test_invalid_slug_exits_2_without_http(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        http = MagicMock()
        rc = _run_main(["mem-zero-cli", "list", "Bad Slug!"], http)
        assert rc == 2
        assert capsys.readouterr().err.startswith("Error: Invalid project slug")
        http.get.assert_not_called()

    def test_delete_url_encodes_memory_id(self) -> None:
        http = MagicMock()
        http.delete.return_value = _resp({"deleted": True, "id": "x/y"})
        rc = _run_main(["mem-zero-cli", "delete", "myproj", "x/y"], http)
        assert rc == 0
        url = http.delete.call_args.args[0]
        assert url == "/api/v1/projects/myproj/memories/x%2Fy"

    def test_delete_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        http = MagicMock()
        http.delete.return_value = _resp({"deleted": True, "id": "abc"})
        rc = _run_main(["mem-zero-cli", "--json", "delete", "myproj", "abc"], http)
        assert rc == 0
        assert json.loads(capsys.readouterr().out) == {"deleted": True, "id": "abc"}
