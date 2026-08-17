from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mem_zero.backends import (
    FallbackBackend,
    LLMBackend,
    OllamaBackend,
    OpenAIBackend,
    create_backend,
)
from mem_zero.config import Config


class TestLLMBackendABC:
    def test_default_is_not_degraded(self) -> None:
        class DummyBackend(LLMBackend):
            async def generate(self, system: str, user: str) -> str:
                return ""

            async def embed(self, texts: list[str]) -> list[list[float]]:
                return []

            @property
            def embedding_dimensions(self) -> int:
                return 768

        backend = DummyBackend()
        assert backend.is_degraded is False

    @pytest.mark.asyncio
    async def test_default_health_ping(self) -> None:
        class DummyBackend(LLMBackend):
            async def generate(self, system: str, user: str) -> str:
                return ""

            async def embed(self, texts: list[str]) -> list[list[float]]:
                return []

            @property
            def embedding_dimensions(self) -> int:
                return 768

        backend = DummyBackend()
        assert await backend.health_ping() is True


class TestFallbackBackend:
    @pytest.fixture
    def primary(self) -> AsyncMock:
        p = AsyncMock(spec=LLMBackend)
        p.embedding_dimensions = 768
        p.is_degraded = False
        p.generate.return_value = '["fact"]'
        p.embed.return_value = [[0.1] * 768]
        return p

    @pytest.fixture
    def fallback_factory(self) -> MagicMock:
        fb = AsyncMock()
        fb.embedding_dimensions = 768
        fb.generate.return_value = '["fallback fact"]'
        fb.embed.return_value = [[0.2] * 768]
        factory = MagicMock(return_value=fb)
        return factory

    @pytest.fixture
    def backend(self, primary: AsyncMock, fallback_factory: MagicMock) -> FallbackBackend:
        # cooldown_seconds=0 keeps the pre-existing tests exercising the
        # retry-primary-on-every-call path; the breaker has its own test below.
        return FallbackBackend(primary, fallback_factory, cooldown_seconds=0.0)

    @pytest.mark.asyncio
    async def test_uses_primary_when_healthy(
        self, backend: FallbackBackend, primary: AsyncMock
    ) -> None:
        result = await backend.generate("sys", "user")
        assert result == '["fact"]'
        primary.generate.assert_called_once_with("sys", "user", None)

    @pytest.mark.asyncio
    async def test_falls_back_on_primary_failure(
        self, backend: FallbackBackend, primary: AsyncMock, fallback_factory: MagicMock
    ) -> None:
        primary.generate.side_effect = Exception("ollama down")
        result = await backend.generate("sys", "user")
        assert result == '["fallback fact"]'
        assert backend.is_degraded is True
        fallback_factory.assert_called_once()

    @pytest.mark.asyncio
    async def test_recovers_when_primary_returns(
        self, backend: FallbackBackend, primary: AsyncMock
    ) -> None:
        primary.generate.side_effect = Exception("down")
        await backend.generate("sys", "user")
        assert backend.is_degraded is True

        primary.generate.side_effect = None
        primary.generate.return_value = '["recovered"]'
        result = await backend.generate("sys", "user")
        assert result == '["recovered"]'
        assert backend.is_degraded is False

    @pytest.mark.asyncio
    async def test_embed_falls_back(
        self, backend: FallbackBackend, primary: AsyncMock, fallback_factory: MagicMock
    ) -> None:
        primary.embed.side_effect = Exception("embed failed")
        result = await backend.embed(["test"])
        assert result == [[0.2] * 768]
        # Embed and generate breakers are independent: an embed outage must
        # NOT report the LLM as degraded (that would make the engine skip
        # dedup for every concurrent add).
        assert backend.embed_degraded is True
        assert backend.is_degraded is False

    @pytest.mark.asyncio
    async def test_health_ping_delegates_to_primary(
        self, backend: FallbackBackend, primary: AsyncMock
    ) -> None:
        primary.health_ping.return_value = True
        assert await backend.health_ping() is True

    @pytest.mark.asyncio
    async def test_close_closes_both(
        self, backend: FallbackBackend, primary: AsyncMock, fallback_factory: MagicMock
    ) -> None:
        primary.generate.side_effect = Exception("down")
        await backend.generate("sys", "user")
        await backend.close()
        primary.close.assert_called_once()
        fallback_factory.return_value.close.assert_called_once()

    def test_embedding_dimensions_from_primary(
        self, backend: FallbackBackend, primary: AsyncMock
    ) -> None:
        assert backend.embedding_dimensions == 768

    @pytest.mark.asyncio
    async def test_circuit_breaker_skips_primary_during_cooldown(
        self, primary: AsyncMock, fallback_factory: MagicMock
    ) -> None:
        backend = FallbackBackend(primary, fallback_factory, cooldown_seconds=60.0)
        primary.generate.side_effect = Exception("down")
        await backend.generate("sys", "user")  # trips the breaker
        assert primary.generate.call_count == 1

        # Primary "recovers", but we are inside the cooldown window, so the next
        # call must fail over instantly WITHOUT touching the primary again.
        primary.generate.side_effect = None
        primary.generate.return_value = '["recovered"]'
        result = await backend.generate("sys", "user")
        assert result == '["fallback fact"]'
        assert primary.generate.call_count == 1  # not retried during cooldown
        assert backend.is_degraded is True

    @pytest.mark.asyncio
    async def test_embed_fallback_rejects_dimension_mismatch(
        self, primary: AsyncMock
    ) -> None:
        fb = AsyncMock()
        fb.embedding_dimensions = 1024  # != primary's 768
        fb.embed.return_value = [[0.3] * 1024]
        backend = FallbackBackend(
            primary, MagicMock(return_value=fb), cooldown_seconds=0.0
        )
        primary.embed.side_effect = Exception("embed down")
        # EmbeddingError (not RuntimeError) so the server maps it to 503 with a
        # hint instead of a bare 500.
        from mem_zero.backends import EmbeddingError
        with pytest.raises(EmbeddingError, match="dim"):
            await backend.embed(["test"])


class TestCreateBackend:
    def test_openai_backend_requires_key(self) -> None:
        config = Config(llm_backend="openai", openai_api_key=None)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            create_backend(config)

    def test_openai_backend_created(self) -> None:
        config = Config(llm_backend="openai", openai_api_key="sk-test")
        backend = create_backend(config)
        assert isinstance(backend, OpenAIBackend)

    def test_ollama_returns_fallback_backend(self) -> None:
        config = Config(llm_backend="ollama")
        with patch("mem_zero.backends.OllamaBackend"):
            backend = create_backend(config)
            assert isinstance(backend, FallbackBackend)

    def test_bundled_backend_created(self) -> None:
        config = Config(llm_backend="bundled")
        with patch("mem_zero.backends.BundledBackend") as mock_bundled:
            mock_bundled.return_value = MagicMock()
            create_backend(config)
            mock_bundled.assert_called_once()


class TestOllamaBackend:
    def test_dimensions(self) -> None:
        backend = OllamaBackend.__new__(OllamaBackend)
        backend._dims = 768
        assert backend.embedding_dimensions == 768

    @staticmethod
    def _backend_with_tags(models: list[str]) -> OllamaBackend:
        backend = OllamaBackend.__new__(OllamaBackend)
        backend._llm_model = "qwen2.5:14b"
        backend._embed_model = "mxbai-embed-large"
        mock_http = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(
            return_value={"models": [{"name": m} for m in models]}
        )
        mock_http.get.return_value = mock_resp
        backend._http = mock_http
        return backend

    @pytest.mark.asyncio
    async def test_health_ping_success(self) -> None:
        backend = self._backend_with_tags(
            ["qwen2.5:14b", "mxbai-embed-large:latest"]
        )
        assert await backend.health_ping() is True

    @pytest.mark.asyncio
    async def test_health_ping_false_when_embed_model_not_pulled(self) -> None:
        # Ollama answers 200 but the configured embedder was never pulled —
        # every embed call fails, so health must report degraded.
        backend = self._backend_with_tags(["qwen2.5:14b", "nomic-embed-text:latest"])
        assert await backend.health_ping() is False

    @pytest.mark.asyncio
    async def test_health_ping_false_when_chat_model_not_pulled(self) -> None:
        backend = self._backend_with_tags(["mxbai-embed-large:latest"])
        assert await backend.health_ping() is False

    @pytest.mark.asyncio
    async def test_health_ping_failure(self) -> None:
        backend = OllamaBackend.__new__(OllamaBackend)
        mock_http = AsyncMock()
        mock_http.get.side_effect = Exception("connection refused")
        backend._http = mock_http
        assert await backend.health_ping() is False


class TestOpenAIBackend:
    def test_dimensions(self) -> None:
        backend = OpenAIBackend.__new__(OpenAIBackend)
        backend._dims = 1536
        assert backend.embedding_dimensions == 1536


class TestFallbackBreakerIndependence:
    @pytest.fixture
    def primary(self) -> AsyncMock:
        p = AsyncMock(spec=LLMBackend)
        p.embedding_dimensions = 768
        p.generate.return_value = '["fact"]'
        p.embed.return_value = [[0.1] * 768]
        return p

    @pytest.fixture
    def factory(self) -> MagicMock:
        fb = AsyncMock()
        fb.embedding_dimensions = 768
        fb.generate.return_value = '["fallback fact"]'
        fb.embed.return_value = [[0.2] * 768]
        return MagicMock(return_value=fb)

    @pytest.mark.asyncio
    async def test_embed_failure_does_not_degrade_generate(
        self, primary: AsyncMock, factory: MagicMock
    ) -> None:
        backend = FallbackBackend(primary, factory, cooldown_seconds=60.0)
        primary.embed.side_effect = Exception("embed down")
        await backend.embed(["x"])
        # generate is untouched: still healthy, still hits the primary.
        assert backend.is_degraded is False
        result = await backend.generate("s", "u")
        assert result == '["fact"]'
        primary.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_cooldown_does_not_block_embed(
        self, primary: AsyncMock, factory: MagicMock
    ) -> None:
        backend = FallbackBackend(primary, factory, cooldown_seconds=60.0)
        primary.generate.side_effect = Exception("llm down")
        await backend.generate("s", "u")  # trips the GENERATE breaker only
        assert backend.is_degraded is True
        # embed must still go to the primary, not the bundled fallback.
        result = await backend.embed(["x"])
        assert result == [[0.1] * 768]
        primary.embed.assert_awaited_once()
        assert backend.embed_degraded is False

    @pytest.mark.asyncio
    async def test_generate_recovery_does_not_heal_embed(
        self, primary: AsyncMock, factory: MagicMock
    ) -> None:
        # The old single flag let a successful generate "heal" a still-broken
        # embed path (and vice-versa).
        backend = FallbackBackend(primary, factory, cooldown_seconds=0.0)
        primary.embed.side_effect = Exception("embed down")
        await backend.embed(["x"])
        assert backend.embed_degraded is True
        await backend.generate("s", "u")  # healthy generate
        assert backend.embed_degraded is True  # unchanged


class TestOllamaRetry:
    @staticmethod
    def _backend() -> OllamaBackend:
        b = OllamaBackend.__new__(OllamaBackend)
        b._llm_model = "m"
        b._embed_model = "e"
        b._dims = 768
        import asyncio
        b._semaphore = asyncio.Semaphore(2)
        return b

    @staticmethod
    def _ok(payload: dict) -> MagicMock:
        r = MagicMock()
        r.status_code = 200
        r.raise_for_status = MagicMock()
        r.json.return_value = payload
        return r

    @pytest.mark.asyncio
    async def test_retries_once_on_connect_error_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        import mem_zero.backends as bk
        monkeypatch.setattr(bk, "_OLLAMA_RETRY_BACKOFF_S", 0)
        b = self._backend()
        b._http = AsyncMock()
        b._http.post.side_effect = [
            httpx.ConnectError("reset"),
            self._ok({"embeddings": [[0.1] * 768]}),
        ]
        out = await b.embed(["x"])
        assert out == [[0.1] * 768]
        assert b._http.post.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_5xx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import mem_zero.backends as bk
        monkeypatch.setattr(bk, "_OLLAMA_RETRY_BACKOFF_S", 0)
        b = self._backend()
        b._http = AsyncMock()
        bad = MagicMock()
        bad.status_code = 503
        bad.request = MagicMock()
        b._http.post.side_effect = [bad, self._ok({"embeddings": [[0.1] * 768]})]
        out = await b.embed(["x"])
        assert out == [[0.1] * 768]
        assert b._http.post.call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_4xx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        import mem_zero.backends as bk
        monkeypatch.setattr(bk, "_OLLAMA_RETRY_BACKOFF_S", 0)
        b = self._backend()
        b._http = AsyncMock()
        bad = MagicMock()
        bad.status_code = 404
        bad.raise_for_status.side_effect = httpx.HTTPStatusError(
            "not found", request=MagicMock(), response=bad
        )
        b._http.post.return_value = bad
        with pytest.raises(httpx.HTTPStatusError):
            await b.embed(["x"])
        assert b._http.post.call_count == 1  # config error: never retried

    @pytest.mark.asyncio
    async def test_gives_up_after_two_transport_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        import mem_zero.backends as bk
        monkeypatch.setattr(bk, "_OLLAMA_RETRY_BACKOFF_S", 0)
        b = self._backend()
        b._http = AsyncMock()
        b._http.post.side_effect = httpx.ConnectError("down")
        with pytest.raises(httpx.ConnectError):
            await b.embed(["x"])
        assert b._http.post.call_count == 2


class TestOpenAIEmbed:
    @staticmethod
    def _backend(model: str = "text-embedding-3-small") -> OpenAIBackend:
        b = OpenAIBackend.__new__(OpenAIBackend)
        b._llm_model = "gpt"
        b._embed_model = model
        b._dims = 1536
        b._send_dimensions = model.startswith("text-embedding-3")
        b._http = AsyncMock()
        return b

    @staticmethod
    def _resp(data: list[dict]) -> MagicMock:
        r = MagicMock()
        r.raise_for_status = MagicMock()
        r.json.return_value = {"data": data}
        return r

    @pytest.mark.asyncio
    async def test_embed_sorted_by_index(self) -> None:
        # The API returns "index" because order is not guaranteed; add() zips
        # vectors against facts, so a reorder would silently mis-attach them.
        b = self._backend()
        b._http.post.return_value = self._resp(
            [{"index": 1, "embedding": [1.0]}, {"index": 0, "embedding": [0.0]}]
        )
        assert await b.embed(["a", "b"]) == [[0.0], [1.0]]

    @pytest.mark.asyncio
    async def test_sends_dimensions_for_v3(self) -> None:
        b = self._backend("text-embedding-3-small")
        b._http.post.return_value = self._resp([{"index": 0, "embedding": [0.0]}])
        await b.embed(["a"])
        assert b._http.post.call_args.kwargs["json"]["dimensions"] == 1536

    @pytest.mark.asyncio
    async def test_omits_dimensions_for_ada(self) -> None:
        # ada-002 (and most OpenAI-compatible proxies) reject the field.
        b = self._backend("text-embedding-ada-002")
        b._http.post.return_value = self._resp([{"index": 0, "embedding": [0.0]}])
        await b.embed(["a"])
        assert "dimensions" not in b._http.post.call_args.kwargs["json"]

    @pytest.mark.asyncio
    async def test_health_ping_uses_models_endpoint(self) -> None:
        b = self._backend()
        ok = MagicMock()
        ok.status_code = 200
        b._http.get.return_value = ok
        assert await b.health_ping() is True
        assert b._http.get.call_args.args[0] == "/models"

    @pytest.mark.asyncio
    async def test_health_ping_false_on_error(self) -> None:
        # A revoked key / dead endpoint must not report healthy.
        b = self._backend()
        b._http.get.side_effect = Exception("401")
        assert await b.health_ping() is False


class TestCreateBackendDimensions:
    def test_openai_defaults_to_model_native_dims(self) -> None:
        # The bug: config default 768 was forced onto OpenAIBackend, whose
        # 1536-dim vectors then failed against a 768-dim collection.
        config = Config(llm_backend="openai", openai_api_key="sk",
                        openai_embed_model="text-embedding-3-large")
        assert create_backend(config).embedding_dimensions == 3072

    def test_openai_small_defaults_1536(self) -> None:
        config = Config(llm_backend="openai", openai_api_key="sk")
        assert create_backend(config).embedding_dimensions == 1536

    def test_openai_explicit_dims_respected(self) -> None:
        config = Config(llm_backend="openai", openai_api_key="sk",
                        embedding_dimensions=512)
        assert create_backend(config).embedding_dimensions == 512

    def test_ollama_defaults_768_when_unset(self) -> None:
        config = Config(llm_backend="ollama")
        with patch("mem_zero.backends.OllamaBackend") as ob:
            create_backend(config)
        assert ob.call_args.kwargs["dimensions"] == 768

    def test_from_env_dims_unset_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EMBEDDER_DIMENSIONS", raising=False)
        assert Config.from_env().embedding_dimensions is None
