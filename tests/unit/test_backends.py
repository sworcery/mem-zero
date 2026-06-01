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
        fb.generate.return_value = '["fallback fact"]'
        fb.embed.return_value = [[0.2] * 768]
        factory = MagicMock(return_value=fb)
        return factory

    @pytest.fixture
    def backend(self, primary: AsyncMock, fallback_factory: MagicMock) -> FallbackBackend:
        return FallbackBackend(primary, fallback_factory)

    @pytest.mark.asyncio
    async def test_uses_primary_when_healthy(
        self, backend: FallbackBackend, primary: AsyncMock
    ) -> None:
        result = await backend.generate("sys", "user")
        assert result == '["fact"]'
        primary.generate.assert_called_once_with("sys", "user")

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
        assert backend.is_degraded is True

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

    @pytest.mark.asyncio
    async def test_health_ping_success(self) -> None:
        backend = OllamaBackend.__new__(OllamaBackend)
        mock_http = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_http.get.return_value = mock_resp
        backend._http = mock_http
        assert await backend.health_ping() is True

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
