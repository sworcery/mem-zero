from __future__ import annotations

import pytest

from mem_zero.config import Config, _ensure_scheme, validate_slug


class TestValidateSlug:
    def test_valid_slugs(self) -> None:
        assert validate_slug("my-project") == "my-project"
        assert validate_slug("project123") == "project123"
        assert validate_slug("a") == "a"
        assert validate_slug("my_project") == "my_project"

    def test_rejects_uppercase(self) -> None:
        with pytest.raises(ValueError, match="Invalid project slug"):
            validate_slug("MyProject")

    def test_rejects_spaces(self) -> None:
        with pytest.raises(ValueError, match="Invalid project slug"):
            validate_slug("my project")

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="Invalid project slug"):
            validate_slug("")

    def test_rejects_special_chars(self) -> None:
        with pytest.raises(ValueError, match="Invalid project slug"):
            validate_slug("my.project")

    def test_rejects_leading_hyphen(self) -> None:
        with pytest.raises(ValueError, match="Invalid project slug"):
            validate_slug("-project")

    def test_rejects_too_long(self) -> None:
        with pytest.raises(ValueError, match="Invalid project slug"):
            validate_slug("a" * 64)

    def test_rejects_trailing_newline(self) -> None:
        with pytest.raises(ValueError, match="Invalid project slug"):
            validate_slug("project\n")


class TestConfig:
    def test_defaults(self) -> None:
        config = Config()
        assert config.port == 8765
        assert config.qdrant_host == "127.0.0.1"
        assert config.qdrant_port == 6333
        assert config.embedder_model == "nomic-embed-text"
        # None = not explicitly set; create_backend resolves a per-backend
        # default (768 for Ollama/bundled, the model's native size for OpenAI).
        assert config.embedding_dimensions is None
        assert config.collection_prefix == "mem-zero"

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PORT", "9999")
        monkeypatch.setenv("QDRANT_HOST", "qdrant.local")
        monkeypatch.setenv("EMBEDDER_MODEL", "bge-small")
        monkeypatch.setenv("EMBEDDER_DIMENSIONS", "384")

        config = Config.from_env()
        assert config.port == 9999
        assert config.qdrant_host == "qdrant.local"
        assert config.embedder_model == "bge-small"
        assert config.embedding_dimensions == 384

    def test_from_env_invalid_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PORT", "abc")
        with pytest.raises(ValueError, match="Invalid integer for PORT"):
            Config.from_env()

    def test_from_env_invalid_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_BACKEND", "ollamaa")
        with pytest.raises(ValueError, match="Invalid LLM_BACKEND"):
            Config.from_env()

    def test_from_env_rejects_non_positive_max_concurrent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OLLAMA_MAX_CONCURRENT", "0")
        with pytest.raises(ValueError, match="OLLAMA_MAX_CONCURRENT must be >= 1"):
            Config.from_env()

    def test_optional_fields_default_none(self) -> None:
        config = Config()
        assert config.qdrant_url is None
        assert config.qdrant_api_key is None

    def test_collection_name(self) -> None:
        config = Config(collection_prefix="mem-zero")
        assert config.collection_name("my-project") == "mem-zero_my-project"

    def test_collection_name_validates_slug(self) -> None:
        config = Config()
        with pytest.raises(ValueError):
            config.collection_name("invalid.slug!")

    def test_from_env_prepends_http_to_ollama_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_BASE_URL", "192.0.2.10:11434")
        monkeypatch.setenv("LLM_BACKEND", "ollama")
        config = Config.from_env()
        assert config.ollama_base_url == "http://192.0.2.10:11434"

    def test_from_env_preserves_explicit_scheme(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.example.com")
        monkeypatch.setenv("LLM_BACKEND", "ollama")
        config = Config.from_env()
        assert config.ollama_base_url == "https://ollama.example.com"


class TestEnsureScheme:
    def test_adds_http_when_missing(self) -> None:
        assert _ensure_scheme("192.0.2.10:11434") == "http://192.0.2.10:11434"

    def test_preserves_existing_http(self) -> None:
        assert _ensure_scheme("http://localhost:11434") == "http://localhost:11434"

    def test_preserves_existing_https(self) -> None:
        assert _ensure_scheme("https://api.example.com") == "https://api.example.com"

    def test_custom_default_scheme(self) -> None:
        assert _ensure_scheme("api.openai.com/v1", "https") == "https://api.openai.com/v1"

    def test_empty_string_unchanged(self) -> None:
        assert _ensure_scheme("") == ""
