from __future__ import annotations

import pytest

from mem_zero.config import Config, validate_slug


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


class TestConfig:
    def test_defaults(self) -> None:
        config = Config()
        assert config.port == 8765
        assert config.qdrant_host == "127.0.0.1"
        assert config.qdrant_port == 6333
        assert config.embedder_model == "nomic-embed-text"
        assert config.embedding_dimensions == 768
        assert config.collection_prefix == "mem0"

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

    def test_optional_fields_default_none(self) -> None:
        config = Config()
        assert config.qdrant_url is None
        assert config.qdrant_api_key is None

    def test_collection_name(self) -> None:
        config = Config(collection_prefix="mem0")
        assert config.collection_name("my-project") == "mem0_my-project"

    def test_collection_name_validates_slug(self) -> None:
        config = Config()
        with pytest.raises(ValueError):
            config.collection_name("invalid.slug!")
