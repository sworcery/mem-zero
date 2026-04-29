from __future__ import annotations

import os
import re
from dataclasses import dataclass

SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


def validate_slug(slug: str) -> str:
    if not SLUG_PATTERN.match(slug):
        raise ValueError(
            f"Invalid project slug: {slug!r}. "
            "Must be 1-63 lowercase alphanumeric chars, hyphens, or underscores."
        )
    return slug


@dataclass(frozen=True)
class Config:
    host: str = "0.0.0.0"
    port: int = 8765

    qdrant_host: str = "127.0.0.1"
    qdrant_port: int = 6333
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None

    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen2.5:7b"
    embedder_model: str = "nomic-embed-text"
    embedding_dimensions: int = 768

    collection_prefix: str = "mem0"

    @staticmethod
    def from_env() -> Config:
        def _str(key: str, default: str) -> str:
            return os.environ.get(key, default)

        def _int(key: str, default: int) -> int:
            raw = os.environ.get(key)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError:
                raise ValueError(f"Invalid integer for {key}: {raw!r}") from None

        def _opt(key: str) -> str | None:
            val = os.environ.get(key, "")
            return val if val else None

        return Config(
            host=_str("HOST", "0.0.0.0"),
            port=_int("PORT", 8765),
            qdrant_host=_str("QDRANT_HOST", "127.0.0.1"),
            qdrant_port=_int("QDRANT_PORT", 6333),
            qdrant_url=_opt("QDRANT_URL"),
            qdrant_api_key=_opt("QDRANT_API_KEY"),
            ollama_base_url=_str("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            llm_model=_str("LLM_MODEL", "qwen2.5:7b"),
            embedder_model=_str("EMBEDDER_MODEL", "nomic-embed-text"),
            embedding_dimensions=_int("EMBEDDER_DIMENSIONS", 768),
            collection_prefix=_str("COLLECTION_PREFIX", "mem0"),
        )

    def collection_name(self, project_slug: str) -> str:
        slug = validate_slug(project_slug)
        return f"{self.collection_prefix}_{slug}"
