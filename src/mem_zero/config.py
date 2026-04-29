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

    # Backend selection: "bundled", "ollama", "openai" (auto-detected if not set)
    llm_backend: str = "bundled"

    # Ollama settings (used when llm_backend=ollama)
    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen2.5:7b"
    embedder_model: str = "nomic-embed-text"
    embedding_dimensions: int = 768

    # Bundled settings (used when llm_backend=bundled, also used as fallback for ollama)
    bundled_model_path: str = "/app/models/qwen2.5-3b-instruct-q4_k_m.gguf"
    bundled_embed_model: str = "nomic-ai/nomic-embed-text-v1.5"
    bundled_threads: int = 4

    # OpenAI-compatible settings (used when llm_backend=openai)
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"

    collection_prefix: str = "mem0"

    dashboard_user: str | None = None
    dashboard_pass: str | None = None

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

        explicit_backend = os.environ.get("LLM_BACKEND", "").strip().lower()
        if explicit_backend:
            backend = explicit_backend
        elif os.environ.get("OPENAI_API_KEY"):
            backend = "openai"
        elif os.environ.get("OLLAMA_BASE_URL"):
            backend = "ollama"
        else:
            backend = "bundled"

        return Config(
            host=_str("HOST", "0.0.0.0"),
            port=_int("PORT", 8765),
            qdrant_host=_str("QDRANT_HOST", "127.0.0.1"),
            qdrant_port=_int("QDRANT_PORT", 6333),
            qdrant_url=_opt("QDRANT_URL"),
            qdrant_api_key=_opt("QDRANT_API_KEY"),
            llm_backend=backend,
            ollama_base_url=_str("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            llm_model=_str("LLM_MODEL", "qwen2.5:7b"),
            embedder_model=_str("EMBEDDER_MODEL", "nomic-embed-text"),
            embedding_dimensions=_int("EMBEDDER_DIMENSIONS", 768),
            bundled_model_path=_str(
                "BUNDLED_MODEL_PATH",
                "/app/models/qwen2.5-3b-instruct-q4_k_m.gguf",
            ),
            bundled_embed_model=_str(
                "BUNDLED_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5"
            ),
            bundled_threads=_int("BUNDLED_THREADS", 4),
            openai_api_key=_opt("OPENAI_API_KEY"),
            openai_base_url=_str("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_model=_str("OPENAI_MODEL", "gpt-4o-mini"),
            openai_embed_model=_str("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            collection_prefix=_str("COLLECTION_PREFIX", "mem0"),
            dashboard_user=_opt("DASHBOARD_USER"),
            dashboard_pass=_opt("DASHBOARD_PASS"),
        )

    def collection_name(self, project_slug: str) -> str:
        slug = validate_slug(project_slug)
        return f"{self.collection_prefix}_{slug}"
