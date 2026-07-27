from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .config import Config
    from .stats import DiagnosticStats, _NullStats

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    @abstractmethod
    async def generate(
        self, system: str, user: str, schema: dict[str, object] | None = None
    ) -> str: ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    @abstractmethod
    def embedding_dimensions(self) -> int: ...

    @property
    def is_degraded(self) -> bool:
        return False

    async def health_ping(self) -> bool:
        return True

    async def close(self) -> None:  # noqa: B027
        pass


class BundledBackend(LLMBackend):

    def __init__(
        self,
        model_path: str,
        embed_model: str = "nomic-ai/nomic-embed-text-v1.5",
        n_threads: int = 4,
        n_ctx: int = 4096,
    ) -> None:
        from fastembed import TextEmbedding
        from llama_cpp import Llama

        logger.info("Loading bundled LLM: %s", model_path)
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
        logger.info("Loading bundled embedder: %s", embed_model)
        self._embedder = TextEmbedding(model_name=embed_model)
        self._dims = len(list(self._embedder.embed(["dim_probe"]))[0])
        logger.info("Bundled backend ready (embed dims=%d)", self._dims)

    @property
    def embedding_dimensions(self) -> int:
        return self._dims

    async def generate(
        self, system: str, user: str, schema: dict[str, object] | None = None
    ) -> str:
        def _run() -> str:
            result = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            return result["choices"][0]["message"]["content"]  # type: ignore[index]

        return await asyncio.to_thread(_run)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        def _run() -> list[list[float]]:
            return [e.tolist() for e in self._embedder.embed(texts)]

        return await asyncio.to_thread(_run)


class OllamaBackend(LLMBackend):

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        llm_model: str = "qwen2.5:7b",
        embed_model: str = "nomic-embed-text",
        dimensions: int = 768,
        max_concurrent: int = 2,
    ) -> None:
        self._http = httpx.AsyncClient(base_url=base_url, timeout=30.0)
        self._llm_model = llm_model
        self._embed_model = embed_model
        self._dims = dimensions
        self._semaphore = asyncio.Semaphore(max_concurrent)
        logger.info("Ollama backend: max_concurrent=%d", max_concurrent)

    @property
    def embedding_dimensions(self) -> int:
        return self._dims

    async def generate(
        self, system: str, user: str, schema: dict[str, object] | None = None
    ) -> str:
        async with self._semaphore:
            resp = await self._http.post(
                "/api/chat",
                json={
                    "model": self._llm_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    # A JSON schema constrains generation to the exact shape,
                    # eliminating most parse failures; plain "json" otherwise.
                    "format": schema if schema is not None else "json",
                    # Keep the model resident so back-to-back calls don't pay a
                    # cold reload (the source of the worst-case latency spikes).
                    "keep_alive": "30m",
                    "options": {
                        # Match the other backends; low temperature is better for
                        # the JSON-constrained extraction/dedup responses.
                        "temperature": 0.1,
                        # Headroom so a long existing-memories block can't
                        # truncate the response mid-JSON.
                        "num_ctx": 8192,
                        "num_predict": 1024,
                    },
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with self._semaphore:
            resp = await self._http.post(
                "/api/embed",
                json={
                    "model": self._embed_model,
                    "input": texts,
                    "keep_alive": "30m",
                },
            )
            resp.raise_for_status()
            return resp.json()["embeddings"]

    @staticmethod
    def _model_available(configured: str, available: list[str]) -> bool:
        # Ollama reports tags like "mxbai-embed-large:latest"; a configured name
        # with no explicit tag should match the implicit ":latest".
        if configured in available:
            return True
        return ":" not in configured and f"{configured}:latest" in available

    async def health_ping(self) -> bool:
        try:
            resp = await self._http.get("/api/tags", timeout=5.0)
            if resp.status_code != 200:
                return False
            available = [m.get("name", "") for m in resp.json().get("models", [])]
        except Exception:
            return False
        # A reachable Ollama isn't enough: if a configured model was never
        # pulled, every request fails while /api/tags still answers 200.
        for model in (self._llm_model, self._embed_model):
            if not self._model_available(model, available):
                logger.warning("Ollama is reachable but model %r is not pulled", model)
                return False
        return True

    async def close(self) -> None:
        await self._http.aclose()


class OpenAIBackend(LLMBackend):

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        llm_model: str = "gpt-4o-mini",
        embed_model: str = "text-embedding-3-small",
        dimensions: int = 1536,
    ) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )
        self._llm_model = llm_model
        self._embed_model = embed_model
        self._dims = dimensions

    @property
    def embedding_dimensions(self) -> int:
        return self._dims

    async def generate(
        self, system: str, user: str, schema: dict[str, object] | None = None
    ) -> str:
        resp = await self._http.post(
            "/chat/completions",
            json={
                "model": self._llm_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._http.post(
            "/embeddings",
            json={"model": self._embed_model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return [item["embedding"] for item in data]

    async def close(self) -> None:
        await self._http.aclose()


class FallbackBackend(LLMBackend):
    """Wraps a primary backend with lazy fallback to bundled on failure."""

    def __init__(
        self,
        primary: LLMBackend,
        fallback_factory: Callable[[], BundledBackend],
        stats: DiagnosticStats | _NullStats | None = None,
        cooldown_seconds: float = 30.0,
    ) -> None:
        self._primary = primary
        self._fallback_factory = fallback_factory
        self._fallback: BundledBackend | None = None
        self._using_fallback = False
        # Circuit breaker: after the primary fails, skip it entirely for this
        # long so every request during an outage fails over instantly instead
        # of each paying the full primary timeout.
        self._cooldown_seconds = cooldown_seconds
        self._cooldown_until = 0.0
        self._fallback_lock = asyncio.Lock()
        from .stats import NULL_STATS

        self._stats = stats or NULL_STATS

    def _primary_in_cooldown(self) -> bool:
        return time.monotonic() < self._cooldown_until

    def _trip_breaker(self) -> None:
        self._cooldown_until = time.monotonic() + self._cooldown_seconds

    async def _get_fallback(self) -> BundledBackend:
        if self._fallback is not None:
            return self._fallback
        # Load the GGUF off the event loop so one Ollama hiccup doesn't freeze
        # the whole server; the lock stops two concurrent fallbacks from
        # loading the multi-GB model twice.
        async with self._fallback_lock:
            if self._fallback is None:
                logger.warning("Initializing bundled fallback backend")
                self._fallback = await asyncio.to_thread(self._fallback_factory)
        return self._fallback

    @property
    def embedding_dimensions(self) -> int:
        return self._primary.embedding_dimensions

    @property
    def is_degraded(self) -> bool:
        return self._using_fallback

    async def generate(
        self, system: str, user: str, schema: dict[str, object] | None = None
    ) -> str:
        if self._primary_in_cooldown():
            fallback = await self._get_fallback()
            return await fallback.generate(system, user, schema)
        try:
            result = await self._primary.generate(system, user, schema)
            if self._using_fallback:
                self._using_fallback = False
                self._stats.inc("backend.recovery")
            return result
        except Exception as exc:
            logger.warning("Primary backend failed (%s), using bundled fallback", exc)
            if not self._using_fallback:
                self._stats.inc("backend.fallback")
            self._using_fallback = True
            self._trip_breaker()
            fallback = await self._get_fallback()
            return await fallback.generate(system, user, schema)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._primary_in_cooldown():
            return await self._embed_fallback(texts)
        try:
            result = await self._primary.embed(texts)
            if self._using_fallback:
                self._using_fallback = False
                self._stats.inc("backend.recovery")
            return result
        except Exception as exc:
            logger.warning("Primary embedding failed (%s), using bundled fallback", exc)
            if not self._using_fallback:
                self._stats.inc("backend.fallback")
            self._using_fallback = True
            self._trip_breaker()
            return await self._embed_fallback(texts)

    async def _embed_fallback(self, texts: list[str]) -> list[list[float]]:
        fallback = await self._get_fallback()
        # A fallback that emits a different vector size than the collection was
        # created with would be silently rejected/corrupt; fail loudly instead.
        if fallback.embedding_dimensions != self._primary.embedding_dimensions:
            raise RuntimeError(
                f"Fallback embedder produces {fallback.embedding_dimensions}-dim "
                f"vectors but the collection expects "
                f"{self._primary.embedding_dimensions}. Align EMBEDDER_DIMENSIONS "
                f"with the bundled model, or use matching embedding models."
            )
        return await fallback.embed(texts)

    async def health_ping(self) -> bool:
        return await self._primary.health_ping()

    async def close(self) -> None:
        await self._primary.close()
        if self._fallback:
            await self._fallback.close()


def create_backend(
    config: Config,
    stats: DiagnosticStats | _NullStats | None = None,
) -> LLMBackend:
    if config.llm_backend == "openai":
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_BACKEND=openai")
        return OpenAIBackend(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            llm_model=config.openai_model,
            embed_model=config.openai_embed_model,
            dimensions=config.embedding_dimensions,
        )

    if config.llm_backend == "ollama":
        primary = OllamaBackend(
            base_url=config.ollama_base_url,
            llm_model=config.llm_model,
            embed_model=config.embedder_model,
            dimensions=config.embedding_dimensions,
            max_concurrent=config.ollama_max_concurrent,
        )

        def _make_fallback() -> BundledBackend:
            return BundledBackend(
                model_path=config.bundled_model_path,
                embed_model=config.bundled_embed_model,
                n_threads=config.bundled_threads,
            )

        return FallbackBackend(primary, _make_fallback, stats=stats)

    return BundledBackend(
        model_path=config.bundled_model_path,
        embed_model=config.bundled_embed_model,
        n_threads=config.bundled_threads,
    )
