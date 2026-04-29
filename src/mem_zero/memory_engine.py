from __future__ import annotations

import asyncio
import logging
import time
import uuid

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from .config import Config
from .models import MemoryRecord, ProjectInfo

logger = logging.getLogger(__name__)

RESERVED_PAYLOAD_KEYS = frozenset({"text", "user_id", "project", "created_at", "updated_at"})


class EmbeddingError(Exception):
    pass


def validate_memory_id(value: str) -> str:
    try:
        uuid.UUID(value, version=4)
    except ValueError:
        raise ValueError(f"Invalid memory ID: {value!r}") from None
    return value


class MemoryEngine:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._http = httpx.AsyncClient(
            base_url=config.ollama_base_url,
            timeout=30.0,
        )

        if config.qdrant_url:
            self._qdrant = AsyncQdrantClient(
                url=config.qdrant_url,
                api_key=config.qdrant_api_key,
            )
        else:
            self._qdrant = AsyncQdrantClient(
                host=config.qdrant_host,
                port=config.qdrant_port,
            )

        self._ensured_collections: set[str] = set()
        self._lock = asyncio.Lock()

    async def close(self) -> None:
        await self._http.aclose()
        await self._qdrant.close()

    async def _ensure_collection(self, project_slug: str) -> str:
        name = self._config.collection_name(project_slug)

        if name in self._ensured_collections:
            return name

        async with self._lock:
            if name in self._ensured_collections:
                return name

            existing = {
                c.name for c in (await self._qdrant.get_collections()).collections
            }
            if name not in existing:
                await self._qdrant.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=self._config.embedding_dimensions,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created collection %s", name)

            self._ensured_collections.add(name)
            return name

    async def health_check(self) -> bool:
        await self._qdrant.get_collections()
        return True

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = await self._http.post(
                "/api/embed",
                json={"model": self._config.embedder_model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            return data["embeddings"]
        except (httpx.HTTPError, KeyError) as exc:
            raise EmbeddingError(f"Ollama embedding failed: {exc}") from exc

    async def add(
        self,
        project_slug: str,
        user_id: str,
        texts: list[str],
        metadata: dict[str, object] | None = None,
    ) -> list[str]:
        collection = await self._ensure_collection(project_slug)
        vectors = await self.embed(texts)
        now = time.time()
        ids: list[str] = []
        points: list[PointStruct] = []

        clean_meta = {
            k: v for k, v in (metadata or {}).items() if k not in RESERVED_PAYLOAD_KEYS
        }

        for text, vector in zip(texts, vectors, strict=True):
            point_id = str(uuid.uuid4())
            ids.append(point_id)
            payload = {
                **clean_meta,
                "text": text,
                "user_id": user_id,
                "project": project_slug,
                "created_at": now,
                "updated_at": now,
            }
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        await self._qdrant.upsert(collection_name=collection, points=points)
        logger.info("Added %d memories to %s", len(points), collection)
        return ids

    async def search(
        self,
        project_slug: str,
        query: str,
        top_k: int = 10,
    ) -> list[MemoryRecord]:
        collection = await self._ensure_collection(project_slug)
        vectors = await self.embed([query])

        results = (
            await self._qdrant.query_points(
                collection_name=collection,
                query=vectors[0],
                limit=top_k,
            )
        ).points

        return [
            MemoryRecord(
                id=str(hit.id),
                text=hit.payload.get("text", ""),
                user_id=hit.payload.get("user_id", ""),
                created_at=hit.payload.get("created_at", 0),
                updated_at=hit.payload.get("updated_at", 0),
                metadata={
                    k: v
                    for k, v in hit.payload.items()
                    if k not in RESERVED_PAYLOAD_KEYS
                },
                score=hit.score,
            )
            for hit in results
        ]

    async def list_all(
        self,
        project_slug: str,
        limit: int = 50,
        offset: int | None = None,
    ) -> list[MemoryRecord]:
        collection = await self._ensure_collection(project_slug)
        points, _next_offset = await self._qdrant.scroll(
            collection_name=collection,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        return [
            MemoryRecord(
                id=str(pt.id),
                text=pt.payload.get("text", ""),
                user_id=pt.payload.get("user_id", ""),
                created_at=pt.payload.get("created_at", 0),
                updated_at=pt.payload.get("updated_at", 0),
                metadata={
                    k: v
                    for k, v in pt.payload.items()
                    if k not in RESERVED_PAYLOAD_KEYS
                },
            )
            for pt in points
        ]

    async def delete(self, project_slug: str, memory_id: str) -> bool:
        validate_memory_id(memory_id)
        collection = await self._ensure_collection(project_slug)
        await self._qdrant.delete(
            collection_name=collection,
            points_selector=[memory_id],
        )
        return True

    async def delete_all(self, project_slug: str) -> int:
        collection = await self._ensure_collection(project_slug)
        info = await self._qdrant.get_collection(collection)
        count = info.points_count or 0
        await self._qdrant.delete_collection(collection)
        self._ensured_collections.discard(collection)
        await self._ensure_collection(project_slug)
        return count

    async def list_projects(self) -> list[ProjectInfo]:
        prefix = f"{self._config.collection_prefix}_"
        projects: list[ProjectInfo] = []
        for col in (await self._qdrant.get_collections()).collections:
            if col.name.startswith(prefix):
                slug = col.name[len(prefix) :]
                info = await self._qdrant.get_collection(col.name)
                projects.append(
                    ProjectInfo(
                        slug=slug,
                        collection=col.name,
                        memory_count=info.points_count or 0,
                    )
                )
        return projects
