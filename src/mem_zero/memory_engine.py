from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .backends import LLMBackend
from .config import Config
from .models import MemoryRecord, ProjectInfo

logger = logging.getLogger(__name__)

RESERVED_PAYLOAD_KEYS = frozenset({"text", "user_id", "project", "created_at", "updated_at"})

EXTRACT_PROMPT = """\
You are a memory extraction system. Extract distinct, atomic facts from the \
input text. Each fact should be a single, self-contained statement that captures \
one piece of information — a preference, habit, biographical detail, technical \
choice, opinion, or anything worth remembering about the user.

Rules:
- Write each fact as a short, third-person statement.
- Do NOT include facts that are purely transient or conversational filler.
- If the text contains no memorable facts, return an empty list.
- Respond with ONLY a JSON array of strings, no other text.

Example input: "I'm a data scientist working at Acme Corp. I prefer Python over R \
and I've been using PostgreSQL for my data pipelines."

Example output: ["User is a data scientist", "User works at Acme Corp", \
"User prefers Python over R", "User uses PostgreSQL for data pipelines"]
"""

DEDUP_PROMPT = """\
You are a memory deduplication system. Compare a NEW fact against EXISTING \
memories and decide what to do.

Respond with ONLY a JSON object:
- If the new fact is truly novel: {{"action": "add"}}
- If it updates/replaces an existing memory: \
{{"action": "update", "id": "<memory_id>", "text": "<merged text>"}}
- If it duplicates an existing memory with no new info: {{"action": "skip"}}

EXISTING MEMORIES:
{existing}

NEW FACT:
{new_fact}
"""


class EmbeddingError(Exception):
    pass


class LLMError(Exception):
    pass


def validate_memory_id(value: str) -> str:
    try:
        uuid.UUID(value, version=4)
    except ValueError:
        raise ValueError(f"Invalid memory ID: {value!r}") from None
    return value


class MemoryEngine:
    def __init__(self, config: Config, backend: LLMBackend) -> None:
        self._config = config
        self._backend = backend

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
        await self._backend.close()
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
                        size=self._backend.embedding_dimensions,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created collection %s", name)

            self._ensured_collections.add(name)
            return name

    async def health_check(self) -> bool:
        await self._qdrant.get_collections()
        return True

    async def _extract_facts(self, text: str) -> list[str]:
        try:
            raw = await self._backend.generate(EXTRACT_PROMPT, text)
        except Exception as exc:
            raise LLMError(f"Fact extraction failed: {exc}") from exc
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                parsed = parsed["facts"] if "facts" in parsed else list(parsed.keys())
            if not isinstance(parsed, list):
                return [str(parsed)]
            facts = [str(f) for f in parsed if f]
            if not facts:
                logger.warning("LLM returned empty fact list, storing raw text")
                return [text]
            return facts
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON for extraction, storing raw text")
            return [text]

    async def _dedup_fact(
        self, collection: str, fact: str, user_id: str
    ) -> tuple[str, str | None, str | None]:
        if self._backend.is_degraded:
            logger.debug("Skipping LLM dedup (degraded mode), adding directly")
            return "add", None, None

        results = await self.search_in_collection(collection, fact, top_k=5, user_id=user_id)
        if not results:
            return "add", None, None

        existing_lines = "\n".join(f"- [id={r.id}] {r.text}" for r in results)
        if not existing_lines:
            return "add", None, None

        prompt = DEDUP_PROMPT.format(existing=existing_lines, new_fact=fact)
        try:
            raw = await self._backend.generate(prompt, "")
        except Exception:
            return "add", None, None
        try:
            result = json.loads(raw)
            action = result.get("action", "add")
            if action == "update":
                return "update", result.get("id"), result.get("text", fact)
            if action == "skip":
                return "skip", None, None
            return "add", None, None
        except json.JSONDecodeError:
            return "add", None, None

    async def search_in_collection(
        self, collection: str, query: str, top_k: int = 10, user_id: str | None = None
    ) -> list[MemoryRecord]:
        vectors = await self._backend.embed([query])
        query_filter = None
        if user_id:
            query_filter = Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            )
        results = (
            await self._qdrant.query_points(
                collection_name=collection,
                query=vectors[0],
                query_filter=query_filter,
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
                    k: v for k, v in hit.payload.items() if k not in RESERVED_PAYLOAD_KEYS
                },
                score=hit.score,
            )
            for hit in results
        ]

    async def add(
        self,
        project_slug: str,
        user_id: str,
        texts: list[str],
        metadata: dict[str, object] | None = None,
    ) -> list[str]:
        collection = await self._ensure_collection(project_slug)
        now = time.time()
        ids: list[str] = []

        clean_meta = {
            k: v for k, v in (metadata or {}).items() if k not in RESERVED_PAYLOAD_KEYS
        }

        all_facts: list[str] = []
        for text in texts:
            facts = await self._extract_facts(text)
            all_facts.extend(facts)

        if not all_facts:
            return ids

        for fact in all_facts:
            action, update_id, merged_text = await self._dedup_fact(
                collection, fact, user_id
            )

            if action == "skip":
                logger.debug("Skipping duplicate: %s", fact[:80])
                continue

            if action == "update" and update_id:
                text_to_store = merged_text or fact
                vectors = await self._backend.embed([text_to_store])
                try:
                    validate_memory_id(update_id)
                    await self._qdrant.upsert(
                        collection_name=collection,
                        points=[
                            PointStruct(
                                id=update_id,
                                vector=vectors[0],
                                payload={
                                    **clean_meta,
                                    "text": text_to_store,
                                    "user_id": user_id,
                                    "project": project_slug,
                                    "created_at": now,
                                    "updated_at": now,
                                },
                            )
                        ],
                    )
                    ids.append(update_id)
                    logger.info("Updated memory %s in %s", update_id, collection)
                except ValueError:
                    action = "add"

            if action == "add":
                vectors = await self._backend.embed([fact])
                point_id = str(uuid.uuid4())
                ids.append(point_id)
                await self._qdrant.upsert(
                    collection_name=collection,
                    points=[
                        PointStruct(
                            id=point_id,
                            vector=vectors[0],
                            payload={
                                **clean_meta,
                                "text": fact,
                                "user_id": user_id,
                                "project": project_slug,
                                "created_at": now,
                                "updated_at": now,
                            },
                        )
                    ],
                )
                logger.info("Added memory %s to %s", point_id, collection)

        return ids

    async def search(
        self,
        project_slug: str,
        query: str,
        top_k: int = 10,
    ) -> list[MemoryRecord]:
        collection = await self._ensure_collection(project_slug)
        return await self.search_in_collection(collection, query, top_k=top_k)

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

    async def reembed_all(self, project_slug: str) -> int:
        collection = await self._ensure_collection(project_slug)
        updated = 0
        offset = None
        while True:
            points, next_offset = await self._qdrant.scroll(
                collection_name=collection,
                limit=50,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            for pt in points:
                text = pt.payload.get("text", "")
                if not text:
                    continue
                vectors = await self._backend.embed([text])
                await self._qdrant.upsert(
                    collection_name=collection,
                    points=[PointStruct(id=pt.id, vector=vectors[0], payload=pt.payload)],
                )
                updated += 1
            if next_offset is None:
                break
            offset = next_offset
        logger.info("Re-embedded %d points in %s", updated, collection)
        return updated

    @staticmethod
    def _clean_dict_text(text: str) -> list[str] | None:
        if not (text.startswith("{") and text.endswith("}")):
            return None
        try:
            import ast
            parsed = ast.literal_eval(text)
            if not isinstance(parsed, dict):
                return None
            keys = [str(k).strip() for k in parsed if str(k).strip()]
            return keys if keys else None
        except (ValueError, SyntaxError):
            return None

    async def cleanup_text(self, project_slug: str) -> dict[str, int]:
        collection = await self._ensure_collection(project_slug)
        cleaned = 0
        split = 0
        skipped = 0
        offset = None
        while True:
            points, next_offset = await self._qdrant.scroll(
                collection_name=collection,
                limit=50,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            for pt in points:
                text = pt.payload.get("text", "")
                facts = self._clean_dict_text(text)
                if facts is None:
                    skipped += 1
                    continue
                payload = dict(pt.payload)
                if len(facts) == 1:
                    payload["text"] = facts[0]
                    vectors = await self._backend.embed([facts[0]])
                    await self._qdrant.upsert(
                        collection_name=collection,
                        points=[PointStruct(id=pt.id, vector=vectors[0], payload=payload)],
                    )
                    cleaned += 1
                else:
                    await self._qdrant.delete(
                        collection_name=collection,
                        points_selector=[str(pt.id)],
                    )
                    now = payload.get("updated_at", payload.get("created_at", 0))
                    for fact in facts:
                        new_payload = {**payload, "text": fact, "updated_at": now}
                        vectors = await self._backend.embed([fact])
                        point_id = str(uuid.uuid4())
                        await self._qdrant.upsert(
                            collection_name=collection,
                            points=[PointStruct(
                                id=point_id, vector=vectors[0], payload=new_payload,
                            )],
                        )
                    split += 1
                    cleaned += 1
            if next_offset is None:
                break
            offset = next_offset
        logger.info(
            "Cleanup %s: %d cleaned, %d split, %d skipped", collection, cleaned, split, skipped
        )
        return {"cleaned": cleaned, "split_into_multiple": split, "skipped": skipped}

    async def delete_project(self, project_slug: str) -> bool:
        collection = self._config.collection_name(project_slug)
        existing = {c.name for c in (await self._qdrant.get_collections()).collections}
        if collection not in existing:
            return False
        await self._qdrant.delete_collection(collection)
        self._ensured_collections.discard(collection)
        return True

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
