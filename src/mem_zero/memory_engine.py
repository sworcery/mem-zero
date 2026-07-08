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
from .stats import NULL_STATS, DiagnosticStats, _NullStats

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

CONSOLIDATE_PROMPT = """\
You are a memory consolidation system. You will receive a group of related \
memory fragments that overlap in topic. Merge them into one or more clear, \
self-contained statements that preserve all distinct, useful information.

Rules:
- Each output statement must stand alone — a reader with no prior context should \
understand it fully.
- Discard fragments that are meaningless on their own (e.g. "Root cause", \
"Solution", single words like "Qdrant").
- Discard implementation details that belong in code or git history (e.g. \
"Updated line 42", "All tests pass", "Changed variable name").
- Combine overlapping fragments into a single coherent statement when they \
describe the same thing.
- If the group contains genuinely distinct facts, output multiple statements.
- It is fine to return fewer statements than inputs — that is the point.
- Respond with ONLY a JSON array of strings, no other text.

MEMORY FRAGMENTS:
{fragments}
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
    def __init__(
        self,
        config: Config,
        backend: LLMBackend,
        stats: DiagnosticStats | _NullStats | None = None,
    ) -> None:
        self._config = config
        self._backend = backend
        self._stats = stats or NULL_STATS

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

    async def _timed_generate(self, system: str, user: str) -> str:
        t0 = time.monotonic()
        try:
            result = await self._backend.generate(system, user)
            self._stats.record_latency(
                "llm_generate", (time.monotonic() - t0) * 1000
            )
            self._stats.inc("llm_generate")
            return result
        except Exception:
            self._stats.record_latency(
                "llm_generate", (time.monotonic() - t0) * 1000
            )
            raise

    async def _timed_embed(self, texts: list[str]) -> list[list[float]]:
        t0 = time.monotonic()
        try:
            result = await self._backend.embed(texts)
            self._stats.record_latency(
                "embed", (time.monotonic() - t0) * 1000
            )
            self._stats.inc("embed")
            self._stats.inc("embed.texts", len(texts))
            for i, vec in enumerate(result):
                if not any(vec):
                    self._stats.inc("embed.zero_vectors")
                    raise EmbeddingError(
                        f"Embedding model returned zero vector for text: {texts[i][:80]!r}"
                    )
            return result
        except EmbeddingError:
            raise
        except Exception:
            self._stats.record_latency(
                "embed", (time.monotonic() - t0) * 1000
            )
            raise

    async def _extract_facts(self, text: str) -> list[str]:
        self._stats.inc("extract_facts")
        try:
            raw = await self._timed_generate(EXTRACT_PROMPT, text)
        except Exception as exc:
            self._stats.record_error("extract_facts", str(exc))
            raise LLMError(f"Fact extraction failed: {exc}") from exc
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                parsed = parsed["facts"] if "facts" in parsed else list(parsed.keys())
            if not isinstance(parsed, list):
                self._stats.inc("extract_facts.produced", 1)
                return [str(parsed)]
            facts = [str(f) for f in parsed if f]
            if not facts:
                logger.warning("LLM returned empty fact list, storing raw text")
                self._stats.inc("extract_facts.empty")
                return [text]
            self._stats.inc("extract_facts.produced", len(facts))
            return facts
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON for extraction, storing raw text")
            self._stats.inc("extract_facts.json_failures")
            return [text]

    async def _dedup_fact(
        self, collection: str, fact: str, user_id: str
    ) -> tuple[str, str | None, str | None]:
        if self._backend.is_degraded:
            logger.debug("Skipping LLM dedup (degraded mode), adding directly")
            self._stats.inc("dedup.degraded_skip")
            return "add", None, None

        self._stats.inc("dedup")
        results = await self.search_in_collection(collection, fact, top_k=5, user_id=user_id)
        if not results:
            self._stats.inc("dedup.add")
            return "add", None, None

        existing_lines = "\n".join(f"- [id={r.id}] {r.text}" for r in results)
        if not existing_lines:
            self._stats.inc("dedup.add")
            return "add", None, None

        prompt = DEDUP_PROMPT.format(existing=existing_lines, new_fact=fact)
        try:
            raw = await self._timed_generate(prompt, "")
        except Exception as exc:
            self._stats.record_error("dedup", str(exc))
            self._stats.inc("dedup.add")
            return "add", None, None
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            self._stats.inc("dedup.json_failures")
            self._stats.inc("dedup.add")
            return "add", None, None
        # Ollama's json mode guarantees valid JSON, not an object shape — a
        # bare array/scalar (e.g. ["add"]) would crash result.get() below.
        if not isinstance(result, dict):
            self._stats.inc("dedup.add")
            return "add", None, None
        action = result.get("action", "add")
        update_id = result.get("id")
        # An "update" with no id would otherwise fall through both branches in
        # add() and silently drop the fact; treat it as a plain add.
        if action == "update" and not update_id:
            action = "add"
        self._stats.inc(f"dedup.{action}")
        if action == "update":
            return "update", update_id, result.get("text", fact)
        if action == "skip":
            return "skip", None, None
        return "add", None, None

    async def search_in_collection(
        self, collection: str, query: str, top_k: int = 10, user_id: str | None = None
    ) -> list[MemoryRecord]:
        vectors = await self._timed_embed([query])
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

        records = [
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
        scores = [r.score for r in records if r.score is not None]
        if scores:
            self._stats.record_search_scores(scores)
        return records

    async def add(
        self,
        project_slug: str,
        user_id: str,
        texts: list[str],
        metadata: dict[str, object] | None = None,
    ) -> list[str]:
        self._stats.inc("add_memory")
        self._stats.inc_project(project_slug, "add_memory")
        self._stats.record_activity(project_slug)
        t0 = time.monotonic()
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
                vectors = await self._timed_embed([text_to_store])
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
                vectors = await self._timed_embed([fact])
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

        self._stats.inc("facts_stored", len(ids))
        self._stats.inc_project(project_slug, "facts_stored", len(ids))
        self._stats.record_latency("add_memory", (time.monotonic() - t0) * 1000)
        return ids

    async def search(
        self,
        project_slug: str,
        query: str,
        top_k: int = 10,
    ) -> list[MemoryRecord]:
        self._stats.inc("search")
        self._stats.inc_project(project_slug, "search")
        self._stats.record_activity(project_slug)
        t0 = time.monotonic()
        collection = await self._ensure_collection(project_slug)
        results = await self.search_in_collection(collection, query, top_k=top_k)
        if not results:
            self._stats.inc("search.zero_results")
        self._stats.record_latency("search", (time.monotonic() - t0) * 1000)
        return results

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
        self._stats.inc("delete")
        self._stats.inc_project(project_slug, "delete")
        collection = await self._ensure_collection(project_slug)
        await self._qdrant.delete(
            collection_name=collection,
            points_selector=[memory_id],
        )
        return True

    async def delete_all(self, project_slug: str) -> int:
        self._stats.inc("delete_all")
        self._stats.inc_project(project_slug, "delete_all")
        collection = await self._ensure_collection(project_slug)
        info = await self._qdrant.get_collection(collection)
        count = info.points_count or 0
        await self._qdrant.delete_collection(collection)
        self._ensured_collections.discard(collection)
        await self._ensure_collection(project_slug)
        return count

    async def reembed_all(self, project_slug: str) -> int:
        self._stats.inc("reembed")
        self._stats.inc_project(project_slug, "reembed")
        collection = await self._ensure_collection(project_slug)

        info = await self._qdrant.get_collection(collection)
        current_dims = self._backend.embedding_dimensions
        collection_dims = info.config.params.vectors.size  # type: ignore[union-attr]
        dimension_change = collection_dims != current_dims

        all_payloads: list[tuple[str | int, dict]] = []
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
                if pt.payload.get("text"):
                    all_payloads.append((pt.id, pt.payload))
            if next_offset is None:
                break
            offset = next_offset

        if dimension_change:
            logger.info(
                "Dimension change %d→%d for %s, recreating collection",
                collection_dims, current_dims, collection,
            )
            await self._qdrant.delete_collection(collection)
            self._ensured_collections.discard(collection)
            collection = await self._ensure_collection(project_slug)

        updated = 0
        for point_id, payload in all_payloads:
            text = payload.get("text", "")
            if not text:
                continue
            vectors = await self._timed_embed([text])
            await self._qdrant.upsert(
                collection_name=collection,
                points=[PointStruct(id=point_id, vector=vectors[0], payload=payload)],
            )
            updated += 1

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
        self._stats.inc("cleanup")
        self._stats.inc_project(project_slug, "cleanup")
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
                    vectors = await self._timed_embed([facts[0]])
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
                        vectors = await self._timed_embed([fact])
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

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _cluster_points(
        self, points: list, threshold: float
    ) -> list[list[int]]:
        n = len(points)
        vectors = [pt.vector for pt in points]
        used: set[int] = set()
        clusters: list[list[int]] = []

        for i in range(n):
            if i in used:
                continue
            cluster = [i]
            used.add(i)
            for j in range(i + 1, n):
                if j in used:
                    continue
                if self._cosine_similarity(vectors[i], vectors[j]) >= threshold:
                    cluster.append(j)
                    used.add(j)
            clusters.append(cluster)

        return clusters

    async def consolidate(
        self,
        project_slug: str,
        similarity_threshold: float = 0.75,
        dry_run: bool = False,
    ) -> dict[str, object]:
        if self._backend.is_degraded:
            raise LLMError("Cannot consolidate in degraded mode — LLM is required")

        self._stats.inc("consolidate")
        self._stats.inc_project(project_slug, "consolidate")
        collection = await self._ensure_collection(project_slug)

        all_points = []
        offset = None
        while True:
            points, next_offset = await self._qdrant.scroll(
                collection_name=collection,
                limit=50,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if not points:
                break
            all_points.extend(points)
            if next_offset is None:
                break
            offset = next_offset

        if len(all_points) < 2:
            return {"clusters": 0, "memories_removed": 0, "memories_created": 0}

        clusters = self._cluster_points(all_points, similarity_threshold)
        merge_clusters = [c for c in clusters if len(c) >= 2]

        if dry_run:
            previews = []
            for cluster in merge_clusters:
                texts = [all_points[i].payload.get("text", "") for i in cluster]
                previews.append({"count": len(cluster), "texts": texts})
            return {"clusters": len(merge_clusters), "previews": previews}

        total_removed = 0
        total_created = 0
        for cluster in merge_clusters:
            pts = [all_points[i] for i in cluster]
            texts = [pt.payload.get("text", "") for pt in pts]

            user_ids = [pt.payload.get("user_id", "default") for pt in pts]
            user_id = max(set(user_ids), key=user_ids.count)

            fragments = "\n".join(f"- {t}" for t in texts)
            prompt = CONSOLIDATE_PROMPT.format(fragments=fragments)

            try:
                raw = await self._timed_generate(prompt, "")
                consolidated = json.loads(raw)
                if isinstance(consolidated, str):
                    consolidated = [consolidated]
                if not isinstance(consolidated, list):
                    continue
                consolidated = [str(f) for f in consolidated if f and str(f).strip()]
            except json.JSONDecodeError:
                self._stats.inc("consolidate.json_failures")
                logger.warning("Consolidation LLM call failed for cluster, skipping")
                continue
            except Exception as exc:
                self._stats.record_error("consolidate", str(exc))
                logger.warning("Consolidation LLM call failed for cluster, skipping")
                continue

            if not consolidated:
                continue

            old_ids = [str(pt.id) for pt in pts]
            await self._qdrant.delete(
                collection_name=collection, points_selector=old_ids
            )
            total_removed += len(old_ids)

            now = time.time()
            earliest = min(
                pt.payload.get("created_at", now) for pt in pts
            )
            for fact in consolidated:
                vectors = await self._timed_embed([fact])
                point_id = str(uuid.uuid4())
                await self._qdrant.upsert(
                    collection_name=collection,
                    points=[
                        PointStruct(
                            id=point_id,
                            vector=vectors[0],
                            payload={
                                "text": fact,
                                "user_id": user_id,
                                "project": project_slug,
                                "created_at": earliest,
                                "updated_at": now,
                            },
                        )
                    ],
                )
                total_created += 1

        logger.info(
            "Consolidated %s: %d clusters, %d removed, %d created",
            collection, len(merge_clusters), total_removed, total_created,
        )
        return {
            "clusters": len(merge_clusters),
            "memories_removed": total_removed,
            "memories_created": total_created,
        }

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
                        last_updated=self._stats.get_last_activity(slug),
                    )
                )
        return projects
