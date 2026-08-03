from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid

import numpy as np
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
You are a memory extraction system for an engineering project's long-term \
notes. Extract facts worth remembering in future work sessions: decisions and \
their reasoning, gotchas, workarounds, dead ends, conventions, and preferences.

Rules for each fact:
- One complete, self-contained sentence. A reader with no other context must \
understand it fully — name systems, tools, and components explicitly.
- Never start a fact with "This", "It", "That", "Also", or a dangling \
reference. Restate the subject instead.
- Keep the reasoning attached to the decision when the input gives one \
("X was chosen over Y because Z").
- Write personal preferences as "User prefers ..." or "User never ...".
- EXCLUDE transient status: test results, version bumps, "all tests pass", \
per-file change lists, greetings, thanks, conversational filler.
- EXCLUDE fragments that cannot stand alone ("Root cause", "Solution", lone \
tool names).
- If nothing qualifies, return {"facts": []}.

Example input: "Fixed the sync bug. Root cause: the API returns timestamps in \
local time, not UTC. Also bumped to 1.2.4, all tests pass. We should always \
normalize to UTC at ingestion from now on."

Example output: {"facts": ["The sync bug was caused by the API returning \
timestamps in local time instead of UTC.", "Timestamps must always be \
normalized to UTC at ingestion."]}
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
You are a memory deduplication system for an engineering project's long-term \
notes. Decide how a NEW fact relates to the EXISTING memories.

Pick exactly one action:
- "skip": the new fact contains no information that is missing from the \
existing memories.
- "update": the new fact adds NEW information to, corrects, or supersedes \
exactly ONE existing memory. Return that memory's "id", and "text": a single \
self-contained statement keeping ALL information from the old memory plus the \
new fact. If old and new conflict, the new fact wins.
- "add": the new fact is about something no existing memory covers. Two \
memories that mention the same tool or system are still different facts if \
they state different things.

If unsure between add and update, choose add. Never merge more than one \
existing memory.

Respond with ONLY a JSON object: {"action": "add"} or {"action": "skip"} or \
{"action": "update", "id": "<memory_id>", "text": "<merged statement>"}
"""

# Schemas passed to the LLM backend to grammar-constrain output to the exact
# expected shape (Ollama supports this) instead of relying on the prompt alone.
# NOTE: never add minLength here — Ollama grammar-enforces it by padding the
# string with degenerate rambling when the model wants to stop early.
EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
    "required": ["facts"],
}

CONSOLIDATE_SCHEMA = {"type": "array", "items": {"type": "string"}}

DEDUP_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["add", "update", "skip"]},
        "id": {"type": "string"},
        "text": {"type": "string"},
    },
    "required": ["action"],
}


# Facts opening with a dangling reference depend on context that is not
# stored with them, so they are meaningless on retrieval.
_CONTEXT_DEPENDENT = re.compile(
    r"^(this|that|it|its|they|these|those|also|additionally|however)\b",
    re.IGNORECASE,
)

# Cosine gates calibrated against mxbai-embed-large on real memory pairs:
# identical 1.00, paraphrase 0.98, supersession 0.91, additive detail 0.81,
# same-topic-different-fact 0.67, unrelated 0.47. Above SKIP the new fact is
# a rewording (the LLM reliably mislabels these "update"); below RELEVANT a
# candidate is noise that only invites bad merges.
DEDUP_AUTO_SKIP_SCORE = 0.95
DEDUP_RELEVANT_SCORE = 0.60


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

    async def _timed_generate(
        self, system: str, user: str, schema: dict[str, object] | None = None
    ) -> str:
        t0 = time.monotonic()
        try:
            result = await self._backend.generate(system, user, schema)
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
        except EmbeddingError as exc:
            # Latency was already recorded above; just surface the failure so an
            # embedding outage shows up in error_rate and recent_errors.
            self._stats.record_error("embed", str(exc))
            raise
        except Exception as exc:
            self._stats.record_latency(
                "embed", (time.monotonic() - t0) * 1000
            )
            self._stats.record_error("embed", str(exc))
            raise

    @staticmethod
    def _facts_from_dict(obj: dict[str, object]) -> list[str]:
        # The model returned an object instead of the requested array. Recover
        # fact strings from whichever shape it used: a {"facts": [...]} list, a
        # {category: [...]} list value, a {label: "fact text"} string value, or
        # a {"fact text": true} shape where the key itself is the fact.
        facts = obj.get("facts")
        if isinstance(facts, list):
            return facts
        collected: list[str] = []
        for key, value in obj.items():
            if isinstance(value, list):
                collected.extend(str(x) for x in value if x)
            elif isinstance(value, str):
                if value.strip():
                    collected.append(value)
            elif not isinstance(value, dict) and isinstance(key, str) and key.strip():
                # scalar value (bool/number/null) carries no text — the key is
                # the fact itself, e.g. {"User likes Python": true}
                collected.append(key)
        return collected

    @staticmethod
    def _valid_fact(fact: str) -> bool:
        # Python-side junk gate: kills ", ", "Root cause", lone tool names,
        # and context-dependent fragments regardless of what the LLM emits.
        fact = fact.strip()
        if len(fact) < 20:
            return False
        if len(fact.split()) < 4:
            return False
        if not any(c.isalpha() for c in fact):
            return False
        return not _CONTEXT_DEPENDENT.match(fact)

    async def _extract_facts(self, text: str) -> list[str]:
        self._stats.inc("extract_facts")
        try:
            raw = await self._timed_generate(EXTRACT_PROMPT, text, schema=EXTRACT_SCHEMA)
        except Exception as exc:
            self._stats.inc("extract_facts.llm_failures")
            self._stats.record_error("extract_facts", str(exc))
            raise LLMError(f"Fact extraction failed: {exc}") from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Rare with schema-constrained output; retry once, then fail the
            # add. Storing the raw input blob (the old fallback) polluted the
            # store with unsearchable conversation dumps.
            self._stats.inc("extract_facts.json_failures")
            try:
                raw = await self._timed_generate(
                    EXTRACT_PROMPT, text, schema=EXTRACT_SCHEMA
                )
                parsed = json.loads(raw)
            except Exception as exc:
                self._stats.inc("extract_facts.llm_failures")
                self._stats.record_error("extract_facts", str(exc))
                raise LLMError(f"Fact extraction failed: {exc}") from exc

        if isinstance(parsed, dict):
            parsed = self._facts_from_dict(parsed)
        if not isinstance(parsed, list):
            parsed = [str(parsed)]
        candidates = [str(f).strip() for f in parsed if f]
        facts = [f for f in candidates if self._valid_fact(f)]
        rejected = len(candidates) - len(facts)
        if rejected:
            self._stats.inc("extract_facts.rejected", rejected)
        if not facts:
            # An empty result is the model correctly declining to store
            # filler — a success, not a failure.
            self._stats.inc("extract_facts.no_facts")
            return []
        self._stats.inc("extract_facts.produced", len(facts))
        return facts

    async def _dedup_fact(
        self,
        collection: str,
        fact: str,
        user_id: str,
        fact_vector: list[float] | None = None,
    ) -> tuple[str, str | None, str | None]:
        if self._backend.is_degraded:
            logger.debug("Skipping LLM dedup (degraded mode), adding directly")
            self._stats.inc("dedup.degraded_skip")
            return "add", None, None

        self._stats.inc("dedup")
        results = await self.search_in_collection(
            collection, fact, top_k=5, user_id=user_id, query_vector=fact_vector
        )
        # Deterministic gates around the LLM call: near-identical rewordings
        # skip without a call (the LLM labels them "update" 6/6 times and
        # rewrites unchanged content), and sub-relevance candidates are
        # dropped so they can't invite bad merges.
        if (
            results
            and results[0].score is not None
            and results[0].score >= DEDUP_AUTO_SKIP_SCORE
        ):
            self._stats.inc("dedup.auto_skip")
            self._stats.inc("dedup.skip")
            return "skip", None, None
        results = [
            r for r in results if r.score is None or r.score >= DEDUP_RELEVANT_SCORE
        ]
        if not results:
            self._stats.inc("dedup.add")
            return "add", None, None

        existing_lines = "\n".join(f"- [id={r.id}] {r.text}" for r in results)
        user_content = f"EXISTING MEMORIES:\n{existing_lines}\n\nNEW FACT:\n{fact}"
        try:
            raw = await self._timed_generate(
                DEDUP_PROMPT, user_content, schema=DEDUP_SCHEMA
            )
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
        # add() and silently drop the fact; an "update" with no merged text
        # would replace the existing memory's content with the bare new fact,
        # destroying the old information. Both degrade to a plain add.
        if action == "update" and not (update_id and result.get("text")):
            action = "add"
        self._stats.inc(f"dedup.{action}")
        if action == "update":
            return "update", update_id, result["text"]
        if action == "skip":
            return "skip", None, None
        return "add", None, None

    async def search_in_collection(
        self,
        collection: str,
        query: str,
        top_k: int = 10,
        user_id: str | None = None,
        query_vector: list[float] | None = None,
    ) -> list[MemoryRecord]:
        if query_vector is None:
            query_vector = (await self._timed_embed([query]))[0]
        query_filter = None
        if user_id:
            query_filter = Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            )
        results = (
            await self._qdrant.query_points(
                collection_name=collection,
                query=query_vector,
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

        # Embed every fact once, up front, in a single batched call. The vector
        # is reused for both the dedup search and storage below — previously
        # each fact was embedded twice (once to search, once to store).
        fact_vectors = await self._timed_embed(all_facts)

        for fact, fact_vector in zip(all_facts, fact_vectors, strict=True):
            action, update_id, merged_text = await self._dedup_fact(
                collection, fact, user_id, fact_vector=fact_vector
            )

            if action == "skip":
                logger.debug("Skipping duplicate: %s", fact[:80])
                continue

            if action == "update" and update_id:
                text_to_store = merged_text or fact
                # Re-embed only if the LLM actually changed the text on merge;
                # otherwise the fact's own vector still applies.
                if text_to_store == fact:
                    vector = fact_vector
                else:
                    vector = (await self._timed_embed([text_to_store]))[0]
                try:
                    validate_memory_id(update_id)
                    # Preserve the original creation time — an update merges into
                    # an existing memory, it does not create a new one.
                    existing = await self._qdrant.retrieve(
                        collection_name=collection,
                        ids=[update_id],
                        with_payload=True,
                    )
                    created_at = (
                        (existing[0].payload or {}).get("created_at", now)
                        if existing
                        else now
                    )
                    await self._qdrant.upsert(
                        collection_name=collection,
                        points=[
                            PointStruct(
                                id=update_id,
                                vector=vector,
                                payload={
                                    **clean_meta,
                                    "text": text_to_store,
                                    "user_id": user_id,
                                    "project": project_slug,
                                    "created_at": created_at,
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
                point_id = str(uuid.uuid4())
                ids.append(point_id)
                await self._qdrant.upsert(
                    collection_name=collection,
                    points=[
                        PointStruct(
                            id=point_id,
                            vector=fact_vector,
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
                limit=256,
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

        # Re-embed everything up front, before any destructive operation, so a
        # mid-way embedding failure (e.g. the LLM backend going down) can never
        # leave the collection deleted-but-not-repopulated. Embeds are batched:
        # one round trip per 64 texts instead of one per memory.
        texts = [payload.get("text", "") for _, payload in all_payloads]
        vectors: list[list[float]] = []
        for i in range(0, len(texts), 64):
            vectors.extend(await self._timed_embed(texts[i : i + 64]))
        new_points = [
            PointStruct(id=point_id, vector=vec, payload=payload)
            for (point_id, payload), vec in zip(all_payloads, vectors, strict=True)
        ]

        if dimension_change:
            logger.info(
                "Dimension change %d→%d for %s, recreating collection",
                collection_dims, current_dims, collection,
            )
            await self._qdrant.delete_collection(collection)
            self._ensured_collections.discard(collection)
            collection = await self._ensure_collection(project_slug)

        for i in range(0, len(new_points), 100):
            await self._qdrant.upsert(
                collection_name=collection, points=new_points[i : i + 100]
            )

        logger.info("Re-embedded %d points in %s", len(new_points), collection)
        return len(new_points)

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
                limit=256,
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
    def _cluster_points(points: list, threshold: float) -> list[list[int]]:
        # Vectorized similarity: the previous pure-Python O(n^2) cosine loop
        # blocked the event loop for seconds once collections passed a few
        # hundred points; the matrix product is milliseconds.
        n = len(points)
        v = np.asarray([pt.vector for pt in points], dtype=np.float64)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        sims = (v / norms) @ (v / norms).T

        used: set[int] = set()
        clusters: list[list[int]] = []
        for i in range(n):
            if i in used:
                continue
            cluster = [i]
            used.add(i)
            for j in range(i + 1, n):
                if j not in used and sims[i, j] >= threshold:
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
                limit=256,
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
                raw = await self._timed_generate(prompt, "", schema=CONSOLIDATE_SCHEMA)
                consolidated = json.loads(raw)
                if isinstance(consolidated, str):
                    consolidated = [consolidated]
                if not isinstance(consolidated, list):
                    continue
                consolidated = [
                    str(f).strip()
                    for f in consolidated
                    if f and self._valid_fact(str(f))
                ]
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

            # Embed the replacements BEFORE deleting the originals, so an
            # embedding failure leaves the cluster untouched instead of losing
            # the old memories with nothing stored in their place.
            try:
                vectors = await self._timed_embed(consolidated)
            except Exception as exc:
                self._stats.record_error("consolidate", str(exc))
                logger.warning("Embedding failed for cluster, skipping")
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
            new_pts = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "text": fact,
                        "user_id": user_id,
                        "project": project_slug,
                        "created_at": earliest,
                        "updated_at": now,
                    },
                )
                for fact, vec in zip(consolidated, vectors, strict=True)
            ]
            await self._qdrant.upsert(collection_name=collection, points=new_pts)
            total_created += len(new_pts)

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
        matching = [
            c.name
            for c in (await self._qdrant.get_collections()).collections
            if c.name.startswith(prefix)
        ]
        # Fetch collection infos concurrently — sequential get_collection calls
        # made this endpoint an N+1 that backed every dashboard refresh.
        infos = await asyncio.gather(
            *(self._qdrant.get_collection(name) for name in matching)
        )
        return [
            ProjectInfo(
                slug=name[len(prefix) :],
                collection=name,
                memory_count=info.points_count or 0,
                last_updated=self._stats.get_last_activity(name[len(prefix) :]),
            )
            for name, info in zip(matching, infos, strict=True)
        ]
