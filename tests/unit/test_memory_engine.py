from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mem_zero.config import Config
from mem_zero.memory_engine import (
    EXTRACT_SCHEMA,
    LLMError,
    MemoryEngine,
    validate_memory_id,
)
from mem_zero.models import MemoryRecord
from mem_zero.stats import DiagnosticStats


@pytest.fixture
def config() -> Config:
    return Config(
        qdrant_host="localhost",
        qdrant_port=6333,
        ollama_base_url="http://localhost:11434",
        embedder_model="nomic-embed-text",
        embedding_dimensions=768,
        collection_prefix="test",
    )


@pytest.fixture
def mock_backend() -> AsyncMock:
    backend = AsyncMock()
    backend.embedding_dimensions = 768
    backend.is_degraded = False
    # Length-aware: batched embed calls get one vector per input text.
    backend.embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
    backend.generate.return_value = '["test memory"]'
    return backend


def _pt(id_: str, text: str, vector: list[float], **payload):
    """A MagicMock Qdrant point with the standard payload shape."""
    p = MagicMock()
    p.id = id_
    p.vector = vector
    p.payload = {
        "text": text,
        "user_id": "john",
        "created_at": 100.0,
        "updated_at": 100.0,
        **payload,
    }
    return p


@pytest.fixture
def mock_qdrant() -> AsyncMock:
    client = AsyncMock()
    client.get_collections.return_value = MagicMock(collections=[])
    client.get_collection.return_value = MagicMock(points_count=0)
    return client


@pytest.fixture
def engine(config: Config, mock_backend: AsyncMock, mock_qdrant: AsyncMock) -> MemoryEngine:
    eng = MemoryEngine(config, mock_backend)
    eng._qdrant = mock_qdrant
    return eng


class TestValidateMemoryId:
    def test_valid_uuid(self) -> None:
        uid = "550e8400-e29b-41d4-a716-446655440000"
        assert validate_memory_id(uid) == uid

    def test_rejects_non_uuid(self) -> None:
        with pytest.raises(ValueError, match="Invalid memory ID"):
            validate_memory_id("not-a-uuid")


class TestCollectionNaming:
    @pytest.mark.asyncio
    async def test_creates_prefixed_name(self, engine: MemoryEngine) -> None:
        name = await engine._ensure_collection("my-project")
        assert name == "test_my-project"

    @pytest.mark.asyncio
    async def test_creates_collection_if_missing(
        self, engine: MemoryEngine, mock_qdrant: AsyncMock
    ) -> None:
        await engine._ensure_collection("new-project")
        mock_qdrant.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_creation_if_exists(
        self, engine: MemoryEngine, mock_qdrant: AsyncMock
    ) -> None:
        existing = MagicMock()
        existing.name = "test_existing"
        mock_qdrant.get_collections.return_value = MagicMock(collections=[existing])
        await engine._ensure_collection("existing")
        mock_qdrant.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_caches_after_first_check(
        self, engine: MemoryEngine, mock_qdrant: AsyncMock
    ) -> None:
        await engine._ensure_collection("cached")
        await engine._ensure_collection("cached")
        assert mock_qdrant.get_collections.call_count == 1


class TestIsolation:
    def test_different_projects_get_different_collections(
        self, engine: MemoryEngine
    ) -> None:
        name_a = engine._config.collection_name("project-a")
        name_b = engine._config.collection_name("project-b")
        assert name_a != name_b
        assert name_a == "test_project-a"
        assert name_b == "test_project-b"


class TestAdd:
    @pytest.mark.asyncio
    async def test_upserts_to_correct_collection(
        self, engine: MemoryEngine, mock_qdrant: AsyncMock
    ) -> None:
        with (
            patch.object(
                engine, "_extract_facts", new_callable=AsyncMock,
                return_value=["test memory"],
            ),
            patch.object(
                engine, "_dedup_fact", new_callable=AsyncMock,
                return_value=("add", None, None),
            ),
        ):
            await engine.add("project-a", "john", ["test memory"])
            call_args = mock_qdrant.upsert.call_args
            assert call_args.kwargs["collection_name"] == "test_project-a"

    @pytest.mark.asyncio
    async def test_returns_ids(
        self, engine: MemoryEngine, mock_qdrant: AsyncMock
    ) -> None:
        with (
            patch.object(
                engine, "_extract_facts", new_callable=AsyncMock,
                return_value=["memory one"],
            ),
            patch.object(
                engine, "_dedup_fact", new_callable=AsyncMock,
                return_value=("add", None, None),
            ),
        ):
            ids = await engine.add("project-a", "john", ["memory one"])
            assert len(ids) == 1
            assert isinstance(ids[0], str)

    @pytest.mark.asyncio
    async def test_reserved_keys_cannot_be_overwritten(
        self, engine: MemoryEngine, mock_qdrant: AsyncMock
    ) -> None:
        with (
            patch.object(
                engine, "_extract_facts", new_callable=AsyncMock,
                return_value=["test"],
            ),
            patch.object(
                engine, "_dedup_fact", new_callable=AsyncMock,
                return_value=("add", None, None),
            ),
        ):
            await engine.add(
                "project-a",
                "john",
                ["test"],
                metadata={"user_id": "attacker", "project": "other"},
            )
            call_args = mock_qdrant.upsert.call_args
            point = call_args.kwargs["points"][0]
            assert point.payload["user_id"] == "john"
            assert point.payload["project"] == "project-a"

    @pytest.mark.asyncio
    async def test_batches_fact_embeddings_into_one_call(
        self, engine: MemoryEngine, mock_backend: AsyncMock, mock_qdrant: AsyncMock
    ) -> None:
        mock_backend.embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
        with (
            patch.object(
                engine, "_extract_facts", new_callable=AsyncMock,
                return_value=["fact a", "fact b", "fact c"],
            ),
            patch.object(
                engine, "_dedup_fact", new_callable=AsyncMock,
                return_value=("add", None, None),
            ),
        ):
            ids = await engine.add("project-a", "john", ["some text"])
        assert len(ids) == 3
        # One batched embed of all three facts — not one call per fact, and no
        # separate embed to store what dedup already embedded.
        assert mock_backend.embed.call_count == 1
        assert mock_backend.embed.call_args.args[0] == ["fact a", "fact b", "fact c"]
        assert mock_qdrant.upsert.call_count == 3


FACT_A = "User prefers Python over R for data pipeline work"
FACT_B = "PostgreSQL was chosen over Redis because sessions need ACID transactions"


class TestValidFact:
    def test_accepts_self_contained_sentence(self) -> None:
        assert MemoryEngine._valid_fact(FACT_A) is True

    @pytest.mark.parametrize(
        "legit",
        [
            # The extraction prompt mandates this exact terse form.
            "User never force-pushes.",
            "User prefers tabs.",
            # Demonstrative + noun is a real subject, not a dangling reference.
            "This project uses Python 3.12 and targets Unraid only.",
            "This repo is owned by Gadsden LLC, not the personal account.",
            "These builds download Minecraft artifacts from Mojang servers.",
            "However, Watchtower is never used on Unraid.",
            "It is required to run docker login ghcr.io before pulling images.",
        ],
    )
    def test_keeps_legitimate_facts(self, legit: str) -> None:
        assert MemoryEngine._valid_fact(legit) is True

    @pytest.mark.parametrize(
        "junk",
        [
            ", ",
            "Root cause",
            "Solution",
            "Qdrant",
            "12345 67890 12345 67890",  # no letters
            "This prevents config loss across container updates",
            "That was the root cause of the outage",
            "Also the registry uses a self-signed certificate",
        ],
    )
    def test_rejects_junk(self, junk: str) -> None:
        assert MemoryEngine._valid_fact(junk) is False


class TestDiffersMaterially:
    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("Server listens on port 8765", "Server listens on port 8766"),
            ("Ollama works with the bundled model", "Ollama does not work with it"),
            ("Watchtower is used for updates", "Watchtower is never used"),
        ],
    )
    def test_detects_corrections(self, a: str, b: str) -> None:
        assert MemoryEngine._differs_materially(a, b) is True

    def test_ignores_pure_rewording(self) -> None:
        assert (
            MemoryEngine._differs_materially(
                "User prefers Python over R for data work",
                "User likes Python more than R for data work",
            )
            is False
        )


class TestExtractFacts:
    @pytest.mark.asyncio
    async def test_empty_list_returns_no_facts(
        self, engine: MemoryEngine, mock_backend: AsyncMock
    ) -> None:
        # An empty extraction is the model correctly declining filler — the
        # raw input must NOT be stored as a memory (the old fallback).
        mock_backend.generate.return_value = '{"facts": []}'
        facts = await engine._extract_facts("ok sounds good, thanks!")
        assert facts == []

    @pytest.mark.asyncio
    async def test_valid_facts_returned(
        self, engine: MemoryEngine, mock_backend: AsyncMock
    ) -> None:
        mock_backend.generate.return_value = json.dumps({"facts": [FACT_A, FACT_B]})
        facts = await engine._extract_facts("some text")
        assert facts == [FACT_A, FACT_B]

    @pytest.mark.asyncio
    async def test_bare_array_still_accepted(
        self, engine: MemoryEngine, mock_backend: AsyncMock
    ) -> None:
        mock_backend.generate.return_value = json.dumps([FACT_A])
        facts = await engine._extract_facts("some text")
        assert facts == [FACT_A]

    @pytest.mark.asyncio
    async def test_junk_facts_filtered_out(
        self, engine: MemoryEngine, mock_backend: AsyncMock
    ) -> None:
        mock_backend.generate.return_value = json.dumps(
            {"facts": [FACT_A, "Root cause", ", "]}
        )
        facts = await engine._extract_facts("some text")
        assert facts == [FACT_A]

    @pytest.mark.asyncio
    async def test_dict_keys_extracted_as_facts(
        self, engine: MemoryEngine, mock_backend: AsyncMock
    ) -> None:
        mock_backend.generate.return_value = json.dumps({FACT_A: True, FACT_B: True})
        facts = await engine._extract_facts("some text")
        assert facts == [FACT_A, FACT_B]

    @pytest.mark.asyncio
    async def test_dict_string_values_used_not_keys(
        self, engine: MemoryEngine, mock_backend: AsyncMock
    ) -> None:
        mock_backend.generate.return_value = json.dumps(
            {"fact1": FACT_A, "fact2": FACT_B}
        )
        facts = await engine._extract_facts("some text")
        assert facts == [FACT_A, FACT_B]

    @pytest.mark.asyncio
    async def test_decode_failure_retries_then_raises(
        self, engine: MemoryEngine, mock_backend: AsyncMock
    ) -> None:
        # Two bad payloads -> LLMError; the raw input must never be stored.
        mock_backend.generate.side_effect = ["not json", "still not json"]
        with pytest.raises(LLMError):
            await engine._extract_facts("some text")
        assert mock_backend.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_decode_failure_retry_can_succeed(
        self, engine: MemoryEngine, mock_backend: AsyncMock
    ) -> None:
        mock_backend.generate.side_effect = [
            "not json",
            json.dumps({"facts": [FACT_A]}),
        ]
        facts = await engine._extract_facts("some text")
        assert facts == [FACT_A]

    @pytest.mark.asyncio
    async def test_passes_extract_schema_to_backend(
        self, engine: MemoryEngine, mock_backend: AsyncMock
    ) -> None:
        mock_backend.generate.return_value = json.dumps({"facts": [FACT_A]})
        await engine._extract_facts("some text")
        assert mock_backend.generate.call_args.args[2] == EXTRACT_SCHEMA


class TestDedupDegraded:
    @pytest.mark.asyncio
    async def test_skips_dedup_when_degraded(
        self, engine: MemoryEngine, mock_backend: AsyncMock, mock_qdrant: AsyncMock
    ) -> None:
        mock_backend.is_degraded = True
        action, _, _ = await engine._dedup_fact("test_proj", "new fact", "john")
        assert action == "add"
        mock_backend.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_runs_dedup_when_not_degraded(
        self, engine: MemoryEngine, mock_backend: AsyncMock, mock_qdrant: AsyncMock
    ) -> None:
        mock_backend.is_degraded = False
        mock_qdrant.query_points.return_value = MagicMock(points=[])
        action, _, _ = await engine._dedup_fact("test_proj", "new fact", "john")
        assert action == "add"


class TestDedupScoreGates:
    @pytest.fixture
    def near_identical(self) -> list[MemoryRecord]:
        return [
            MemoryRecord(
                id="550e8400-e29b-41d4-a716-446655440000",
                text="User prefers Python over R for data work",
                user_id="john",
                created_at=0.0,
                updated_at=0.0,
                score=0.98,
            )
        ]

    @pytest.mark.asyncio
    async def test_paraphrase_skipped_without_llm_call(
        self,
        engine: MemoryEngine,
        mock_backend: AsyncMock,
        near_identical: list[MemoryRecord],
    ) -> None:
        with patch.object(
            engine, "search_in_collection", new_callable=AsyncMock,
            return_value=near_identical,
        ):
            action, _, _ = await engine._dedup_fact(
                "test_proj", "User likes Python more than R for data work", "john"
            )
        assert action == "skip"
        mock_backend.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_low_score_candidates_add_without_llm_call(
        self, engine: MemoryEngine, mock_backend: AsyncMock
    ) -> None:
        noise = [
            MemoryRecord(
                id="550e8400-e29b-41d4-a716-446655440000",
                text="An unrelated memory about container updates",
                user_id="john",
                created_at=0.0,
                updated_at=0.0,
                score=0.45,
            )
        ]
        with patch.object(
            engine, "search_in_collection", new_callable=AsyncMock,
            return_value=noise,
        ):
            action, _, _ = await engine._dedup_fact(
                "test_proj", "A brand new fact about the deploy pipeline", "john"
            )
        assert action == "add"
        mock_backend.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_ambiguous_band_still_asks_llm(
        self, engine: MemoryEngine, mock_backend: AsyncMock
    ) -> None:
        candidate = [
            MemoryRecord(
                id="550e8400-e29b-41d4-a716-446655440000",
                text="Qwen 7b is used for extraction",
                user_id="john",
                created_at=0.0,
                updated_at=0.0,
                score=0.85,
            )
        ]
        mock_backend.generate.return_value = '{"action": "skip"}'
        with patch.object(
            engine, "search_in_collection", new_callable=AsyncMock,
            return_value=candidate,
        ):
            action, _, _ = await engine._dedup_fact(
                "test_proj", "Qwen 14b is now used for extraction", "john"
            )
        assert action == "skip"
        mock_backend.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_without_text_becomes_add(
        self, engine: MemoryEngine, mock_backend: AsyncMock
    ) -> None:
        # update with an id but no merged text would overwrite the existing
        # memory with the bare new fact, destroying its content.
        candidate = [
            MemoryRecord(
                id="550e8400-e29b-41d4-a716-446655440000",
                text="existing memory about the deploy pipeline",
                user_id="john",
                created_at=0.0,
                updated_at=0.0,
                score=0.80,
            )
        ]
        mock_backend.generate.return_value = (
            '{"action": "update", "id": "550e8400-e29b-41d4-a716-446655440000"}'
        )
        with patch.object(
            engine, "search_in_collection", new_callable=AsyncMock,
            return_value=candidate,
        ):
            action, update_id, _ = await engine._dedup_fact(
                "test_proj", "new fact about the deploy pipeline", "john"
            )
        assert action == "add"
        assert update_id is None


class TestDedupMalformedResponses:
    """Dedup must survive valid-but-unexpected LLM JSON without crashing or
    silently dropping the fact. Ollama's json mode enforces valid JSON, not a
    particular shape, so these responses are all reachable in production."""

    @pytest.fixture
    def existing(self) -> list[MemoryRecord]:
        return [
            MemoryRecord(
                id="550e8400-e29b-41d4-a716-446655440000",
                text="existing memory",
                user_id="john",
                created_at=0.0,
                updated_at=0.0,
            )
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw", ['["add"]', '"skip"', "42", "true"])
    async def test_non_object_json_defaults_to_add(
        self,
        engine: MemoryEngine,
        mock_backend: AsyncMock,
        existing: list[MemoryRecord],
        raw: str,
    ) -> None:
        mock_backend.generate.return_value = raw
        with patch.object(
            engine, "search_in_collection", new_callable=AsyncMock,
            return_value=existing,
        ):
            action, update_id, _ = await engine._dedup_fact(
                "test_proj", "new fact", "john"
            )
        assert action == "add"
        assert update_id is None

    @pytest.mark.asyncio
    async def test_update_without_id_becomes_add(
        self,
        engine: MemoryEngine,
        mock_backend: AsyncMock,
        existing: list[MemoryRecord],
    ) -> None:
        mock_backend.generate.return_value = '{"action": "update"}'
        with patch.object(
            engine, "search_in_collection", new_callable=AsyncMock,
            return_value=existing,
        ):
            action, update_id, _ = await engine._dedup_fact(
                "test_proj", "new fact", "john"
            )
        assert action == "add"
        assert update_id is None

    @pytest.mark.asyncio
    async def test_valid_update_is_preserved(
        self,
        engine: MemoryEngine,
        mock_backend: AsyncMock,
        existing: list[MemoryRecord],
    ) -> None:
        mock_backend.generate.return_value = (
            '{"action": "update", "id": "550e8400-e29b-41d4-a716-446655440000",'
            ' "text": "merged text"}'
        )
        with patch.object(
            engine, "search_in_collection", new_callable=AsyncMock,
            return_value=existing,
        ):
            action, update_id, text = await engine._dedup_fact(
                "test_proj", "new fact", "john"
            )
        assert action == "update"
        assert update_id == "550e8400-e29b-41d4-a716-446655440000"
        assert text == "merged text"


class TestSearch:
    @pytest.mark.asyncio
    async def test_searches_correct_collection(
        self, engine: MemoryEngine, mock_qdrant: AsyncMock
    ) -> None:
        mock_qdrant.query_points.return_value = MagicMock(points=[])
        await engine.search("project-a", "test query")
        call_args = mock_qdrant.query_points.call_args
        assert call_args.kwargs["collection_name"] == "test_project-a"

    @pytest.mark.asyncio
    async def test_rerank_disabled_by_default(
        self, engine: MemoryEngine, mock_qdrant: AsyncMock
    ) -> None:
        mock_qdrant.query_points.return_value = MagicMock(points=[])
        await engine.search("project-a", "q", top_k=10)
        # No over-fetch when reranking is off.
        assert mock_qdrant.query_points.call_args.kwargs["limit"] == 10


class TestRerank:
    @pytest.fixture
    def rerank_engine(
        self, mock_backend: AsyncMock, mock_qdrant: AsyncMock
    ) -> MemoryEngine:
        cfg = Config(collection_prefix="test", rerank_enabled=True)
        eng = MemoryEngine(cfg, mock_backend)
        eng._qdrant = mock_qdrant
        return eng

    @staticmethod
    def _record(id_: str, text: str, score: float) -> MemoryRecord:
        return MemoryRecord(
            id=id_, text=text, user_id="john",
            created_at=0.0, updated_at=0.0, score=score,
        )

    @pytest.mark.asyncio
    async def test_reorders_by_cross_encoder_and_slices(
        self, rerank_engine: MemoryEngine
    ) -> None:
        # Vector order: a, b, c. Cross-encoder strongly prefers c.
        candidates = [
            self._record("a", "memory a", 0.60),
            self._record("b", "memory b", 0.55),
            self._record("c", "memory c", 0.50),
        ]
        fake = MagicMock()
        fake.rerank.return_value = [-5.0, -8.0, 6.0]
        with (
            patch.object(
                rerank_engine, "search_in_collection", new_callable=AsyncMock,
                return_value=candidates,
            ) as sic,
            patch.object(
                rerank_engine, "_get_reranker", new_callable=AsyncMock,
                return_value=fake,
            ),
        ):
            results = await rerank_engine.search("proj", "query", top_k=2)
        # Over-fetched candidates for the reranker...
        assert sic.call_args.kwargs["top_k"] == 6
        # ...c wins with a calibrated sigmoid score, sliced to top_k.
        assert [r.id for r in results] == ["c", "a"]
        assert results[0].score is not None and results[0].score > 0.99

    @pytest.mark.asyncio
    async def test_large_top_k_not_capped_at_25(
        self, rerank_engine: MemoryEngine
    ) -> None:
        # REST/MCP allow top_k up to 100; reranking must not silently cap it.
        with patch.object(
            rerank_engine, "search_in_collection", new_callable=AsyncMock,
            return_value=[],
        ) as sic:
            await rerank_engine.search("proj", "query", top_k=100)
        assert sic.call_args.kwargs["top_k"] == 100

    @pytest.mark.asyncio
    async def test_score_count_mismatch_falls_back(
        self, rerank_engine: MemoryEngine
    ) -> None:
        # A library returning the wrong number of scores must not 500.
        candidates = [
            self._record("a", "memory a", 0.60),
            self._record("b", "memory b", 0.55),
        ]
        fake = MagicMock()
        fake.rerank.return_value = [1.0]  # one score for two documents
        with (
            patch.object(
                rerank_engine, "search_in_collection", new_callable=AsyncMock,
                return_value=candidates,
            ),
            patch.object(
                rerank_engine, "_get_reranker", new_callable=AsyncMock,
                return_value=fake,
            ),
        ):
            results = await rerank_engine.search("proj", "query", top_k=2)
        assert [r.id for r in results] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_rerank_failure_falls_back_to_vector_order(
        self, rerank_engine: MemoryEngine
    ) -> None:
        candidates = [
            self._record("a", "memory a", 0.60),
            self._record("b", "memory b", 0.55),
        ]
        with (
            patch.object(
                rerank_engine, "search_in_collection", new_callable=AsyncMock,
                return_value=candidates,
            ),
            patch.object(
                rerank_engine, "_get_reranker", new_callable=AsyncMock,
                side_effect=RuntimeError("model download failed"),
            ),
        ):
            results = await rerank_engine.search("proj", "query", top_k=2)
        assert [r.id for r in results] == ["a", "b"]


class TestDeleteAll:
    @pytest.mark.asyncio
    async def test_deletes_and_recreates_collection(
        self, engine: MemoryEngine, mock_qdrant: AsyncMock
    ) -> None:
        mock_qdrant.get_collection.return_value = MagicMock(points_count=5)
        count = await engine.delete_all("project-a")
        assert count == 5
        mock_qdrant.delete_collection.assert_called_once_with("test_project-a")


class TestEmbedFailureIsRecorded:
    @pytest.mark.asyncio
    async def test_embed_error_recorded_in_stats(
        self,
        config: Config,
        mock_backend: AsyncMock,
        mock_qdrant: AsyncMock,
        tmp_path: Path,
    ) -> None:
        # An embedding outage must surface in error_rate/recent_errors; it was
        # previously invisible, so a total outage still reported 0.0% errors.
        stats = DiagnosticStats(str(tmp_path / "stats.json"))
        engine = MemoryEngine(config, mock_backend, stats=stats)
        engine._qdrant = mock_qdrant
        mock_backend.embed.side_effect = Exception("ollama embed down")
        with pytest.raises(Exception, match="ollama embed down"):
            await engine._timed_embed(["some text"])
        assert stats._counters.get("errors.embed") == 1
        assert any(e.get("operation") == "embed" for e in stats._recent_errors)


class TestReembedAtomicity:
    @pytest.mark.asyncio
    async def test_embed_failure_does_not_delete_collection(
        self, engine: MemoryEngine, mock_backend: AsyncMock, mock_qdrant: AsyncMock
    ) -> None:
        # Dimension change so a successful reembed WOULD recreate the collection.
        info = MagicMock()
        info.config.params.vectors.size = 512  # != backend's 768
        mock_qdrant.get_collection.return_value = info
        pt1 = MagicMock(id="id1", payload={"text": "a"})
        pt2 = MagicMock(id="id2", payload={"text": "b"})
        mock_qdrant.scroll.return_value = ([pt1, pt2], None)
        # Embedding fails before any re-embed completes.
        mock_backend.embed.side_effect = Exception("ollama down")
        with pytest.raises(Exception, match="ollama down"):
            await engine.reembed_all("proj")
        # The collection must NOT have been deleted — data is preserved.
        mock_qdrant.delete_collection.assert_not_called()


class TestUpdatePreservesCreatedAt:
    @pytest.mark.asyncio
    async def test_update_keeps_original_created_at(
        self, engine: MemoryEngine, mock_backend: AsyncMock, mock_qdrant: AsyncMock
    ) -> None:
        uid = "550e8400-e29b-41d4-a716-446655440000"
        existing_pt = MagicMock()
        existing_pt.payload = {"created_at": 12345.0}
        mock_qdrant.retrieve.return_value = [existing_pt]
        with (
            patch.object(
                engine, "_extract_facts", new_callable=AsyncMock,
                return_value=["new fact"],
            ),
            patch.object(
                engine, "_dedup_fact", new_callable=AsyncMock,
                return_value=("update", uid, "merged text"),
            ),
        ):
            await engine.add("proj", "john", ["some text"])
        payload = mock_qdrant.upsert.call_args.kwargs["points"][0].payload
        assert payload["created_at"] == 12345.0
        assert payload["updated_at"] != 12345.0


class TestConsolidate:
    @pytest.mark.asyncio
    async def test_merges_similar_cluster(
        self, engine: MemoryEngine, mock_backend: AsyncMock, mock_qdrant: AsyncMock
    ) -> None:
        a = _pt("id-a", "likes python", [1.0, 0.0])
        b = _pt("id-b", "user likes python", [0.99, 0.1])
        c = _pt("id-c", "deploys on unraid", [0.0, 1.0])
        mock_qdrant.scroll.return_value = ([a, b, c], None)
        merged = "User likes Python for backend scripting work"
        mock_backend.generate.return_value = json.dumps([merged])
        result = await engine.consolidate("proj", similarity_threshold=0.75)
        assert result == {
            "clusters": 1, "memories_removed": 2, "memories_created": 1,
        }
        del_call = mock_qdrant.delete.call_args
        assert sorted(del_call.kwargs["points_selector"]) == ["id-a", "id-b"]
        new_payload = mock_qdrant.upsert.call_args.kwargs["points"][0].payload
        assert new_payload["text"] == merged
        # Provenance: the merged memory keeps the earliest created_at.
        assert new_payload["created_at"] == 100.0

    @pytest.mark.asyncio
    async def test_dry_run_makes_no_changes(
        self, engine: MemoryEngine, mock_qdrant: AsyncMock
    ) -> None:
        a = _pt("id-a", "likes python", [1.0, 0.0])
        b = _pt("id-b", "user likes python", [0.99, 0.1])
        mock_qdrant.scroll.return_value = ([a, b], None)
        result = await engine.consolidate("proj", dry_run=True)
        assert result["clusters"] == 1
        assert result["previews"][0]["texts"] == ["likes python", "user likes python"]
        mock_qdrant.delete.assert_not_called()
        mock_qdrant.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_failure_skips_cluster_without_deleting(
        self, engine: MemoryEngine, mock_backend: AsyncMock, mock_qdrant: AsyncMock
    ) -> None:
        a = _pt("id-a", "likes python", [1.0, 0.0])
        b = _pt("id-b", "user likes python", [0.99, 0.1])
        mock_qdrant.scroll.return_value = ([a, b], None)
        mock_backend.generate.side_effect = Exception("ollama down")
        result = await engine.consolidate("proj")
        assert result["memories_removed"] == 0
        mock_qdrant.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_statement_skips_whole_cluster(
        self, engine: MemoryEngine, mock_backend: AsyncMock, mock_qdrant: AsyncMock
    ) -> None:
        # Storing only the valid statement would destroy the rejected one's
        # information once the originals are deleted.
        a = _pt("id-a", "likes python", [1.0, 0.0])
        b = _pt("id-b", "user likes python", [0.99, 0.1])
        mock_qdrant.scroll.return_value = ([a, b], None)
        mock_backend.generate.return_value = json.dumps(
            ["User prefers Python for backend scripting work", "Root cause"]
        )
        result = await engine.consolidate("proj")
        assert result["memories_removed"] == 0
        mock_qdrant.delete.assert_not_called()
        mock_qdrant.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_stores_replacements_before_deleting_originals(
        self, engine: MemoryEngine, mock_backend: AsyncMock, mock_qdrant: AsyncMock
    ) -> None:
        calls: list[str] = []
        mock_qdrant.upsert.side_effect = lambda **kw: calls.append("upsert")
        mock_qdrant.delete.side_effect = lambda **kw: calls.append("delete")
        a = _pt("id-a", "likes python", [1.0, 0.0])
        b = _pt("id-b", "user likes python", [0.99, 0.1])
        mock_qdrant.scroll.return_value = ([a, b], None)
        mock_backend.generate.return_value = json.dumps(
            ["User prefers Python for backend scripting work"]
        )
        await engine.consolidate("proj")
        # Upsert first: a failure between the two leaves duplicates, not loss.
        assert calls == ["upsert", "delete"]

    @pytest.mark.asyncio
    async def test_embed_failure_skips_cluster_without_deleting(
        self, engine: MemoryEngine, mock_backend: AsyncMock, mock_qdrant: AsyncMock
    ) -> None:
        # Replacements are embedded BEFORE the originals are deleted; an
        # embedding outage must leave the cluster untouched.
        a = _pt("id-a", "likes python", [1.0, 0.0])
        b = _pt("id-b", "user likes python", [0.99, 0.1])
        mock_qdrant.scroll.return_value = ([a, b], None)
        mock_backend.generate.return_value = '["merged fact"]'
        mock_backend.embed.side_effect = Exception("embed down")
        result = await engine.consolidate("proj")
        assert result["memories_removed"] == 0
        mock_qdrant.delete.assert_not_called()
        mock_qdrant.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_degraded_mode_refuses(
        self, engine: MemoryEngine, mock_backend: AsyncMock
    ) -> None:
        mock_backend.is_degraded = True
        with pytest.raises(LLMError, match="degraded"):
            await engine.consolidate("proj")

    @pytest.mark.asyncio
    async def test_fewer_than_two_points_noop(
        self, engine: MemoryEngine, mock_qdrant: AsyncMock
    ) -> None:
        mock_qdrant.scroll.return_value = ([_pt("id-a", "x", [1.0, 0.0])], None)
        result = await engine.consolidate("proj")
        assert result == {"clusters": 0, "memories_removed": 0, "memories_created": 0}


class TestCleanupText:
    @pytest.mark.asyncio
    async def test_single_key_dict_rewritten_in_place(
        self, engine: MemoryEngine, mock_qdrant: AsyncMock
    ) -> None:
        pt = _pt("id-a", "{'User prefers dark mode': True}", [0.1])
        mock_qdrant.scroll.return_value = ([pt], None)
        result = await engine.cleanup_text("proj")
        assert result == {"cleaned": 1, "split_into_multiple": 0, "skipped": 0}
        payload = mock_qdrant.upsert.call_args.kwargs["points"][0].payload
        assert payload["text"] == "User prefers dark mode"
        mock_qdrant.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_multi_key_dict_split_into_points(
        self, engine: MemoryEngine, mock_qdrant: AsyncMock
    ) -> None:
        pt = _pt("id-a", "{'fact one': True, 'fact two': True}", [0.1])
        mock_qdrant.scroll.return_value = ([pt], None)
        result = await engine.cleanup_text("proj")
        assert result == {"cleaned": 1, "split_into_multiple": 1, "skipped": 0}
        mock_qdrant.delete.assert_called_once()
        texts = [
            c.kwargs["points"][0].payload["text"]
            for c in mock_qdrant.upsert.call_args_list
        ]
        assert texts == ["fact one", "fact two"]

    @pytest.mark.asyncio
    async def test_normal_text_skipped(
        self, engine: MemoryEngine, mock_qdrant: AsyncMock
    ) -> None:
        pt = _pt("id-a", "a normal memory", [0.1])
        mock_qdrant.scroll.return_value = ([pt], None)
        result = await engine.cleanup_text("proj")
        assert result == {"cleaned": 0, "split_into_multiple": 0, "skipped": 1}
        mock_qdrant.upsert.assert_not_called()


class TestDeleteProject:
    @pytest.mark.asyncio
    async def test_deletes_existing(
        self, engine: MemoryEngine, mock_qdrant: AsyncMock
    ) -> None:
        col = MagicMock()
        col.name = "test_proj"
        mock_qdrant.get_collections.return_value = MagicMock(collections=[col])
        assert await engine.delete_project("proj") is True
        mock_qdrant.delete_collection.assert_called_once_with("test_proj")

    @pytest.mark.asyncio
    async def test_missing_returns_false(
        self, engine: MemoryEngine, mock_qdrant: AsyncMock
    ) -> None:
        assert await engine.delete_project("ghost") is False
        mock_qdrant.delete_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_clears_ensured_cache(
        self, engine: MemoryEngine, mock_qdrant: AsyncMock
    ) -> None:
        # A stale cache entry would let the next add() upsert into a
        # collection that no longer exists.
        await engine._ensure_collection("proj")
        col = MagicMock()
        col.name = "test_proj"
        mock_qdrant.get_collections.return_value = MagicMock(collections=[col])
        await engine.delete_project("proj")
        assert "test_proj" not in engine._ensured_collections


class TestListProjects:
    @pytest.mark.asyncio
    async def test_only_prefixed_collections(
        self, engine: MemoryEngine, mock_qdrant: AsyncMock
    ) -> None:
        mine = MagicMock()
        mine.name = "test_alpha"
        other = MagicMock()
        other.name = "unrelated_thing"
        mock_qdrant.get_collections.return_value = MagicMock(
            collections=[mine, other]
        )
        mock_qdrant.get_collection.return_value = MagicMock(points_count=7)
        projects = await engine.list_projects()
        assert len(projects) == 1
        assert projects[0].slug == "alpha"
        assert projects[0].memory_count == 7


class TestReembedPaging:
    @pytest.mark.asyncio
    async def test_follows_scroll_offsets(
        self, engine: MemoryEngine, mock_backend: AsyncMock, mock_qdrant: AsyncMock
    ) -> None:
        info = MagicMock()
        info.config.params.vectors.size = 768  # no dimension change
        mock_qdrant.get_collection.return_value = info
        page1 = [_pt("id-1", "a", [0.1]), _pt("id-2", "b", [0.1])]
        page2 = [_pt("id-3", "c", [0.1])]
        mock_qdrant.scroll.side_effect = [(page1, "cursor"), (page2, None)]
        count = await engine.reembed_all("proj")
        assert count == 3
        assert mock_qdrant.scroll.call_count == 2
        mock_qdrant.delete_collection.assert_not_called()
