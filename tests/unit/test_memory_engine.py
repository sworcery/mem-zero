from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mem_zero.config import Config
from mem_zero.memory_engine import MemoryEngine, validate_memory_id
from mem_zero.models import MemoryRecord


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
    backend.embed.return_value = [[0.1] * 768]
    backend.generate.return_value = '["test memory"]'
    return backend


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


class TestExtractFacts:
    @pytest.mark.asyncio
    async def test_empty_list_falls_back_to_raw_text(
        self, engine: MemoryEngine, mock_backend: AsyncMock
    ) -> None:
        mock_backend.generate.return_value = "[]"
        facts = await engine._extract_facts("some user input")
        assert facts == ["some user input"]

    @pytest.mark.asyncio
    async def test_valid_facts_returned(
        self, engine: MemoryEngine, mock_backend: AsyncMock
    ) -> None:
        mock_backend.generate.return_value = '["fact one", "fact two"]'
        facts = await engine._extract_facts("some text")
        assert facts == ["fact one", "fact two"]

    @pytest.mark.asyncio
    async def test_dict_keys_extracted_as_facts(
        self, engine: MemoryEngine, mock_backend: AsyncMock
    ) -> None:
        mock_backend.generate.return_value = (
            '{"User likes Python": true, "User works at Acme": true}'
        )
        facts = await engine._extract_facts("some text")
        assert facts == ["User likes Python", "User works at Acme"]

    @pytest.mark.asyncio
    async def test_dict_with_facts_key(
        self, engine: MemoryEngine, mock_backend: AsyncMock
    ) -> None:
        mock_backend.generate.return_value = '{"facts": ["fact one", "fact two"]}'
        facts = await engine._extract_facts("some text")
        assert facts == ["fact one", "fact two"]

    @pytest.mark.asyncio
    async def test_dict_list_value_used(
        self, engine: MemoryEngine, mock_backend: AsyncMock
    ) -> None:
        mock_backend.generate.return_value = '{"greetings": ["hello", "world"]}'
        facts = await engine._extract_facts("some text")
        assert facts == ["hello", "world"]

    @pytest.mark.asyncio
    async def test_dict_string_values_used_not_keys(
        self, engine: MemoryEngine, mock_backend: AsyncMock
    ) -> None:
        mock_backend.generate.return_value = (
            '{"fact1": "User likes Python", "fact2": "User uses Qdrant"}'
        )
        facts = await engine._extract_facts("some text")
        assert facts == ["User likes Python", "User uses Qdrant"]


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


class TestDeleteAll:
    @pytest.mark.asyncio
    async def test_deletes_and_recreates_collection(
        self, engine: MemoryEngine, mock_qdrant: AsyncMock
    ) -> None:
        mock_qdrant.get_collection.return_value = MagicMock(points_count=5)
        count = await engine.delete_all("project-a")
        assert count == 5
        mock_qdrant.delete_collection.assert_called_once_with("test_project-a")


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
