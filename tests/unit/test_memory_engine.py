from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mem_zero.config import Config
from mem_zero.memory_engine import MemoryEngine, validate_memory_id


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


class TestMigrateCollectionPrefix:
    @pytest.fixture
    def migrate_engine(self, mock_backend: AsyncMock) -> tuple[MemoryEngine, AsyncMock]:
        config = Config(collection_prefix="mem-zero")
        eng = MemoryEngine(config, mock_backend)
        mock_q = AsyncMock()
        eng._qdrant = mock_q
        return eng, mock_q

    def _make_points(self, specs: list[tuple[str, list[float], dict]]) -> list[MagicMock]:
        points = []
        for pid, vec, pay in specs:
            pt = MagicMock()
            pt.id = pid
            pt.vector = vec
            pt.payload = pay
            points.append(pt)
        return points

    @pytest.mark.asyncio
    async def test_migrates_and_verifies_1to1(
        self, migrate_engine: tuple[MemoryEngine, AsyncMock]
    ) -> None:
        engine, mock_q = migrate_engine
        old_col = MagicMock()
        old_col.name = "mem0_my-project"
        mock_q.get_collections.return_value = MagicMock(collections=[old_col])

        vector_config = MagicMock()
        mock_q.get_collection.return_value = MagicMock(
            points_count=2, config=MagicMock(params=MagicMock(vectors=vector_config)),
        )

        pts = self._make_points([
            ("id-1", [0.1] * 768, {"text": "fact one", "user_id": "john"}),
            ("id-2", [0.2] * 768, {"text": "fact two", "user_id": "john"}),
        ])
        mock_q.scroll.return_value = (pts, None)

        count = await engine.migrate_collection_prefix("mem0")

        assert count == 1
        assert mock_q.upsert.call_count == 2
        mock_q.delete_collection.assert_called_once_with("mem0_my-project")
        assert mock_q.scroll.call_count == 2

    @pytest.mark.asyncio
    async def test_preserves_point_ids_vectors_payloads(
        self, migrate_engine: tuple[MemoryEngine, AsyncMock]
    ) -> None:
        engine, mock_q = migrate_engine
        old_col = MagicMock()
        old_col.name = "mem0_test-proj"
        mock_q.get_collections.return_value = MagicMock(collections=[old_col])
        mock_q.get_collection.return_value = MagicMock(
            points_count=1, config=MagicMock(params=MagicMock(vectors=MagicMock())),
        )

        original_payload = {"text": "important fact", "user_id": "john", "created_at": 12345.0}
        original_vector = [0.5] * 768
        pts = self._make_points([("preserve-me", original_vector, original_payload)])
        mock_q.scroll.return_value = (pts, None)

        await engine.migrate_collection_prefix("mem0")

        upsert_call = mock_q.upsert.call_args
        migrated_point = upsert_call.kwargs["points"][0]
        assert migrated_point.id == "preserve-me"
        assert migrated_point.vector == original_vector
        assert migrated_point.payload == original_payload

    @pytest.mark.asyncio
    async def test_keeps_both_on_payload_mismatch(
        self, migrate_engine: tuple[MemoryEngine, AsyncMock]
    ) -> None:
        engine, mock_q = migrate_engine
        old_col = MagicMock()
        old_col.name = "mem0_mismatch"
        mock_q.get_collections.return_value = MagicMock(collections=[old_col])
        mock_q.get_collection.return_value = MagicMock(
            points_count=1, config=MagicMock(params=MagicMock(vectors=MagicMock())),
        )

        old_pt = self._make_points([("id-1", [0.1] * 768, {"text": "original"})])
        bad_pt = self._make_points([("id-1", [0.1] * 768, {"text": "corrupted"})])
        mock_q.scroll.side_effect = [
            (old_pt, None),
            (bad_pt, None),
        ]

        count = await engine.migrate_collection_prefix("mem0")

        assert count == 0
        delete_calls = [c.args[0] for c in mock_q.delete_collection.call_args_list]
        assert "mem0_mismatch" not in delete_calls

    @pytest.mark.asyncio
    async def test_keeps_both_on_missing_point(
        self, migrate_engine: tuple[MemoryEngine, AsyncMock]
    ) -> None:
        engine, mock_q = migrate_engine
        old_col = MagicMock()
        old_col.name = "mem0_missing"
        mock_q.get_collections.return_value = MagicMock(collections=[old_col])
        mock_q.get_collection.return_value = MagicMock(
            points_count=2, config=MagicMock(params=MagicMock(vectors=MagicMock())),
        )

        both_pts = self._make_points([
            ("id-1", [0.1] * 768, {"text": "fact one"}),
            ("id-2", [0.2] * 768, {"text": "fact two"}),
        ])
        only_one = self._make_points([("id-1", [0.1] * 768, {"text": "fact one"})])
        mock_q.scroll.side_effect = [
            (both_pts, None),
            (only_one, None),
        ]

        count = await engine.migrate_collection_prefix("mem0")

        assert count == 0
        delete_calls = [c.args[0] for c in mock_q.delete_collection.call_args_list]
        assert "mem0_missing" not in delete_calls

    @pytest.mark.asyncio
    async def test_retries_after_previous_attempt(
        self, migrate_engine: tuple[MemoryEngine, AsyncMock]
    ) -> None:
        engine, mock_q = migrate_engine
        old_col = MagicMock()
        old_col.name = "mem0_retry"
        leftover = MagicMock()
        leftover.name = "mem-zero_retry"
        mock_q.get_collections.return_value = MagicMock(collections=[old_col, leftover])
        mock_q.get_collection.return_value = MagicMock(
            points_count=1, config=MagicMock(params=MagicMock(vectors=MagicMock())),
        )

        pts = self._make_points([("id-1", [0.1] * 768, {"text": "fact"})])
        mock_q.scroll.return_value = (pts, None)

        count = await engine.migrate_collection_prefix("mem0")

        assert count == 1
        delete_calls = [c.args[0] for c in mock_q.delete_collection.call_args_list]
        assert "mem-zero_retry" in delete_calls
        assert "mem0_retry" in delete_calls

    @pytest.mark.asyncio
    async def test_cleans_up_on_exception(
        self, migrate_engine: tuple[MemoryEngine, AsyncMock]
    ) -> None:
        engine, mock_q = migrate_engine
        old_col = MagicMock()
        old_col.name = "mem0_fails"
        mock_q.get_collections.return_value = MagicMock(collections=[old_col])
        mock_q.get_collection.return_value = MagicMock(
            points_count=10, config=MagicMock(params=MagicMock(vectors=MagicMock())),
        )
        mock_q.scroll.side_effect = Exception("connection lost")

        count = await engine.migrate_collection_prefix("mem0")

        assert count == 0
        mock_q.delete_collection.assert_called_once_with("mem-zero_fails")

    @pytest.mark.asyncio
    async def test_deletes_empty_old_collections(
        self, migrate_engine: tuple[MemoryEngine, AsyncMock]
    ) -> None:
        engine, mock_q = migrate_engine
        old_col = MagicMock()
        old_col.name = "mem0_empty-proj"
        mock_q.get_collections.return_value = MagicMock(collections=[old_col])
        mock_q.get_collection.return_value = MagicMock(points_count=0)

        count = await engine.migrate_collection_prefix("mem0")

        assert count == 0
        mock_q.delete_collection.assert_called_once_with("mem0_empty-proj")
        mock_q.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignores_unrelated_collections(
        self, migrate_engine: tuple[MemoryEngine, AsyncMock]
    ) -> None:
        engine, mock_q = migrate_engine
        unrelated = MagicMock()
        unrelated.name = "other_collection"
        mock_q.get_collections.return_value = MagicMock(collections=[unrelated])

        count = await engine.migrate_collection_prefix("mem0")

        assert count == 0
        mock_q.get_collection.assert_not_called()
        mock_q.create_collection.assert_not_called()
