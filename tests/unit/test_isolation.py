from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mem_zero.config import Config
from mem_zero.memory_engine import MemoryEngine


@pytest.fixture
def config() -> Config:
    return Config(collection_prefix="iso")


@pytest.fixture
def engine(config: Config) -> MemoryEngine:
    eng = MemoryEngine(config)
    eng._qdrant = AsyncMock()
    eng._qdrant.get_collections.return_value = MagicMock(collections=[])
    eng._qdrant.get_collection.return_value = MagicMock(points_count=0)
    return eng


class TestProjectIsolation:
    def test_collection_names_are_distinct(self, config: Config) -> None:
        assert config.collection_name("alpha") == "iso_alpha"
        assert config.collection_name("beta") == "iso_beta"
        assert config.collection_name("alpha") != config.collection_name("beta")

    @pytest.mark.asyncio
    async def test_add_targets_correct_collection(self, engine: MemoryEngine) -> None:
        with (
            patch.object(
                engine, "embed", new_callable=AsyncMock, return_value=[[0.1] * 768]
            ),
            patch.object(
                engine, "_extract_facts", new_callable=AsyncMock,
                return_value=["memory for alpha"],
            ),
            patch.object(
                engine, "_dedup_fact", new_callable=AsyncMock,
                return_value=("add", None, None),
            ),
        ):
            await engine.add("alpha", "user1", ["memory for alpha"])
            call = engine._qdrant.upsert.call_args
            assert call.kwargs["collection_name"] == "iso_alpha"

    @pytest.mark.asyncio
    async def test_search_targets_correct_collection(self, engine: MemoryEngine) -> None:
        engine._qdrant.query_points.return_value = MagicMock(points=[])
        with patch.object(
            engine, "embed", new_callable=AsyncMock, return_value=[[0.1] * 768]
        ):
            await engine.search("beta", "query")
            call = engine._qdrant.query_points.call_args
            assert call.kwargs["collection_name"] == "iso_beta"

    @pytest.mark.asyncio
    async def test_list_targets_correct_collection(self, engine: MemoryEngine) -> None:
        engine._qdrant.scroll.return_value = ([], None)
        await engine.list_all("alpha")
        call = engine._qdrant.scroll.call_args
        assert call.kwargs["collection_name"] == "iso_alpha"

    @pytest.mark.asyncio
    async def test_delete_targets_correct_collection(self, engine: MemoryEngine) -> None:
        uid = "550e8400-e29b-41d4-a716-446655440000"
        await engine.delete("alpha", uid)
        call = engine._qdrant.delete.call_args
        assert call.kwargs["collection_name"] == "iso_alpha"

    @pytest.mark.asyncio
    async def test_delete_all_targets_correct_collection(self, engine: MemoryEngine) -> None:
        engine._qdrant.get_collection.return_value = MagicMock(points_count=3)
        await engine.delete_all("beta")
        engine._qdrant.delete_collection.assert_called_once_with("iso_beta")

    @pytest.mark.asyncio
    async def test_no_cross_collection_access(self, engine: MemoryEngine) -> None:
        engine._qdrant.scroll.return_value = ([], None)
        await engine.list_all("alpha")
        await engine.list_all("beta")
        calls = engine._qdrant.scroll.call_args_list
        assert calls[0].kwargs["collection_name"] == "iso_alpha"
        assert calls[1].kwargs["collection_name"] == "iso_beta"
