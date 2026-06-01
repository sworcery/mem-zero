from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from mem_zero.graph import MemoryGraph
from mem_zero.models import MemoryRecord


@pytest.fixture
def mock_engine() -> AsyncMock:
    engine = AsyncMock()
    engine.list_all.return_value = []
    engine.search.return_value = []
    return engine


@pytest.fixture
def graph(mock_engine: AsyncMock) -> MemoryGraph:
    return MemoryGraph(mock_engine)


def _mem(id: str, text: str, score: float | None = None) -> MemoryRecord:
    return MemoryRecord(
        id=id, text=text, user_id="u",
        created_at=1700000000, updated_at=1700000000,
        score=score,
    )


class TestFindRelated:
    @pytest.mark.asyncio
    async def test_no_results_for_missing_memory(
        self, graph: MemoryGraph, mock_engine: AsyncMock
    ) -> None:
        mock_engine.list_all.return_value = [_mem("a", "hello")]
        edges = await graph.find_related("proj", "nonexistent")
        assert edges == []

    @pytest.mark.asyncio
    async def test_finds_related_memories(
        self, graph: MemoryGraph, mock_engine: AsyncMock
    ) -> None:
        mock_engine.list_all.return_value = [
            _mem("a", "Python is great"),
            _mem("b", "Python is awesome"),
            _mem("c", "Unrelated topic"),
        ]
        mock_engine.search.return_value = [
            _mem("a", "Python is great", score=1.0),
            _mem("b", "Python is awesome", score=0.85),
            _mem("c", "Unrelated topic", score=0.3),
        ]
        edges = await graph.find_related("proj", "a", threshold=0.7)
        assert len(edges) == 1
        assert edges[0].target_id == "b"
        assert edges[0].similarity == 0.85

    @pytest.mark.asyncio
    async def test_excludes_self(
        self, graph: MemoryGraph, mock_engine: AsyncMock
    ) -> None:
        mock_engine.list_all.return_value = [_mem("a", "test")]
        mock_engine.search.return_value = [
            _mem("a", "test", score=1.0),
        ]
        edges = await graph.find_related("proj", "a")
        assert edges == []


class TestBuildGraph:
    @pytest.mark.asyncio
    async def test_empty_project(
        self, graph: MemoryGraph, mock_engine: AsyncMock
    ) -> None:
        result = await graph.build_graph("proj")
        assert result["nodes"] == []
        assert result["edges"] == []

    @pytest.mark.asyncio
    async def test_builds_nodes_and_edges(
        self, graph: MemoryGraph, mock_engine: AsyncMock
    ) -> None:
        mock_engine.list_all.return_value = [
            _mem("a", "Python programming"),
            _mem("b", "Python coding"),
        ]

        def search_side_effect(slug, text, top_k=10):
            if "programming" in text:
                return [
                    _mem("a", "Python programming", score=1.0),
                    _mem("b", "Python coding", score=0.85),
                ]
            return [
                _mem("b", "Python coding", score=1.0),
                _mem("a", "Python programming", score=0.85),
            ]

        mock_engine.search.side_effect = search_side_effect
        result = await graph.build_graph("proj", threshold=0.7)
        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1
        assert result["stats"]["node_count"] == 2
        assert result["stats"]["edge_count"] == 1


class TestDetectClusters:
    def test_finds_connected_components(self) -> None:
        nodes = [
            {"id": "a"}, {"id": "b"}, {"id": "c"},
            {"id": "x"}, {"id": "y"},
            {"id": "z"},
        ]
        edges = [
            {"source": "a", "target": "b", "similarity": 0.9},
            {"source": "b", "target": "c", "similarity": 0.8},
            {"source": "x", "target": "y", "similarity": 0.7},
        ]
        clusters = MemoryGraph._detect_clusters(nodes, edges)
        assert len(clusters) == 2
        sizes = sorted([c["size"] for c in clusters], reverse=True)
        assert sizes == [3, 2]

    def test_no_clusters_when_no_edges(self) -> None:
        nodes = [{"id": "a"}, {"id": "b"}]
        clusters = MemoryGraph._detect_clusters(nodes, [])
        assert clusters == []


class TestCrossProjectSearch:
    @pytest.mark.asyncio
    async def test_searches_multiple_projects(
        self, graph: MemoryGraph, mock_engine: AsyncMock
    ) -> None:
        def search_side_effect(slug, query, top_k=10):
            if slug == "alpha":
                return [_mem("a1", "alpha result", score=0.9)]
            return [_mem("b1", "beta result", score=0.7)]

        mock_engine.search.side_effect = search_side_effect
        results = await graph.cross_project_search(
            "test query", ["alpha", "beta"]
        )
        assert len(results) == 2
        assert results[0]["project"] == "alpha"
        assert results[0]["score"] == 0.9

    @pytest.mark.asyncio
    async def test_handles_project_errors(
        self, graph: MemoryGraph, mock_engine: AsyncMock
    ) -> None:
        mock_engine.search.side_effect = Exception("fail")
        results = await graph.cross_project_search("test", ["alpha"])
        assert results == []


class TestFindDuplicates:
    @pytest.mark.asyncio
    async def test_finds_near_duplicates(
        self, graph: MemoryGraph, mock_engine: AsyncMock
    ) -> None:
        mock_engine.list_all.return_value = [
            _mem("a", "User prefers dark mode"),
            _mem("b", "User likes dark mode"),
        ]

        def search_side_effect(slug, text, top_k=10):
            if "prefers" in text:
                return [
                    _mem("a", "User prefers dark mode", score=1.0),
                    _mem("b", "User likes dark mode", score=0.95),
                ]
            return [
                _mem("b", "User likes dark mode", score=1.0),
                _mem("a", "User prefers dark mode", score=0.95),
            ]

        mock_engine.search.side_effect = search_side_effect
        dupes = await graph.find_duplicates("proj", threshold=0.90)
        assert len(dupes) == 1
        assert dupes[0]["similarity"] == 0.95

    @pytest.mark.asyncio
    async def test_no_duplicates(
        self, graph: MemoryGraph, mock_engine: AsyncMock
    ) -> None:
        mock_engine.list_all.return_value = [
            _mem("a", "Python"),
            _mem("b", "Cooking recipes"),
        ]

        def search_side_effect(slug, text, top_k=10):
            if "Python" in text:
                return [
                    _mem("a", "Python", score=1.0),
                    _mem("b", "Cooking recipes", score=0.1),
                ]
            return [
                _mem("b", "Cooking recipes", score=1.0),
                _mem("a", "Python", score=0.1),
            ]

        mock_engine.search.side_effect = search_side_effect
        dupes = await graph.find_duplicates("proj", threshold=0.90)
        assert dupes == []
