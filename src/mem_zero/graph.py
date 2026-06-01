from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .memory_engine import MemoryEngine

logger = logging.getLogger(__name__)


@dataclass
class MemoryNode:
    id: str
    text: str
    project: str
    user_id: str
    created_at: float
    connections: list[str] = field(default_factory=list)


@dataclass
class MemoryEdge:
    source_id: str
    target_id: str
    similarity: float
    relationship: str = "similar"


@dataclass
class MemoryCluster:
    id: int
    theme: str
    memory_ids: list[str]
    avg_similarity: float


class MemoryGraph:
    def __init__(self, engine: MemoryEngine) -> None:
        self._engine = engine

    async def find_related(
        self,
        project_slug: str,
        memory_id: str,
        threshold: float = 0.70,
        limit: int = 10,
    ) -> list[MemoryEdge]:
        memories = await self._engine.list_all(project_slug, limit=500)
        source = None
        for mem in memories:
            if mem.id == memory_id:
                source = mem
                break

        if not source:
            return []

        results = await self._engine.search(project_slug, source.text, top_k=limit + 1)
        edges: list[MemoryEdge] = []
        for result in results:
            if result.id == memory_id:
                continue
            if result.score is not None and result.score >= threshold:
                edges.append(MemoryEdge(
                    source_id=memory_id,
                    target_id=result.id,
                    similarity=round(result.score, 4),
                ))
            if len(edges) >= limit:
                break
        return edges

    async def build_graph(
        self,
        project_slug: str,
        threshold: float = 0.70,
        max_memories: int = 200,
    ) -> dict[str, Any]:
        memories = await self._engine.list_all(project_slug, limit=max_memories)
        if not memories:
            return {"nodes": [], "edges": [], "clusters": []}

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str]] = set()

        for mem in memories:
            nodes.append({
                "id": mem.id,
                "text": mem.text[:200],
                "user_id": mem.user_id,
                "created_at": mem.created_at,
            })

        for mem in memories:
            results = await self._engine.search(project_slug, mem.text, top_k=6)
            for result in results:
                if result.id == mem.id:
                    continue
                edge_key = tuple(sorted([mem.id, result.id]))
                if edge_key in seen_edges:
                    continue
                if result.score is not None and result.score >= threshold:
                    seen_edges.add(edge_key)
                    edges.append({
                        "source": mem.id,
                        "target": result.id,
                        "similarity": round(result.score, 4),
                    })

        clusters = self._detect_clusters(nodes, edges)

        return {
            "nodes": nodes,
            "edges": edges,
            "clusters": clusters,
            "stats": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "cluster_count": len(clusters),
                "avg_connections": (
                    round(len(edges) * 2 / len(nodes), 1) if nodes else 0
                ),
            },
        }

    @staticmethod
    def _detect_clusters(
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        adjacency: dict[str, set[str]] = {n["id"]: set() for n in nodes}
        for edge in edges:
            adjacency[edge["source"]].add(edge["target"])
            adjacency[edge["target"]].add(edge["source"])

        visited: set[str] = set()
        clusters: list[dict[str, Any]] = []
        cluster_id = 0

        for node_id in adjacency:
            if node_id in visited:
                continue
            if not adjacency[node_id]:
                continue

            component: list[str] = []
            queue = [node_id]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            if len(component) >= 2:
                relevant_edges = [
                    e for e in edges
                    if e["source"] in component and e["target"] in component
                ]
                avg_sim = (
                    sum(e["similarity"] for e in relevant_edges) / len(relevant_edges)
                    if relevant_edges
                    else 0
                )
                clusters.append({
                    "id": cluster_id,
                    "memory_ids": component,
                    "size": len(component),
                    "avg_similarity": round(avg_sim, 4),
                })
                cluster_id += 1

        clusters.sort(key=lambda c: c["size"], reverse=True)
        return clusters

    async def cross_project_search(
        self,
        query: str,
        project_slugs: list[str],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        all_results: list[dict[str, Any]] = []
        for slug in project_slugs:
            try:
                results = await self._engine.search(slug, query, top_k=top_k)
                for r in results:
                    all_results.append({
                        "project": slug,
                        "id": r.id,
                        "text": r.text,
                        "score": r.score,
                        "user_id": r.user_id,
                        "created_at": r.created_at,
                    })
            except Exception:
                logger.warning("Cross-project search failed for %s", slug)
                continue

        all_results.sort(key=lambda r: r.get("score", 0) or 0, reverse=True)
        return all_results[:top_k]

    async def find_duplicates(
        self,
        project_slug: str,
        threshold: float = 0.90,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        memories = await self._engine.list_all(project_slug, limit=limit)
        duplicates: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, str]] = set()

        for mem in memories:
            results = await self._engine.search(project_slug, mem.text, top_k=3)
            for result in results:
                if result.id == mem.id:
                    continue
                pair_key = tuple(sorted([mem.id, result.id]))
                if pair_key in seen_pairs:
                    continue
                if result.score is not None and result.score >= threshold:
                    seen_pairs.add(pair_key)
                    duplicates.append({
                        "memory_a": {"id": mem.id, "text": mem.text},
                        "memory_b": {"id": result.id, "text": result.text},
                        "similarity": round(result.score, 4),
                    })

        duplicates.sort(key=lambda d: d["similarity"], reverse=True)
        return duplicates
