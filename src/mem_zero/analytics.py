from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .stats import DiagnosticStats, _NullStats


@dataclass
class GrowthPoint:
    date: str
    total: int
    delta: int = 0


@dataclass
class ProjectHealth:
    slug: str
    memory_count: int
    avg_search_score: float | None
    staleness_days: float | None
    operations_total: int
    growth_rate_per_day: float


@dataclass
class UsagePattern:
    peak_operations: dict[str, int] = field(default_factory=dict)
    most_active_projects: list[tuple[str, int]] = field(default_factory=list)
    avg_facts_per_add: float | None = None
    dedup_effectiveness: float | None = None
    search_hit_rate: float | None = None


class AnalyticsEngine:
    def __init__(self, stats: DiagnosticStats | _NullStats) -> None:
        self._stats = stats

    def project_growth(self, project: str | None = None) -> list[GrowthPoint]:
        snap = self._stats.snapshot()
        daily = snap.get("daily_snapshots", [])
        if not daily:
            return []

        points: list[GrowthPoint] = []
        prev_total = 0
        for entry in daily:
            total = (
                entry.get("projects", {}).get(project, 0) if project
                else entry.get("total", 0)
            )
            delta = total - prev_total
            points.append(GrowthPoint(date=entry["date"], total=total, delta=delta))
            prev_total = total
        return points

    def project_health(self) -> list[ProjectHealth]:
        snap = self._stats.snapshot()
        projects = snap.get("projects", {})
        now = time.time()
        results: list[ProjectHealth] = []

        search_avg = snap.get("accuracy", {}).get("search", {}).get("avg_score")

        for slug, counters in projects.items():
            ops = sum(
                counters.get(k, 0)
                for k in ("add_memory", "search", "delete", "delete_all")
            )
            last_ts = counters.get("last_activity_ts")
            staleness = (now - last_ts) / 86400 if last_ts else None

            daily = snap.get("daily_snapshots", [])
            mem_count = 0
            growth_rate = 0.0
            project_days: list[int] = []
            for entry in daily:
                count = entry.get("projects", {}).get(slug, 0)
                project_days.append(count)
                mem_count = count

            if len(project_days) >= 2 and project_days[0] > 0:
                span = len(project_days)
                growth_rate = (project_days[-1] - project_days[0]) / span

            results.append(ProjectHealth(
                slug=slug,
                memory_count=mem_count,
                avg_search_score=search_avg,
                staleness_days=round(staleness, 1) if staleness is not None else None,
                operations_total=ops,
                growth_rate_per_day=round(growth_rate, 2),
            ))

        results.sort(key=lambda h: h.operations_total, reverse=True)
        return results

    def usage_patterns(self) -> UsagePattern:
        snap = self._stats.snapshot()
        usage = snap.get("usage", {})
        accuracy = snap.get("accuracy", {})

        projects = snap.get("projects", {})
        ranked = sorted(
            ((slug, sum(c.get(k, 0) for k in ("add_memory", "search")))
             for slug, c in projects.items()),
            key=lambda x: x[1],
            reverse=True,
        )

        extract = accuracy.get("extraction", {})
        avg_facts = extract.get("avg_facts_per_input")

        dedup = accuracy.get("dedup", {})
        dedup_total = dedup.get("total_checked", 0)
        dedup_eff = None
        if dedup_total > 0:
            effective = dedup.get("skipped", 0) + dedup.get("updated", 0)
            dedup_eff = round(effective / dedup_total * 100, 1)

        search = accuracy.get("search", {})
        total_searches = search.get("total_queries", 0)
        zero_results = search.get("zero_result_queries", 0)
        hit_rate = None
        if total_searches > 0:
            hit_rate = round((total_searches - zero_results) / total_searches * 100, 1)

        ops_by_type: dict[str, int] = {}
        for key in ("add_memory", "search", "delete", "delete_all", "reembed", "consolidate"):
            val = usage.get(f"total_{key}s" if key == "search" else key, 0)
            if key == "add_memory":
                val = usage.get("add_operations", 0)
            elif key == "search":
                val = usage.get("search_operations", 0)
            elif key == "delete":
                val = usage.get("total_deletes", 0)
            elif key == "reembed":
                val = usage.get("total_reembeds", 0)
            elif key == "consolidate":
                val = usage.get("total_consolidations", 0)
            if val:
                ops_by_type[key] = val

        return UsagePattern(
            peak_operations=ops_by_type,
            most_active_projects=ranked[:10],
            avg_facts_per_add=avg_facts,
            dedup_effectiveness=dedup_eff,
            search_hit_rate=hit_rate,
        )

    def search_quality(self) -> dict[str, Any]:
        snap = self._stats.snapshot()
        search = snap.get("accuracy", {}).get("search", {})
        dist = search.get("score_distribution", {})

        total = sum(dist.values())
        high_quality = dist.get("0.8-1.0", 0) + dist.get("0.6-0.8", 0)
        low_quality = dist.get("0.0-0.2", 0) + dist.get("0.2-0.4", 0)

        quality_ratio = round(high_quality / total * 100, 1) if total > 0 else None

        return {
            "avg_score": search.get("avg_score"),
            "total_queries": search.get("total_queries", 0),
            "zero_result_rate": (
                round(search.get("zero_result_queries", 0)
                      / search["total_queries"] * 100, 1)
                if search.get("total_queries", 0) > 0
                else None
            ),
            "score_distribution": dist,
            "high_quality_pct": quality_ratio,
            "low_quality_count": low_quality,
            "high_quality_count": high_quality,
        }

    def summary(self) -> dict[str, Any]:
        health = self.project_health()
        patterns = self.usage_patterns()
        quality = self.search_quality()

        return {
            "projects": [
                {
                    "slug": h.slug,
                    "memories": h.memory_count,
                    "staleness_days": h.staleness_days,
                    "operations": h.operations_total,
                    "growth_rate": h.growth_rate_per_day,
                }
                for h in health
            ],
            "usage": {
                "operations_by_type": patterns.peak_operations,
                "most_active": patterns.most_active_projects[:5],
                "avg_facts_per_add": patterns.avg_facts_per_add,
                "dedup_effectiveness_pct": patterns.dedup_effectiveness,
                "search_hit_rate_pct": patterns.search_hit_rate,
            },
            "search_quality": quality,
        }
