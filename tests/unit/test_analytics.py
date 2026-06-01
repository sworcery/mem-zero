from __future__ import annotations

from pathlib import Path

import pytest

from mem_zero.analytics import AnalyticsEngine
from mem_zero.stats import DiagnosticStats


@pytest.fixture
def stats(tmp_path: Path) -> DiagnosticStats:
    return DiagnosticStats(tmp_path / "stats.json")


@pytest.fixture
def analytics(stats: DiagnosticStats) -> AnalyticsEngine:
    return AnalyticsEngine(stats)


class TestProjectGrowth:
    def test_empty_snapshots(self, analytics: AnalyticsEngine) -> None:
        assert analytics.project_growth() == []

    def test_tracks_deltas(self, analytics: AnalyticsEngine, stats: DiagnosticStats) -> None:
        stats.record_daily_snapshot({"alpha": 5})
        stats._daily_snapshots[-1]["date"] = "2026-05-30"
        stats.record_daily_snapshot({"alpha": 8})
        growth = analytics.project_growth("alpha")
        assert len(growth) == 2
        assert growth[0].total == 5
        assert growth[1].total == 8
        assert growth[1].delta == 3

    def test_global_growth(self, analytics: AnalyticsEngine, stats: DiagnosticStats) -> None:
        stats.record_daily_snapshot({"a": 5, "b": 3})
        growth = analytics.project_growth()
        assert growth[0].total == 8


class TestProjectHealth:
    def test_empty(self, analytics: AnalyticsEngine) -> None:
        assert analytics.project_health() == []

    def test_reports_operations(
        self, analytics: AnalyticsEngine, stats: DiagnosticStats
    ) -> None:
        stats.inc_project("alpha", "add_memory", 10)
        stats.inc_project("alpha", "search", 5)
        stats.record_activity("alpha")

        health = analytics.project_health()
        assert len(health) == 1
        assert health[0].slug == "alpha"
        assert health[0].operations_total == 15

    def test_staleness(self, analytics: AnalyticsEngine, stats: DiagnosticStats) -> None:
        stats.inc_project("old", "search", 1)
        stats.record_activity("old")
        health = analytics.project_health()
        assert health[0].staleness_days is not None
        assert health[0].staleness_days < 1

    def test_sorted_by_operations(
        self, analytics: AnalyticsEngine, stats: DiagnosticStats
    ) -> None:
        stats.inc_project("low", "search", 1)
        stats.inc_project("high", "search", 100)
        health = analytics.project_health()
        assert health[0].slug == "high"
        assert health[1].slug == "low"


class TestUsagePatterns:
    def test_empty_stats(self, analytics: AnalyticsEngine) -> None:
        patterns = analytics.usage_patterns()
        assert patterns.dedup_effectiveness is None
        assert patterns.search_hit_rate is None

    def test_dedup_effectiveness(
        self, analytics: AnalyticsEngine, stats: DiagnosticStats
    ) -> None:
        stats.inc("dedup.add", 6)
        stats.inc("dedup.skip", 3)
        stats.inc("dedup.update", 1)
        patterns = analytics.usage_patterns()
        assert patterns.dedup_effectiveness == 40.0

    def test_search_hit_rate(
        self, analytics: AnalyticsEngine, stats: DiagnosticStats
    ) -> None:
        stats.inc("search", 10)
        stats.inc("search.zero_results", 2)
        patterns = analytics.usage_patterns()
        assert patterns.search_hit_rate == 80.0

    def test_most_active_projects(
        self, analytics: AnalyticsEngine, stats: DiagnosticStats
    ) -> None:
        stats.inc_project("busy", "add_memory", 50)
        stats.inc_project("busy", "search", 30)
        stats.inc_project("quiet", "search", 2)
        patterns = analytics.usage_patterns()
        assert len(patterns.most_active_projects) == 2
        assert patterns.most_active_projects[0][0] == "busy"


class TestSearchQuality:
    def test_empty(self, analytics: AnalyticsEngine) -> None:
        quality = analytics.search_quality()
        assert quality["avg_score"] is None
        assert quality["high_quality_pct"] is None

    def test_with_scores(
        self, analytics: AnalyticsEngine, stats: DiagnosticStats
    ) -> None:
        stats.record_search_scores([0.9, 0.85, 0.7, 0.3, 0.1])
        stats.inc("search", 5)
        quality = analytics.search_quality()
        assert quality["high_quality_count"] == 3
        assert quality["low_quality_count"] == 2
        assert quality["high_quality_pct"] == 60.0


class TestSummary:
    def test_returns_all_sections(
        self, analytics: AnalyticsEngine, stats: DiagnosticStats
    ) -> None:
        stats.inc_project("test", "search", 5)
        stats.inc("search", 5)
        summary = analytics.summary()
        assert "projects" in summary
        assert "usage" in summary
        assert "search_quality" in summary
