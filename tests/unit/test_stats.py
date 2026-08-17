from __future__ import annotations

from pathlib import Path

import pytest

from mem_zero.stats import NULL_STATS, DiagnosticStats


class TestNullStats:
    def test_inc_is_noop(self) -> None:
        NULL_STATS.inc("anything", 5)

    def test_inc_project_is_noop(self) -> None:
        NULL_STATS.inc_project("proj", "key", 3)

    def test_record_latency_is_noop(self) -> None:
        NULL_STATS.record_latency("key", 123.4)

    def test_record_activity_is_noop(self) -> None:
        NULL_STATS.record_activity("proj")

    def test_get_last_activity_returns_none(self) -> None:
        assert NULL_STATS.get_last_activity("proj") is None

    def test_snapshot_returns_empty(self) -> None:
        assert NULL_STATS.snapshot() == {}

    def test_flush_is_noop(self) -> None:
        NULL_STATS.flush()

    def test_reset_is_noop(self) -> None:
        NULL_STATS.reset()


class TestDiagnosticStats:
    @pytest.fixture
    def stats_path(self, tmp_path: Path) -> Path:
        return tmp_path / "stats.json"

    def test_flush_prunes_stale_projects(self, stats_path: Path) -> None:
        import time as _time

        stats = DiagnosticStats(str(stats_path))
        stats.inc_project("ancient", "add_memory")
        stats._project_counters["ancient"]["last_activity_ts"] = (
            _time.time() - 200 * 86400
        )
        stats.record_activity("fresh")
        stats.inc_project("no-timestamp", "add_memory")  # never pruned
        stats.flush()
        assert "ancient" not in stats._project_counters
        assert "fresh" in stats._project_counters
        assert "no-timestamp" in stats._project_counters

    def test_load_coerces_null_and_wrong_types(self, stats_path: Path) -> None:
        # A hand-edited or version-drifted file with null / wrong-typed fields
        # must not poison the counters and crash the always-on hot path.
        stats_path.write_text(
            '{"counters": null, "project_counters": "oops", '
            '"daily_snapshots": null, "search_score_sum": null}'
        )
        stats = DiagnosticStats(str(stats_path))
        stats.inc("add_memory")  # must not raise
        assert stats._counters.get("add_memory") == 1
        assert stats._project_counters == {}
        assert stats._daily_snapshots == []
        assert stats._search_score_sum == 0.0

    @pytest.fixture
    def stats(self, stats_path: Path) -> DiagnosticStats:
        return DiagnosticStats(stats_path)

    def test_inc_counter(self, stats: DiagnosticStats) -> None:
        stats.inc("test_counter")
        stats.inc("test_counter")
        stats.inc("test_counter", 3)
        snap = stats.snapshot()
        assert snap["usage"]["total_operations"] == 0

    def test_inc_project(self, stats: DiagnosticStats) -> None:
        stats.inc_project("myproj", "search", 5)
        snap = stats.snapshot()
        assert snap["projects"]["myproj"]["search"] == 5

    def test_record_latency(self, stats: DiagnosticStats) -> None:
        stats.record_latency("search", 50.0)
        stats.record_latency("search", 100.0)
        stats.record_latency("search", 200.0)
        snap = stats.snapshot()
        assert "search" in snap["performance"]
        assert snap["performance"]["search"]["count"] == 3
        assert snap["performance"]["search"]["min"] == 50.0

    def test_record_search_scores(self, stats: DiagnosticStats) -> None:
        stats.record_search_scores([0.1, 0.5, 0.9, 0.85])
        snap = stats.snapshot()
        dist = snap["accuracy"]["search"]["score_distribution"]
        assert dist["0.0-0.2"] == 1
        assert dist["0.4-0.6"] == 1
        assert dist["0.8-1.0"] == 2

    def test_record_error(self, stats: DiagnosticStats) -> None:
        stats.record_error("search", "connection timeout")
        snap = stats.snapshot()
        errors = snap["reliability"]["recent_errors"]
        assert len(errors) == 1
        assert errors[0]["operation"] == "search"
        assert "timeout" in errors[0]["error"]

    def test_record_activity_and_get(self, stats: DiagnosticStats) -> None:
        stats.record_activity("myproj")
        ts = stats.get_last_activity("myproj")
        assert ts is not None
        assert ts > 0

    def test_get_last_activity_missing_project(self, stats: DiagnosticStats) -> None:
        assert stats.get_last_activity("nonexistent") is None

    def test_flush_and_reload(self, stats_path: Path) -> None:
        stats = DiagnosticStats(stats_path)
        stats.inc("add_memory", 10)
        stats.inc("search", 5)
        stats.record_latency("embed", 42.0)
        stats.flush()

        assert stats_path.exists()

        reloaded = DiagnosticStats(stats_path)
        snap = reloaded.snapshot()
        assert snap["usage"]["total_add_memory_calls"] == 10
        assert snap["usage"]["total_searches"] == 5

    def test_reset_clears_all(self, stats: DiagnosticStats) -> None:
        stats.inc("add_memory", 100)
        stats.record_latency("search", 50.0)
        stats.record_error("embed", "fail")
        stats.reset()

        snap = stats.snapshot()
        assert snap["usage"]["total_operations"] == 0
        assert snap["reliability"]["total_errors"] == 0
        assert snap["performance"] == {}

    def test_daily_snapshot(self, stats: DiagnosticStats) -> None:
        stats.record_daily_snapshot({"proj-a": 10, "proj-b": 5})
        snap = stats.snapshot()
        assert len(snap["daily_snapshots"]) == 1
        assert snap["daily_snapshots"][0]["total"] == 15

    def test_daily_snapshot_replaces_same_day(self, stats: DiagnosticStats) -> None:
        stats.record_daily_snapshot({"proj-a": 10})
        stats.record_daily_snapshot({"proj-a": 15})
        snap = stats.snapshot()
        assert len(snap["daily_snapshots"]) == 1
        assert snap["daily_snapshots"][0]["total"] == 15

    def test_percentiles_with_single_sample(self, stats: DiagnosticStats) -> None:
        stats.record_latency("single", 42.5)
        snap = stats.snapshot()
        perf = snap["performance"]["single"]
        assert perf["p50"] == 42.5
        assert perf["min"] == 42.5
        assert perf["max"] == 42.5

    def test_project_specific_snapshot(self, stats: DiagnosticStats) -> None:
        stats.inc_project("alpha", "search", 3)
        stats.inc_project("alpha", "add_memory", 7)
        snap = stats.snapshot(project="alpha")
        assert snap["project"] == "alpha"
        assert snap["counters"]["search"] == 3
        assert snap["counters"]["add_memory"] == 7

    def test_dedup_effectiveness(self, stats: DiagnosticStats) -> None:
        stats.inc("dedup.add", 5)
        stats.inc("dedup.skip", 3)
        stats.inc("dedup.update", 2)
        snap = stats.snapshot()
        assert snap["accuracy"]["dedup"]["total_checked"] == 10
        assert snap["accuracy"]["dedup"]["effectiveness"] == "50.0%"

    def test_error_rate(self, stats: DiagnosticStats) -> None:
        stats.inc("add_memory", 10)
        stats.record_error("add", "fail1")
        stats.record_error("add", "fail2")
        snap = stats.snapshot()
        assert snap["reliability"]["error_rate"] == "20.0%"

    def test_flush_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested_path = tmp_path / "deep" / "nested" / "stats.json"
        stats = DiagnosticStats(nested_path)
        stats.inc("test", 1)
        stats.flush()
        assert nested_path.exists()


class TestLoadValueCoercion:
    def test_null_counter_values_are_dropped_not_crashing(self, tmp_path: Path) -> None:
        # Container-level checks let {"counters": {"search": null}} through;
        # the first inc() then crashed the always-on hot path.
        p = tmp_path / "s.json"
        p.write_text(
            '{"counters": {"search": null, "add_memory": "3", "ok": 2, "flag": true},'
            ' "latency_totals": {"embed": null}, "latency_counts": {"embed": "x"},'
            ' "project_counters": {"proj": {"add_memory": null, "n": 1}, "bad": 5},'
            ' "daily_snapshots": [{"date": "2026-01-01", "total": 1}, "junk", {"nodate": 1}]}'
        )
        stats = DiagnosticStats(str(p))
        stats.inc("search")  # must not raise
        assert stats._counters == {"ok": 2, "search": 1}
        assert stats._latency_totals == {}
        assert stats._latency_counts == {}
        assert stats._project_counters == {"proj": {"n": 1}}
        assert stats._daily_snapshots == [{"date": "2026-01-01", "total": 1}]


class TestResetAndForget:
    def test_reset_clears_daily_snapshots(self, tmp_path: Path) -> None:
        stats = DiagnosticStats(str(tmp_path / "s.json"))
        stats.record_daily_snapshot({"p": 3})
        assert stats._daily_snapshots
        stats.reset()
        assert stats._daily_snapshots == []

    def test_forget_project(self, tmp_path: Path) -> None:
        stats = DiagnosticStats(str(tmp_path / "s.json"))
        stats.inc_project("gone", "add_memory")
        stats.forget_project("gone")
        assert "gone" not in stats._project_counters
        stats.forget_project("never-existed")  # no-op, no raise


class TestFlushLoopSnapshot:
    @pytest.mark.asyncio
    async def test_flush_loop_records_snapshot_via_provider(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio

        import mem_zero.stats as st

        monkeypatch.setattr(st, "_FLUSH_INTERVAL", 0.01)
        monkeypatch.setattr(st, "_SNAPSHOT_INTERVAL", 0)
        stats = DiagnosticStats(str(tmp_path / "s.json"))

        async def provider() -> dict[str, int]:
            return {"p": 2}

        stats.set_project_count_provider(provider)
        await stats.start_flush_loop()
        await asyncio.sleep(0.05)
        await stats.shutdown()
        assert stats._daily_snapshots
        assert stats._daily_snapshots[-1]["projects"] == {"p": 2}

    @pytest.mark.asyncio
    async def test_provider_failure_does_not_kill_loop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio

        import mem_zero.stats as st

        monkeypatch.setattr(st, "_FLUSH_INTERVAL", 0.01)
        monkeypatch.setattr(st, "_SNAPSHOT_INTERVAL", 0)
        stats = DiagnosticStats(str(tmp_path / "s.json"))

        async def broken() -> dict[str, int]:
            raise RuntimeError("qdrant down")

        stats.set_project_count_provider(broken)
        await stats.start_flush_loop()
        await asyncio.sleep(0.05)
        assert not stats._flush_task.done()  # still running despite errors
        await stats.shutdown()
