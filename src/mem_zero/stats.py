from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_FLUSH_INTERVAL = 60
# The flush loop takes a project-count snapshot at most this often (seconds);
# record_daily_snapshot itself dedups to one row per calendar day.
_SNAPSHOT_INTERVAL = 900
_MAX_LATENCY_SAMPLES = 200
_MAX_RECENT_ERRORS = 50


class _NullStats:
    def inc(self, key: str, amount: int = 1) -> None:
        pass

    def inc_project(self, project: str, key: str, amount: int = 1) -> None:
        pass

    def record_latency(self, key: str, ms: float) -> None:
        pass

    def record_activity(self, project: str) -> None:
        pass

    def get_last_activity(self, project: str) -> float | None:
        return None

    def record_search_scores(self, scores: list[float]) -> None:
        pass

    def record_error(self, operation: str, error: str) -> None:
        pass

    def snapshot(self, project: str | None = None) -> dict[str, Any]:
        return {}

    def flush(self) -> None:
        pass

    async def start_flush_loop(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def forget_project(self, project: str) -> None:
        pass

    def set_project_count_provider(
        self, provider: Callable[[], Awaitable[dict[str, int]]]
    ) -> None:
        pass


NULL_STATS = _NullStats()


class DiagnosticStats:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._started_at = time.time()
        self._flush_task: asyncio.Task[None] | None = None

        self._counters: dict[str, int] = {}
        self._latencies: dict[str, deque[float]] = {}
        self._latency_totals: dict[str, float] = {}
        self._latency_counts: dict[str, int] = {}

        self._search_score_sum: float = 0.0
        self._search_score_count: int = 0
        self._search_score_buckets: dict[str, int] = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0,
        }

        self._recent_errors: deque[dict[str, Any]] = deque(maxlen=_MAX_RECENT_ERRORS)
        self._project_counters: dict[str, dict[str, int]] = {}
        self._daily_snapshots: list[dict[str, Any]] = []
        self._project_count_provider: Callable[[], Awaitable[dict[str, int]]] | None = None
        self._last_snapshot_ts: float = 0.0

        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            if not isinstance(data, dict):
                raise ValueError("diagnostics file is not a JSON object")

            # Coerce every field to its expected type: a hand-edited or
            # version-drifted file with a null/wrong-typed value must not
            # poison the counters and crash the always-on hot path later.
            def _as_dict(key: str) -> dict[str, Any]:
                value = data.get(key)
                return value if isinstance(value, dict) else {}

            def _as_list(key: str) -> list[Any]:
                value = data.get(key)
                return value if isinstance(value, list) else []

            def _num(v: object) -> bool:
                # bool is an int subclass; a stray true/false is not a count.
                return isinstance(v, (int, float)) and not isinstance(v, bool)

            def _as_num_dict(key: str) -> dict[str, float]:
                # Container-level checks were not enough: {"counters":
                # {"search": null}} survived load and crashed inc() on the
                # always-on hot path. Keep only str -> number entries.
                return {
                    k: v for k, v in _as_dict(key).items()
                    if isinstance(k, str) and _num(v)
                }

            started = data.get("started_at")
            if isinstance(started, (int, float)):
                self._started_at = started
            self._counters = {k: int(v) for k, v in _as_num_dict("counters").items()}
            self._latency_totals = _as_num_dict("latency_totals")
            self._latency_counts = {
                k: int(v) for k, v in _as_num_dict("latency_counts").items()
            }
            for key, samples in _as_dict("latency_samples").items():
                if isinstance(key, str) and isinstance(samples, list):
                    self._latencies[key] = deque(
                        [x for x in samples if _num(x)], maxlen=_MAX_LATENCY_SAMPLES
                    )
            score_sum = data.get("search_score_sum")
            self._search_score_sum = (
                float(score_sum) if isinstance(score_sum, (int, float)) else 0.0
            )
            score_count = data.get("search_score_count")
            self._search_score_count = (
                int(score_count) if isinstance(score_count, int) else 0
            )
            self._search_score_buckets = {
                **self._search_score_buckets,
                **{k: int(v) for k, v in _as_num_dict("search_score_buckets").items()},
            }
            for err in _as_list("recent_errors"):
                if isinstance(err, dict):
                    self._recent_errors.append(err)
            self._project_counters = {
                slug: {k: v for k, v in pc.items() if isinstance(k, str) and _num(v)}
                for slug, pc in _as_dict("project_counters").items()
                if isinstance(slug, str) and isinstance(pc, dict)
            }
            self._daily_snapshots = [
                snap for snap in _as_list("daily_snapshots")
                if isinstance(snap, dict) and isinstance(snap.get("date"), str)
            ]
            logger.info("Loaded diagnostics from %s", self._path)
        except Exception:
            logger.warning(
                "Failed to load diagnostics from %s, starting fresh", self._path
            )

    def _prune_stale_projects(self) -> None:
        # Slugs accumulate forever otherwise (deleted/renamed projects, probes).
        # Anything inactive for 180 days stops being interesting diagnostics.
        cutoff = time.time() - 180 * 86400
        stale = [
            slug
            for slug, pc in self._project_counters.items()
            if isinstance(pc.get("last_activity_ts"), (int, float))
            and pc["last_activity_ts"] < cutoff
        ]
        for slug in stale:
            del self._project_counters[slug]

    def flush(self) -> None:
        self._prune_stale_projects()
        data = {
            "started_at": self._started_at,
            "last_flush": time.time(),
            "counters": self._counters,
            "latency_totals": self._latency_totals,
            "latency_counts": self._latency_counts,
            "latency_samples": {k: list(v) for k, v in self._latencies.items()},
            "search_score_sum": self._search_score_sum,
            "search_score_count": self._search_score_count,
            "search_score_buckets": self._search_score_buckets,
            "recent_errors": list(self._recent_errors),
            "project_counters": self._project_counters,
            "daily_snapshots": self._daily_snapshots,
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(self._path)
        except Exception:
            logger.warning("Failed to flush diagnostics to %s", self._path)

    def inc(self, key: str, amount: int = 1) -> None:
        self._counters[key] = self._counters.get(key, 0) + amount

    def inc_project(self, project: str, key: str, amount: int = 1) -> None:
        if project not in self._project_counters:
            self._project_counters[project] = {}
        pc = self._project_counters[project]
        pc[key] = pc.get(key, 0) + amount

    def record_activity(self, project: str) -> None:
        if project not in self._project_counters:
            self._project_counters[project] = {}
        self._project_counters[project]["last_activity_ts"] = time.time()

    def get_last_activity(self, project: str) -> float | None:
        pc = self._project_counters.get(project, {})
        return pc.get("last_activity_ts")

    def record_daily_snapshot(self, projects: dict[str, int]) -> None:
        today = time.strftime("%Y-%m-%d")
        total = sum(projects.values())
        entry = {"date": today, "total": total, "projects": projects}
        if self._daily_snapshots and self._daily_snapshots[-1].get("date") == today:
            self._daily_snapshots[-1] = entry
        else:
            self._daily_snapshots.append(entry)
        self._daily_snapshots = self._daily_snapshots[-90:]

    def record_latency(self, key: str, ms: float) -> None:
        if key not in self._latencies:
            self._latencies[key] = deque(maxlen=_MAX_LATENCY_SAMPLES)
        self._latencies[key].append(ms)
        self._latency_totals[key] = self._latency_totals.get(key, 0.0) + ms
        self._latency_counts[key] = self._latency_counts.get(key, 0) + 1

    def record_search_scores(self, scores: list[float]) -> None:
        for s in scores:
            self._search_score_sum += s
            self._search_score_count += 1
            if s < 0.2:
                bucket = "0.0-0.2"
            elif s < 0.4:
                bucket = "0.2-0.4"
            elif s < 0.6:
                bucket = "0.4-0.6"
            elif s < 0.8:
                bucket = "0.6-0.8"
            else:
                bucket = "0.8-1.0"
            self._search_score_buckets[bucket] = (
                self._search_score_buckets.get(bucket, 0) + 1
            )

    def record_error(self, operation: str, error: str) -> None:
        self._recent_errors.append(
            {
                "timestamp": time.time(),
                "operation": operation,
                "error": error[:500],
            }
        )
        self.inc(f"errors.{operation}")

    def _percentiles(self, key: str) -> dict[str, float | None]:
        samples = self._latencies.get(key)
        if not samples:
            return {"p50": None, "p95": None, "p99": None, "min": None, "max": None}
        s = sorted(samples)
        n = len(s)
        return {
            "p50": round(s[int(n * 0.5)], 1),
            "p95": round(s[min(int(n * 0.95), n - 1)], 1),
            "p99": round(s[min(int(n * 0.99), n - 1)], 1),
            "min": round(s[0], 1),
            "max": round(s[-1], 1),
        }

    def snapshot(self, project: str | None = None) -> dict[str, Any]:
        now = time.time()
        uptime = now - self._started_at

        if project:
            return {
                "project": project,
                "counters": dict(self._project_counters.get(project, {})),
            }

        total_ops = (
            self._counters.get("add_memory", 0)
            + self._counters.get("search", 0)
            + self._counters.get("delete", 0)
            + self._counters.get("delete_all", 0)
            + self._counters.get("reembed", 0)
            + self._counters.get("cleanup", 0)
            + self._counters.get("consolidate", 0)
        )
        total_errors = sum(
            v for k, v in self._counters.items() if k.startswith("errors.")
        )
        days_up = max(uptime / 86400, 0.01)

        dedup_add = self._counters.get("dedup.add", 0)
        dedup_skip = self._counters.get("dedup.skip", 0)
        dedup_update = self._counters.get("dedup.update", 0)
        dedup_total = dedup_add + dedup_skip + dedup_update
        dedup_effective = dedup_skip + dedup_update

        extract_total = self._counters.get("extract_facts", 0)
        # Empty extractions are the model correctly declining filler — a
        # success. Failures are parse errors and LLM-call errors only.
        extract_failures = (
            self._counters.get("extract_facts.json_failures", 0)
            + self._counters.get("extract_facts.llm_failures", 0)
        )
        facts_produced = self._counters.get("extract_facts.produced", 0)

        avg_score = (
            round(self._search_score_sum / self._search_score_count, 4)
            if self._search_score_count
            else None
        )

        latency_report: dict[str, Any] = {}
        for key in sorted(set(self._latency_counts) | set(self._latencies)):
            count = self._latency_counts.get(key, 0)
            total = self._latency_totals.get(key, 0.0)
            latency_report[key] = {
                "count": count,
                "avg_ms": round(total / count, 1) if count else None,
                **self._percentiles(key),
            }

        return {
            "uptime_seconds": round(uptime, 1),
            "started_at": self._started_at,
            "usage": {
                "total_operations": total_ops,
                "operations_per_day": round(total_ops / days_up, 1),
                "total_add_memory_calls": self._counters.get("add_memory", 0),
                "add_operations": self._counters.get("add_memory", 0),
                "total_facts_stored": self._counters.get("facts_stored", 0),
                "total_searches": self._counters.get("search", 0),
                "search_operations": self._counters.get("search", 0),
                "dedup_hits": dedup_effective,
                "total_embeddings": self._counters.get("embed", 0),
                "total_deletes": (
                    self._counters.get("delete", 0)
                    + self._counters.get("delete_all", 0)
                ),
                "total_reembeds": self._counters.get("reembed", 0),
                "total_cleanups": self._counters.get("cleanup", 0),
                "total_consolidations": self._counters.get("consolidate", 0),
            },
            "accuracy": {
                "dedup": {
                    "total_checked": dedup_total,
                    "added": dedup_add,
                    "skipped": dedup_skip,
                    "updated": dedup_update,
                    "degraded_skips": self._counters.get(
                        "dedup.degraded_skip", 0
                    ),
                    "json_parse_failures": self._counters.get(
                        "dedup.json_failures", 0
                    ),
                    "effectiveness": (
                        f"{dedup_effective / dedup_total * 100:.1f}%"
                        if dedup_total
                        else "N/A"
                    ),
                },
                "extraction": {
                    "total_calls": extract_total,
                    "json_parse_failures": self._counters.get(
                        "extract_facts.json_failures", 0
                    ),
                    "no_facts_inputs": self._counters.get(
                        "extract_facts.no_facts", 0
                    ),
                    "rejected_facts": self._counters.get(
                        "extract_facts.rejected", 0
                    ),
                    "llm_failures": self._counters.get(
                        "extract_facts.llm_failures", 0
                    ),
                    "failure_rate": (
                        f"{extract_failures / extract_total * 100:.1f}%"
                        if extract_total
                        else "N/A"
                    ),
                    "total_facts_produced": facts_produced,
                    "avg_facts_per_input": (
                        round(facts_produced / extract_total, 1)
                        if extract_total
                        else None
                    ),
                },
                "search": {
                    "total_queries": self._counters.get("search", 0),
                    "zero_result_queries": self._counters.get(
                        "search.zero_results", 0
                    ),
                    "avg_score": avg_score,
                    "score_distribution": dict(self._search_score_buckets),
                },
            },
            "performance": latency_report,
            "reliability": {
                "total_errors": total_errors,
                "error_rate": (
                    f"{min(total_errors / total_ops * 100, 100.0):.1f}%"
                    if total_ops
                    else "N/A"
                ),
                "backend_fallback_activations": self._counters.get(
                    "backend.fallback", 0
                ),
                "backend_primary_recoveries": self._counters.get(
                    "backend.recovery", 0
                ),
                "recent_errors": list(self._recent_errors)[-10:],
            },
            "projects": {
                slug: dict(counters)
                for slug, counters in self._project_counters.items()
            },
            "daily_snapshots": self._daily_snapshots[-30:],
        }

    def reset(self) -> None:
        self._started_at = time.time()
        self._counters.clear()
        self._latencies.clear()
        self._latency_totals.clear()
        self._latency_counts.clear()
        self._search_score_sum = 0.0
        self._search_score_count = 0
        self._search_score_buckets = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0,
        }
        self._recent_errors.clear()
        self._project_counters.clear()
        # A reset that left 90 days of history behind was not a reset.
        self._daily_snapshots.clear()
        self.flush()

    def forget_project(self, project: str) -> None:
        self._project_counters.pop(project, None)

    def set_project_count_provider(
        self, provider: Callable[[], Awaitable[dict[str, int]]]
    ) -> None:
        # Lets the flush loop take daily snapshots on its own. Previously a
        # snapshot was only recorded as a side effect of someone opening the
        # dashboard, so days without a visit simply had no data point.
        self._project_count_provider = provider

    async def start_flush_loop(self) -> None:
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def _maybe_snapshot(self) -> None:
        if self._project_count_provider is None:
            return
        now = time.time()
        if now - self._last_snapshot_ts < _SNAPSHOT_INTERVAL:
            return
        self._last_snapshot_ts = now
        try:
            self.record_daily_snapshot(await self._project_count_provider())
        except Exception:
            logger.debug("Daily snapshot skipped", exc_info=True)

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(_FLUSH_INTERVAL)
            await self._maybe_snapshot()
            self.flush()

    async def shutdown(self) -> None:
        if self._flush_task:
            self._flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._flush_task
        self.flush()
