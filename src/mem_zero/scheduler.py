from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .memory_engine import MemoryEngine
    from .webhooks import WebhookManager

logger = logging.getLogger(__name__)


@dataclass
class RetentionPolicy:
    max_memories_per_project: int | None = None
    max_age_days: float | None = None
    auto_consolidate_threshold: float = 0.80
    stale_after_days: float = 30.0


@dataclass
class MaintenanceResult:
    task: str
    project: str
    timestamp: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None


class MaintenanceScheduler:
    def __init__(
        self,
        engine: MemoryEngine,
        interval_hours: float = 24.0,
        policy: RetentionPolicy | None = None,
        webhooks: WebhookManager | None = None,
    ) -> None:
        self._engine = engine
        self._interval = interval_hours * 3600
        self._policy = policy or RetentionPolicy()
        self._webhooks = webhooks
        self._task: asyncio.Task[None] | None = None
        self._results: list[MaintenanceResult] = []
        self._max_results = 500
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def policy(self) -> RetentionPolicy:
        return self._policy

    def update_policy(self, **kwargs: Any) -> RetentionPolicy:
        for key, value in kwargs.items():
            if hasattr(self._policy, key):
                object.__setattr__(self._policy, key, value)
        return self._policy

    def recent_results(self, limit: int = 50) -> list[dict[str, Any]]:
        return [
            {
                "task": r.task,
                "project": r.project,
                "timestamp": r.timestamp,
                "details": r.details,
                "success": r.success,
                "error": r.error,
            }
            for r in self._results[-limit:]
        ]

    def _record(self, result: MaintenanceResult) -> None:
        self._results.append(result)
        if len(self._results) > self._max_results:
            self._results = self._results[-self._max_results:]

    async def detect_stale(self, project_slug: str) -> list[dict[str, Any]]:
        if self._policy.stale_after_days <= 0:
            return []

        cutoff = time.time() - (self._policy.stale_after_days * 86400)
        memories = await self._engine.list_all(project_slug, limit=1000)
        stale = []
        for mem in memories:
            updated = mem.updated_at or mem.created_at
            if updated < cutoff:
                stale.append({
                    "id": mem.id,
                    "text": mem.text[:100],
                    "last_updated": updated,
                    "age_days": round((time.time() - updated) / 86400, 1),
                })
        return stale

    async def enforce_retention(self, project_slug: str) -> MaintenanceResult:
        result = MaintenanceResult(task="retention", project=project_slug)
        deleted_count = 0

        try:
            if self._policy.max_age_days:
                cutoff = time.time() - (self._policy.max_age_days * 86400)
                memories = await self._engine.list_all(project_slug, limit=1000)
                for mem in memories:
                    updated = mem.updated_at or mem.created_at
                    if updated < cutoff:
                        await self._engine.delete(project_slug, mem.id)
                        deleted_count += 1

            if self._policy.max_memories_per_project:
                memories = await self._engine.list_all(
                    project_slug, limit=self._policy.max_memories_per_project + 100
                )
                if len(memories) > self._policy.max_memories_per_project:
                    sorted_mems = sorted(memories, key=lambda m: m.updated_at or m.created_at)
                    excess = len(memories) - self._policy.max_memories_per_project
                    for mem in sorted_mems[:excess]:
                        await self._engine.delete(project_slug, mem.id)
                        deleted_count += 1

            result.details = {"deleted": deleted_count}
        except Exception as exc:
            result.success = False
            result.error = str(exc)[:200]

        self._record(result)
        return result

    async def auto_consolidate(self, project_slug: str) -> MaintenanceResult:
        result = MaintenanceResult(task="consolidate", project=project_slug)
        try:
            outcome = await self._engine.consolidate(
                project_slug,
                similarity_threshold=self._policy.auto_consolidate_threshold,
            )
            result.details = outcome
            if self._webhooks and outcome.get("clusters", 0) > 0:
                await self._webhooks.fire("consolidation_complete", {
                    "project": project_slug,
                    **outcome,
                })
        except Exception as exc:
            result.success = False
            result.error = str(exc)[:200]

        self._record(result)
        return result

    async def run_maintenance(self) -> list[MaintenanceResult]:
        results: list[MaintenanceResult] = []
        try:
            projects = await self._engine.list_projects()
        except Exception as exc:
            logger.error("Failed to list projects for maintenance: %s", exc)
            return results

        for project in projects:
            slug = project.slug

            retention_result = await self.enforce_retention(slug)
            results.append(retention_result)

            consolidate_result = await self.auto_consolidate(slug)
            results.append(consolidate_result)

        logger.info(
            "Maintenance complete: %d tasks across %d projects",
            len(results), len(projects),
        )
        return results

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Maintenance scheduler started (interval=%.1fh)", self._interval / 3600)

    async def _loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._interval)
            try:
                await self.run_maintenance()
            except Exception:
                logger.exception("Maintenance cycle failed")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
