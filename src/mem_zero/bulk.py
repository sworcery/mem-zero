from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .memory_engine import MemoryEngine

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BulkJob:
    id: str
    operation: str
    project: str
    status: JobStatus = JobStatus.PENDING
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    result: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def progress_pct(self) -> float:
        if self.total_items == 0:
            return 0.0
        return round(self.processed_items / self.total_items * 100, 1)

    @property
    def elapsed_seconds(self) -> float | None:
        if not self.started_at:
            return None
        end = self.completed_at or time.time()
        return round(end - self.started_at, 2)


class BulkOperations:
    def __init__(self, engine: MemoryEngine, max_concurrent: int = 5) -> None:
        self._engine = engine
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._jobs: dict[str, BulkJob] = {}
        self._max_jobs = 100

    def get_job(self, job_id: str) -> BulkJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        jobs = sorted(
            self._jobs.values(),
            key=lambda j: j.created_at,
            reverse=True,
        )[:limit]
        return [
            {
                "id": j.id,
                "operation": j.operation,
                "project": j.project,
                "status": j.status.value,
                "progress_pct": j.progress_pct,
                "total_items": j.total_items,
                "processed_items": j.processed_items,
                "failed_items": j.failed_items,
                "elapsed_seconds": j.elapsed_seconds,
                "created_at": j.created_at,
            }
            for j in jobs
        ]

    def _create_job(self, operation: str, project: str, total: int) -> BulkJob:
        job = BulkJob(
            id=str(uuid.uuid4()),
            operation=operation,
            project=project,
            total_items=total,
        )
        self._jobs[job.id] = job
        if len(self._jobs) > self._max_jobs:
            oldest = min(self._jobs.values(), key=lambda j: j.created_at)
            if oldest.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                del self._jobs[oldest.id]
        return job

    async def bulk_add(
        self,
        project_slug: str,
        texts: list[str],
        user_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> BulkJob:
        job = self._create_job("bulk_add", project_slug, len(texts))
        asyncio.create_task(self._run_bulk_add(job, project_slug, texts, user_id, metadata))
        return job

    async def _run_bulk_add(
        self,
        job: BulkJob,
        project_slug: str,
        texts: list[str],
        user_id: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        all_ids: list[str] = []

        for text in texts:
            async with self._semaphore:
                try:
                    ids = await self._engine.add(
                        project_slug, user_id, [text], metadata
                    )
                    all_ids.extend(ids)
                    job.processed_items += 1
                except Exception as exc:
                    job.failed_items += 1
                    job.errors.append(f"{text[:50]}: {exc}")
                    job.processed_items += 1

        job.status = JobStatus.COMPLETED
        job.completed_at = time.time()
        job.result = {"stored_ids": all_ids, "total_stored": len(all_ids)}

    async def bulk_delete(
        self,
        project_slug: str,
        memory_ids: list[str],
    ) -> BulkJob:
        job = self._create_job("bulk_delete", project_slug, len(memory_ids))
        asyncio.create_task(self._run_bulk_delete(job, project_slug, memory_ids))
        return job

    async def _run_bulk_delete(
        self,
        job: BulkJob,
        project_slug: str,
        memory_ids: list[str],
    ) -> None:
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        deleted = 0

        for mid in memory_ids:
            async with self._semaphore:
                try:
                    await self._engine.delete(project_slug, mid)
                    deleted += 1
                    job.processed_items += 1
                except Exception as exc:
                    job.failed_items += 1
                    job.errors.append(f"{mid}: {exc}")
                    job.processed_items += 1

        job.status = JobStatus.COMPLETED
        job.completed_at = time.time()
        job.result = {"deleted": deleted}

    async def bulk_search(
        self,
        project_slug: str,
        queries: list[str],
        top_k: int = 10,
    ) -> dict[str, Any]:
        results: dict[str, list[dict[str, Any]]] = {}

        async def search_one(query: str) -> None:
            async with self._semaphore:
                hits = await self._engine.search(project_slug, query, top_k=top_k)
                results[query] = [
                    {
                        "id": h.id,
                        "text": h.text,
                        "score": h.score,
                    }
                    for h in hits
                ]

        await asyncio.gather(
            *(search_one(q) for q in queries),
            return_exceptions=True,
        )

        return {
            "queries": len(queries),
            "results": results,
            "total_hits": sum(len(v) for v in results.values()),
        }

    def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return False
        job.status = JobStatus.CANCELLED
        job.completed_at = time.time()
        return True

    def cleanup_finished(self, max_age_hours: float = 24.0) -> int:
        cutoff = time.time() - (max_age_hours * 3600)
        to_remove = [
            jid for jid, job in self._jobs.items()
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
            and job.created_at < cutoff
        ]
        for jid in to_remove:
            del self._jobs[jid]
        return len(to_remove)
