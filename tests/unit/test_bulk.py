from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from mem_zero.bulk import BulkOperations, JobStatus
from mem_zero.models import MemoryRecord


@pytest.fixture
def mock_engine() -> AsyncMock:
    engine = AsyncMock()
    engine.add.return_value = ["id-1"]
    engine.delete.return_value = True
    engine.search.return_value = []
    return engine


@pytest.fixture
def bulk(mock_engine: AsyncMock) -> BulkOperations:
    return BulkOperations(mock_engine, max_concurrent=2)


def _mem(id: str, text: str, score: float | None = None) -> MemoryRecord:
    return MemoryRecord(
        id=id, text=text, user_id="u",
        created_at=1700000000, updated_at=1700000000,
        score=score,
    )


class TestBulkAdd:
    @pytest.mark.asyncio
    async def test_creates_job(
        self, bulk: BulkOperations, mock_engine: AsyncMock
    ) -> None:
        job = await bulk.bulk_add("proj", ["text1", "text2", "text3"])
        assert job.operation == "bulk_add"
        assert job.total_items == 3
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_job_completes(
        self, bulk: BulkOperations, mock_engine: AsyncMock
    ) -> None:
        job = await bulk.bulk_add("proj", ["text1"])
        await asyncio.sleep(0.2)
        updated = bulk.get_job(job.id)
        assert updated is not None
        assert updated.status == JobStatus.COMPLETED
        assert updated.processed_items == 1

    @pytest.mark.asyncio
    async def test_job_tracks_failures(
        self, bulk: BulkOperations, mock_engine: AsyncMock
    ) -> None:
        mock_engine.add.side_effect = [["id-1"], Exception("fail")]
        job = await bulk.bulk_add("proj", ["good", "bad"])
        await asyncio.sleep(0.2)
        updated = bulk.get_job(job.id)
        assert updated is not None
        assert updated.processed_items == 2
        assert updated.failed_items == 1


class TestBulkDelete:
    @pytest.mark.asyncio
    async def test_deletes_all(
        self, bulk: BulkOperations, mock_engine: AsyncMock
    ) -> None:
        job = await bulk.bulk_delete(
            "proj", ["550e8400-e29b-41d4-a716-446655440000"]
        )
        await asyncio.sleep(0.2)
        updated = bulk.get_job(job.id)
        assert updated is not None
        assert updated.status == JobStatus.COMPLETED
        assert updated.result["deleted"] == 1


class TestBulkSearch:
    @pytest.mark.asyncio
    async def test_searches_multiple_queries(
        self, bulk: BulkOperations, mock_engine: AsyncMock
    ) -> None:
        mock_engine.search.return_value = [_mem("a", "result", score=0.9)]
        result = await bulk.bulk_search("proj", ["query1", "query2"])
        assert result["queries"] == 2
        assert result["total_hits"] == 2
        assert "query1" in result["results"]
        assert "query2" in result["results"]

    @pytest.mark.asyncio
    async def test_empty_search(
        self, bulk: BulkOperations, mock_engine: AsyncMock
    ) -> None:
        result = await bulk.bulk_search("proj", [])
        assert result["queries"] == 0
        assert result["total_hits"] == 0


class TestJobManagement:
    def test_list_jobs(self, bulk: BulkOperations) -> None:
        assert bulk.list_jobs() == []

    @pytest.mark.asyncio
    async def test_list_shows_jobs(
        self, bulk: BulkOperations, mock_engine: AsyncMock
    ) -> None:
        await bulk.bulk_add("proj", ["text"])
        await asyncio.sleep(0.1)
        jobs = bulk.list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["operation"] == "bulk_add"

    @pytest.mark.asyncio
    async def test_cancel_job(
        self, bulk: BulkOperations, mock_engine: AsyncMock
    ) -> None:
        job = await bulk.bulk_add("proj", ["text"])
        result = bulk.cancel_job(job.id)
        assert result is True

    def test_cancel_nonexistent(self, bulk: BulkOperations) -> None:
        assert bulk.cancel_job("fake-id") is False

    def test_cleanup_finished(self, bulk: BulkOperations) -> None:
        removed = bulk.cleanup_finished()
        assert removed == 0


class TestJobProgress:
    @pytest.mark.asyncio
    async def test_progress_tracking(
        self, bulk: BulkOperations, mock_engine: AsyncMock
    ) -> None:
        job = await bulk.bulk_add("proj", ["a", "b", "c", "d"])
        assert job.progress_pct == 0.0
        await asyncio.sleep(0.3)
        updated = bulk.get_job(job.id)
        assert updated is not None
        assert updated.progress_pct == 100.0

    @pytest.mark.asyncio
    async def test_elapsed_time(
        self, bulk: BulkOperations, mock_engine: AsyncMock
    ) -> None:
        job = await bulk.bulk_add("proj", ["text"])
        await asyncio.sleep(0.2)
        updated = bulk.get_job(job.id)
        assert updated is not None
        assert updated.elapsed_seconds is not None
        assert updated.elapsed_seconds >= 0
