from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from mem_zero.models import MemoryRecord, ProjectInfo
from mem_zero.scheduler import MaintenanceScheduler, RetentionPolicy


@pytest.fixture
def mock_engine() -> AsyncMock:
    engine = AsyncMock()
    engine.list_projects.return_value = [
        ProjectInfo(slug="test-proj", collection="mem-zero_test-proj", memory_count=10),
    ]
    engine.list_all.return_value = []
    engine.delete.return_value = True
    engine.consolidate.return_value = {"clusters": 0, "memories_removed": 0, "memories_created": 0}
    return engine


@pytest.fixture
def scheduler(mock_engine: AsyncMock) -> MaintenanceScheduler:
    return MaintenanceScheduler(mock_engine, interval_hours=24.0)


class TestRetentionPolicy:
    def test_defaults(self) -> None:
        policy = RetentionPolicy()
        assert policy.max_memories_per_project is None
        assert policy.max_age_days is None
        assert policy.auto_consolidate_threshold == 0.80
        assert policy.stale_after_days == 30.0

    def test_custom_policy(self) -> None:
        policy = RetentionPolicy(
            max_memories_per_project=500,
            max_age_days=90.0,
        )
        assert policy.max_memories_per_project == 500
        assert policy.max_age_days == 90.0


class TestDetectStale:
    @pytest.mark.asyncio
    async def test_no_stale_memories(
        self, scheduler: MaintenanceScheduler, mock_engine: AsyncMock
    ) -> None:
        mock_engine.list_all.return_value = [
            MemoryRecord(
                id="abc", text="recent", user_id="u",
                created_at=time.time(), updated_at=time.time(),
            ),
        ]
        stale = await scheduler.detect_stale("test-proj")
        assert stale == []

    @pytest.mark.asyncio
    async def test_finds_stale_memories(
        self, scheduler: MaintenanceScheduler, mock_engine: AsyncMock
    ) -> None:
        old_ts = time.time() - (60 * 86400)
        mock_engine.list_all.return_value = [
            MemoryRecord(
                id="old1", text="ancient memory", user_id="u",
                created_at=old_ts, updated_at=old_ts,
            ),
            MemoryRecord(
                id="new1", text="fresh memory", user_id="u",
                created_at=time.time(), updated_at=time.time(),
            ),
        ]
        stale = await scheduler.detect_stale("test-proj")
        assert len(stale) == 1
        assert stale[0]["id"] == "old1"
        assert stale[0]["age_days"] > 30


class TestEnforceRetention:
    @pytest.mark.asyncio
    async def test_deletes_old_memories(self, mock_engine: AsyncMock) -> None:
        policy = RetentionPolicy(max_age_days=30.0)
        sched = MaintenanceScheduler(mock_engine, policy=policy)

        old_ts = time.time() - (60 * 86400)
        mock_engine.list_all.return_value = [
            MemoryRecord(
                id="550e8400-e29b-41d4-a716-446655440000",
                text="old", user_id="u",
                created_at=old_ts, updated_at=old_ts,
            ),
        ]
        result = await sched.enforce_retention("test-proj")
        assert result.success is True
        assert result.details["deleted"] == 1
        mock_engine.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_enforces_max_count(self, mock_engine: AsyncMock) -> None:
        policy = RetentionPolicy(max_memories_per_project=2)
        sched = MaintenanceScheduler(mock_engine, policy=policy)

        now = time.time()
        mock_engine.list_all.return_value = [
            MemoryRecord(
                id=f"550e8400-e29b-41d4-a716-44665544000{i}",
                text=f"mem {i}", user_id="u",
                created_at=now - (i * 1000), updated_at=now - (i * 1000),
            )
            for i in range(4)
        ]
        result = await sched.enforce_retention("test-proj")
        assert result.success is True
        assert result.details["deleted"] == 2

    @pytest.mark.asyncio
    async def test_no_deletions_when_under_limit(
        self, scheduler: MaintenanceScheduler, mock_engine: AsyncMock
    ) -> None:
        result = await scheduler.enforce_retention("test-proj")
        assert result.details["deleted"] == 0


class TestAutoConsolidate:
    @pytest.mark.asyncio
    async def test_runs_consolidation(
        self, scheduler: MaintenanceScheduler, mock_engine: AsyncMock
    ) -> None:
        mock_engine.consolidate.return_value = {
            "clusters": 2, "memories_removed": 4, "memories_created": 2,
        }
        result = await scheduler.auto_consolidate("test-proj")
        assert result.success is True
        assert result.details["clusters"] == 2

    @pytest.mark.asyncio
    async def test_handles_consolidation_error(
        self, scheduler: MaintenanceScheduler, mock_engine: AsyncMock
    ) -> None:
        mock_engine.consolidate.side_effect = Exception("LLM failed")
        result = await scheduler.auto_consolidate("test-proj")
        assert result.success is False
        assert "LLM failed" in (result.error or "")


class TestRunMaintenance:
    @pytest.mark.asyncio
    async def test_runs_all_tasks(
        self, scheduler: MaintenanceScheduler, mock_engine: AsyncMock
    ) -> None:
        results = await scheduler.run_maintenance()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_results_recorded(
        self, scheduler: MaintenanceScheduler, mock_engine: AsyncMock
    ) -> None:
        await scheduler.run_maintenance()
        recent = scheduler.recent_results()
        assert len(recent) == 2
        tasks = {r["task"] for r in recent}
        assert "retention" in tasks
        assert "consolidate" in tasks


class TestUpdatePolicy:
    def test_update_values(self, scheduler: MaintenanceScheduler) -> None:
        scheduler.update_policy(
            max_memories_per_project=100,
            stale_after_days=7.0,
        )
        assert scheduler.policy.max_memories_per_project == 100
        assert scheduler.policy.stale_after_days == 7.0


class TestLifecycle:
    def test_not_running_by_default(self, scheduler: MaintenanceScheduler) -> None:
        assert scheduler.is_running is False
