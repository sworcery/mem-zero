from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mem_zero.backends import BackendUnavailable
from mem_zero.config import Config
from mem_zero.memory_engine import MemoryEngine, QueuedIds
from mem_zero.pending import MAX_ATTEMPTS, PendingStore


class TestPendingStore:
    def test_enqueue_roundtrip_and_fifo(self, tmp_path: Path) -> None:
        store = PendingStore(str(tmp_path / "q"))
        for i in range(5):
            store.enqueue("proj", "john", f"note {i}", {"k": i})
        items = store.list()
        assert [i.text for i in items] == [f"note {i}" for i in range(5)]
        assert items[0].project == "proj"
        assert items[0].metadata == {"k": 0}
        assert items[0].attempts == 0
        assert store.count() == 5

    def test_atomic_write_leaves_no_tmp(self, tmp_path: Path) -> None:
        store = PendingStore(str(tmp_path))
        path = store.enqueue("proj", "john", "x", None)
        assert [p.name for p in tmp_path.iterdir()] == [path.name]
        assert json.loads(path.read_text())["text"] == "x"

    def test_count_missing_dir(self, tmp_path: Path) -> None:
        assert PendingStore(str(tmp_path / "nope")).count() == 0

    def test_dead_letter_after_max_attempts(self, tmp_path: Path) -> None:
        store = PendingStore(str(tmp_path))
        store.enqueue("proj", "john", "x", None)
        item = store.list()[0]
        for n in range(MAX_ATTEMPTS - 1):
            assert store.record_failure(item, "boom") is False
            assert store.list()[0].attempts == n + 1
        assert store.record_failure(item, "boom") is True
        assert store.count() == 0
        assert len(list((tmp_path / "failed").glob("*.json"))) == 1

    def test_corrupt_file_moved_to_failed(self, tmp_path: Path) -> None:
        (tmp_path / "1-a.json").write_text("{not json")
        store = PendingStore(str(tmp_path))
        assert store.list() == []
        assert (tmp_path / "failed" / "1-a.json").exists()


@pytest.fixture
def engine(tmp_path: Path) -> MemoryEngine:
    config = Config(
        qdrant_host="localhost",
        qdrant_port=6333,
        ollama_base_url="http://localhost:11434",
        embedder_model="nomic-embed-text",
        embedding_dimensions=768,
        collection_prefix="test",
        pending_dir=str(tmp_path / "pending"),
    )
    backend = AsyncMock()
    backend.embedding_dimensions = 768
    backend.is_degraded = False
    eng = MemoryEngine(config, backend)
    qdrant = AsyncMock()
    qdrant.get_collections.return_value = MagicMock(collections=[])
    qdrant.get_collection.return_value = MagicMock(points_count=0)
    eng._qdrant = qdrant
    return eng


class TestEnqueueOnUnavailable:
    async def test_extract_outage_is_queued(self, engine: MemoryEngine) -> None:
        engine._backend.generate.side_effect = BackendUnavailable("down")
        result = await engine.add("proj", "john", ["remember this"], {"a": 1})
        assert isinstance(result, QueuedIds)
        assert result == []
        item = engine._pending.list()[0]
        assert (item.project, item.user_id, item.text) == ("proj", "john", "remember this")
        assert item.metadata == {"a": 1}
        assert await engine.pending_count() == 1

    async def test_embed_outage_is_queued(self, engine: MemoryEngine) -> None:
        engine._backend.generate.return_value = '["a fact that is long enough"]'
        engine._backend.embed.side_effect = BackendUnavailable("cooldown")
        assert isinstance(await engine.add("proj", "john", ["t"]), QueuedIds)

    async def test_qdrant_error_is_not_queued(self, engine: MemoryEngine) -> None:
        engine._backend.generate.return_value = '["a fact that is long enough"]'
        engine._backend.embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
        engine._qdrant.query_points.return_value = MagicMock(points=[])
        engine._qdrant.upsert.side_effect = RuntimeError("qdrant down")
        with pytest.raises(RuntimeError):
            await engine.add("proj", "john", ["t"])
        assert await engine.pending_count() == 0

    async def test_add_path_uses_primary_only_methods(self) -> None:
        # A real LLMBackend subclass: the add path must call the *_primary
        # methods, which is what keeps it off the bundled fallback.
        from mem_zero.backends import LLMBackend

        class Spy(LLMBackend):
            def __init__(self) -> None:
                self.calls: list[str] = []

            async def generate(self, system, user, schema=None):  # type: ignore[no-untyped-def]
                self.calls.append("generate")
                return "[]"

            async def generate_primary(self, system, user, schema=None):  # type: ignore[no-untyped-def]
                self.calls.append("generate_primary")
                raise BackendUnavailable("down")

            async def embed(self, texts):  # type: ignore[no-untyped-def]
                self.calls.append("embed")
                return []

            @property
            def embedding_dimensions(self) -> int:
                return 768

        spy = Spy()
        cfg = Config(embedding_dimensions=768, pending_dir="/nonexistent-unused")
        eng = MemoryEngine(cfg, spy)
        eng._qdrant = AsyncMock()
        eng._ensure_collection = AsyncMock(return_value="c")  # type: ignore[method-assign]
        eng._pending = MagicMock()
        eng._pending.count.return_value = 0
        assert isinstance(await eng.add("proj", "john", ["t"]), QueuedIds)
        assert spy.calls == ["generate_primary"]
        eng._pending.enqueue.assert_called_once()


class TestReplay:
    async def test_success_deletes_file(self, engine: MemoryEngine) -> None:
        engine._pending.enqueue("proj", "john", "t", None)
        engine._add_primary = AsyncMock(return_value=["id"])  # type: ignore[method-assign]
        assert await engine.replay_pending() == 1
        assert await engine.pending_count() == 0

    async def test_outage_with_failed_ping_never_charges(
        self, engine: MemoryEngine
    ) -> None:
        for t in ("a", "b", "c"):
            engine._pending.enqueue("proj", "john", t, None)
        engine._backend.health_ping.return_value = False
        engine._add_primary = AsyncMock(  # type: ignore[method-assign]
            side_effect=[["id"], BackendUnavailable("down")]
        )
        assert await engine.replay_pending() == 1
        left = engine._pending.list()
        assert [i.text for i in left] == ["b", "c"]
        assert all(i.attempts == 0 for i in left)
        assert engine._add_primary.await_count == 2

    async def test_timeout_with_healthy_ping_dead_letters_head(
        self, engine: MemoryEngine, tmp_path: Path
    ) -> None:
        engine._pending.enqueue("proj", "john", "huge", None)
        engine._pending.enqueue("proj", "john", "other", None)
        engine._backend.health_ping.return_value = True
        engine._add_primary = AsyncMock(  # type: ignore[method-assign]
            side_effect=BackendUnavailable("ReadTimeout")
        )
        for n in range(MAX_ATTEMPTS):
            await engine.replay_pending()
            if n < MAX_ATTEMPTS - 1:
                assert engine._pending.list()[0].attempts == n + 1
        assert [i.text for i in engine._pending.list()] == ["other"]
        assert engine._pending.list()[0].attempts == 0
        assert len(list((tmp_path / "pending" / "failed").glob("*.json"))) == 1

    async def test_non_outage_failure_charges_only_head(
        self, engine: MemoryEngine
    ) -> None:
        for t in ("a", "b", "c"):
            engine._pending.enqueue("proj", "john", t, None)
        engine._add_primary = AsyncMock(side_effect=ValueError("poison"))  # type: ignore[method-assign]
        await engine.replay_pending()
        assert [i.attempts for i in engine._pending.list()] == [1, 0, 0]
        assert engine._add_primary.await_count == 1

    async def test_live_add_queues_while_queue_nonempty(
        self, engine: MemoryEngine
    ) -> None:
        engine._pending.enqueue("proj", "john", "older", None)
        engine._add_primary = AsyncMock(return_value=["id"])  # type: ignore[method-assign]
        result = await engine.add("proj", "john", ["newer"])
        assert isinstance(result, QueuedIds)
        engine._add_primary.assert_not_called()
        assert [i.text for i in engine._pending.list()] == ["older", "newer"]

    async def test_replay_stamps_enqueued_at(self, engine: MemoryEngine) -> None:
        engine._pending.enqueue("proj", "john", "t", None)
        enq = engine._pending.list()[0].enqueued_at
        engine._add_primary = AsyncMock(return_value=[])  # type: ignore[method-assign]
        await engine.replay_pending()
        assert engine._add_primary.await_args.args[-1] == enq

    async def test_add_texts_uses_given_timestamp(self, engine: MemoryEngine) -> None:
        from unittest.mock import patch

        with (
            patch.object(
                engine, "_extract_facts", new_callable=AsyncMock,
                return_value=["a fact that is long enough"],
            ),
            patch.object(
                engine, "_dedup_fact", new_callable=AsyncMock,
                return_value=("add", None, None),
            ),
        ):
            engine._backend.embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
            await engine._add_texts("proj", "john", ["t"], None, 1234.5)
        point = engine._qdrant.upsert.call_args.kwargs["points"][0]
        assert point.payload["created_at"] == 1234.5
        assert point.payload["updated_at"] == 1234.5

    async def test_fifo_order(self, engine: MemoryEngine) -> None:
        for t in ("a", "b", "c"):
            engine._pending.enqueue("proj", "john", t, None)
        engine._add_primary = AsyncMock(return_value=[])  # type: ignore[method-assign]
        await engine.replay_pending()
        texts = [c.args[2][0] for c in engine._add_primary.await_args_list]
        assert texts == ["a", "b", "c"]

    async def test_dead_letters_after_repeated_failures(
        self, engine: MemoryEngine, tmp_path: Path
    ) -> None:
        engine._pending.enqueue("proj", "john", "bad", None)
        engine._add_primary = AsyncMock(side_effect=ValueError("poison"))  # type: ignore[method-assign]
        for _ in range(MAX_ATTEMPTS):
            await engine.replay_pending()
        assert await engine.pending_count() == 0
        assert len(list((tmp_path / "pending" / "failed").glob("*.json"))) == 1
