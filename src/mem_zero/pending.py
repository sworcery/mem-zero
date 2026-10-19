from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Failures that are not a backend outage (bad input, Qdrant rejecting a
# point). After this many the add is parked in failed/ for a human.
MAX_ATTEMPTS = 10


@dataclass
class PendingAdd:
    path: Path
    project: str
    user_id: str
    text: str
    metadata: dict[str, object]
    enqueued_at: float
    attempts: int
    last_error: str | None


class PendingStore:
    """Durable FIFO of raw adds waiting for the LLM backend, one JSON file each."""

    def __init__(self, directory: str) -> None:
        self._dir = Path(directory)
        self._failed = self._dir / "failed"

    def enqueue(
        self, project: str, user_id: str, text: str, metadata: dict[str, object] | None
    ) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        # time_ns first so a plain sort is arrival order; the uuid only breaks
        # ties between adds landing in the same nanosecond tick.
        path = self._dir / f"{time.time_ns()}-{uuid.uuid4()}.json"
        self._write(
            path,
            {
                "project": project,
                "user_id": user_id,
                "text": text,
                "metadata": metadata or {},
                "enqueued_at": time.time(),
                "attempts": 0,
                "last_error": None,
            },
        )
        return path

    @staticmethod
    def _write(path: Path, data: dict[str, object]) -> None:
        # Write-then-rename so a crash never leaves a half-written file that
        # the replay worker would choke on.
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(data))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def count(self) -> int:
        if not self._dir.is_dir():
            return 0
        return sum(1 for _ in self._dir.glob("*.json"))

    def list(self) -> list[PendingAdd]:
        if not self._dir.is_dir():
            return []
        items: list[PendingAdd] = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                items.append(
                    PendingAdd(
                        path=path,
                        project=d["project"],
                        user_id=d["user_id"],
                        text=d["text"],
                        metadata=d.get("metadata") or {},
                        enqueued_at=d.get("enqueued_at", 0.0),
                        attempts=d.get("attempts", 0),
                        last_error=d.get("last_error"),
                    )
                )
            except (OSError, ValueError, KeyError, TypeError):
                logger.exception("Unreadable pending file %s; moving to failed/", path)
                self._move_to_failed(path)
        return items

    def delete(self, item: PendingAdd) -> None:
        item.path.unlink(missing_ok=True)

    def record_failure(self, item: PendingAdd, error: str) -> bool:
        """Bump attempts; dead-letter at MAX_ATTEMPTS. True if dead-lettered."""
        item.attempts += 1
        item.last_error = error
        self._write(
            item.path,
            {
                "project": item.project,
                "user_id": item.user_id,
                "text": item.text,
                "metadata": item.metadata,
                "enqueued_at": item.enqueued_at,
                "attempts": item.attempts,
                "last_error": error,
            },
        )
        if item.attempts >= MAX_ATTEMPTS:
            self._move_to_failed(item.path)
            return True
        return False

    def _move_to_failed(self, path: Path) -> None:
        self._failed.mkdir(parents=True, exist_ok=True)
        os.replace(path, self._failed / path.name)
