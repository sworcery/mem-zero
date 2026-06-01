from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

EVENT_TYPES = frozenset({
    "memory_added",
    "memory_deleted",
    "memories_cleared",
    "consolidation_complete",
    "project_deleted",
    "health_degraded",
    "health_recovered",
})

MAX_RETRIES = 3
RETRY_DELAYS = [1.0, 5.0, 15.0]


@dataclass
class WebhookConfig:
    id: str
    url: str
    events: list[str]
    secret: str | None = None
    enabled: bool = True
    created_at: float = field(default_factory=time.time)


@dataclass
class WebhookDelivery:
    webhook_id: str
    event: str
    status_code: int | None
    success: bool
    attempt: int
    timestamp: float = field(default_factory=time.time)
    error: str | None = None


class WebhookManager:
    def __init__(self, config_path: str | Path | None = None) -> None:
        self._hooks: dict[str, WebhookConfig] = {}
        self._deliveries: list[WebhookDelivery] = []
        self._max_deliveries = 200
        self._config_path = Path(config_path) if config_path else None
        self._http = httpx.AsyncClient(timeout=10.0)
        self._load()

    def _load(self) -> None:
        if not self._config_path or not self._config_path.exists():
            return
        try:
            data = json.loads(self._config_path.read_text())
            for hook_data in data.get("webhooks", []):
                hook = WebhookConfig(**hook_data)
                self._hooks[hook.id] = hook
            logger.info("Loaded %d webhooks from %s", len(self._hooks), self._config_path)
        except Exception:
            logger.warning("Failed to load webhooks from %s", self._config_path)

    def _save(self) -> None:
        if not self._config_path:
            return
        data = {
            "webhooks": [
                {
                    "id": h.id,
                    "url": h.url,
                    "events": h.events,
                    "secret": h.secret,
                    "enabled": h.enabled,
                    "created_at": h.created_at,
                }
                for h in self._hooks.values()
            ]
        }
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._config_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(self._config_path)
        except Exception:
            logger.warning("Failed to save webhooks to %s", self._config_path)

    def register(
        self,
        url: str,
        events: list[str],
        secret: str | None = None,
    ) -> WebhookConfig:
        for event in events:
            if event not in EVENT_TYPES:
                raise ValueError(f"Unknown event type: {event!r}")
        hook = WebhookConfig(
            id=str(uuid.uuid4()),
            url=url,
            events=events,
            secret=secret,
        )
        self._hooks[hook.id] = hook
        self._save()
        return hook

    def unregister(self, hook_id: str) -> bool:
        if hook_id not in self._hooks:
            return False
        del self._hooks[hook_id]
        self._save()
        return True

    def list_hooks(self) -> list[WebhookConfig]:
        return list(self._hooks.values())

    def get_hook(self, hook_id: str) -> WebhookConfig | None:
        return self._hooks.get(hook_id)

    def update(
        self,
        hook_id: str,
        url: str | None = None,
        events: list[str] | None = None,
        enabled: bool | None = None,
    ) -> WebhookConfig | None:
        hook = self._hooks.get(hook_id)
        if not hook:
            return None
        if url is not None:
            hook.url = url
        if events is not None:
            for event in events:
                if event not in EVENT_TYPES:
                    raise ValueError(f"Unknown event type: {event!r}")
            hook.events = events
        if enabled is not None:
            hook.enabled = enabled
        self._save()
        return hook

    def recent_deliveries(self, limit: int = 50) -> list[dict[str, Any]]:
        return [
            {
                "webhook_id": d.webhook_id,
                "event": d.event,
                "status_code": d.status_code,
                "success": d.success,
                "attempt": d.attempt,
                "timestamp": d.timestamp,
                "error": d.error,
            }
            for d in self._deliveries[-limit:]
        ]

    def _record_delivery(self, delivery: WebhookDelivery) -> None:
        self._deliveries.append(delivery)
        if len(self._deliveries) > self._max_deliveries:
            self._deliveries = self._deliveries[-self._max_deliveries:]

    async def _deliver(
        self, hook: WebhookConfig, event: str, payload: dict[str, Any]
    ) -> None:
        body = {
            "event": event,
            "timestamp": time.time(),
            "data": payload,
        }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if hook.secret:
            import hashlib
            import hmac
            sig = hmac.new(
                hook.secret.encode(), json.dumps(body).encode(), hashlib.sha256
            ).hexdigest()
            headers["X-Webhook-Signature"] = sig

        for attempt in range(MAX_RETRIES):
            try:
                resp = await self._http.post(hook.url, json=body, headers=headers)
                self._record_delivery(WebhookDelivery(
                    webhook_id=hook.id,
                    event=event,
                    status_code=resp.status_code,
                    success=200 <= resp.status_code < 300,
                    attempt=attempt + 1,
                ))
                if 200 <= resp.status_code < 300:
                    return
            except Exception as exc:
                self._record_delivery(WebhookDelivery(
                    webhook_id=hook.id,
                    event=event,
                    status_code=None,
                    success=False,
                    attempt=attempt + 1,
                    error=str(exc)[:200],
                ))

            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAYS[attempt])

        logger.warning(
            "Webhook %s delivery failed after %d attempts for event %s",
            hook.id[:8], MAX_RETRIES, event,
        )

    async def fire(self, event: str, payload: dict[str, Any]) -> int:
        if event not in EVENT_TYPES:
            return 0
        tasks = []
        for hook in self._hooks.values():
            if not hook.enabled:
                continue
            if event in hook.events:
                tasks.append(self._deliver(hook, event, payload))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)

    async def close(self) -> None:
        await self._http.aclose()
