from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from mem_zero.webhooks import EVENT_TYPES, WebhookManager


@pytest.fixture
def manager(tmp_path: Path) -> WebhookManager:
    return WebhookManager(config_path=tmp_path / "webhooks.json")


@pytest.fixture
def manager_no_persist() -> WebhookManager:
    return WebhookManager(config_path=None)


class TestRegistration:
    def test_register_webhook(self, manager: WebhookManager) -> None:
        hook = manager.register(
            url="http://example.com/hook",
            events=["memory_added"],
        )
        assert hook.id
        assert hook.url == "http://example.com/hook"
        assert hook.events == ["memory_added"]
        assert hook.enabled is True

    def test_register_with_secret(self, manager: WebhookManager) -> None:
        hook = manager.register(
            url="http://example.com/hook",
            events=["memory_added"],
            secret="mysecret",
        )
        assert hook.secret == "mysecret"

    def test_reject_unknown_event(self, manager: WebhookManager) -> None:
        with pytest.raises(ValueError, match="Unknown event type"):
            manager.register(url="http://x.com", events=["fake_event"])

    def test_unregister(self, manager: WebhookManager) -> None:
        hook = manager.register(url="http://x.com", events=["memory_added"])
        assert manager.unregister(hook.id) is True
        assert manager.get_hook(hook.id) is None

    def test_unregister_nonexistent(self, manager: WebhookManager) -> None:
        assert manager.unregister("fake-id") is False

    def test_list_hooks(self, manager: WebhookManager) -> None:
        manager.register(url="http://a.com", events=["memory_added"])
        manager.register(url="http://b.com", events=["memory_deleted"])
        hooks = manager.list_hooks()
        assert len(hooks) == 2


class TestUpdate:
    def test_update_url(self, manager: WebhookManager) -> None:
        hook = manager.register(url="http://old.com", events=["memory_added"])
        updated = manager.update(hook.id, url="http://new.com")
        assert updated is not None
        assert updated.url == "http://new.com"

    def test_update_events(self, manager: WebhookManager) -> None:
        hook = manager.register(url="http://x.com", events=["memory_added"])
        updated = manager.update(hook.id, events=["memory_deleted", "project_deleted"])
        assert updated is not None
        assert "memory_deleted" in updated.events

    def test_disable_hook(self, manager: WebhookManager) -> None:
        hook = manager.register(url="http://x.com", events=["memory_added"])
        updated = manager.update(hook.id, enabled=False)
        assert updated is not None
        assert updated.enabled is False

    def test_update_nonexistent(self, manager: WebhookManager) -> None:
        assert manager.update("fake-id", url="http://x.com") is None

    def test_update_rejects_bad_event(self, manager: WebhookManager) -> None:
        hook = manager.register(url="http://x.com", events=["memory_added"])
        with pytest.raises(ValueError):
            manager.update(hook.id, events=["bad_event"])


class TestPersistence:
    def test_save_and_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "hooks.json"
        mgr1 = WebhookManager(config_path=path)
        mgr1.register(url="http://test.com", events=["memory_added"], secret="s3cret")
        mgr1.register(url="http://other.com", events=["memory_deleted"])

        mgr2 = WebhookManager(config_path=path)
        hooks = mgr2.list_hooks()
        assert len(hooks) == 2
        urls = {h.url for h in hooks}
        assert "http://test.com" in urls
        assert "http://other.com" in urls


class TestFire:
    @pytest.mark.asyncio
    async def test_fires_matching_hooks(self, manager: WebhookManager) -> None:
        manager.register(url="http://a.com", events=["memory_added"])
        manager.register(url="http://b.com", events=["memory_deleted"])

        with patch.object(manager, "_deliver", new_callable=AsyncMock) as mock_deliver:
            count = await manager.fire("memory_added", {"text": "hello"})
            assert count == 1
            mock_deliver.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_disabled_hooks(self, manager: WebhookManager) -> None:
        hook = manager.register(url="http://a.com", events=["memory_added"])
        manager.update(hook.id, enabled=False)

        with patch.object(manager, "_deliver", new_callable=AsyncMock) as mock_deliver:
            count = await manager.fire("memory_added", {"text": "hello"})
            assert count == 0
            mock_deliver.assert_not_called()

    @pytest.mark.asyncio
    async def test_fire_unknown_event(self, manager: WebhookManager) -> None:
        count = await manager.fire("nonexistent", {})
        assert count == 0

    @pytest.mark.asyncio
    async def test_delivery_recorded(self, manager_no_persist: WebhookManager) -> None:
        mgr = manager_no_persist
        mgr.register(url="http://example.com/hook", events=["memory_added"])

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        with patch.object(mgr._http, "post", return_value=mock_resp):
            await mgr.fire("memory_added", {"test": True})

        deliveries = mgr.recent_deliveries()
        assert len(deliveries) == 1
        assert deliveries[0]["success"] is True
        assert deliveries[0]["status_code"] == 200


class TestEventTypes:
    def test_all_events_documented(self) -> None:
        assert "memory_added" in EVENT_TYPES
        assert "memory_deleted" in EVENT_TYPES
        assert "consolidation_complete" in EVENT_TYPES
        assert "project_deleted" in EVENT_TYPES
        assert "health_degraded" in EVENT_TYPES
        assert "health_recovered" in EVENT_TYPES
