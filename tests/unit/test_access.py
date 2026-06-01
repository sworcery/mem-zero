from __future__ import annotations

import time
from pathlib import Path

import pytest

from mem_zero.access import AccessManager, Permission


@pytest.fixture
def manager(tmp_path: Path) -> AccessManager:
    return AccessManager(config_path=tmp_path / "keys.json")


@pytest.fixture
def manager_no_persist() -> AccessManager:
    return AccessManager(config_path=None)


class TestCreateKey:
    def test_creates_key(self, manager: AccessManager) -> None:
        raw_key, record = manager.create_key("test-key")
        assert len(raw_key) == 64
        assert record.name == "test-key"
        assert record.enabled is True
        assert record.key_prefix == raw_key[:8]

    def test_default_permissions(self, manager: AccessManager) -> None:
        _, record = manager.create_key("default")
        assert Permission.READ.value in record.permissions
        assert Permission.WRITE.value in record.permissions
        assert Permission.ADMIN.value not in record.permissions

    def test_custom_permissions(self, manager: AccessManager) -> None:
        _, record = manager.create_key(
            "readonly", permissions=[Permission.READ.value]
        )
        assert record.permissions == [Permission.READ.value]

    def test_project_restriction(self, manager: AccessManager) -> None:
        _, record = manager.create_key(
            "scoped", projects=["alpha", "beta"]
        )
        assert record.projects == ["alpha", "beta"]

    def test_expiration(self, manager: AccessManager) -> None:
        _, record = manager.create_key("expiring", expires_in_days=30)
        assert record.expires_at is not None
        assert record.expires_at > time.time()

    def test_reject_invalid_permission(self, manager: AccessManager) -> None:
        with pytest.raises(ValueError, match="Invalid permission"):
            manager.create_key("bad", permissions=["superuser"])


class TestValidateKey:
    def test_valid_key(self, manager: AccessManager) -> None:
        raw_key, _ = manager.create_key("test")
        record = manager.validate_key(raw_key)
        assert record is not None
        assert record.name == "test"

    def test_invalid_key(self, manager: AccessManager) -> None:
        assert manager.validate_key("bogus-key") is None

    def test_disabled_key(self, manager: AccessManager) -> None:
        raw_key, record = manager.create_key("disabled")
        manager.revoke_key(record.id)
        assert manager.validate_key(raw_key) is None

    def test_expired_key(self, manager: AccessManager) -> None:
        raw_key, record = manager.create_key("expired", expires_in_days=0.0001)
        record.expires_at = time.time() - 1
        assert manager.validate_key(raw_key) is None

    def test_tracks_usage(self, manager: AccessManager) -> None:
        raw_key, _ = manager.create_key("tracked")
        manager.validate_key(raw_key)
        manager.validate_key(raw_key)
        record = manager.validate_key(raw_key)
        assert record is not None
        assert record.usage_count == 3
        assert record.last_used_at is not None


class TestCheckPermission:
    def test_read_allowed(self, manager: AccessManager) -> None:
        _, record = manager.create_key("rw")
        assert manager.check_permission(record, "read") is True

    def test_write_allowed(self, manager: AccessManager) -> None:
        _, record = manager.create_key("rw")
        assert manager.check_permission(record, "write") is True

    def test_admin_allows_everything(self, manager: AccessManager) -> None:
        _, record = manager.create_key("admin", permissions=["admin"])
        assert manager.check_permission(record, "read") is True
        assert manager.check_permission(record, "write") is True
        assert manager.check_permission(record, "admin") is True

    def test_readonly_blocks_write(self, manager: AccessManager) -> None:
        _, record = manager.create_key("ro", permissions=["read"])
        assert manager.check_permission(record, "read") is True
        assert manager.check_permission(record, "write") is False

    def test_project_scoping(self, manager: AccessManager) -> None:
        _, record = manager.create_key("scoped", projects=["alpha"])
        assert manager.check_permission(record, "read", project="alpha") is True
        assert manager.check_permission(record, "read", project="beta") is False

    def test_no_project_restriction(self, manager: AccessManager) -> None:
        _, record = manager.create_key("global")
        assert manager.check_permission(record, "read", project="anything") is True


class TestKeyManagement:
    def test_revoke_key(self, manager: AccessManager) -> None:
        _, record = manager.create_key("revokable")
        assert manager.revoke_key(record.id) is True
        keys = manager.list_keys()
        disabled = [k for k in keys if k["id"] == record.id]
        assert disabled[0]["enabled"] is False

    def test_revoke_nonexistent(self, manager: AccessManager) -> None:
        assert manager.revoke_key("fake") is False

    def test_delete_key(self, manager: AccessManager) -> None:
        _, record = manager.create_key("deletable")
        assert manager.delete_key(record.id) is True
        assert len(manager.list_keys()) == 0

    def test_delete_nonexistent(self, manager: AccessManager) -> None:
        assert manager.delete_key("fake") is False

    def test_list_keys(self, manager: AccessManager) -> None:
        manager.create_key("key1")
        manager.create_key("key2")
        keys = manager.list_keys()
        assert len(keys) == 2
        names = {k["name"] for k in keys}
        assert names == {"key1", "key2"}

    def test_list_keys_excludes_hash(self, manager: AccessManager) -> None:
        manager.create_key("secure")
        keys = manager.list_keys()
        assert "key_hash" not in keys[0]


class TestPersistence:
    def test_save_and_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "keys.json"
        mgr1 = AccessManager(config_path=path)
        raw_key, _ = mgr1.create_key("persistent", permissions=["read"])

        mgr2 = AccessManager(config_path=path)
        record = mgr2.validate_key(raw_key)
        assert record is not None
        assert record.name == "persistent"
        assert record.permissions == ["read"]


class TestKeyStats:
    def test_stats(self, manager: AccessManager) -> None:
        manager.create_key("active1")
        manager.create_key("active2")
        raw, record = manager.create_key("disabled")
        manager.revoke_key(record.id)

        stats = manager.key_stats()
        assert stats["total_keys"] == 3
        assert stats["active_keys"] == 2
        assert stats["disabled_keys"] == 1
