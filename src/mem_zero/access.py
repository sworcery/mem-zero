from __future__ import annotations

import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Permission(str, Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


@dataclass
class APIKeyRecord:
    id: str
    name: str
    key_hash: str
    key_prefix: str
    permissions: list[str]
    projects: list[str] | None
    created_at: float = field(default_factory=time.time)
    last_used_at: float | None = None
    expires_at: float | None = None
    enabled: bool = True
    usage_count: int = 0


class AccessManager:
    def __init__(self, config_path: str | Path | None = None) -> None:
        self._keys: dict[str, APIKeyRecord] = {}
        self._config_path = Path(config_path) if config_path else None
        self._load()

    @staticmethod
    def _hash_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def _load(self) -> None:
        if not self._config_path or not self._config_path.exists():
            return
        try:
            data = json.loads(self._config_path.read_text())
            for rec in data.get("keys", []):
                record = APIKeyRecord(**rec)
                self._keys[record.id] = record
            logger.info("Loaded %d API keys from %s", len(self._keys), self._config_path)
        except Exception:
            logger.warning("Failed to load API keys from %s", self._config_path)

    def _save(self) -> None:
        if not self._config_path:
            return
        data = {
            "keys": [
                {
                    "id": k.id,
                    "name": k.name,
                    "key_hash": k.key_hash,
                    "key_prefix": k.key_prefix,
                    "permissions": k.permissions,
                    "projects": k.projects,
                    "created_at": k.created_at,
                    "last_used_at": k.last_used_at,
                    "expires_at": k.expires_at,
                    "enabled": k.enabled,
                    "usage_count": k.usage_count,
                }
                for k in self._keys.values()
            ]
        }
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._config_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(self._config_path)
        except Exception:
            logger.warning("Failed to save API keys to %s", self._config_path)

    def create_key(
        self,
        name: str,
        permissions: list[str] | None = None,
        projects: list[str] | None = None,
        expires_in_days: float | None = None,
    ) -> tuple[str, APIKeyRecord]:
        for perm in (permissions or []):
            if perm not in {p.value for p in Permission}:
                raise ValueError(f"Invalid permission: {perm!r}")

        raw_key = secrets.token_hex(32)
        key_id = secrets.token_hex(8)
        expires_at = None
        if expires_in_days:
            expires_at = time.time() + (expires_in_days * 86400)

        record = APIKeyRecord(
            id=key_id,
            name=name,
            key_hash=self._hash_key(raw_key),
            key_prefix=raw_key[:8],
            permissions=permissions or [Permission.READ.value, Permission.WRITE.value],
            projects=projects,
            expires_at=expires_at,
        )
        self._keys[key_id] = record
        self._save()
        return raw_key, record

    def revoke_key(self, key_id: str) -> bool:
        if key_id not in self._keys:
            return False
        self._keys[key_id].enabled = False
        self._save()
        return True

    def delete_key(self, key_id: str) -> bool:
        if key_id not in self._keys:
            return False
        del self._keys[key_id]
        self._save()
        return True

    def list_keys(self) -> list[dict[str, Any]]:
        return [
            {
                "id": k.id,
                "name": k.name,
                "key_prefix": k.key_prefix,
                "permissions": k.permissions,
                "projects": k.projects,
                "enabled": k.enabled,
                "created_at": k.created_at,
                "last_used_at": k.last_used_at,
                "expires_at": k.expires_at,
                "usage_count": k.usage_count,
            }
            for k in self._keys.values()
        ]

    def validate_key(self, raw_key: str) -> APIKeyRecord | None:
        key_hash = self._hash_key(raw_key)
        for record in self._keys.values():
            if not secrets.compare_digest(record.key_hash, key_hash):
                continue
            if not record.enabled:
                return None
            if record.expires_at and time.time() > record.expires_at:
                return None
            record.last_used_at = time.time()
            record.usage_count += 1
            return record
        return None

    def check_permission(
        self,
        record: APIKeyRecord,
        permission: str,
        project: str | None = None,
    ) -> bool:
        if Permission.ADMIN.value in record.permissions:
            return True
        if permission not in record.permissions:
            return False
        return not (project and record.projects is not None and project not in record.projects)

    def key_stats(self) -> dict[str, Any]:
        total = len(self._keys)
        active = sum(1 for k in self._keys.values() if k.enabled)
        expired = sum(
            1 for k in self._keys.values()
            if k.expires_at and time.time() > k.expires_at
        )
        return {
            "total_keys": total,
            "active_keys": active,
            "disabled_keys": total - active,
            "expired_keys": expired,
            "total_usage": sum(k.usage_count for k in self._keys.values()),
        }
