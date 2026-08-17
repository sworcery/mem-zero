from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# Metadata is written straight into the Qdrant payload and returned on every
# read; keep it small and flat so a client can't bloat every point.
_META_MAX_KEYS = 32
_META_MAX_KEY_LEN = 64
_META_MAX_STR_LEN = 2000
_META_SCALARS = (str, int, float, bool, type(None))


def _validate_metadata(meta: dict[str, object]) -> dict[str, object]:
    if len(meta) > _META_MAX_KEYS:
        raise ValueError(f"metadata may have at most {_META_MAX_KEYS} keys")
    for key, value in meta.items():
        if not isinstance(key, str) or not key or len(key) > _META_MAX_KEY_LEN:
            raise ValueError(f"metadata key {key!r} must be 1-{_META_MAX_KEY_LEN} chars")
        if isinstance(value, str):
            if len(value) > _META_MAX_STR_LEN:
                raise ValueError(
                    f"metadata value for {key!r} exceeds {_META_MAX_STR_LEN} chars"
                )
        elif isinstance(value, list):
            if len(value) > _META_MAX_KEYS:
                raise ValueError(f"metadata list {key!r} may have at most {_META_MAX_KEYS} items")
            for item in value:
                if not isinstance(item, _META_SCALARS) or (
                    isinstance(item, str) and len(item) > _META_MAX_STR_LEN
                ):
                    raise ValueError(f"metadata list {key!r} may only contain short scalars")
        elif not isinstance(value, _META_SCALARS):
            raise ValueError(
                f"metadata value for {key!r} must be a scalar or a list of scalars"
            )
    return meta


class MemoryRecord(BaseModel):
    id: str
    text: str
    user_id: str
    created_at: float
    updated_at: float
    metadata: dict[str, object] = Field(default_factory=dict)
    score: float | None = None


class MemoryCreate(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _check_metadata(cls, v: dict[str, object]) -> dict[str, object]:
        return _validate_metadata(v)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=100)


class ProjectInfo(BaseModel):
    slug: str
    collection: str
    memory_count: int
    last_updated: float | None = None
