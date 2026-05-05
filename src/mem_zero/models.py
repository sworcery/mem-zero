from __future__ import annotations

from pydantic import BaseModel, Field


class MemoryRecord(BaseModel):
    id: str
    text: str
    user_id: str
    created_at: float
    updated_at: float
    metadata: dict[str, object] = Field(default_factory=dict)
    score: float | None = None


class MemoryCreate(BaseModel):
    text: str = Field(..., max_length=50000)
    metadata: dict[str, object] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)


class ProjectInfo(BaseModel):
    slug: str
    collection: str
    memory_count: int
    last_updated: float | None = None
