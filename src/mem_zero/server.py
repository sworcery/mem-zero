from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Path, Query

from .config import Config, validate_slug
from .mcp_server import mcp_router, set_engine
from .memory_engine import MemoryEngine
from .models import MemoryCreate, MemoryRecord, ProjectInfo, SearchRequest

logger = logging.getLogger(__name__)

config = Config.from_env()
engine = MemoryEngine(config)
set_engine(engine)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    await engine.close()


app = FastAPI(
    title="mem-zero",
    description="Project-isolated MCP memory server",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(mcp_router, prefix="/mcp")


def _validated_slug(slug: str) -> str:
    try:
        return validate_slug(slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
async def health() -> dict[str, str]:
    try:
        await engine.health_check()
        return {"status": "ok"}
    except Exception as exc:
        logger.exception("Health check failed")
        raise HTTPException(status_code=503, detail="Service unavailable") from exc


@app.get("/api/v1/projects")
async def list_projects() -> list[ProjectInfo]:
    return await engine.list_projects()


@app.get("/api/v1/projects/{slug}/memories")
async def get_memories(
    slug: str = Path(...),
    limit: int = Query(default=50, ge=1, le=1000),
) -> list[MemoryRecord]:
    slug = _validated_slug(slug)
    return await engine.list_all(slug, limit=limit)


@app.post("/api/v1/projects/{slug}/memories")
async def create_memory(
    body: MemoryCreate,
    slug: str = Path(...),
    user_id: str = "default",
) -> dict[str, object]:
    slug = _validated_slug(slug)
    try:
        ids = await engine.add(slug, user_id, [body.text], body.metadata or None)
    except Exception as exc:
        logger.exception("Failed to add memory")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"stored": len(ids), "ids": ids}


@app.post("/api/v1/projects/{slug}/search")
async def search_memories(
    body: SearchRequest,
    slug: str = Path(...),
) -> list[MemoryRecord]:
    slug = _validated_slug(slug)
    return await engine.search(slug, body.query, top_k=body.top_k)


@app.delete("/api/v1/projects/{slug}/memories/{memory_id}")
async def remove_memory(
    slug: str = Path(...),
    memory_id: str = Path(...),
) -> dict[str, bool]:
    slug = _validated_slug(slug)
    await engine.delete(slug, memory_id)
    return {"deleted": True}


@app.delete("/api/v1/projects/{slug}/memories")
async def remove_all_memories(slug: str = Path(...)) -> dict[str, int]:
    slug = _validated_slug(slug)
    count = await engine.delete_all(slug)
    return {"deleted": count}
