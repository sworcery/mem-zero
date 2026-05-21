from __future__ import annotations

import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path as FilePath

from fastapi import FastAPI, HTTPException, Path, Query, Request, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from .backends import create_backend
from .config import Config, validate_slug
from .mcp_server import mcp_router, set_engine
from .memory_engine import MemoryEngine
from .models import MemoryCreate, MemoryRecord, ProjectInfo, SearchRequest
from .stats import DiagnosticStats

logger = logging.getLogger(__name__)

config = Config.from_env()
stats = DiagnosticStats(config.stats_path)
backend = create_backend(config, stats=stats)
engine = MemoryEngine(config, backend, stats=stats)
set_engine(engine)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await stats.start_flush_loop()
    yield
    await stats.shutdown()
    await engine.close()


app = FastAPI(
    title="mem-zero",
    description="Project-isolated MCP memory server",
    version="0.1.37",
    lifespan=lifespan,
)

_ALWAYS_OPEN = ("/health", "/icon.png")
_API_PREFIXES = ("/api/", "/mcp/", "/debug/")


class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app_: FastAPI, api_key: str) -> None:
        super().__init__(app_)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        path = request.url.path
        if any(path.startswith(p) for p in _ALWAYS_OPEN):
            return await call_next(request)
        if any(path.startswith(p) for p in _API_PREFIXES):
            if getattr(request.state, "dashboard_authenticated", False):
                return await call_next(request)
            auth = request.headers.get("authorization", "")
            if auth.startswith("Bearer "):
                token = auth[7:]
                if secrets.compare_digest(token, self._api_key):
                    return await call_next(request)
            key = request.query_params.get("api_key", "")
            if key and secrets.compare_digest(key, self._api_key):
                return await call_next(request)
            return Response(
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="mem-zero"'},
                content="Invalid or missing API key",
            )
        return await call_next(request)


class DashboardAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app_: FastAPI, username: str, password: str) -> None:
        super().__init__(app_)
        self._username = username
        self._password = password

    def _valid_basic_auth(self, request: Request) -> bool:
        import base64

        auth = request.headers.get("authorization", "")
        if not auth.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(auth[6:]).decode()
            user, passwd = decoded.split(":", 1)
            return (
                secrets.compare_digest(user, self._username)
                and secrets.compare_digest(passwd, self._password)
            )
        except Exception:
            return False

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        path = request.url.path
        if any(path.startswith(p) for p in _ALWAYS_OPEN):
            return await call_next(request)

        if any(path.startswith(p) for p in _API_PREFIXES):
            if self._valid_basic_auth(request):
                request.state.dashboard_authenticated = True
                return await call_next(request)
            return await call_next(request)

        if self._valid_basic_auth(request):
            return await call_next(request)

        return Response(
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="mem-zero"'},
            content="Unauthorized",
        )


if config.api_key:
    app.add_middleware(APIKeyMiddleware, api_key=config.api_key)
else:
    logger.warning("API_KEY is not set — all API and MCP endpoints are unauthenticated")

if config.dashboard_user and config.dashboard_pass:
    app.add_middleware(
        DashboardAuthMiddleware,
        username=config.dashboard_user,
        password=config.dashboard_pass,
    )

app.include_router(mcp_router, prefix="/mcp")


def _validated_slug(slug: str) -> str:
    try:
        return validate_slug(slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
async def health() -> dict[str, object]:
    services: dict[str, bool] = {}
    try:
        await engine.health_check()
        services["qdrant"] = True
    except Exception:
        services["qdrant"] = False
    services["llm"] = await backend.health_ping()
    overall = services["qdrant"] and services["llm"]
    if not overall:
        raise HTTPException(status_code=503, detail="Service degraded")
    return {"status": "ok", "version": app.version, "services": services}


@app.get("/debug/config")
async def debug_config() -> dict[str, object]:
    info: dict[str, object] = {"llm_backend": config.llm_backend}
    if config.llm_backend == "ollama":
        info["ollama_base_url"] = config.ollama_base_url
        info["llm_model"] = config.llm_model
        info["embedder_model"] = config.embedder_model
    elif config.llm_backend == "openai":
        info["openai_base_url"] = config.openai_base_url
        info["openai_model"] = config.openai_model
        info["openai_embed_model"] = config.openai_embed_model
    else:
        info["bundled_model_path"] = config.bundled_model_path
        info["bundled_embed_model"] = config.bundled_embed_model
        info["bundled_threads"] = config.bundled_threads
    info["embedding_dimensions"] = backend.embedding_dimensions
    return info


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


@app.delete("/api/v1/projects/{slug}")
async def remove_project(slug: str = Path(...)) -> dict[str, bool]:
    slug = _validated_slug(slug)
    removed = await engine.delete_project(slug)
    if not removed:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"deleted": True}


@app.post("/api/v1/projects/{slug}/reembed")
async def reembed_memories(slug: str = Path(...)) -> dict[str, int]:
    slug = _validated_slug(slug)
    try:
        count = await engine.reembed_all(slug)
    except Exception as exc:
        logger.exception("Re-embed failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"reembedded": count}


@app.post("/api/v1/projects/{slug}/cleanup")
async def cleanup_memories(slug: str = Path(...)) -> dict[str, int]:
    slug = _validated_slug(slug)
    try:
        result = await engine.cleanup_text(slug)
    except Exception as exc:
        logger.exception("Cleanup failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@app.post("/api/v1/projects/{slug}/consolidate")
async def consolidate_memories(
    slug: str = Path(...),
    threshold: float = Query(default=0.75, ge=0.5, le=1.0),
    dry_run: bool = Query(default=False),
) -> dict[str, object]:
    slug = _validated_slug(slug)
    try:
        result = await engine.consolidate(
            slug, similarity_threshold=threshold, dry_run=dry_run
        )
    except Exception as exc:
        logger.exception("Consolidation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@app.get("/api/v1/diagnostics")
async def get_diagnostics() -> dict[str, object]:
    if not config.diagnostics_enabled:
        raise HTTPException(status_code=404, detail="Diagnostics disabled")
    try:
        projects = await engine.list_projects()
        stats.record_daily_snapshot({p.slug: p.memory_count for p in projects})
    except Exception:
        pass
    return stats.snapshot()


@app.get("/api/v1/projects/{slug}/diagnostics")
async def get_project_diagnostics(slug: str = Path(...)) -> dict[str, object]:
    if not config.diagnostics_enabled:
        raise HTTPException(status_code=404, detail="Diagnostics disabled")
    slug = _validated_slug(slug)
    return stats.snapshot(project=slug)


@app.post("/api/v1/diagnostics/reset")
async def reset_diagnostics() -> dict[str, str]:
    if not config.diagnostics_enabled:
        raise HTTPException(status_code=404, detail="Diagnostics disabled")
    stats.reset()
    return {"status": "reset"}


@app.post("/api/v1/diagnostics/export")
async def export_diagnostics() -> dict[str, object]:
    if not config.diagnostics_enabled:
        raise HTTPException(status_code=404, detail="Diagnostics disabled")
    return stats.snapshot()


_static_dir = FilePath(__file__).parent / "static"
if _static_dir.is_dir():
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
