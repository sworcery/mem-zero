from __future__ import annotations

import logging
import secrets
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from pathlib import Path as FilePath
from typing import TypeVar

from fastapi import FastAPI, HTTPException, Path, Query, Request, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from . import __version__
from .backends import create_backend
from .config import Config, validate_slug, validate_user_id
from .mcp_server import mcp_router, set_engine
from .memory_engine import (
    ConsolidationTooLargeError,
    DimensionMismatchError,
    EmbeddingError,
    LLMError,
    MaintenanceInProgressError,
    MemoryEngine,
    validate_memory_id,
)
from .models import MemoryCreate, MemoryRecord, ProjectInfo, SearchRequest
from .stats import DiagnosticStats

logger = logging.getLogger(__name__)

config = Config.from_env()
stats = DiagnosticStats(config.stats_path)
backend = create_backend(config, stats=stats)
engine = MemoryEngine(config, backend, stats=stats)
set_engine(engine)


async def _project_counts() -> dict[str, int]:
    return {p.slug: p.memory_count for p in await engine.list_projects()}


# The flush loop takes daily snapshots itself now; previously they were only
# recorded when someone happened to open the dashboard.
stats.set_project_count_provider(_project_counts)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await stats.start_flush_loop()
    yield
    await stats.shutdown()
    await engine.close()


app = FastAPI(
    title="mem-zero",
    description="Project-isolated MCP memory server",
    version=__version__,
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
        if path in _ALWAYS_OPEN:
            return await call_next(request)
        if any(path.startswith(p) for p in _API_PREFIXES):
            if getattr(request.state, "dashboard_authenticated", False):
                return await call_next(request)
            auth = request.headers.get("authorization", "")
            if auth.startswith("Bearer "):
                token = auth[7:]
                if secrets.compare_digest(token.encode(), self._api_key.encode()):
                    return await call_next(request)
            key = request.query_params.get("api_key", "")
            if key and secrets.compare_digest(key.encode(), self._api_key.encode()):
                return await call_next(request)
            return Response(
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="mem-zero"'},
                content="Invalid or missing API key",
            )
        return await call_next(request)


class DashboardAuthMiddleware(BaseHTTPMiddleware):
    def __init__(
        self, app_: FastAPI, username: str, password: str, api_key_configured: bool
    ) -> None:
        super().__init__(app_)
        self._username = username
        self._password = password
        # When an API key exists, unauthenticated /api requests fall through to
        # APIKeyMiddleware (Bearer). When it does NOT, this middleware is the
        # only gate — falling through would leave the whole API and MCP open
        # behind a dashboard that LOOKS password-protected.
        self._api_key_configured = api_key_configured

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
        if path in _ALWAYS_OPEN:
            return await call_next(request)

        if any(path.startswith(p) for p in _API_PREFIXES):
            if self._valid_basic_auth(request):
                request.state.dashboard_authenticated = True
                return await call_next(request)
            if self._api_key_configured:
                return await call_next(request)  # Bearer is checked next
            return Response(
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="mem-zero"'},
                content="Unauthorized",
            )

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
        api_key_configured=bool(config.api_key),
    )
    if not config.api_key:
        logger.warning(
            "DASHBOARD_USER/PASS set without API_KEY: /api, /mcp and /debug "
            "require the dashboard Basic-auth credentials"
        )
elif config.dashboard_user or config.dashboard_pass:
    # AO: half-configured used to silently disable dashboard auth.
    logger.warning(
        "Only one of DASHBOARD_USER / DASHBOARD_PASS is set — dashboard "
        "authentication is DISABLED until both are provided"
    )

app.include_router(mcp_router, prefix="/mcp")


def _validated_slug(slug: str) -> str:
    try:
        return validate_slug(slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


T = TypeVar("T")


async def _run(coro: Awaitable[T], *, op: str) -> T:
    """One error ladder for every engine call.

    Our own validation messages (ValueError) are safe to echo. Internal
    failures are logged with the traceback and returned as a generic message —
    the previous detail=str(exc) leaked Qdrant URLs and backend error bodies.
    """
    try:
        return await coro
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (MaintenanceInProgressError, DimensionMismatchError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ConsolidationTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except (LLMError, EmbeddingError) as exc:
        logger.warning("%s: backend unavailable: %s", op, exc)
        raise HTTPException(
            status_code=503, detail="Language model or embedder unavailable"
        ) from exc
    except Exception as exc:
        logger.exception("%s failed", op)
        raise HTTPException(status_code=500, detail=f"Internal error in {op}") from exc


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
    if not config.diagnostics_enabled:
        raise HTTPException(status_code=404, detail="Not found")
    info: dict[str, object] = {"llm_backend": config.llm_backend}
    if config.llm_backend == "ollama":
        info["ollama_base_url"] = config.ollama_base_url
        info["llm_model"] = config.llm_model
        info["embedder_model"] = config.embedder_model
        info["fallback_backend"] = "bundled"
    elif config.llm_backend == "openai":
        info["openai_base_url"] = config.openai_base_url
        info["openai_model"] = config.openai_model
        info["openai_embed_model"] = config.openai_embed_model
        info["fallback_backend"] = None
    else:
        info["bundled_model_path"] = config.bundled_model_path
        info["bundled_embed_model"] = config.bundled_embed_model
        info["bundled_threads"] = config.bundled_threads
        info["fallback_backend"] = None
    info["embedding_dimensions"] = backend.embedding_dimensions
    # Booleans/names only — never the key itself. The dashboard's Config tab
    # read these keys for months and they were never returned (dead tiles).
    info["degraded"] = bool(getattr(backend, "is_degraded", False))
    info["embed_degraded"] = bool(getattr(backend, "embed_degraded", False))
    info["api_key_enabled"] = bool(config.api_key)
    info["dashboard_auth_enabled"] = bool(config.dashboard_user and config.dashboard_pass)
    info["rerank_enabled"] = config.rerank_enabled
    info["rerank_model"] = config.rerank_model
    return info


@app.get("/api/v1/projects")
async def list_projects() -> list[ProjectInfo]:
    return await _run(engine.list_projects(), op="list_projects")


@app.get("/api/v1/projects/{slug}/memories")
async def get_memories(
    response: Response,
    slug: str = Path(...),
    limit: int = Query(default=50, ge=1, le=1000),
    offset: str | None = Query(default=None),
) -> list[MemoryRecord]:
    slug = _validated_slug(slug)
    cursor: str | None = None
    if offset:
        try:
            cursor = validate_memory_id(offset)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    records, next_offset = await _run(
        engine.list_page(slug, limit=limit, offset=cursor), op="list_memories"
    )
    # Body stays a plain list for dashboard/CLI compatibility; the cursor rides
    # in a header. Absent header = last page.
    if next_offset:
        response.headers["X-Next-Offset"] = next_offset
    return records


@app.post("/api/v1/projects/{slug}/memories")
async def create_memory(
    body: MemoryCreate,
    slug: str = Path(...),
    user_id: str = Query(default="default", max_length=63),
) -> dict[str, object]:
    slug = _validated_slug(slug)
    try:
        user_id = validate_user_id(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    ids = await _run(
        engine.add(slug, user_id, [body.text], body.metadata or None), op="add_memory"
    )
    return {"stored": len(ids), "ids": ids}


@app.post("/api/v1/projects/{slug}/search")
async def search_memories(
    body: SearchRequest,
    slug: str = Path(...),
) -> list[MemoryRecord]:
    slug = _validated_slug(slug)
    return await _run(engine.search(slug, body.query, top_k=body.top_k), op="search")


@app.delete("/api/v1/projects/{slug}/memories/{memory_id}")
async def remove_memory(
    slug: str = Path(...),
    memory_id: str = Path(...),
) -> dict[str, bool]:
    slug = _validated_slug(slug)
    deleted = await _run(engine.delete(slug, memory_id), op="delete_memory")
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"deleted": True}


@app.delete("/api/v1/projects/{slug}/memories")
async def remove_all_memories(slug: str = Path(...)) -> dict[str, int]:
    slug = _validated_slug(slug)
    count = await _run(engine.delete_all(slug), op="delete_all")
    return {"deleted": count}


@app.delete("/api/v1/projects/{slug}")
async def remove_project(slug: str = Path(...)) -> dict[str, bool]:
    slug = _validated_slug(slug)
    removed = await _run(engine.delete_project(slug), op="delete_project")
    if not removed:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"deleted": True}


@app.post("/api/v1/projects/{slug}/reembed")
async def reembed_memories(slug: str = Path(...)) -> dict[str, int]:
    slug = _validated_slug(slug)
    count = await _run(engine.reembed_all(slug), op="reembed")
    return {"reembedded": count}


@app.post("/api/v1/projects/{slug}/cleanup")
async def cleanup_memories(slug: str = Path(...)) -> dict[str, int]:
    slug = _validated_slug(slug)
    return await _run(engine.cleanup_text(slug), op="cleanup")


@app.post("/api/v1/projects/{slug}/consolidate")
async def consolidate_memories(
    slug: str = Path(...),
    threshold: float = Query(default=0.75, ge=0.5, le=1.0),
    dry_run: bool = Query(default=False),
) -> dict[str, object]:
    slug = _validated_slug(slug)
    return await _run(
        engine.consolidate(slug, similarity_threshold=threshold, dry_run=dry_run),
        op="consolidate",
    )


@app.get("/api/v1/diagnostics")
async def get_diagnostics() -> dict[str, object]:
    if not config.diagnostics_enabled:
        raise HTTPException(status_code=404, detail="Diagnostics disabled")
    try:
        projects = await engine.list_projects()
        stats.record_daily_snapshot({p.slug: p.memory_count for p in projects})
    except Exception:
        logger.debug("On-demand daily snapshot skipped", exc_info=True)
    return stats.snapshot()


@app.get("/api/v1/projects/{slug}/diagnostics")
async def get_project_diagnostics(slug: str = Path(...)) -> dict[str, object]:
    slug = _validated_slug(slug)
    if not config.diagnostics_enabled:
        raise HTTPException(status_code=404, detail="Diagnostics disabled")
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
