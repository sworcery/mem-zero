from __future__ import annotations

import contextvars
import json
import logging
from datetime import datetime, timezone
from typing import Annotated

import anyio
from fastapi import APIRouter, Request, Response
from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http import StreamableHTTPServerTransport
from pydantic import Field

from .config import validate_slug, validate_user_id
from .memory_engine import MemoryEngine, validate_memory_id

logger = logging.getLogger(__name__)

client_name_var: contextvars.ContextVar[str] = contextvars.ContextVar("client_name")
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_id")

mcp = FastMCP("mem-zero")
_engine: MemoryEngine | None = None


def set_engine(engine: MemoryEngine) -> None:
    global _engine
    _engine = engine


def _get_engine() -> MemoryEngine:
    if _engine is None:
        raise RuntimeError("MemoryEngine not initialized")
    return _engine


def _get_context() -> tuple[str, str]:
    project = client_name_var.get(None)
    user = user_id_var.get(None)
    if not project or not user:
        raise RuntimeError("Missing project or user context")
    return project, user


def _format_record(record: dict) -> dict:
    for key in ("created_at", "updated_at"):
        ts = record.get(key)
        if isinstance(ts, (int, float)) and ts > 0:
            record[key] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    return record


@mcp.tool(
    description=(
        "Store notes for the current project's long-term memory. Facts are "
        "extracted, deduplicated against existing memories, and embedded; "
        "conversational filler is dropped, so `stored` may legitimately be 0. "
        "Call this at natural checkpoints (a decision made, a bug fixed, a "
        "gotcha discovered), not continuously. Send focused notes — decisions "
        "and their reasoning, workarounds, dead ends, preferences — rather than "
        "whole transcripts; very long inputs are truncated by the model."
    )
)
async def add_memories(text: Annotated[str, Field(min_length=1, max_length=50000)]) -> str:
    project, user = _get_context()
    engine = _get_engine()
    ids = await engine.add(project, user, [text])
    return json.dumps({"stored": len(ids), "ids": ids})


@mcp.tool(
    description=(
        "Semantic (meaning-based, not keyword) search over this project's "
        "memories. Use it at the start of a task to recall prior decisions and "
        "gotchas before reading code — it surfaces things the code does not "
        "explain. Phrase the query as the topic you care about, not exact "
        "words. Returns up to top_k results, each with a 0-1 relevance score."
    )
)
async def search_memory(
    query: Annotated[str, Field(min_length=1, max_length=2000)],
    top_k: Annotated[int, Field(ge=1, le=100)] = 10,
) -> str:
    project, _ = _get_context()
    engine = _get_engine()
    results = await engine.search(project, query, top_k=top_k)
    return json.dumps([_format_record(r.model_dump()) for r in results])


@mcp.tool(
    description=(
        "Page through this project's stored memories (an inventory, in storage "
        "order — not ranked). Prefer search_memory unless you need to see "
        "everything. Returns {memories, total, truncated}: `total` is the "
        "project's full count and `truncated` is true when more exist beyond "
        "`limit`."
    )
)
async def list_memories(limit: Annotated[int, Field(ge=1, le=1000)] = 100) -> str:
    project, _ = _get_context()
    engine = _get_engine()
    results = await engine.list_all(project, limit=limit)
    total = await engine.count_memories(project)
    return json.dumps(
        {
            "memories": [_format_record(r.model_dump()) for r in results],
            "total": total,
            "truncated": total > len(results),
        }
    )


@mcp.tool(
    description=(
        "Delete specific memories by id (get ids from search_memory or "
        "list_memories). Every id is validated before any deletion happens, so "
        "one bad id fails the whole call rather than leaving a partial delete. "
        "Returns the number actually deleted; ids that did not exist are not "
        "counted. Ask the user before calling this."
    )
)
async def delete_memories(memory_ids: list[str]) -> str:
    project, _ = _get_context()
    engine = _get_engine()
    # Validate every id before deleting any, so one bad id fails the whole
    # call cleanly instead of leaving a partially-applied delete.
    for mid in memory_ids:
        validate_memory_id(mid)
    deleted = 0
    for mid in memory_ids:
        if await engine.delete(project, mid):
            deleted += 1
    return json.dumps({"deleted": deleted})


@mcp.tool(
    description=(
        "IRREVERSIBLE: delete every memory in the current project. This is a "
        "two-step tool. Call it with no arguments first to see how many "
        "memories would be destroyed; then, only with explicit user approval, "
        "call it again with confirm=true to execute. Never call with "
        "confirm=true on your own initiative."
    )
)
async def delete_all_memories(confirm: bool = False) -> str:
    project, _ = _get_context()
    engine = _get_engine()
    if not confirm:
        count = await engine.count_memories(project)
        # ValueError -> FastMCP returns an isError result the model can read.
        raise ValueError(
            f"Refusing to delete {count} memories in project {project!r} without "
            f"confirm=true. This cannot be undone — get explicit user approval, "
            f"then call again with confirm=true."
        )
    count = await engine.delete_all(project)
    return json.dumps({"deleted": count})


mcp_router = APIRouter()


@mcp_router.api_route(
    "/{client_name}/http/{user_id}",
    # POST only: this is a stateless JSON server, so the optional GET SSE
    # stream would just hang and buffer forever. GET/DELETE now 405 cleanly.
    methods=["POST"],
)
async def handle_mcp(request: Request, client_name: str, user_id: str) -> Response:
    try:
        validate_slug(client_name)
    except ValueError:
        return Response(
            content=f"Invalid project slug: {client_name!r}",
            status_code=400,
        )

    try:
        validate_user_id(user_id)
    except ValueError:
        return Response(
            content=f"Invalid user_id: {user_id!r}",
            status_code=400,
        )

    client_token = client_name_var.set(client_name)
    user_token = user_id_var.set(user_id)

    response_started = False
    response_status = 200
    response_headers: list[tuple[bytes, bytes]] = []
    response_body = bytearray()

    async def capture_send(message: dict) -> None:
        nonlocal response_started, response_status
        if message["type"] == "http.response.start":
            response_started = True
            response_status = message["status"]
            response_headers.extend(message.get("headers", []))
        elif message["type"] == "http.response.body":
            response_body.extend(message.get("body", b""))

    try:
        transport = StreamableHTTPServerTransport(
            mcp_session_id=None,
            is_json_response_enabled=True,
        )

        async with anyio.create_task_group() as tg:

            async def run_server(*, task_status=anyio.TASK_STATUS_IGNORED) -> None:
                async with transport.connect() as (read_stream, write_stream):
                    task_status.started()
                    await mcp._mcp_server.run(
                        read_stream,
                        write_stream,
                        mcp._mcp_server.create_initialization_options(),
                        stateless=True,
                    )

            await tg.start(run_server)
            await transport.handle_request(request.scope, request.receive, capture_send)
            await transport.terminate()
            tg.cancel_scope.cancel()

    finally:
        client_name_var.reset(client_token)
        user_id_var.reset(user_token)

    if not response_started:
        # The transport never emitted http.response.start — returning the
        # initialised 200 with an empty body would tell the client "success".
        return Response(
            content=json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32603, "message": "MCP transport produced no response"},
                }
            ),
            status_code=502,
            media_type="application/json",
        )
    return Response(
        content=bytes(response_body),
        status_code=response_status,
        headers={k.decode(): v.decode() for k, v in response_headers},
    )
