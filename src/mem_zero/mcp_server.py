from __future__ import annotations

import contextvars
import json
import logging

import anyio
from fastapi import APIRouter, Request, Response
from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http import StreamableHTTPServerTransport

from .config import validate_slug
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


@mcp.tool(description="Store a memory for the current project.")
async def add_memories(text: str) -> str:
    project, user = _get_context()
    engine = _get_engine()
    ids = await engine.add(project, user, [text])
    return json.dumps({"stored": len(ids), "ids": ids})


@mcp.tool(description="Semantic search across memories in the current project.")
async def search_memory(query: str, top_k: int = 10) -> str:
    project, _ = _get_context()
    engine = _get_engine()
    results = await engine.search(project, query, top_k=top_k)
    return json.dumps([r.model_dump() for r in results])


@mcp.tool(description="List all memories stored for the current project.")
async def list_memories() -> str:
    project, _ = _get_context()
    engine = _get_engine()
    results = await engine.list_all(project)
    return json.dumps([r.model_dump() for r in results])


@mcp.tool(description="Delete specific memories by their IDs.")
async def delete_memories(memory_ids: list[str]) -> str:
    project, _ = _get_context()
    engine = _get_engine()
    deleted = 0
    for mid in memory_ids:
        validate_memory_id(mid)
        if await engine.delete(project, mid):
            deleted += 1
    return json.dumps({"deleted": deleted})


@mcp.tool(description="Delete all memories for the current project.")
async def delete_all_memories() -> str:
    project, _ = _get_context()
    engine = _get_engine()
    count = await engine.delete_all(project)
    return json.dumps({"deleted": count})


mcp_router = APIRouter()


@mcp_router.api_route(
    "/{client_name}/http/{user_id}",
    methods=["POST", "GET", "DELETE"],
)
async def handle_mcp(request: Request, client_name: str, user_id: str) -> Response:
    try:
        validate_slug(client_name)
    except ValueError:
        return Response(
            content=f"Invalid project slug: {client_name!r}",
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

    return Response(
        content=bytes(response_body),
        status_code=response_status,
        headers={k.decode(): v.decode() for k, v in response_headers},
    )
