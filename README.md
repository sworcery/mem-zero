# mem-zero

Project-isolated MCP memory server for Claude Code. Each project gets its own Qdrant vector collection — no cross-pollination between projects.

## How it works

Claude Code projects connect via MCP at `/mcp/{project-slug}/http/{user-id}`. The `project-slug` from the URL maps to a dedicated Qdrant collection (`mem0_{slug}`), so memories written by one project can never be read by another.

**Stack:** FastAPI + Qdrant (bundled) + Ollama (external, for embeddings)

## Quick start (Docker)

```bash
docker run -d \
  --name mem-zero \
  -p 8765:8765 \
  -v mem-zero-data:/mem0/storage \
  -e OLLAMA_BASE_URL=http://your-ollama-host:11434 \
  192.168.1.10:5000/mem-zero:dev
```

Then in your Claude Code project:

```bash
claude mcp add mem0 --transport http \
  "http://your-host:8765/mcp/your-project-slug/http/john" \
  -s local
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `add_memories(text)` | Store a memory in the current project |
| `search_memory(query, top_k)` | Semantic search within the current project |
| `list_memories()` | List all memories for the current project |
| `delete_memories(memory_ids)` | Delete specific memories by ID |
| `delete_all_memories()` | Delete all memories for the current project |

## REST API

For debugging and direct access:

```
GET  /health                              — health check
GET  /api/v1/projects                     — list all projects
GET  /api/v1/projects/{slug}/memories     — list memories
POST /api/v1/projects/{slug}/memories     — add memory
POST /api/v1/projects/{slug}/search       — search memories
DELETE /api/v1/projects/{slug}/memories/{id} — delete one
DELETE /api/v1/projects/{slug}/memories   — delete all
```

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `QDRANT_HOST` | `127.0.0.1` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_URL` | — | Full Qdrant URL (overrides host/port) |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama API |
| `EMBEDDER_MODEL` | `nomic-embed-text` | Embedding model |
| `EMBEDDER_DIMENSIONS` | `768` | Vector dimensions |
| `COLLECTION_PREFIX` | `mem0` | Qdrant collection name prefix |
| `PORT` | `8765` | Server port |

## Migration from mem0-aio

If you have existing memories in a mem0-aio instance:

```bash
# Preview what would be migrated
docker exec mem-zero python -m mem_zero.migration --source-collection mem0 --dry-run

# Run the migration
docker exec mem-zero python -m mem_zero.migration --source-collection mem0
```

This reads the old collection, groups memories by their `app_name` metadata, and copies them into per-project collections. Non-destructive — the old collection is left untouched.
