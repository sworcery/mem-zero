# mem-zero

Project-isolated MCP memory server for Claude Code. Each project gets its own Qdrant vector collection — no cross-pollination between projects.

When you store a memory, the text is sent through an LLM (Ollama) to extract atomic facts, deduplicate against existing memories, and embed for semantic search. The result is a clean, searchable memory store per project.

## How it works

Claude Code projects connect via MCP at `/mcp/{project-slug}/http/{user-id}`. The `project-slug` maps to a dedicated Qdrant collection (`mem0_{slug}`), so memories written by one project are invisible to another.

**Memory pipeline:**

1. Text comes in via MCP `add_memories` or REST API
2. LLM extracts atomic facts (e.g. "User prefers Python over R")
3. Each fact is checked against existing memories for duplicates
4. Novel facts are embedded and stored; duplicates are merged or skipped

**Stack:** FastAPI + Qdrant (bundled) + Ollama (external) + s6-overlay (process supervision)

## Quick start (Docker)

```bash
docker run -d \
  --name mem-zero \
  -p 8765:8765 \
  -v mem-zero-data:/mem0/storage \
  -e OLLAMA_BASE_URL=http://your-ollama-host:11434 \
  -e LLM_MODEL=qwen2.5:7b \
  192.168.1.10:5000/mem-zero:dev
```

Or with docker-compose:

```yaml
services:
  mem-zero:
    image: 192.168.1.10:5000/mem-zero:dev
    ports:
      - "8765:8765"
    volumes:
      - mem-zero-storage:/mem0/storage
    environment:
      - OLLAMA_BASE_URL=http://your-ollama-host:11434
      - LLM_MODEL=qwen2.5:7b
    restart: unless-stopped

volumes:
  mem-zero-storage:
```

## Connecting Claude Code

Add the MCP server to your project:

```bash
claude mcp add mem0 --transport http \
  "http://your-host:8765/mcp/your-project-slug/http/your-user-id" \
  -s local
```

The project slug must be lowercase alphanumeric with hyphens or underscores (1-63 chars). Each unique slug creates an isolated collection.

## Web dashboard

A management UI is served at the root URL (`http://your-host:8765/`). From the dashboard you can:

- Browse all projects and their memory counts
- View, search, and delete memories per project
- Add new memories manually

## MCP tools

| Tool | Description |
|------|-------------|
| `add_memories(text)` | Extract and store facts from text |
| `search_memory(query, top_k)` | Semantic search within the project |
| `list_memories()` | List all memories for the project |
| `delete_memories(memory_ids)` | Delete specific memories by ID |
| `delete_all_memories()` | Delete all memories for the project |

## REST API

```
GET    /health                                  — health check
GET    /api/v1/projects                         — list all projects
GET    /api/v1/projects/{slug}/memories          — list memories
POST   /api/v1/projects/{slug}/memories          — add memory {"text": "..."}
POST   /api/v1/projects/{slug}/search            — search {"query": "...", "top_k": 10}
DELETE /api/v1/projects/{slug}/memories/{id}      — delete one
DELETE /api/v1/projects/{slug}/memories           — delete all memories
DELETE /api/v1/projects/{slug}                    — delete entire project
```

## Configuration

All settings are via environment variables.

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama API URL |
| `LLM_MODEL` | `qwen2.5:7b` | Model for fact extraction and dedup |
| `EMBEDDER_MODEL` | `nomic-embed-text` | Embedding model |
| `EMBEDDER_DIMENSIONS` | `768` | Vector dimensions |
| `QDRANT_HOST` | `127.0.0.1` | Qdrant host (bundled) |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_URL` | — | Full Qdrant URL (overrides host/port) |
| `QDRANT_API_KEY` | — | Qdrant API key (if using external) |
| `COLLECTION_PREFIX` | `mem0` | Qdrant collection name prefix |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8765` | Server port |

## Architecture

The container bundles everything into a single image using s6-overlay for process supervision:

- **Qdrant** — embedded vector database, data persisted to `/mem0/storage`
- **FastAPI** — HTTP server handling MCP transport, REST API, and static dashboard
- **Ollama** — external dependency for LLM inference and embeddings (not bundled)

Project isolation is enforced at the Qdrant collection level. Each project slug maps to `{prefix}_{slug}`, and all queries are scoped to a single collection.

## Migration from mem0-aio

If you have existing memories in a mem0-aio instance:

```bash
# Preview what would be migrated
docker exec mem-zero python -m mem_zero.migration --source-collection mem0 --dry-run

# Run the migration
docker exec mem-zero python -m mem_zero.migration --source-collection mem0
```

Reads the old collection, groups memories by `app_name` metadata, and copies them into per-project collections. Non-destructive — the source collection is left untouched.
