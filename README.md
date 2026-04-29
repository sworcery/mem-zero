# mem-zero

Project-isolated MCP memory server for Claude Code. Each project gets its own Qdrant vector collection — no cross-pollination between projects.

When you store a memory, the text is sent through an LLM to extract atomic facts, deduplicate against existing memories, and embed for semantic search. The result is a clean, searchable memory store per project.

## How it works

Claude Code projects connect via MCP at `/mcp/{project-slug}/http/{user-id}`. The `project-slug` maps to a dedicated Qdrant collection (`mem0_{slug}`), so memories written by one project are invisible to another.

**Memory pipeline:**

1. Text comes in via MCP `add_memories` or REST API
2. LLM extracts atomic facts (e.g. "User prefers Python over R")
3. Each fact is checked against existing memories for duplicates
4. Novel facts are embedded and stored; duplicates are merged or skipped

## LLM backends

mem-zero supports three LLM backends. The default is **bundled** — no external dependencies needed.

| Backend | LLM | Embeddings | Setup |
|---------|-----|------------|-------|
| **bundled** (default) | Qwen2.5-3B (built-in) | nomic-embed-text via fastembed | Zero config — just run the container |
| **ollama** | Any Ollama model | Any Ollama embedding model | Set `OLLAMA_BASE_URL` |
| **openai** | Any OpenAI-compatible API | Any OpenAI-compatible embeddings | Set `OPENAI_API_KEY` |

**Auto-detection:** If `LLM_BACKEND` is not set, the backend is chosen automatically:
- `OPENAI_API_KEY` present → `openai`
- `OLLAMA_BASE_URL` present → `ollama`
- Neither → `bundled`

**Fallback:** When using `ollama`, if the Ollama server is unreachable, requests automatically fall back to the bundled model. The fallback is lazy — the bundled model only loads into memory on the first failure.

## Quick start (Docker)

```bash
# Self-contained — no external LLM needed
docker run -d \
  --name mem-zero \
  -p 8765:8765 \
  -v mem-zero-data:/mem0/storage \
  192.168.1.10:5000/mem-zero:dev
```

With Ollama:

```bash
docker run -d \
  --name mem-zero \
  -p 8765:8765 \
  -v mem-zero-data:/mem0/storage \
  -e OLLAMA_BASE_URL=http://your-ollama-host:11434 \
  -e LLM_MODEL=qwen2.5:7b \
  192.168.1.10:5000/mem-zero:dev
```

## Connecting Claude Code

Add the MCP server to your project:

```bash
claude mcp add mem-zero --transport http \
  "http://your-host:8765/mcp/your-project-slug/http/your-user-id" \
  -s local
```

The project slug must be lowercase alphanumeric with hyphens or underscores (1-63 chars). Each unique slug creates an isolated collection.

## Web dashboard

A management UI is served at the root URL (`http://your-host:8765/`). From the dashboard you can:

- Browse all projects and their memory counts
- View, search, and delete memories per project
- Delete entire projects
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

### General

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_BACKEND` | auto-detect | `bundled`, `ollama`, or `openai` |
| `EMBEDDER_DIMENSIONS` | `768` | Vector dimensions |
| `COLLECTION_PREFIX` | `mem0` | Qdrant collection name prefix |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8765` | Server port |
| `DASHBOARD_USER` | — | Dashboard login username (auth disabled if empty) |
| `DASHBOARD_PASS` | — | Dashboard login password |

### Bundled backend

| Variable | Default | Purpose |
|----------|---------|---------|
| `BUNDLED_MODEL_PATH` | `/mem0/storage/models/qwen2.5-3b-instruct-q4_k_m.gguf` | Path to GGUF model |
| `BUNDLED_EMBED_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | fastembed model name |
| `BUNDLED_THREADS` | `4` | CPU threads for inference |

### Ollama backend

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama API URL |
| `LLM_MODEL` | `qwen2.5:7b` | Model for fact extraction and dedup |
| `EMBEDDER_MODEL` | `nomic-embed-text` | Embedding model |

### OpenAI backend

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | API key (required) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API base URL |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | Embedding model |

### Qdrant

| Variable | Default | Purpose |
|----------|---------|---------|
| `QDRANT_HOST` | `127.0.0.1` | Qdrant host (bundled) |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_URL` | — | Full Qdrant URL (overrides host/port) |
| `QDRANT_API_KEY` | — | Qdrant API key (if using external) |

## Architecture

The container bundles everything into a single image using s6-overlay for process supervision:

- **Qdrant** — embedded vector database, data persisted to `/mem0/storage`
- **FastAPI** — HTTP server handling MCP transport, REST API, and static dashboard
- **Qwen2.5-3B** — bundled LLM for fact extraction and dedup (CPU-only, ~1.8 GB RAM)
- **fastembed** — bundled embedding model (nomic-embed-text, ~270 MB)

External LLMs (Ollama, OpenAI) are supported as alternatives. When using Ollama, the bundled model serves as an automatic fallback if Ollama is unreachable.

Project isolation is enforced at the Qdrant collection level. Each project slug maps to `{prefix}_{slug}`, and all queries are scoped to a single collection.
