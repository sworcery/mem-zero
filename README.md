<p align="center">
  <img src="banner.png" alt="mem-zero" />
</p>

Self-hosted memory server for AI coding assistants. Store, search, and manage persistent context across sessions — so your tools remember what happened last week without stuffing everything into the context window.

Each project gets its own isolated vector collection. When you store a memory, an LLM extracts atomic facts, deduplicates against existing memories, and embeds them for semantic search. The result is a clean, searchable memory store per project that any MCP client or HTTP-capable tool can query.

## Why?

AI coding assistants forget everything between sessions. Every new conversation starts from scratch — you re-explain decisions, re-discover bugs, repeat yourself. mem-zero fixes that by giving your assistant persistent project memory:

- **Session 50 knows what session 1 figured out** — bugs fixed, decisions made, preferences learned
- **Selective retrieval** — semantic search pulls in only what's relevant, not entire conversation logs
- **Project-isolated** — memories from one project never leak into another
- **Self-contained** — runs as a single Docker container with an embedded LLM, no external dependencies required

## Quick start

```bash
docker run -d \
  --name mem-zero \
  -p 8765:8765 \
  -v mem-zero-data:/mem-zero/storage \
  ghcr.io/sworcery/mem-zero:latest
```

That's it. The bundled LLM handles fact extraction and embeddings out of the box. First startup downloads models (~2 GB) and takes a few minutes — subsequent starts are fast.

## Connecting your tools

### Claude Code (MCP)

```bash
claude mcp add mem-zero --transport http \
  "http://your-host:8765/mcp/your-project-slug/http/your-user-id" \
  -s local
```

### CLI

```bash
pip install mem-zero
mem-zero-cli --url http://your-host:8765 --api-key your-key projects
```

The CLI provides direct terminal access to all memory operations:

```bash
# List projects
mem-zero-cli projects

# Add a memory
mem-zero-cli add my-project "Chose PostgreSQL over Redis for session storage"

# Pipe text from stdin
echo "User prefers dark mode" | mem-zero-cli add my-project -

# Search
mem-zero-cli search my-project "database decision"

# List memories
mem-zero-cli list my-project

# Export/import for backup and migration
mem-zero-cli export my-project -o backup.json
mem-zero-cli import backup.json --project new-project

# Health check
mem-zero-cli health

# Diagnostics
mem-zero-cli stats
mem-zero-cli stats --project my-project
```

All commands support `--json` for machine-readable output. Run `mem-zero-cli --help` for full usage.

### Any MCP client (Claude Desktop, Cursor, Windsurf, etc.)

Add the MCP server URL to your client's configuration:

```
http://your-host:8765/mcp/your-project-slug/http/your-user-id
```

### REST API (any tool)

Anything that can make HTTP requests can use mem-zero directly:

```bash
# Store a memory
curl -X POST http://your-host:8765/api/v1/projects/my-project/memories \
  -H "Content-Type: application/json" \
  -d '{"text": "Switched from Redis to PostgreSQL for session storage because we need ACID transactions"}'

# Search memories
curl -X POST http://your-host:8765/api/v1/projects/my-project/search \
  -H "Content-Type: application/json" \
  -d '{"query": "database decision", "top_k": 5}'
```

The project slug must start with a letter or number, followed by lowercase alphanumeric characters, hyphens, or underscores (1-63 chars). Each unique slug creates an isolated collection.

## How it works

1. Text comes in via MCP or REST API
2. LLM extracts atomic facts (e.g. "User prefers Python over R")
3. Each fact is checked against existing memories for duplicates
4. Novel facts are embedded and stored; duplicates are merged or skipped

## LLM backends

mem-zero supports three LLM backends for fact extraction and deduplication.

| Backend | LLM | Embeddings | Setup |
|---------|-----|------------|-------|
| **ollama** (recommended) | Any Ollama model | Any Ollama embedding model | Set `OLLAMA_BASE_URL` |
| **bundled** (default) | Qwen2.5-3B (built-in, CPU) | nomic-embed-text via fastembed | Zero config — just run the container |
| **openai** (beta) | Any OpenAI-compatible API | Any OpenAI-compatible embeddings | Set `OPENAI_API_KEY` |

**Auto-detection:** If `LLM_BACKEND` is not set, the backend is chosen automatically:
- `OPENAI_API_KEY` present → `openai`
- `OLLAMA_BASE_URL` present → `ollama`
- Neither → `bundled`

### Using with Ollama (recommended)

Ollama gives the best results. The bundled 3B model works for basic use, but a 7B+ model on GPU produces significantly better fact extraction. If you have a dedicated GPU, `qwen2.5:14b` is the sweet spot for quality vs. resource usage.

```bash
docker run -d \
  --name mem-zero \
  -p 8765:8765 \
  -v mem-zero-data:/mem-zero/storage \
  -e OLLAMA_BASE_URL=http://your-ollama-host:11434 \
  -e LLM_MODEL=qwen2.5:14b \
  ghcr.io/sworcery/mem-zero:latest
```

**Fallback:** When using Ollama, if the server is unreachable, requests automatically fall back to the bundled 3B model. The fallback is lazy — it only loads into memory on the first failure. Extraction quality is reduced but the service stays available.

### Using the bundled backend

The bundled backend runs a quantized Qwen2.5-3B model on CPU with no external dependencies. It handles embeddings well and provides basic fact extraction. For better extraction quality, use Ollama or an OpenAI-compatible API.

### Using with OpenAI-compatible APIs (beta)

```bash
docker run -d \
  --name mem-zero \
  -p 8765:8765 \
  -v mem-zero-data:/mem-zero/storage \
  -e OPENAI_API_KEY=sk-... \
  ghcr.io/sworcery/mem-zero:latest
```

Works with any OpenAI-compatible API (OpenAI, Groq, Together, etc.) by setting `OPENAI_BASE_URL`. This backend has not been extensively tested — if you encounter issues, please report them.

## Web dashboard

A management UI is served at the root URL (`http://your-host:8765/`). From the dashboard you can:

- Monitor system health, uptime, and live performance charts
- Browse all projects and their memory counts
- View, search, and delete memories per project
- Consolidate similar memory fragments
- Delete entire projects
- Add new memories manually

Enable `DIAGNOSTICS_ENABLED=true` to see performance metrics, accuracy stats, score distributions, and error tracking on the home page.

Optionally protect it with basic auth via `DASHBOARD_USER` and `DASHBOARD_PASS`.

## Authentication

Set `API_KEY` to protect all API and MCP endpoints. When set, requests must include the key as a Bearer token:

```bash
curl -H "Authorization: Bearer your-api-key" \
  http://your-host:8765/api/v1/projects
```

For MCP clients, add the header to your client config. In Claude Code's `.mcp.json`:

```json
{
  "mem-zero": {
    "type": "http",
    "url": "http://your-host:8765/mcp/my-project/http/my-user",
    "headers": {
      "Authorization": "Bearer your-api-key"
    }
  }
}
```

A query parameter (`?api_key=your-key`) is also accepted for clients that can't set headers.

If `API_KEY` is not set, all endpoints are open — suitable for trusted networks.

The dashboard has its own basic auth (`DASHBOARD_USER`/`DASHBOARD_PASS`) since browsers need a login prompt rather than Bearer tokens.

## MCP tools

| Tool | Description |
|------|-------------|
| `add_memories(text)` | Extract and store facts from text |
| `search_memory(query, top_k)` | Semantic search within the project |
| `list_memories()` | List all memories for the project |
| `delete_memories(memory_ids)` | Delete specific memories by ID |
| `delete_all_memories()` | Delete all memories for the project |

## Export and import

Back up project memories to a JSON file, or migrate between servers:

```bash
# Export via CLI
mem-zero-cli export my-project -o backup.json

# Import to same or different server
mem-zero-cli import backup.json
mem-zero-cli import backup.json --project different-project

# Export via REST
curl http://your-host:8765/api/v1/projects/my-project/memories?limit=1000 > backup.json
```

The export format includes project metadata, timestamps, and all memory content. Importing re-processes text through the LLM pipeline (extraction and dedup), so imported memories are properly deduplicated against existing content.

## Best practices

mem-zero supplements conversations — it's not a transcript. Store things a future session would need that aren't obvious from reading the code or git history.

**Search first.** At the start of every conversation, search mem-zero for prior context. A well-maintained memory store means you never start from scratch.

**Store decisions, not play-by-play.** "Chose skopeo over docker push because the local registry uses a self-signed cert" is useful. "Updated line 42 in server.py" is not — that's what `git log` is for.

**Store dead ends.** If you spend 30 minutes debugging something that turned out to be a red herring, store that. It prevents future sessions from going down the same path.

**Quality over quantity.** Each memory should be a complete, self-contained statement. One memory per logical change or decision — not one per file touched. Fragments like "Root cause" or "Solution" without context are useless noise.

**Let the code speak.** Don't store function signatures, file structure, what a method does, or test results. The codebase is the authoritative source for those. Store the *why*, not the *what*.

For Claude Code, add instructions to your `CLAUDE.md` telling the assistant to use mem-zero proactively. Without explicit instructions, most assistants won't store memories on their own.

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
POST   /api/v1/projects/{slug}/reembed            — regenerate embeddings for all memories
POST   /api/v1/projects/{slug}/cleanup            — fix garbled text and split multi-fact entries
POST   /api/v1/projects/{slug}/consolidate        — merge similar fragments into clean summaries
GET    /api/v1/diagnostics                        — performance and accuracy metrics
```

## Configuration

All settings are via environment variables.

### General

| Variable | Default | Purpose |
|----------|---------|---------|
| `API_KEY` | — | API key for MCP and REST endpoints (disabled if empty) |
| `LLM_BACKEND` | auto-detect | `bundled`, `ollama`, or `openai` |
| `EMBEDDER_DIMENSIONS` | `768` | Vector dimensions |
| `COLLECTION_PREFIX` | `mem-zero` | Qdrant collection name prefix |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8765` | Server port |
| `DASHBOARD_USER` | — | Dashboard login username (auth disabled if empty) |
| `DASHBOARD_PASS` | — | Dashboard login password |
| `DIAGNOSTICS_ENABLED` | `false` | Enable performance and accuracy metrics on the dashboard |

### Bundled backend

| Variable | Default | Purpose |
|----------|---------|---------|
| `BUNDLED_MODEL_PATH` | `/mem-zero/storage/models/qwen2.5-3b-instruct-q4_k_m.gguf` | Path to GGUF model |
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

- **Qdrant** — embedded vector database, data persisted to `/mem-zero/storage`
- **FastAPI** — HTTP server handling MCP transport, REST API, and static dashboard
- **Qwen2.5-3B** — bundled LLM for fact extraction and dedup (CPU-only, ~1.8 GB RAM)
- **fastembed** — bundled embedding model (nomic-embed-text, ~270 MB)

External LLMs (Ollama, OpenAI) are supported as alternatives. When using Ollama, the bundled model serves as an automatic fallback if Ollama is unreachable.

Project isolation is enforced at the Qdrant collection level. Each project slug maps to `{prefix}_{slug}`, and all queries are scoped to a single collection.

## Unraid

An Unraid Docker template is included at `unraid-template.xml`. Install it through Community Applications or manually add the template to your Docker configuration.

## License

AGPL-3.0 — see [LICENSE](LICENSE).
