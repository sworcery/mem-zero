# AGENTS.md

Guidance for coding agents working on mem-zero itself. (To connect an agent to a
running mem-zero server for project memory, see the README instead.)

## Project overview

A self-hosted MCP memory server. FastAPI serves a REST API, an MCP endpoint, and a
static dashboard; Qdrant stores one vector collection per project; a pluggable LLM
backend does fact extraction, deduplication, and embeddings.

- `src/mem_zero/server.py` — FastAPI app, auth middleware, REST routes
- `src/mem_zero/mcp_server.py` — MCP tools and the ASGI transport bridge
- `src/mem_zero/memory_engine.py` — the core: extract → dedup → embed → store, plus search and maintenance
- `src/mem_zero/backends.py` — Ollama / OpenAI-compatible / bundled backends, with fallback and a circuit breaker
- `src/mem_zero/stats.py` — diagnostics counters, persisted to JSON
- `src/mem_zero/static/index.html` — the entire dashboard, vanilla JS, no build step

## Setup and commands

```bash
pip install -e ".[test]"

PYTHONPATH=src python3 -m pytest -q     # full suite
python3 -m ruff check src/ tests/       # lint
python3 -m mypy src/mem_zero            # types
```

All three must be run before committing.

The Docker image installs the pinned set in `requirements.txt` (generated,
do not hand-edit). After changing `pyproject.toml` dependencies, refresh it
with `make lock` (needs `uv`: `pip install --user uv`) and commit both.
`pyproject.toml` deliberately keeps floating ranges for library and CI-test
consumers; only the image is locked. The suite is fast (~7s) and fully mocked —
no network, no running Qdrant or Ollama required.

pytest and ruff must be clean. mypy currently reports ~46 pre-existing errors
(mostly `union-attr` on Qdrant payloads, which are safe because every read uses
`with_payload=True`); treat that count as the baseline and don't add to it.

## Conventions

- **Version lives in exactly one place**: `src/mem_zero/__init__.py`. `pyproject.toml`
  reads it via hatchling's dynamic version and `server.py` imports it. Never hardcode
  it elsewhere.
- Local commits use a 4th version segment (`0.1.41.1`); promote to a clean 3-segment
  version when publishing.
- Line length 100, ruff rules `E,F,I,N,UP,B,SIM`, `from __future__ import annotations`
  at the top of every module.
- Tests mirror the module they cover (`tests/unit/test_<module>.py`), use `AsyncMock`
  backends and `MagicMock` Qdrant clients, and `asyncio_mode = "auto"` is set.
- Comments explain *why*, not *what*. Don't narrate the code.

## Gotchas worth knowing before you change things

- **`mcp<2` is load-bearing.** mcp 2.0 restructured the package and removed
  `mcp.server.fastmcp`, which `mcp_server.py` imports. Don't relax that pin.
- **The MCP route is POST-only** by design. This is a stateless JSON server, so the
  optional GET SSE stream would hang and buffer forever. `tests/unit/test_mcp_server.py`
  pins both the 405 and the full client handshake — external clients depend on it.
- **Never put `minLength` in a JSON schema sent to Ollama.** It grammar-enforces the
  constraint and pads short strings with degenerate text. Validate lengths in Python.
- **Destructive maintenance must embed and upsert before deleting.** `reembed_all`
  and `consolidate` both compute everything up front so an LLM or Qdrant failure
  can't leave a collection emptied. Preserve that ordering.
- **Heavy imports are lazy on purpose.** `llama_cpp` and `fastembed` are imported
  inside `BundledBackend.__init__` and the reranker loader so CI can skip them; the
  CI job installs deps derived from `pyproject.toml` minus those two.
- Backends must stay usable independently: chat and embeddings can come from
  different providers, and `FallbackBackend` loads the bundled model off the event
  loop via `asyncio.to_thread`.

## Making changes

Match the surrounding style, keep changes scoped to the request, and add a
regression test that fails before your fix and passes after. If you touch the
dashboard, verify the inline script still parses (`node --check` on the extracted
`<script>` block) — there's no build step to catch syntax errors.
