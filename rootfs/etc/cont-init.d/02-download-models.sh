#!/usr/bin/env bash
set -euo pipefail

# Pick up any paths 01-bootstrap.sh restored (Unraid can blank template fields).
[[ -f /var/run/mem-zero.env ]] && source /var/run/mem-zero.env

MODEL_DIR="/mem-zero/storage/models"
GGUF_PATH="${BUNDLED_MODEL_PATH:-$MODEL_DIR/qwen2.5-3b-instruct-q4_k_m.gguf}"
GGUF_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
FASTEMBED_CACHE="${FASTEMBED_CACHE_PATH:-$MODEL_DIR/fastembed}"
EMBED_MODEL="${BUNDLED_EMBED_MODEL:-nomic-ai/nomic-embed-text-v1.5}"

# Resolve the active backend the same way config.py does. When it isn't
# "bundled", the bundled model is only a lazy fallback, so a download failure
# must not be fatal — otherwise an unreachable huggingface.co crash-loops a
# container whose real backend (Ollama/OpenAI) never touches these files.
BACKEND="${LLM_BACKEND:-}"
if [[ -z "$BACKEND" ]]; then
    if [[ -n "${OPENAI_API_KEY:-}" ]]; then
        BACKEND="openai"
    elif [[ -n "${OLLAMA_BASE_URL:-}" ]]; then
        BACKEND="ollama"
    else
        BACKEND="bundled"
    fi
fi

mkdir -p "$MODEL_DIR"

if [[ ! -f "$GGUF_PATH" ]]; then
    echo "[mem-zero] Downloading bundled LLM model (this only happens once)..."
    if curl -fSL --retry 3 -o "${GGUF_PATH}.tmp" "$GGUF_URL"; then
        mv "${GGUF_PATH}.tmp" "$GGUF_PATH"
        echo "[mem-zero] LLM model downloaded: $GGUF_PATH"
    else
        rm -f "${GGUF_PATH}.tmp"
        if [[ "$BACKEND" == "bundled" ]]; then
            echo "[mem-zero] ERROR: could not download the required bundled model" >&2
            exit 1
        fi
        echo "[mem-zero] WARN: fallback model download failed; continuing (backend is '$BACKEND')" >&2
    fi
else
    echo "[mem-zero] LLM model already present: $GGUF_PATH"
fi

export FASTEMBED_CACHE_PATH="$FASTEMBED_CACHE"
if [[ ! -d "$FASTEMBED_CACHE" ]] || [[ -z "$(ls -A "$FASTEMBED_CACHE" 2>/dev/null)" ]]; then
    echo "[mem-zero] Downloading bundled embedding model (this only happens once)..."
    if /opt/venv/bin/python3 -c "from fastembed import TextEmbedding; TextEmbedding('$EMBED_MODEL')"; then
        echo "[mem-zero] Embedding model downloaded to: $FASTEMBED_CACHE"
    elif [[ "$BACKEND" == "bundled" ]]; then
        echo "[mem-zero] ERROR: could not download the required embedding model" >&2
        exit 1
    else
        echo "[mem-zero] WARN: fallback embedding download failed; continuing (backend is '$BACKEND')" >&2
    fi
else
    echo "[mem-zero] Embedding model already present: $FASTEMBED_CACHE"
fi

echo "[mem-zero] model setup complete"
