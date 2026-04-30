#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="/mem-zero/storage/models"
GGUF_PATH="${BUNDLED_MODEL_PATH:-$MODEL_DIR/qwen2.5-3b-instruct-q4_k_m.gguf}"
GGUF_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
FASTEMBED_CACHE="${FASTEMBED_CACHE_PATH:-$MODEL_DIR/fastembed}"
EMBED_MODEL="${BUNDLED_EMBED_MODEL:-nomic-ai/nomic-embed-text-v1.5}"

mkdir -p "$MODEL_DIR"

if [[ ! -f "$GGUF_PATH" ]]; then
    echo "[mem-zero] Downloading bundled LLM model (this only happens once)..."
    curl -fSL --retry 3 -o "${GGUF_PATH}.tmp" "$GGUF_URL"
    mv "${GGUF_PATH}.tmp" "$GGUF_PATH"
    echo "[mem-zero] LLM model downloaded: $GGUF_PATH"
else
    echo "[mem-zero] LLM model already present: $GGUF_PATH"
fi

export FASTEMBED_CACHE_PATH="$FASTEMBED_CACHE"
if [[ ! -d "$FASTEMBED_CACHE" ]] || [[ -z "$(ls -A "$FASTEMBED_CACHE" 2>/dev/null)" ]]; then
    echo "[mem-zero] Downloading bundled embedding model (this only happens once)..."
    /opt/venv/bin/python3 -c "from fastembed import TextEmbedding; TextEmbedding('$EMBED_MODEL')"
    echo "[mem-zero] Embedding model downloaded to: $FASTEMBED_CACHE"
else
    echo "[mem-zero] Embedding model already present: $FASTEMBED_CACHE"
fi

echo "[mem-zero] model setup complete"
