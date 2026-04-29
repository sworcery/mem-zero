#!/usr/bin/env bash
set -euo pipefail

mkdir -p /mem0/storage/qdrant

: >/var/run/mem-zero.env

env_vars=(
    QDRANT_HOST
    QDRANT_PORT
    QDRANT_URL
    QDRANT_API_KEY
    LLM_BACKEND
    OLLAMA_BASE_URL
    LLM_MODEL
    EMBEDDER_MODEL
    EMBEDDER_DIMENSIONS
    BUNDLED_MODEL_PATH
    BUNDLED_EMBED_MODEL
    BUNDLED_THREADS
    FASTEMBED_CACHE_PATH
    OPENAI_API_KEY
    OPENAI_BASE_URL
    OPENAI_MODEL
    OPENAI_EMBED_MODEL
    COLLECTION_PREFIX
    DASHBOARD_USER
    DASHBOARD_PASS
    HOST
    PORT
)

for name in "${env_vars[@]}"; do
    value="${!name-}"
    if [[ -n ${value} ]]; then
        printf '%s=%q\n' "${name}" "${value}" >>/var/run/mem-zero.env
    fi
done

chmod 600 /var/run/mem-zero.env
echo "[mem-zero] bootstrap complete"
