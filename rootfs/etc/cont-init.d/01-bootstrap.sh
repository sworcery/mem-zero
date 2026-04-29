#!/usr/bin/env bash
set -euo pipefail

mkdir -p /mem0/storage/qdrant

: >/var/run/mem-zero.env

env_vars=(
    QDRANT_HOST
    QDRANT_PORT
    QDRANT_URL
    QDRANT_API_KEY
    OLLAMA_BASE_URL
    EMBEDDER_MODEL
    EMBEDDER_DIMENSIONS
    COLLECTION_PREFIX
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
