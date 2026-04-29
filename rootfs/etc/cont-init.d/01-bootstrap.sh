#!/usr/bin/env bash
set -euo pipefail

mkdir -p /mem0/storage/qdrant

SAVED_ENV="/mem0/storage/.env.saved"

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
    API_KEY
    DASHBOARD_USER
    DASHBOARD_PASS
    HOST
    PORT
)

# Restore previously saved values for any env vars that Unraid blanked on update
if [[ -f "$SAVED_ENV" ]]; then
    while IFS='=' read -r key value; do
        [[ -z "$key" || "$key" == \#* ]] && continue
        current="${!key-}"
        if [[ -z "$current" && -n "$value" ]]; then
            eval "export ${key}=${value}"
            echo "[mem-zero] restored ${key} from saved config"
        fi
    done < "$SAVED_ENV"
fi

# Write current values to both runtime env and persistent storage
: >/var/run/mem-zero.env
: >"${SAVED_ENV}.tmp"

for name in "${env_vars[@]}"; do
    value="${!name-}"
    if [[ -n ${value} ]]; then
        printf '%s=%q\n' "${name}" "${value}" >>/var/run/mem-zero.env
        printf '%s=%q\n' "${name}" "${value}" >>"${SAVED_ENV}.tmp"
    fi
done

mv "${SAVED_ENV}.tmp" "$SAVED_ENV"
chmod 600 /var/run/mem-zero.env "$SAVED_ENV"
echo "[mem-zero] bootstrap complete"
