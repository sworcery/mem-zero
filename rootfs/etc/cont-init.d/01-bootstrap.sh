#!/usr/bin/env bash
set -euo pipefail

# Overridable so the script can be exercised against a temp dir in tests.
STORAGE="${MEMZERO_STORAGE_DIR:-/mem-zero/storage}"
RUNTIME_ENV="${MEMZERO_RUNTIME_ENV:-/var/run/mem-zero.env}"
SAVED_ENV="$STORAGE/.env.saved"

mkdir -p "$STORAGE/qdrant"

env_vars=(
    QDRANT_HOST
    QDRANT_PORT
    QDRANT_URL
    QDRANT_API_KEY
    LLM_BACKEND
    OLLAMA_BASE_URL
    OLLAMA_MAX_CONCURRENT
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
    RERANK_ENABLED
    RERANK_MODEL
    COLLECTION_PREFIX
    API_KEY
    DASHBOARD_USER
    DASHBOARD_PASS
    DIAGNOSTICS_ENABLED
    STATS_PATH
    HOST
    PORT
)

_is_persisted_var() {
    local n
    for n in "${env_vars[@]}"; do
        [[ "$n" == "$1" ]] && return 0
    done
    return 1
}

# Restore previously saved values for any env vars that Unraid blanked on
# update. The saved file lives on a user-writable share, so it is treated as
# untrusted input: only allowlisted keys are accepted, values must be in the
# exact shape printf %q produced (backslash escapes over a safe charset), and
# nothing is ever eval'd. A bad line is skipped with a warning; it must never
# abort cont-init (S6_BEHAVIOUR_IF_STAGE2_FAILS=2 would kill the container).
restore_saved_env() {
    local line key value stripped decoded lineno=0 bad=0
    [[ -f "$SAVED_ENV" ]] || return 0
    while IFS= read -r line || [[ -n "$line" ]]; do
        lineno=$((lineno + 1))
        [[ -z "$line" || "$line" == \#* ]] && continue
        if [[ "$line" != *=* ]]; then
            echo "[mem-zero] WARN: $SAVED_ENV line $lineno is not KEY=VALUE; skipped" >&2
            bad=1
            continue
        fi
        key="${line%%=*}"
        value="${line#*=}"
        if [[ ! "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || ! _is_persisted_var "$key"; then
            echo "[mem-zero] WARN: $SAVED_ENV line $lineno: key '$key' is not a persisted setting; skipped" >&2
            bad=1
            continue
        fi
        [[ -z "$value" ]] && continue
        # Environment always wins over the saved copy.
        [[ -n "${!key-}" ]] && continue
        # Shape check: after dropping every backslash-escaped char, nothing
        # shell-significant may remain. Rejects hand edits, quoted forms,
        # $'..' forms and anything that could be an injection.
        stripped="${value//\\?/}"
        if [[ "$stripped" == *[\\\'\"\$\`[:space:]]* ]]; then
            echo "[mem-zero] WARN: $SAVED_ENV line $lineno: value for $key is not in the expected encoding; skipped" >&2
            bad=1
            continue
        fi
        # Decode the backslash escapes without eval: read (without -r) strips
        # exactly one level of backslashes and leaves everything else as-is.
        decoded=""
        # shellcheck disable=SC2162  # no -r on purpose: the backslash processing IS the decoder
        IFS= read -d '' decoded < <(printf '%s' "$value") || true
        export "$key=$decoded"
        echo "[mem-zero] restored $key from saved config"
    done < "$SAVED_ENV"
    if [[ "$bad" -ne 0 ]]; then
        cp -f "$SAVED_ENV" "${SAVED_ENV}.corrupt" 2>/dev/null && chmod 600 "${SAVED_ENV}.corrupt" || true
        echo "[mem-zero] WARN: $SAVED_ENV had unreadable lines; copy kept at ${SAVED_ENV}.corrupt, file regenerated from current environment" >&2
    fi
    return 0
}

restore_saved_env

# Write current values to both runtime env and persistent storage
: >"$RUNTIME_ENV"
: >"${SAVED_ENV}.tmp"

for name in "${env_vars[@]}"; do
    value="${!name-}"
    if [[ -n ${value} ]]; then
        printf '%s=%q\n' "${name}" "${value}" >>"$RUNTIME_ENV"
        printf '%s=%q\n' "${name}" "${value}" >>"${SAVED_ENV}.tmp"
    fi
done

mv "${SAVED_ENV}.tmp" "$SAVED_ENV"
chmod 600 "$RUNTIME_ENV" "$SAVED_ENV"
echo "[mem-zero] bootstrap complete"
