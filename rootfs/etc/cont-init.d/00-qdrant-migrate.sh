#!/usr/bin/env bash
set -euo pipefail

STORAGE_PATH="${QDRANT__STORAGE__STORAGE_PATH:-/mem-zero/storage/qdrant}"
VERSION_FILE="${STORAGE_PATH}/.qdrant_version"
TARGET_VERSION="1.17.1"
MIGRATE_DIR="/usr/local/bin/qdrant-migrate"

mkdir -p "$STORAGE_PATH"

if [[ ! -d "${STORAGE_PATH}/collections" ]]; then
    echo "$TARGET_VERSION" > "$VERSION_FILE"
    echo "[qdrant-migrate] Fresh install, no migration needed"
    exit 0
fi

if [[ -f "$VERSION_FILE" ]]; then
    CURRENT=$(cat "$VERSION_FILE")
else
    CURRENT="1.13.2"
    echo "[qdrant-migrate] No version file found, assuming v${CURRENT}"
fi

if [[ "$CURRENT" == "$TARGET_VERSION" ]]; then
    echo "[qdrant-migrate] Storage already at v${TARGET_VERSION}"
    exit 0
fi

echo "============================================="
echo "[qdrant-migrate] STORAGE MIGRATION REQUIRED"
echo "[qdrant-migrate] Current: v${CURRENT}"
echo "[qdrant-migrate] Target:  v${TARGET_VERSION}"
echo "[qdrant-migrate] This may take several minutes."
echo "============================================="

BACKUP_DIR="${STORAGE_PATH}.backup-v${CURRENT}"
if [[ ! -d "$BACKUP_DIR" ]]; then
    echo "[qdrant-migrate] Backing up storage to ${BACKUP_DIR}..."
    cp -a "$STORAGE_PATH" "$BACKUP_DIR"
    echo "[qdrant-migrate] Backup complete"
else
    echo "[qdrant-migrate] Backup already exists at ${BACKUP_DIR}"
fi

VERSIONS=("1.14.1" "1.15.5" "1.16.3" "1.17.1")

for VER in "${VERSIONS[@]}"; do
    if [[ "$(printf '%s\n' "$VER" "$CURRENT" | sort -V | tail -1)" == "$CURRENT" ]]; then
        continue
    fi

    if [[ "$VER" == "$TARGET_VERSION" ]]; then
        BINARY="/usr/local/bin/qdrant"
    else
        BINARY="${MIGRATE_DIR}/${VER}"
    fi

    if [[ ! -x "$BINARY" ]]; then
        echo "[qdrant-migrate] ERROR: Binary not found: ${BINARY}" >&2
        echo "[qdrant-migrate] Restore backup: cp -a ${BACKUP_DIR} ${STORAGE_PATH}" >&2
        exit 1
    fi

    echo "[qdrant-migrate] Step: v${CURRENT} -> v${VER}..."

    cd /qdrant
    "$BINARY" &
    QDRANT_PID=$!

    READY=false
    for _ in $(seq 1 180); do
        if curl -fsS "http://127.0.0.1:6333/readyz" >/dev/null 2>&1; then
            READY=true
            break
        fi
        if ! kill -0 "$QDRANT_PID" 2>/dev/null; then
            wait "$QDRANT_PID" || true
            echo "[qdrant-migrate] ERROR: Qdrant v${VER} exited during migration" >&2
            echo "[qdrant-migrate] Restore backup: cp -a ${BACKUP_DIR} ${STORAGE_PATH}" >&2
            exit 1
        fi
        sleep 1
    done

    if [[ "$READY" != "true" ]]; then
        echo "[qdrant-migrate] ERROR: Qdrant v${VER} not ready after 180s" >&2
        kill "$QDRANT_PID" 2>/dev/null || true
        wait "$QDRANT_PID" 2>/dev/null || true
        echo "[qdrant-migrate] Restore backup: cp -a ${BACKUP_DIR} ${STORAGE_PATH}" >&2
        exit 1
    fi

    echo "[qdrant-migrate] v${VER} ready, shutting down cleanly..."
    kill "$QDRANT_PID"
    wait "$QDRANT_PID" 2>/dev/null || true
    sleep 2

    echo "$VER" > "$VERSION_FILE"
    CURRENT="$VER"
    echo "[qdrant-migrate] Migrated to v${VER} [OK]"
done

echo "============================================="
echo "[qdrant-migrate] MIGRATION COMPLETE"
echo "[qdrant-migrate] Storage now at v${TARGET_VERSION}"
echo "[qdrant-migrate] Backup preserved at ${BACKUP_DIR}"
echo "============================================="
