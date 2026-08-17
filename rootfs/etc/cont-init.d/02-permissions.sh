#!/usr/bin/env bash
# Remap the memzero service account to PUID/PGID and make sure the persistent
# volume is writable by it. Everything here is best-effort and NEVER fatal: a
# permissions oddity must degrade to a service error, not an unbootable
# container (S6_BEHAVIOUR_IF_STAGE2_FAILS=2 would exit on any non-zero here).
set -uo pipefail

STORAGE="${MEMZERO_STORAGE_DIR:-/mem-zero/storage}"
RUNTIME_ENV="${MEMZERO_RUNTIME_ENV:-/var/run/mem-zero.env}"
USER_FILE="${MEMZERO_USER_FILE:-/var/run/mem-zero.user}"
DRY_RUN="${MEMZERO_PERMS_DRY_RUN:-0}"     # local testing: print instead of act

run() { if [[ "$DRY_RUN" == 1 ]]; then echo "+ $*"; else "$@"; fi; }

# Pick up PUID/PGID that 01-bootstrap restored from the saved config.
# shellcheck disable=SC1090
[[ -f "$RUNTIME_ENV" ]] && source "$RUNTIME_ENV"

PUID="${PUID:-99}"
PGID="${PGID:-100}"
if ! [[ "$PUID" =~ ^[0-9]+$ && "$PGID" =~ ^[0-9]+$ ]]; then
    echo "[mem-zero] WARN: PUID/PGID must be numeric (got '$PUID'/'$PGID'); using 99/100" >&2
    PUID=99
    PGID=100
fi

if [[ "$PUID" == 0 ]]; then
    # Explicit escape hatch: behave exactly like releases before non-root.
    echo "[mem-zero] PUID=0: services will run as root (legacy mode, no ownership changes)"
    printf 'root\n' >"$USER_FILE"
    exit 0
fi

if [[ "$(id -g memzero 2>/dev/null)" != "$PGID" ]]; then
    run groupmod -o -g "$PGID" memzero || echo "[mem-zero] WARN: groupmod failed" >&2
fi
if [[ "$(id -u memzero 2>/dev/null)" != "$PUID" ]]; then
    run usermod -o -u "$PUID" memzero || echo "[mem-zero] WARN: usermod failed" >&2
fi
printf 'memzero\n' >"$USER_FILE"

want="$PUID:$PGID"
run mkdir -p "$STORAGE/qdrant" "$STORAGE/models"

# One-time recursive fix: first boot after the non-root change (existing
# volumes are root-owned) or a PUID/PGID change. The sentinel keeps every later
# boot O(1) instead of walking a volume with thousands of Qdrant segment files.
sentinel="$STORAGE/.owner"
have="$(cat "$sentinel" 2>/dev/null || true)"
top="$(stat -c '%u:%g' "$STORAGE" 2>/dev/null || echo '?')"
if [[ "$have" != "$want" || "$top" != "$want" ]]; then
    echo "[mem-zero] setting ownership of $STORAGE to $want (one-time; may take a while on large volumes)"
    if run chown -R "$want" "$STORAGE"; then
        [[ "$DRY_RUN" == 1 ]] || printf '%s\n' "$want" >"$sentinel"
    else
        echo "[mem-zero] WARN: chown -R $STORAGE failed; services may be unable to write. Delete $sentinel to retry, or set PUID=0." >&2
    fi
fi
# Cheap per-boot: dirs created above as root, plus qdrant's cwd (it writes its
# .qdrant-initialized sentinel there).
run chown "$want" "$STORAGE" "$STORAGE/qdrant" "$STORAGE/models" /qdrant \
    || echo "[mem-zero] WARN: chown of storage dirs failed" >&2
echo "[mem-zero] permissions ready (uid:gid $want)"
