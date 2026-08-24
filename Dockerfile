# syntax=docker/dockerfile:1
FROM qdrant/qdrant:v1.19.0 AS qdrant-bin

############################ builder ############################
# Compiles llama-cpp-python (needs build-essential + cmake) into a venv. None
# of the toolchain reaches the runtime image.
FROM ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates python3 python3-pip python3-venv build-essential cmake \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build

# Deps layer: pinned by requirements.txt (see `make lock`), so this expensive
# layer — the llama-cpp compile lives here — is only invalidated when the lock
# changes, never by a source edit.
COPY requirements.txt .
RUN pip install -r requirements.txt

# App wheel: cheap, rebuilt on source changes. Build isolation fetches
# hatchling itself.
COPY pyproject.toml .
COPY src/ src/
RUN pip wheel --no-deps -w /wheels .

############################ runtime ############################
# MUST stay on the same base tag as the builder: the venv's interpreter path
# and ABI have to match. Dependabot bumps both FROM lines together — check.
FROM ubuntu:24.04
ARG S6_OVERLAY_VERSION=3.2.0.0
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# %q in the bootstrap script must emit UTF-8 bytes verbatim, not $'\303\251'.
ENV LANG=C.UTF-8

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# No compiler here. libgomp1 is the OpenMP runtime the compiled llama.cpp libs
# link against (it used to arrive transitively via gcc); libunwind8 is for
# qdrant. curl serves the healthcheck and model download.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    xz-utils \
    python3 \
    libgomp1 \
    libunwind8 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=qdrant-bin /qdrant/qdrant /usr/local/bin/qdrant
COPY --from=qdrant-bin /qdrant/config /qdrant/config
# /qdrant/static (the Qdrant web dashboard) is deliberately NOT copied. It is a
# bundled JS app whose package manifests are the only Node content in this
# image, and Trivy flags their CVEs even though no Node runtime exists here.
# It was already unreachable: Qdrant binds loopback (below) and mem-zero serves
# its own dashboard on 8765. Qdrant treats the folder as optional
# (web_ui_folder() -> Option; missing dir just skips the service), and
# ENABLE_STATIC_CONTENT=false below stops it warning about the absence.

RUN set -e && \
    S6_ARCH="${TARGETARCH:-$(uname -m)}" && \
    if [ "$S6_ARCH" = "x86_64" ]; then S6_ARCH="x86_64"; \
    elif [ "$S6_ARCH" = "amd64" ]; then S6_ARCH="x86_64"; \
    elif [ "$S6_ARCH" = "aarch64" ]; then S6_ARCH="aarch64"; \
    elif [ "$S6_ARCH" = "arm64" ]; then S6_ARCH="aarch64"; \
    else echo "Unsupported architecture: $S6_ARCH" && exit 1; fi && \
    curl -fsSL -o /tmp/s6-overlay-noarch.tar.xz \
        "https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-noarch.tar.xz" && \
    tar -C / -Jxpf /tmp/s6-overlay-noarch.tar.xz && \
    curl -fsSL -o "/tmp/s6-overlay-${S6_ARCH}.tar.xz" \
        "https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-${S6_ARCH}.tar.xz" && \
    tar -C / -Jxpf "/tmp/s6-overlay-${S6_ARCH}.tar.xz" && \
    rm /tmp/s6-overlay-*.tar.xz

# Deps venv (large layer, stable across source changes) then the app wheel
# (small layer), preserving the "source change doesn't re-download deps" property.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY --from=builder /wheels /tmp/wheels
RUN pip install --no-deps --no-index /tmp/wheels/*.whl && rm -rf /tmp/wheels
# Fail the BUILD, not the first boot, if a native runtime library is missing.
# The traceback names the missing .so if libgomp1 was not the only one.
RUN python3 -c "import llama_cpp, fastembed, uvicorn, qdrant_client, mem_zero; print('runtime import ok')"

# Unprivileged service account; ids are remapped at boot from PUID/PGID
# (Unraid convention 99/100). Home is deliberately non-existent so a runtime
# `usermod -u` never walks a directory tree.
RUN groupadd -g 911 memzero && \
    useradd -u 911 -g memzero -M -d /nonexistent -s /usr/sbin/nologin memzero
ENV PUID=99
ENV PGID=100

WORKDIR /app
COPY rootfs/ /
RUN find /etc/cont-init.d -type f -exec chmod +x {} \; && \
    find /etc/services.d -type f -name "run" -exec chmod +x {} \;

ENV QDRANT__STORAGE__STORAGE_PATH=/mem-zero/storage/qdrant
ENV QDRANT__TELEMETRY_DISABLED=true
# Qdrant is an internal component: bind to loopback so it is unreachable from
# outside the container even if 6333 is published. Operators who really want
# direct access can override with -e QDRANT__SERVICE__HOST=0.0.0.0.
ENV QDRANT__SERVICE__HOST=127.0.0.1
# No web dashboard: the static assets are not shipped (see the COPY block above).
ENV QDRANT__SERVICE__ENABLE_STATIC_CONTENT=false
# On an exit-137 during collection load, restart once in recovery mode
# (handled by the qdrant run script) instead of crash-looping forever.
ENV QDRANT_ALLOW_RECOVERY_MODE=true
ENV QDRANT_HOST=127.0.0.1
ENV QDRANT_PORT=6333
# EMBEDDER_DIMENSIONS is deliberately NOT set here: unset means the app picks
# the right default per backend (768 for Ollama/bundled, the model's native
# size for OpenAI). A baked-in 768 would be forced onto OpenAI and break it.
ENV HOST=0.0.0.0
ENV PORT=8765

# Bundled model defaults (models stored on persistent volume)
ENV BUNDLED_MODEL_PATH=/mem-zero/storage/models/qwen2.5-3b-instruct-q4_k_m.gguf
ENV BUNDLED_EMBED_MODEL=nomic-ai/nomic-embed-text-v1.5
ENV FASTEMBED_CACHE_PATH=/mem-zero/storage/models/fastembed

VOLUME ["/mem-zero/storage"]
EXPOSE 8765

ENV S6_KEEP_ENV=1
ENV S6_BEHAVIOUR_IF_STAGE2_FAILS=2
# Give Qdrant time to flush and exit cleanly on shutdown (values in ms).
# Total worst case (15s TERM wait + 5s stage-3 kill grace) stays under the
# 30s Docker stop budget set in compose / the Unraid template.
ENV S6_SERVICES_GRACETIME=15000
ENV S6_KILL_GRACETIME=5000

# start-period covers a worst-case first boot: ~2.5GB of model downloads on a
# slow connection before services even start. A passing probe flips the
# container healthy immediately regardless, so the generosity costs nothing.
HEALTHCHECK --interval=30s --timeout=10s --start-period=1800s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8765}/health" || exit 1

ENTRYPOINT ["/init"]
