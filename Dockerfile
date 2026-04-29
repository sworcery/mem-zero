FROM qdrant/qdrant:v1.13.2 AS qdrant-bin

FROM ubuntu:24.04
ARG S6_OVERLAY_VERSION=3.2.0.0
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    xz-utils \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    cmake \
    libunwind8 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=qdrant-bin /qdrant/qdrant /usr/local/bin/qdrant
COPY --from=qdrant-bin /qdrant/config /qdrant/config
COPY --from=qdrant-bin /qdrant/static /qdrant/static

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

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install Python dependencies (cached unless pyproject.toml changes)
COPY pyproject.toml .
RUN python3 -c "\
import tomllib, pathlib; \
deps = tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['dependencies']; \
print('\n'.join(deps))" > /tmp/deps.txt && \
    pip install -r /tmp/deps.txt && rm /tmp/deps.txt

# Download bundled GGUF model (~1.8 GB, cached in its own layer)
RUN mkdir -p /app/models && \
    curl -fSL --retry 3 -o /app/models/qwen2.5-3b-instruct-q4_k_m.gguf \
    "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"

# Pre-download fastembed embedding model (~270 MB, cached)
ENV FASTEMBED_CACHE_PATH=/app/models/fastembed
RUN python3 -c "from fastembed import TextEmbedding; TextEmbedding('nomic-ai/nomic-embed-text-v1.5')"

# Install the package (re-runs on source changes but skips dep download)
COPY src/ src/
RUN pip install --no-deps .

COPY rootfs/ /
RUN find /etc/cont-init.d -type f -exec chmod +x {} \; && \
    find /etc/services.d -type f -name "run" -exec chmod +x {} \;

ENV QDRANT__STORAGE__STORAGE_PATH=/mem0/storage/qdrant
ENV QDRANT__TELEMETRY_DISABLED=true
ENV QDRANT_HOST=127.0.0.1
ENV QDRANT_PORT=6333
ENV EMBEDDER_DIMENSIONS=768
ENV HOST=0.0.0.0
ENV PORT=8765
ENV BUNDLED_MODEL_PATH=/app/models/qwen2.5-3b-instruct-q4_k_m.gguf
ENV BUNDLED_EMBED_MODEL=nomic-ai/nomic-embed-text-v1.5

VOLUME ["/mem0/storage"]
EXPOSE 8765 6333

ENV S6_KEEP_ENV=1
ENV S6_CMD_WAIT_FOR_SERVICES_MAXTIME=300000
ENV S6_BEHAVIOUR_IF_STAGE2_FAILS=2

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8765/health || exit 1

ENTRYPOINT ["/init"]
