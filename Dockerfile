# ─────────────────────────────────────────────────────────────────────────────
# NeuralClaw — Production Dockerfile
# Multi-stage build: deps → runtime
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Build ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for building native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only what's needed for pip install (layer caching)
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Install the package + all optional deps into a venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir ".[all,dev]"

# Install Playwright browsers
RUN playwright install chromium --with-deps 2>/dev/null || true


# ── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="NeuralClaw"
LABEL description="NeuralClaw — Local-first autonomous AI agent platform"
LABEL version="1.0.0"

WORKDIR /app

# Minimal system deps for runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git jq \
    # Playwright runtime deps
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxrandr2 libgbm1 \
    libpango-1.0-0 libcairo2 libasound2 libxshmfence1 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy Playwright browsers from builder (glob pattern avoids failure if missing)
COPY --from=builder /root/.cache/ms-playwrigh[t] /root/.cache/ms-playwright/

# Copy application source
COPY src/ ./src/
COPY pyproject.toml README.md LICENSE ./

# Create data directories
RUN mkdir -p data/chroma data/sqlite data/logs data/agent_files data/clawhub/skills

# Environment defaults
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NEURALCLAW_CONFIG=/app/src/neuralclaw/config/config.yaml

# Health check — verifies the package is importable
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import neuralclaw; print(neuralclaw.__version__)" || exit 1

# Expose gateway port
EXPOSE 9090
# Expose webui port
EXPOSE 8080

# Default: CLI interface
ENTRYPOINT ["python", "-m", "neuralclaw.main"]
CMD ["--interface", "cli"]
