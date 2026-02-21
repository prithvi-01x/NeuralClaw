# ─────────────────────────────────────────────────────────────────────────────
# NeuralClaw — Dockerfile
# docker/Dockerfile
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget grep findutils jq \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (for future browser tool)
RUN playwright install chromium --with-deps 2>/dev/null || true

# Copy source
COPY . .

# Data directories
RUN mkdir -p data/chroma data/sqlite data/logs data/agent_files

CMD ["python", "main.py"]