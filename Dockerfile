# ATLAS Production Dockerfile
# Built on Render's servers, NOT your laptop. You never run Docker locally.
# This file is a recipe — Render reads it, builds the image, runs the container.

# Use Python 3.11 slim (smaller image, faster builds)
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required for SQLite + scientific Python
# build-essential: needed to compile some Python packages on arm64
# libsqlite3-dev: SQLite development headers
# curl: needed for Render's health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsqlite3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching — if requirements don't
# change, this layer is cached and builds are faster)
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: smaller image size
# --upgrade: ensure latest compatible versions
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/
COPY pyproject.toml .
COPY README.md .

# Copy the pre-built vector store (SQLite + embeddings)
# This is critical — without it, the agent has no knowledge base
COPY data/ ./data/

# Copy eval files (for anyone who clones and wants to verify)
COPY eval/ ./eval/

# Install the atlas package in editable mode
RUN pip install -e .

# Expose the port FastAPI will run on
# Render assigns a PORT env var — we use that in the start command
EXPOSE $PORT

# Health check — Render pings this every 30s
# If it fails 3 times, Render restarts the container
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Start command: run uvicorn with the PORT Render provides
# --host 0.0.0.0: listen on all interfaces (required for containers)
# --workers 2: handle 2 concurrent requests (free tier gets 512MB RAM)
# --timeout-keep-alive 5: close idle connections after 5s
CMD uvicorn atlas.api.main:app --host 0.0.0.0 --port $PORT --workers 2 --timeout-keep-alive 5