# LatentGate Dockerfile
# Multi-stage build for smaller image size

# Stage 1: Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY latent_gate/ ./latent_gate/

# Install the package with API dependencies
RUN pip install --no-cache-dir --user ".[api]"

# Stage 2: Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY latent_gate/ ./latent_gate/
COPY pyproject.toml README.md LICENSE ./

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Make local packages accessible to appuser
RUN cp -r /root/.local /home/appuser/.local && \
    chown -R appuser:appuser /home/appuser/.local

# Set environment variables
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV LATENTGATE_LOG_LEVEL=INFO
ENV LATENTGATE_REMOTE_PROVIDER=ollama
ENV LATENTGATE_REMOTE_MODEL=llama3:8b
ENV LATENTGATE_VISION_MODEL=llava:7b
ENV LATENTGATE_TEXT_FAST_MODEL=phi3:mini
ENV LATENTGATE_TEXT_SMART_MODEL=qwen2:7b
ENV LATENTGATE_EMBEDDING_MODEL=nomic-embed-text
ENV LATENTGATE_OLLAMA_BASE_URL=http://ollama:11434

USER appuser

# Create cache directory
RUN mkdir -p /home/appuser/.latentgate_cache

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command: run API server
CMD ["latent-gate-api"]
