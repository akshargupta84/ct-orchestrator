# =============================================================================
# CT Orchestrator - Production Dockerfile
# Multi-stage build for slim production image
# =============================================================================

# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system deps for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt requirements-local.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Stage 2: Production image
FROM python:3.11-slim AS production

# Labels
LABEL maintainer="Akshar Gupta"
LABEL description="CT Orchestrator — AI-powered creative testing for media agencies"
LABEL version="1.0"

# HF Spaces requires user with UID 1000
RUN useradd -m -u 1000 user

WORKDIR /app

# Install runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY agents/ ./agents/
COPY services/ ./services/
COPY models/ ./models/
COPY utils/ ./utils/
COPY workflows/ ./workflows/
COPY frontend/ ./frontend/
COPY demo_data/ ./demo_data/
COPY historical_data/ ./historical_data/
COPY master_data/ ./master_data/
COPY requirements.txt requirements-local.txt ./
COPY .env.example ./.env.example

# Create data directories with correct permissions
RUN mkdir -p /app/data/plans/draft \
             /app/data/plans/approved \
             /app/data/results/raw \
             /app/data/results/analyzed \
             /app/data/chat_history \
             /app/data/chroma \
             /app/data/logs \
    && chown -R user:user /app

# Streamlit config — disable telemetry, set port for HF (7860)
RUN mkdir -p /home/user/.streamlit && chown -R user:user /home/user
COPY <<EOF /home/user/.streamlit/config.toml
[server]
port = 7860
address = "0.0.0.0"
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
base = "dark"
EOF

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

# Switch to non-root user
USER user

# Expose HF Spaces required port
EXPOSE 7860

# Default environment
ENV DEMO_MODE=true
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# Entry point
CMD ["streamlit", "run", "frontend/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
