# Multi-stage build for VectorSmuggle
# Stage 1: Build dependencies and compile requirements
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION=latest
ARG VCS_REF

# Add metadata labels
LABEL org.opencontainers.image.title="VectorSmuggle" \
      org.opencontainers.image.description="Vector embedding exfiltration proof-of-concept" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="Security Research" \
      org.opencontainers.image.licenses="MIT"

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Stage 2: Runtime image
FROM python:3.11-slim as runtime

# Set build arguments for runtime
ARG BUILD_DATE
ARG VERSION=latest
ARG VCS_REF

# Add metadata labels
LABEL org.opencontainers.image.title="VectorSmuggle" \
      org.opencontainers.image.description="Vector embedding exfiltration proof-of-concept" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user with specific UID/GID for security
RUN groupadd -r -g 1001 vectorsmuggle && \
    useradd -r -g vectorsmuggle -u 1001 -m -d /home/vectorsmuggle -s /bin/bash vectorsmuggle

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application files with proper ownership
COPY --chown=vectorsmuggle:vectorsmuggle . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/faiss_index /app/.query_cache /app/logs /app/temp && \
    chown -R vectorsmuggle:vectorsmuggle /app

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VECTOR_DB=faiss \
    LOG_LEVEL=INFO \
    FAISS_INDEX_PATH=/app/faiss_index \
    QUERY_CACHE_DIR=/app/.query_cache

# Expose port for health checks (if needed)
EXPOSE 8080

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; from config import get_config; get_config(); sys.exit(0)" || exit 1

# Switch to non-root user
USER vectorsmuggle

# Set signal handling
STOPSIGNAL SIGTERM

# Default command
CMD ["/bin/bash"]
