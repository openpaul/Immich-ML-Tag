# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY README.md ./

# Install dependencies and build the package
RUN uv build

# Runtime stage
FROM python:3.13-slim-bookworm

WORKDIR /app

# Install runtime dependencies for psycopg2
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 && \
    rm -rf /var/lib/apt/lists/*

# Copy uv from builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy the built wheel and install it
COPY --from=builder /app/dist/*.whl /tmp/
RUN uv pip install --system /tmp/*.whl && \
    rm /tmp/*.whl

# Create directory for ML resources
RUN mkdir -p /data/ml_resources

# Set environment variables
ENV ML_RESOURCE_PATH=/data/ml_resources
ENV UV_SYSTEM_PYTHON=1

# Environment variables for configuration (with defaults)
ENV TRAIN_TIME=02:00
ENV INFERENCE_INTERVAL=5
ENV THRESHOLD=0.5
ENV MIN_SAMPLES=10

# Set entrypoint to serve mode by default
ENTRYPOINT ["immich-ml-tag"]
CMD ["serve", "--train-on-start"]
