# Multi-stage build for the PR Review Agent
FROM python:3.11-slim AS base

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null || pip install --no-cache-dir .

# Copy source code
COPY . .

# Default command (overridden by docker-compose per service)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
