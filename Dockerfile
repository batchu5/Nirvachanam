# PR Review Agent — API, worker and MCP server share this image
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install the package and its deps (needs pyproject + src). Failures must
# break the build — don't hide them behind `|| true`.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

# Copy the rest (prompts, scripts). Secrets are excluded via .dockerignore —
# provide .env and the GitHub App key at runtime (env_file / volume / secret).
COPY . .

# Default command (overridden by docker-compose per service)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
