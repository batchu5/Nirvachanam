"""FastAPI application — entry point for the PR Review Agent.

Mounts the webhook router, configures structured logging, and manages
lifecycle events (Redis/DB pool creation/teardown).
"""

import logging

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from src.config import settings
from src.db.connection import close_db_pool, get_db_pool, run_migrations
from src.queue.redis_pool import close_redis_pool, get_redis_pool
from src.webhook.handler import router as webhook_router

# ---------------------------------------------------------------------------
# Structured logging setup
# ---------------------------------------------------------------------------

log_level_int = getattr(logging, settings.log_level.upper(), logging.INFO)

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
        if settings.environment == "development"
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(log_level_int),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app startup/shutdown — create and tear down connection pools."""
    logger.info("app.starting", environment=settings.environment)

    # Startup: initialize pools
    await get_redis_pool()
    await get_db_pool()

    # Run DB migrations
    try:
        await run_migrations()
    except Exception:
        logger.exception("app.migration_failed")
        # Don't crash — migrations may already be applied

    logger.info("app.started")

    yield  # App is running

    # Shutdown: close pools
    await close_redis_pool()
    await close_db_pool()
    logger.info("app.stopped")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI PR Review Agent",
    description="Multi-agent AI code reviewer — LangGraph + FastAPI + Redis",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount routers
app.include_router(webhook_router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "ok",
        "version": "0.1.0",
        "environment": settings.environment,
    }
