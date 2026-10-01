"""PostgresSaver checkpointer setup for LangGraph.

Uses Supabase Postgres (or local Postgres) to persist graph state
across runs. This enables:
  - Resuming interrupted reviews
  - Backfill jobs that pick up from where a quota-skipped agent left off
  - Debugging by inspecting graph state at any point

References: PRD §2 (PostgresSaver from day 1), §13 (database).
"""

from __future__ import annotations

import structlog
from psycopg_pool import AsyncConnectionPool

from src.config import settings

logger = structlog.get_logger()

# Global pool reference
_pool: AsyncConnectionPool | None = None


async def get_checkpointer_pool() -> AsyncConnectionPool:
    """Get or create the async connection pool for the checkpointer.

    Uses the same Postgres instance as the main app (Supabase in prod,
    local Postgres via Docker in dev).
    """
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            conninfo=settings.database_url,
            min_size=1,
            max_size=3,
            open=False,
        )
        await _pool.open()
        logger.info("checkpointer.pool_created")
    return _pool


async def close_checkpointer_pool() -> None:
    """Close the checkpointer connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("checkpointer.pool_closed")


async def create_postgres_checkpointer():
    """Create a PostgresSaver checkpointer instance.

    Lazily imports langgraph to avoid hard dependency when not using
    the full pipeline (e.g. in unit tests).

    Returns:
        PostgresSaver instance, or None if setup fails.
    """
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        pool = await get_checkpointer_pool()
        checkpointer = AsyncPostgresSaver(pool)

        # Create checkpoint tables if they don't exist
        await checkpointer.setup()

        logger.info("checkpointer.postgres_ready")
        return checkpointer

    except ImportError:
        logger.warning(
            "checkpointer.langgraph_postgres_not_installed",
            hint="pip install langgraph-checkpoint-postgres",
        )
        return None
    except Exception as e:
        logger.error("checkpointer.setup_failed", error=str(e))
        return None


async def create_memory_checkpointer():
    """Create an in-memory checkpointer for testing.

    Returns:
        MemorySaver instance.
    """
    try:
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()
    except ImportError:
        logger.warning("checkpointer.langgraph_not_installed")
        return None
