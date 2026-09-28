"""Postgres connection pool via asyncpg.

Manages a connection pool to the Postgres database (local Docker container
in dev, Supabase in production). The pool is created lazily on first use.

References: PRD §13 (database schema).
"""

from __future__ import annotations

from pathlib import Path

import asyncpg
import structlog

from src.config import settings

logger = structlog.get_logger()

_pool: asyncpg.Pool | None = None

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


async def get_db_pool() -> asyncpg.Pool:
    """Get or create the shared asyncpg connection pool.

    Returns:
        asyncpg connection pool instance.
    """
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=settings.database_url,
            min_size=2,
            max_size=10,
        )
        logger.info("db.pool_created", dsn=_redact_dsn(settings.database_url))
    return _pool


async def close_db_pool() -> None:
    """Close the database pool on app shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("db.pool_closed")


async def run_migrations() -> None:
    """Run all SQL migration files in order.

    Simple migration runner — reads .sql files from the migrations directory
    and executes them in filename order. Each migration is idempotent
    (uses IF NOT EXISTS where possible).

    For production, consider using Alembic or a proper migration tool.
    """
    pool = await get_db_pool()

    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not migration_files:
        logger.warning("db.no_migrations_found", dir=str(MIGRATIONS_DIR))
        return

    async with pool.acquire() as conn:
        for migration_file in migration_files:
            logger.info("db.running_migration", file=migration_file.name)
            sql = migration_file.read_text(encoding="utf-8")
            await conn.execute(sql)

    logger.info("db.migrations_complete", count=len(migration_files))


async def check_health() -> bool:
    """Check database connectivity.

    Returns:
        True if the database is reachable.
    """
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            return result == 1
    except Exception:
        logger.exception("db.health_check_failed")
        return False


def _redact_dsn(dsn: str) -> str:
    """Redact password from DSN for safe logging."""
    try:
        # postgresql://user:pass@host:port/db → postgresql://user:***@host:port/db
        if "@" in dsn and ":" in dsn.split("@")[0]:
            prefix, rest = dsn.rsplit("@", 1)
            user_part = prefix.rsplit(":", 1)[0]
            return f"{user_part}:***@{rest}"
    except Exception:
        pass
    return "***"
