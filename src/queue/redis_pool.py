"""Redis connection pool for arq task queue.

Uses local Redis by default — no Upstash, no SSL, no auth required for
local development. The pool is created lazily on first use and reused
across the app lifetime.

References: PRD §5 (Redis usage pattern).
"""

from __future__ import annotations

import structlog
from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from src.config import settings

logger = structlog.get_logger()

# Module-level pool — lazily initialized
_pool: ArqRedis | None = None


def get_redis_settings() -> RedisSettings:
    """Build arq RedisSettings from app config.

    Local Redis: no SSL, no password.
    """
    return RedisSettings(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password,
        ssl=False,  # Local Redis — no TLS
    )


async def get_redis_pool() -> ArqRedis:
    """Get or create the shared arq Redis connection pool.

    Returns:
        ArqRedis connection pool instance.
    """
    global _pool
    if _pool is None:
        _pool = await create_pool(get_redis_settings())
        logger.info(
            "redis.pool_created",
            host=settings.redis_host,
            port=settings.redis_port,
        )
    return _pool


async def close_redis_pool() -> None:
    """Close the Redis pool on app shutdown."""
    global _pool
    if _pool is not None:
        _pool.close()
        await _pool.wait_closed()
        _pool = None
        logger.info("redis.pool_closed")
