"""Redis-backed sliding window quota tracker per LLM provider.

Tracks RPM (requests-per-minute) and RPD (requests-per-day) usage so the
fallback router can proactively skip exhausted providers WITHOUT wasting an
API call to discover it.

References: PRD §1c (proactive quota tracking).
"""

from __future__ import annotations

import structlog
from arq.connections import ArqRedis

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Provider rate limits (free tier)
# ---------------------------------------------------------------------------

PROVIDER_LIMITS: dict[str, dict[str, int]] = {
    "gemini": {"rpm": 15, "rpd": 1500},   # Gemini AI Studio free tier
    "groq":   {"rpm": 30, "rpd": 14400},  # Groq free tier
}


class ProviderQuotaTracker:
    """Redis-backed sliding window quota tracker per LLM provider.

    Uses simple Redis counters with TTLs (not true sliding windows) — good
    enough for our scale and keeps Redis command count low.
    """

    def __init__(self, redis: ArqRedis):
        self.redis = redis

    async def can_call(self, provider: str) -> bool:
        """Check if provider has remaining quota. Called BEFORE routing.

        Reserves 20% headroom for retries.

        Args:
            provider: Provider name ("gemini" or "groq").

        Returns:
            True if the provider has capacity for another call.
        """
        limits = PROVIDER_LIMITS.get(provider)
        if not limits:
            logger.warning("quota.unknown_provider", provider=provider)
            return True  # Unknown provider — allow (fail open)

        # Check exhaustion flag first (cheaper than two GETs)
        if await self.is_exhausted(provider):
            return False

        rpm_key = f"quota:{provider}:rpm"
        rpd_key = f"quota:{provider}:rpd"

        rpm_count = int(await self.redis.get(rpm_key) or 0)
        rpd_count = int(await self.redis.get(rpd_key) or 0)

        # Reserve 20% headroom for retries
        rpm_ok = rpm_count < limits["rpm"] * 0.8
        rpd_ok = rpd_count < limits["rpd"] * 0.8

        if not rpm_ok:
            logger.info("quota.rpm_near_limit", provider=provider, rpm=rpm_count)
        if not rpd_ok:
            logger.info("quota.rpd_near_limit", provider=provider, rpd=rpd_count)

        return rpm_ok and rpd_ok

    async def record_call(self, provider: str, tokens_used: int = 0) -> None:
        """Increment counters after a successful call.

        Args:
            provider: Provider name.
            tokens_used: Total tokens consumed (for logging/tracking).
        """
        pipe = self.redis.pipeline()
        rpm_key = f"quota:{provider}:rpm"
        rpd_key = f"quota:{provider}:rpd"

        pipe.incr(rpm_key)
        pipe.expire(rpm_key, 60)       # 1-minute window
        pipe.incr(rpd_key)
        pipe.expire(rpd_key, 86400)    # 24-hour window
        await pipe.execute()

        logger.debug(
            "quota.recorded_call",
            provider=provider,
            tokens_used=tokens_used,
        )

    async def record_exhaustion(self, provider: str, retry_after: int = 60) -> None:
        """Mark provider as exhausted with a cooldown TTL.

        Args:
            provider: Provider name.
            retry_after: Seconds until provider should be retried.
        """
        key = f"quota:{provider}:exhausted"
        await self.redis.set(key, "1", ex=retry_after)
        logger.warning(
            "quota.provider_exhausted",
            provider=provider,
            retry_after=retry_after,
        )

    async def is_exhausted(self, provider: str) -> bool:
        """Check if provider is currently marked as exhausted.

        Args:
            provider: Provider name.

        Returns:
            True if provider is in cooldown.
        """
        return bool(await self.redis.exists(f"quota:{provider}:exhausted"))
