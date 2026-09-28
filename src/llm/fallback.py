"""Quota-aware fallback LLM — proactive provider routing with tiered error handling.

Wraps multiple LLM providers in priority order and routes calls through the
first available provider. Uses ProviderQuotaTracker to skip exhausted providers
WITHOUT wasting an API call.

Error classification (PRD §1c):
  - Transient (429 short, 503, timeout) → retry same provider → fallback
  - Quota exhaustion (429 long retry-after) → skip provider → fallback
  - Hard failure (401, 400, schema error) → raise immediately

References: PRD §1c (QuotaAwareFallbackLLM).
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from src.llm.providers import LLMProvider, LLMResponse
from src.llm.quota import ProviderQuotaTracker

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Error types for classification
# ---------------------------------------------------------------------------

class RateLimitError(Exception):
    """Provider returned 429 — rate limited."""
    def __init__(self, message: str = "", retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class AuthError(Exception):
    """Provider returned 401/403 — bad credentials."""


class SchemaValidationError(Exception):
    """LLM response failed pydantic validation after repair retry."""


class AllProvidersExhaustedError(Exception):
    """All providers failed or are exhausted."""


# ---------------------------------------------------------------------------
# Classify exceptions from provider SDKs
# ---------------------------------------------------------------------------

def _classify_provider_error(error: Exception) -> tuple[str, int | None]:
    """Classify a provider SDK exception into our error taxonomy.

    Returns:
        Tuple of (error_type, retry_after_seconds).
        error_type: "rate_limit", "auth", "transient", or "hard".
    """
    error_str = str(error).lower()
    error_type_name = type(error).__name__.lower()

    # Rate limit (429)
    if "429" in error_str or "rate" in error_str and "limit" in error_str:
        # Try to extract retry-after
        retry_after = None
        if hasattr(error, "response"):
            resp = getattr(error, "response", None)
            if resp and hasattr(resp, "headers"):
                ra = resp.headers.get("retry-after")
                if ra and ra.isdigit():
                    retry_after = int(ra)
        return ("rate_limit", retry_after or 60)

    # Auth errors (401, 403)
    if "401" in error_str or "403" in error_str or "auth" in error_type_name:
        return ("auth", None)

    # Bad request (400) — usually malformed input
    if "400" in error_str:
        return ("hard", None)

    # Server errors (500, 502, 503) — transient
    if any(code in error_str for code in ("500", "502", "503", "unavailable")):
        return ("transient", None)

    # Timeout
    if "timeout" in error_str or isinstance(error, asyncio.TimeoutError):
        return ("transient", None)

    # Default to transient (fail-safe: try fallback rather than crash)
    return ("transient", None)


class QuotaAwareFallbackLLM:
    """Wraps multiple LLM providers with proactive quota checking and fallback.

    Usage:
        llm = QuotaAwareFallbackLLM(
            providers=[gemini_provider, groq_provider],
            quota=quota_tracker,
        )
        response = await llm.invoke(messages, response_schema=schema)
    """

    def __init__(
        self,
        providers: list[LLMProvider],
        quota: ProviderQuotaTracker,
    ):
        self.providers = providers
        self.quota = quota

    async def invoke(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        timeout: int = 30,
    ) -> LLMResponse | None:
        """Call the first available provider, with fallback on failure.

        Args:
            messages: List of message dicts with "role" and "content".
            response_schema: Optional JSON schema for structured output.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            timeout: Hard timeout per provider call (seconds).

        Returns:
            LLMResponse from the first successful provider, or None if all
            providers failed (agent should be skipped, not crashed).
        """
        last_error: Exception | None = None

        for provider in self.providers:
            # Proactive check 1: exhaustion flag
            if await self.quota.is_exhausted(provider.name):
                logger.info(
                    "fallback.skipping_exhausted",
                    provider=provider.name,
                )
                continue

            # Proactive check 2: near quota limit
            if not await self.quota.can_call(provider.name):
                logger.info(
                    "fallback.skipping_near_limit",
                    provider=provider.name,
                )
                continue

            try:
                response = await asyncio.wait_for(
                    provider.invoke(
                        messages=messages,
                        response_schema=response_schema,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ),
                    timeout=timeout,
                )
                # Record successful call
                await self.quota.record_call(provider.name, response.tokens_used)
                return response

            except asyncio.TimeoutError:
                logger.warning(
                    "fallback.timeout",
                    provider=provider.name,
                    timeout=timeout,
                )
                last_error = asyncio.TimeoutError(
                    f"{provider.name} timed out after {timeout}s"
                )
                continue

            except Exception as e:
                error_type, retry_after = _classify_provider_error(e)

                if error_type == "rate_limit":
                    if retry_after and retry_after > 60:
                        # Quota exhaustion — mark and skip
                        await self.quota.record_exhaustion(
                            provider.name, retry_after
                        )
                        logger.warning(
                            "fallback.quota_exhaustion",
                            provider=provider.name,
                            retry_after=retry_after,
                        )
                        continue
                    else:
                        # Transient 429 — brief backoff, then fallback
                        wait = min(retry_after or 5, 10)
                        logger.info(
                            "fallback.transient_rate_limit",
                            provider=provider.name,
                            wait=wait,
                        )
                        await asyncio.sleep(wait)
                        last_error = e
                        continue

                elif error_type == "auth":
                    logger.error(
                        "fallback.auth_error",
                        provider=provider.name,
                        error=str(e),
                    )
                    raise AuthError(f"Auth error on {provider.name}: {e}") from e

                elif error_type == "hard":
                    logger.error(
                        "fallback.hard_error",
                        provider=provider.name,
                        error=str(e),
                    )
                    last_error = e
                    continue

                else:  # transient
                    logger.warning(
                        "fallback.transient_error",
                        provider=provider.name,
                        error=str(e),
                    )
                    last_error = e
                    continue

        # All providers failed
        if last_error:
            logger.error(
                "fallback.all_providers_failed",
                last_error=str(last_error),
            )

        return None
