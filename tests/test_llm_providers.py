"""Tests for LLM providers, quota tracker, and fallback chain.

Uses mocked provider responses — no real API calls.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.fallback import (
    AllProvidersExhaustedError,
    AuthError,
    QuotaAwareFallbackLLM,
    _classify_provider_error,
)
from src.llm.providers import LLMResponse
from src.llm.quota import PROVIDER_LIMITS, ProviderQuotaTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class FakeRedis:
    """Minimal fake Redis for testing quota tracker."""

    def __init__(self):
        self._data: dict[str, str] = {}
        self._ttls: dict[str, int] = {}

    async def get(self, key: str) -> str | None:
        return self._data.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._data[key] = value
        if ex:
            self._ttls[key] = ex

    async def incr(self, key: str) -> int:
        val = int(self._data.get(key, "0")) + 1
        self._data[key] = str(val)
        return val

    async def expire(self, key: str, seconds: int) -> None:
        self._ttls[key] = seconds

    async def exists(self, key: str) -> bool:
        return key in self._data

    def pipeline(self):
        return FakePipeline(self)


class FakePipeline:
    """Fake Redis pipeline that executes commands immediately."""

    def __init__(self, redis: FakeRedis):
        self._redis = redis
        self._ops: list = []

    def incr(self, key: str):
        self._ops.append(("incr", key))
        return self

    def expire(self, key: str, seconds: int):
        self._ops.append(("expire", key, seconds))
        return self

    async def execute(self):
        for op in self._ops:
            if op[0] == "incr":
                await self._redis.incr(op[1])
            elif op[0] == "expire":
                await self._redis.expire(op[1], op[2])
        self._ops.clear()


class FakeProvider:
    """Fake LLM provider for testing."""

    def __init__(self, provider_name: str = "fake", response: LLMResponse | None = None, error: Exception | None = None):
        self._name = provider_name
        self._response = response
        self._error = error
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    async def invoke(self, messages, response_schema=None, temperature=0.2, max_tokens=2000):
        self.call_count += 1
        if self._error:
            raise self._error
        if self._response:
            return self._response
        return LLMResponse(
            content='{"findings": []}',
            tokens_used=100,
            input_tokens=80,
            output_tokens=20,
            model="fake-model",
            provider=self._name,
        )


@pytest.fixture
def fake_redis():
    return FakeRedis()


@pytest.fixture
def quota_tracker(fake_redis):
    return ProviderQuotaTracker(fake_redis)


# ---------------------------------------------------------------------------
# ProviderQuotaTracker tests
# ---------------------------------------------------------------------------

class TestProviderQuotaTracker:
    """Tests for the Redis-backed quota tracker."""

    async def test_can_call_fresh_provider(self, quota_tracker):
        """Fresh provider with no usage should be allowed."""
        assert await quota_tracker.can_call("gemini") is True
        assert await quota_tracker.can_call("groq") is True

    async def test_can_call_unknown_provider(self, quota_tracker):
        """Unknown providers should be allowed (fail open)."""
        assert await quota_tracker.can_call("unknown_provider") is True

    async def test_record_and_check_call(self, quota_tracker, fake_redis):
        """Recording a call should increment the counter."""
        await quota_tracker.record_call("gemini", tokens_used=100)

        rpm_count = int(await fake_redis.get("quota:gemini:rpm") or 0)
        rpd_count = int(await fake_redis.get("quota:gemini:rpd") or 0)
        assert rpm_count == 1
        assert rpd_count == 1

    async def test_can_call_near_limit(self, quota_tracker, fake_redis):
        """Provider near RPM limit should be rejected (20% headroom)."""
        # Gemini RPM limit is 15, 80% = 12
        await fake_redis.set("quota:gemini:rpm", "12")
        assert await quota_tracker.can_call("gemini") is False

    async def test_can_call_under_limit(self, quota_tracker, fake_redis):
        """Provider under limit should be allowed."""
        await fake_redis.set("quota:gemini:rpm", "10")
        assert await quota_tracker.can_call("gemini") is True

    async def test_exhaustion_flag(self, quota_tracker, fake_redis):
        """Exhausted provider should be rejected."""
        await quota_tracker.record_exhaustion("gemini", retry_after=120)
        assert await quota_tracker.is_exhausted("gemini") is True
        assert await quota_tracker.can_call("gemini") is False

    async def test_not_exhausted(self, quota_tracker):
        """Non-exhausted provider should be fine."""
        assert await quota_tracker.is_exhausted("gemini") is False


# ---------------------------------------------------------------------------
# Error classification tests
# ---------------------------------------------------------------------------

class TestErrorClassification:
    """Tests for provider error classification."""

    def test_rate_limit_error(self):
        error_type, retry_after = _classify_provider_error(Exception("429 Too Many Requests"))
        assert error_type == "rate_limit"

    def test_auth_error(self):
        error_type, _ = _classify_provider_error(Exception("401 Unauthorized"))
        assert error_type == "auth"

    def test_server_error(self):
        error_type, _ = _classify_provider_error(Exception("503 Service Unavailable"))
        assert error_type == "transient"

    def test_timeout_error(self):
        error_type, _ = _classify_provider_error(asyncio.TimeoutError())
        assert error_type == "transient"

    def test_bad_request_error(self):
        error_type, _ = _classify_provider_error(Exception("400 Bad Request"))
        assert error_type == "hard"


# ---------------------------------------------------------------------------
# QuotaAwareFallbackLLM tests
# ---------------------------------------------------------------------------

class TestQuotaAwareFallbackLLM:
    """Tests for the fallback LLM chain."""

    async def test_first_provider_succeeds(self, quota_tracker):
        """When first provider works, use it."""
        provider1 = FakeProvider("gemini")
        provider2 = FakeProvider("groq")

        llm = QuotaAwareFallbackLLM(
            providers=[provider1, provider2],
            quota=quota_tracker,
        )

        response = await llm.invoke([{"role": "user", "content": "test"}])
        assert response is not None
        assert response.provider == "gemini"
        assert provider1.call_count == 1
        assert provider2.call_count == 0

    async def test_fallback_on_failure(self, quota_tracker):
        """When first provider fails, fall back to second."""
        provider1 = FakeProvider(
            "gemini",
            error=Exception("503 Service Unavailable"),
        )
        provider2 = FakeProvider("groq")

        llm = QuotaAwareFallbackLLM(
            providers=[provider1, provider2],
            quota=quota_tracker,
        )

        response = await llm.invoke([{"role": "user", "content": "test"}])
        assert response is not None
        assert response.provider == "groq"
        assert provider1.call_count == 1
        assert provider2.call_count == 1

    async def test_skip_exhausted_provider(self, quota_tracker, fake_redis):
        """Exhausted provider should be skipped without calling it."""
        await quota_tracker.record_exhaustion("gemini", retry_after=120)

        provider1 = FakeProvider("gemini")
        provider2 = FakeProvider("groq")

        llm = QuotaAwareFallbackLLM(
            providers=[provider1, provider2],
            quota=quota_tracker,
        )

        response = await llm.invoke([{"role": "user", "content": "test"}])
        assert response is not None
        assert response.provider == "groq"
        assert provider1.call_count == 0  # Skipped entirely

    async def test_all_providers_fail_returns_none(self, quota_tracker):
        """When all providers fail, return None (graceful degradation)."""
        provider1 = FakeProvider("gemini", error=Exception("503"))
        provider2 = FakeProvider("groq", error=Exception("503"))

        llm = QuotaAwareFallbackLLM(
            providers=[provider1, provider2],
            quota=quota_tracker,
        )

        response = await llm.invoke([{"role": "user", "content": "test"}])
        assert response is None

    async def test_auth_error_raises(self, quota_tracker):
        """Auth errors should be raised immediately (no retry/fallback)."""
        provider1 = FakeProvider("gemini", error=Exception("401 Unauthorized"))

        llm = QuotaAwareFallbackLLM(
            providers=[provider1],
            quota=quota_tracker,
        )

        with pytest.raises(AuthError):
            await llm.invoke([{"role": "user", "content": "test"}])

    async def test_timeout_triggers_fallback(self, quota_tracker):
        """Timeout should trigger fallback to next provider."""

        class SlowProvider:
            @property
            def name(self):
                return "slow"

            async def invoke(self, messages, response_schema=None, temperature=0.2, max_tokens=2000):
                await asyncio.sleep(10)  # Will be killed by timeout

        provider1 = SlowProvider()
        provider2 = FakeProvider("groq")

        llm = QuotaAwareFallbackLLM(
            providers=[provider1, provider2],
            quota=quota_tracker,
        )

        response = await llm.invoke(
            [{"role": "user", "content": "test"}],
            timeout=1,  # 1 second timeout
        )
        assert response is not None
        assert response.provider == "groq"
