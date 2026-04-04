"""Tests for the RateLimiter class."""

import asyncio
import time

import pytest
from src.utils.rate_limiter import RateLimiter, DailyRateLimiter


class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.mark.asyncio
    async def test_initial_tokens(self):
        """Test that rate limiter starts with full tokens."""
        limiter = RateLimiter(requests_per_second=10, burst_size=10)
        assert limiter.available_tokens == 10

    @pytest.mark.asyncio
    async def test_acquire_reduces_tokens(self):
        """Test that acquiring tokens reduces available tokens."""
        limiter = RateLimiter(requests_per_second=10, burst_size=10)
        await limiter.acquire(1)
        assert limiter.available_tokens < 10

    @pytest.mark.asyncio
    async def test_burst_allowed(self):
        """Test that burst requests are allowed."""
        limiter = RateLimiter(requests_per_second=10, burst_size=5)

        start = time.monotonic()
        for _ in range(5):
            await limiter.acquire(1)
        elapsed = time.monotonic() - start

        # Should be nearly instant (within 0.1s)
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_rate_limiting_kicks_in(self):
        """Test that rate limiting works after burst."""
        limiter = RateLimiter(requests_per_second=100, burst_size=2)

        start = time.monotonic()
        # First 2 should be instant (burst)
        await limiter.acquire(1)
        await limiter.acquire(1)
        # Third should wait
        await limiter.acquire(1)
        elapsed = time.monotonic() - start

        # Should have waited at least some time
        assert elapsed >= 0.005

    def test_try_acquire_success(self):
        """Test try_acquire when tokens available."""
        limiter = RateLimiter(requests_per_second=10, burst_size=5)
        assert limiter.try_acquire(1)
        assert limiter.try_acquire(1)

    def test_try_acquire_failure(self):
        """Test try_acquire when no tokens available."""
        limiter = RateLimiter(requests_per_second=10, burst_size=2)
        limiter.try_acquire(2)
        assert not limiter.try_acquire(1)


class TestDailyRateLimiter:
    """Tests for DailyRateLimiter."""

    def test_initial_state(self):
        """Test initial state allows requests."""
        limiter = DailyRateLimiter(daily_limit=100)
        assert limiter.can_make_request()
        assert limiter.remaining_calls == 100

    def test_recording_reduces_remaining(self):
        """Test that recording requests reduces remaining."""
        limiter = DailyRateLimiter(daily_limit=100)
        limiter.record_request()
        assert limiter.remaining_calls == 99

    def test_limit_enforcement(self):
        """Test that limit is enforced."""
        limiter = DailyRateLimiter(daily_limit=3)
        limiter.record_request()
        limiter.record_request()
        limiter.record_request()
        assert not limiter.can_make_request()
        assert limiter.remaining_calls == 0

    def test_remaining_never_negative(self):
        """Test remaining never goes negative."""
        limiter = DailyRateLimiter(daily_limit=1)
        limiter.record_request()
        limiter.record_request()  # Over limit
        assert limiter.remaining_calls == 0
