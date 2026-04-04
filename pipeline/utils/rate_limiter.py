"""Token bucket rate limiter for API requests."""

import asyncio
import time
from typing import Optional


class RateLimiter:
    """Token bucket rate limiter for controlling API request rates.

    This implements a token bucket algorithm where tokens are added at a fixed rate
    and each request consumes one token. Requests wait if no tokens are available.
    """

    def __init__(
        self,
        requests_per_second: float = 10.0,
        burst_size: Optional[int] = None,
    ):
        """Initialize the rate limiter.

        Args:
            requests_per_second: Maximum sustained request rate
            burst_size: Maximum burst size (defaults to requests_per_second)
        """
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size or int(requests_per_second)
        self.tokens = float(self.burst_size)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    def _add_tokens(self) -> None:
        """Add tokens based on elapsed time since last update."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(
            self.burst_size,
            self.tokens + elapsed * self.requests_per_second,
        )
        self.last_update = now

    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire (default 1)
        """
        async with self._lock:
            self._add_tokens()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return

            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.requests_per_second

            await asyncio.sleep(wait_time)

            self._add_tokens()
            self.tokens -= tokens

    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens without waiting.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False otherwise
        """
        self._add_tokens()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True

        return False

    @property
    def available_tokens(self) -> float:
        """Get the current number of available tokens."""
        self._add_tokens()
        return self.tokens


class DailyRateLimiter:
    """Rate limiter that tracks daily API call limits."""

    def __init__(self, daily_limit: int):
        """Initialize daily rate limiter.

        Args:
            daily_limit: Maximum API calls per day
        """
        self.daily_limit = daily_limit
        self.calls_today = 0
        self.reset_date = time.strftime("%Y-%m-%d")

    def _check_reset(self) -> None:
        """Reset counter if it's a new day."""
        today = time.strftime("%Y-%m-%d")
        if today != self.reset_date:
            self.calls_today = 0
            self.reset_date = today

    def can_make_request(self) -> bool:
        """Check if we can make another request today."""
        self._check_reset()
        return self.calls_today < self.daily_limit

    def record_request(self) -> None:
        """Record that a request was made."""
        self._check_reset()
        self.calls_today += 1

    @property
    def remaining_calls(self) -> int:
        """Get remaining API calls for today."""
        self._check_reset()
        return max(0, self.daily_limit - self.calls_today)
