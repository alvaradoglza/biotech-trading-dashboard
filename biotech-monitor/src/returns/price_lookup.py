"""Price lookup with trading day handling for return calculations."""

import bisect
from datetime import date, timedelta
from typing import Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)


class PriceLookup:
    """
    Efficient price lookup with trading day handling.

    Prices are stored as a sorted list for binary search efficiency.
    Handles weekends and holidays by finding the closest trading day.
    """

    def __init__(self, prices: list[tuple[date, float]]):
        """
        Initialize with price data.

        Args:
            prices: List of (date, adjusted_close) tuples, sorted by date ascending

        Raises:
            ValueError: If price list is empty
        """
        if not prices:
            raise ValueError("Price list cannot be empty")

        # Sort by date to ensure correct order
        sorted_prices = sorted(prices, key=lambda p: p[0])

        self.dates = [p[0] for p in sorted_prices]
        self.prices = {p[0]: p[1] for p in sorted_prices}

        self.min_date = self.dates[0]
        self.max_date = self.dates[-1]

    def __len__(self) -> int:
        """Return number of price points."""
        return len(self.dates)

    def get_start_price(self, announcement_date: date) -> Optional[tuple[date, float]]:
        """
        Get price for T+1 (next trading day after announcement).

        Rules:
        - Use the NEXT trading day AFTER announcement_date (T+1)
        - This accounts for after-hours announcements (earnings calls, FDA decisions, etc.)
        - The first moment an investor can act on the news is the next trading day

        Args:
            announcement_date: The announcement date

        Returns:
            Tuple of (actual_date_used, price) or None if no trading days after announcement
        """
        # Find first trading day AFTER announcement_date
        idx = bisect.bisect_right(self.dates, announcement_date)

        if idx >= len(self.dates):
            # No trading days after announcement date
            logger.debug(
                "No start price found - no trading days after announcement",
                announcement_date=str(announcement_date),
                max_date=str(self.max_date),
            )
            return None

        next_trading_day = self.dates[idx]
        logger.debug(
            "Using next trading day (T+1) for start",
            announcement_date=str(announcement_date),
            actual_date=str(next_trading_day),
        )
        return (next_trading_day, self.prices[next_trading_day])

    def get_end_price(
        self, announcement_date: date, days_after: int
    ) -> Optional[tuple[date, float]]:
        """
        Get price for end of return calculation.

        Rules:
        - Target date is announcement_date + days_after calendar days
        - If target is a trading day, use that day's close
        - If weekend/holiday, use NEXT available trading day

        Args:
            announcement_date: The announcement date
            days_after: Number of calendar days after announcement (30, 60, or 90)

        Returns:
            Tuple of (actual_date_used, price) or None if not found
        """
        target_date = announcement_date + timedelta(days=days_after)

        # Find position where target_date would be inserted
        idx = bisect.bisect_left(self.dates, target_date)

        if idx >= len(self.dates):
            # No trading days on or after target date
            logger.debug(
                "No end price found - target date beyond available data",
                announcement_date=str(announcement_date),
                target_date=str(target_date),
                max_date=str(self.max_date),
            )
            return None

        # Check if exact date exists
        if self.dates[idx] == target_date:
            return (target_date, self.prices[target_date])

        # Use next trading day
        next_date = self.dates[idx]
        logger.debug(
            "Using next trading day for end",
            target_date=str(target_date),
            actual_date=str(next_date),
        )
        return (next_date, self.prices[next_date])

    def has_sufficient_data(self, announcement_date: date, days_after: int = 90) -> bool:
        """
        Check if we have enough data to calculate returns.

        Args:
            announcement_date: The announcement date
            days_after: Maximum days after to check (default 90)

        Returns:
            True if we have data for both start and end dates
        """
        start = self.get_start_price(announcement_date)
        if start is None:
            return False

        end = self.get_end_price(announcement_date, days_after)
        return end is not None

    def get_price_on_date(self, target_date: date) -> Optional[float]:
        """
        Get the exact price on a specific date.

        Args:
            target_date: The date to look up

        Returns:
            Price if available, None otherwise
        """
        return self.prices.get(target_date)

    def get_date_range(self) -> tuple[date, date]:
        """
        Get the date range of available price data.

        Returns:
            Tuple of (min_date, max_date)
        """
        return (self.min_date, self.max_date)
