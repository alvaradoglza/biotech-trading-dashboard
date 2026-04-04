"""Return calculation logic for post-announcement stock performance."""

from dataclasses import dataclass
from datetime import date
from typing import Optional

from src.returns.price_lookup import PriceLookup
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReturnResult:
    """Result of a return calculation."""

    announcement_date: date
    days_after: int
    start_date: date
    start_price: float
    end_date: date
    end_price: float
    return_pct: float  # Percentage return (e.g., 15.43 for 15.43%)

    def __str__(self) -> str:
        return (
            f"Return({self.days_after}d): {self.return_pct:+.2f}% "
            f"({self.start_date} ${self.start_price:.2f} -> "
            f"{self.end_date} ${self.end_price:.2f})"
        )


class ReturnCalculator:
    """
    Calculate post-announcement stock returns.

    Calculates returns at 30, 60, and 90 calendar days after announcements
    using adjusted close prices.
    """

    # Standard return periods in calendar days
    RETURN_PERIODS = [5, 30, 60, 90]

    def __init__(self, price_lookup: PriceLookup):
        """
        Initialize calculator with price data.

        Args:
            price_lookup: PriceLookup instance with historical prices
        """
        self.price_lookup = price_lookup

    def calculate_return(
        self, announcement_date: date, days_after: int
    ) -> Optional[ReturnResult]:
        """
        Calculate return for a specific period.

        Formula: ((end_price - start_price) / start_price) * 100

        Args:
            announcement_date: Date of the announcement
            days_after: Number of calendar days after announcement

        Returns:
            ReturnResult if calculation successful, None otherwise
        """
        # Get start price (announcement day or previous trading day)
        start_result = self.price_lookup.get_start_price(announcement_date)
        if start_result is None:
            logger.debug(
                "Cannot calculate return - no start price",
                announcement_date=str(announcement_date),
                days_after=days_after,
            )
            return None

        start_date, start_price = start_result

        # Get end price (target day or next trading day)
        end_result = self.price_lookup.get_end_price(announcement_date, days_after)
        if end_result is None:
            logger.debug(
                "Cannot calculate return - no end price",
                announcement_date=str(announcement_date),
                days_after=days_after,
            )
            return None

        end_date, end_price = end_result

        # Calculate return percentage
        return_pct = ((end_price - start_price) / start_price) * 100

        # Round to 2 decimal places
        return_pct = round(return_pct, 2)

        return ReturnResult(
            announcement_date=announcement_date,
            days_after=days_after,
            start_date=start_date,
            start_price=start_price,
            end_date=end_date,
            end_price=end_price,
            return_pct=return_pct,
        )

    def calculate_all_returns(
        self, announcement_date: date
    ) -> dict[int, Optional[ReturnResult]]:
        """
        Calculate returns for all standard periods (30, 60, 90 days).

        Args:
            announcement_date: Date of the announcement

        Returns:
            Dictionary mapping days_after -> ReturnResult (or None if unavailable)
        """
        results = {}
        for days in self.RETURN_PERIODS:
            results[days] = self.calculate_return(announcement_date, days)
        return results

    def calculate_batch(
        self, announcement_dates: list[date]
    ) -> dict[date, dict[int, Optional[ReturnResult]]]:
        """
        Calculate returns for multiple announcements.

        Args:
            announcement_dates: List of announcement dates

        Returns:
            Dictionary mapping announcement_date -> {days_after -> ReturnResult}
        """
        results = {}
        for ann_date in announcement_dates:
            results[ann_date] = self.calculate_all_returns(ann_date)
        return results

    def get_return_values(
        self, announcement_date: date
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Get just the return percentages for 5, 30, 60, 90 days.

        Convenience method for updating CSV fields.

        Args:
            announcement_date: Date of the announcement

        Returns:
            Tuple of (return_5d, return_30d, return_60d, return_90d) - values are None if unavailable
        """
        results = self.calculate_all_returns(announcement_date)

        return_5d = results[5].return_pct if results[5] else None
        return_30d = results[30].return_pct if results[30] else None
        return_60d = results[60].return_pct if results[60] else None
        return_90d = results[90].return_pct if results[90] else None

        return (return_5d, return_30d, return_60d, return_90d)
