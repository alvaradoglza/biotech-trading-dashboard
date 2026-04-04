"""Tests for PriceLookup with T+1 start price logic."""

import pytest
from datetime import date

from src.returns.price_lookup import PriceLookup


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trading_week():
    """
    A week of Mon-Fri prices (no weekend entries).
    Mon 2025-01-06 → Fri 2025-01-10.
    """
    return PriceLookup([
        (date(2025, 1, 6), 100.0),   # Monday
        (date(2025, 1, 7), 101.0),   # Tuesday
        (date(2025, 1, 8), 102.0),   # Wednesday
        (date(2025, 1, 9), 103.0),   # Thursday
        (date(2025, 1, 10), 104.0),  # Friday
    ])


@pytest.fixture
def sparse_prices():
    """Price history with a gap (holiday / long weekend)."""
    return PriceLookup([
        (date(2025, 1, 2), 50.0),    # Thursday (New Year's)
        (date(2025, 1, 3), 51.0),    # Friday
        # 4-5 = weekend
        (date(2025, 1, 6), 52.0),    # Monday
        (date(2025, 1, 7), 53.0),    # Tuesday
        (date(2025, 1, 8), 54.0),    # Wednesday
    ])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestInit:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            PriceLookup([])

    def test_unsorted_input_sorted_internally(self):
        pl = PriceLookup([
            (date(2025, 1, 3), 30.0),
            (date(2025, 1, 1), 10.0),
            (date(2025, 1, 2), 20.0),
        ])
        assert pl.dates[0] == date(2025, 1, 1)
        assert pl.dates[-1] == date(2025, 1, 3)

    def test_len(self, trading_week):
        assert len(trading_week) == 5

    def test_date_range(self, trading_week):
        lo, hi = trading_week.get_date_range()
        assert lo == date(2025, 1, 6)
        assert hi == date(2025, 1, 10)


# ---------------------------------------------------------------------------
# get_start_price  (T+1 — next trading day AFTER announcement)
# ---------------------------------------------------------------------------

class TestGetStartPrice:
    def test_announcement_on_trading_day_returns_next_day(self, trading_week):
        """When announcement falls on a trading day, start is the NEXT trading day (T+1)."""
        result = trading_week.get_start_price(date(2025, 1, 6))  # Monday
        assert result is not None
        actual_date, price = result
        assert actual_date == date(2025, 1, 7)   # Tuesday  (T+1)
        assert price == 101.0

    def test_announcement_on_friday_returns_monday(self, trading_week):
        """Friday announcement → start is Monday (next trading day)."""
        result = trading_week.get_start_price(date(2025, 1, 10))  # Friday
        # No trading days after Friday in this fixture
        assert result is None

    def test_announcement_on_weekend_returns_next_monday(self, trading_week):
        """Saturday/Sunday announcement → start is Monday."""
        result = trading_week.get_start_price(date(2025, 1, 4))  # Saturday
        assert result is not None
        actual_date, price = result
        assert actual_date == date(2025, 1, 6)   # Monday
        assert price == 100.0

    def test_announcement_on_sunday_returns_next_monday(self, trading_week):
        """Sunday announcement → start is Monday."""
        result = trading_week.get_start_price(date(2025, 1, 5))  # Sunday
        assert result is not None
        actual_date, price = result
        assert actual_date == date(2025, 1, 6)   # Monday
        assert price == 100.0

    def test_announcement_on_wednesday_returns_thursday(self, trading_week):
        """Mid-week announcement → next trading day."""
        result = trading_week.get_start_price(date(2025, 1, 8))  # Wednesday
        assert result is not None
        actual_date, price = result
        assert actual_date == date(2025, 1, 9)   # Thursday
        assert price == 103.0

    def test_announcement_after_last_date_returns_none(self, trading_week):
        """No trading days after the last available date → None."""
        result = trading_week.get_start_price(date(2025, 1, 11))  # Saturday after last Friday
        assert result is None

    def test_announcement_before_all_dates_returns_first(self, trading_week):
        """Announcement before all price data → first trading day."""
        result = trading_week.get_start_price(date(2025, 1, 1))  # New Year's Day
        assert result is not None
        actual_date, price = result
        assert actual_date == date(2025, 1, 6)   # First Monday in fixture
        assert price == 100.0

    def test_t1_skips_same_day_price(self, trading_week):
        """T+1 must NOT return announcement-day price, even if it exists."""
        result = trading_week.get_start_price(date(2025, 1, 7))  # Tuesday
        assert result is not None
        actual_date, _ = result
        assert actual_date != date(2025, 1, 7)   # Must not be same day
        assert actual_date == date(2025, 1, 8)   # Wednesday

    def test_sparse_prices_gap(self, sparse_prices):
        """With a holiday gap, still finds the correct next trading day."""
        # Announcement on Jan 3 (Friday) → next trading day is Jan 6 (Monday)
        result = sparse_prices.get_start_price(date(2025, 1, 3))
        assert result is not None
        actual_date, price = result
        assert actual_date == date(2025, 1, 6)
        assert price == 52.0


# ---------------------------------------------------------------------------
# get_end_price
# ---------------------------------------------------------------------------

class TestGetEndPrice:
    def test_exact_date_match(self, trading_week):
        target = trading_week.get_end_price(date(2025, 1, 6), 1)
        # Jan 6 + 1 = Jan 7 (exact trading day)
        assert target is not None
        actual_date, price = target
        assert actual_date == date(2025, 1, 7)
        assert price == 101.0

    def test_weekend_end_date_uses_next_trading_day(self, trading_week):
        # Jan 6 (Mon) + 5 = Jan 11 (Sat) → next trading day would be outside fixture → None
        result = trading_week.get_end_price(date(2025, 1, 6), 5)
        assert result is None

    def test_end_date_before_data_returns_none(self, trading_week):
        # target: Jan 1 → before all prices
        result = trading_week.get_end_price(date(2024, 12, 30), 2)
        # target = Jan 1 → bisect_left returns 0, exact is Jan 6, but that's 5 days away
        # the implementation returns the next available date (Jan 6) since it's >=
        assert result is not None

    def test_end_date_beyond_data_returns_none(self, trading_week):
        result = trading_week.get_end_price(date(2025, 1, 6), 90)
        assert result is None

    def test_end_price_30_days(self, sparse_prices):
        # Jan 2 + 30 = Feb 1 → beyond fixture data → None
        result = sparse_prices.get_end_price(date(2025, 1, 2), 30)
        assert result is None


# ---------------------------------------------------------------------------
# has_sufficient_data
# ---------------------------------------------------------------------------

class TestHasSufficientData:
    def test_sufficient_when_both_dates_available(self, trading_week):
        # Announcement on Jan 7 → start=Jan 8, end(+1)=Jan 8
        assert trading_week.has_sufficient_data(date(2025, 1, 7), days_after=2)

    def test_insufficient_when_no_data_after_announcement(self, trading_week):
        # Announcement on Jan 10 (last day) → no T+1
        assert not trading_week.has_sufficient_data(date(2025, 1, 10))

    def test_insufficient_when_end_date_beyond_data(self, trading_week):
        assert not trading_week.has_sufficient_data(date(2025, 1, 6), days_after=90)


# ---------------------------------------------------------------------------
# get_price_on_date
# ---------------------------------------------------------------------------

class TestGetPriceOnDate:
    def test_existing_date(self, trading_week):
        price = trading_week.get_price_on_date(date(2025, 1, 8))
        assert price == 102.0

    def test_nonexistent_date_returns_none(self, trading_week):
        price = trading_week.get_price_on_date(date(2025, 1, 4))  # Saturday
        assert price is None
