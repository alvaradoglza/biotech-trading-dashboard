"""Tests for the StockFilter class."""

import pytest
from src.clients.eodhd import StockFilter
from src.models.stock import Stock


class TestStockFilterIndustry:
    """Tests for industry filtering."""

    @pytest.fixture
    def filter(self):
        """Create default stock filter."""
        return StockFilter()

    def test_biotechnology_matches(self, filter):
        """Test that biotechnology matches."""
        assert filter.is_biopharma_industry("Biotechnology")
        assert filter.is_biopharma_industry("biotechnology")
        assert filter.is_biopharma_industry("BIOTECHNOLOGY")

    def test_pharmaceuticals_matches(self, filter):
        """Test that pharmaceuticals matches."""
        assert filter.is_biopharma_industry("Pharmaceuticals")
        assert filter.is_biopharma_industry("Pharmaceutical")

    def test_drug_manufacturers_matches(self, filter):
        """Test that drug manufacturers matches."""
        assert filter.is_biopharma_industry("Drug Manufacturers")
        assert filter.is_biopharma_industry("Drug Manufacturers - General")
        assert filter.is_biopharma_industry("Drug Manufacturers - Specialty & Generic")
        assert filter.is_biopharma_industry("Drug Manufacturers—General")

    def test_diagnostics_matches(self, filter):
        """Test that diagnostics & research matches."""
        assert filter.is_biopharma_industry("Diagnostics & Research")

    def test_non_biopharma_does_not_match(self, filter):
        """Test that non-biopharma industries don't match."""
        assert not filter.is_biopharma_industry("Consumer Electronics")
        assert not filter.is_biopharma_industry("Software")
        assert not filter.is_biopharma_industry("Banking")
        assert not filter.is_biopharma_industry("Automotive")

    def test_none_does_not_match(self, filter):
        """Test that None doesn't match."""
        assert not filter.is_biopharma_industry(None)

    def test_empty_does_not_match(self, filter):
        """Test that empty string doesn't match."""
        assert not filter.is_biopharma_industry("")


class TestStockFilterMarketCap:
    """Tests for market cap filtering."""

    def test_default_max_is_2b(self):
        """Test default max market cap is $2B."""
        filter = StockFilter()
        assert filter.max_market_cap == 2_000_000_000

    def test_under_2b_passes(self):
        """Test that under $2B passes."""
        filter = StockFilter()
        assert filter.passes_market_cap_filter(1_500_000_000)
        assert filter.passes_market_cap_filter(100_000_000)
        assert filter.passes_market_cap_filter(0)

    def test_exactly_2b_passes(self):
        """Test that exactly $2B passes."""
        filter = StockFilter()
        assert filter.passes_market_cap_filter(2_000_000_000)

    def test_over_2b_fails(self):
        """Test that over $2B fails."""
        filter = StockFilter()
        assert not filter.passes_market_cap_filter(2_000_000_001)
        assert not filter.passes_market_cap_filter(15_000_000_000)

    def test_none_passes_by_default(self):
        """Test that None passes by default."""
        filter = StockFilter()
        assert filter.passes_market_cap_filter(None)

    def test_none_fails_when_disabled(self):
        """Test that None fails when include_missing_market_cap is False."""
        filter = StockFilter(include_missing_market_cap=False)
        assert not filter.passes_market_cap_filter(None)

    def test_custom_market_cap_range(self):
        """Test custom market cap range."""
        filter = StockFilter(min_market_cap=100_000_000, max_market_cap=500_000_000)
        assert not filter.passes_market_cap_filter(50_000_000)
        assert filter.passes_market_cap_filter(200_000_000)
        assert not filter.passes_market_cap_filter(600_000_000)


class TestStockFilterSymbol:
    """Tests for symbol pre-filtering."""

    @pytest.fixture
    def filter(self):
        """Create stock filter with exchange restrictions."""
        return StockFilter(exchanges=["NASDAQ", "NYSE"])

    def test_common_stock_passes(self, filter):
        """Test that common stock passes."""
        assert filter.is_common_stock("Common Stock")
        assert filter.is_common_stock("common stock")

    def test_non_common_stock_fails(self, filter):
        """Test that non-common stock fails."""
        assert not filter.is_common_stock("ETF")
        assert not filter.is_common_stock("Preferred Stock")
        assert not filter.is_common_stock("REIT")
        assert not filter.is_common_stock(None)

    def test_filter_symbol_common_stock_nasdaq(self, filter):
        """Test filter_symbol with common stock on NASDAQ."""
        symbol = {"Code": "TEST", "Type": "Common Stock", "Exchange": "NASDAQ"}
        assert filter.filter_symbol(symbol)

    def test_filter_symbol_etf_fails(self, filter):
        """Test filter_symbol rejects ETFs."""
        symbol = {"Code": "SPY", "Type": "ETF", "Exchange": "NYSE"}
        assert not filter.filter_symbol(symbol)

    def test_filter_symbol_wrong_exchange_fails(self, filter):
        """Test filter_symbol rejects wrong exchange."""
        symbol = {"Code": "TEST", "Type": "Common Stock", "Exchange": "LSE"}
        assert not filter.filter_symbol(symbol)


class TestStockFilterFull:
    """Tests for full stock filtering."""

    @pytest.fixture
    def filter(self):
        """Create default stock filter."""
        return StockFilter(exchanges=["NASDAQ", "NYSE"])

    def test_biopharma_small_cap_passes(self, filter, sample_small_cap_fundamentals):
        """Test that biopharma small cap passes."""
        stock = Stock.from_eodhd_fundamentals(
            "SNDX", "NASDAQ", sample_small_cap_fundamentals
        )
        assert filter.filter_stock(stock)

    def test_biopharma_large_cap_fails(self, filter, sample_eodhd_fundamentals):
        """Test that biopharma large cap fails (over $2B)."""
        stock = Stock.from_eodhd_fundamentals(
            "MRNA", "NASDAQ", sample_eodhd_fundamentals
        )
        # MRNA has $15B market cap
        assert not filter.filter_stock(stock)

    def test_non_biopharma_fails(self, filter, sample_non_biopharma_fundamentals):
        """Test that non-biopharma fails."""
        stock = Stock.from_eodhd_fundamentals(
            "AAPL", "NASDAQ", sample_non_biopharma_fundamentals
        )
        assert not filter.filter_stock(stock)

    def test_sic_based_filtering_passes(self, filter):
        """Test that stocks with biopharma SIC codes pass."""
        stock = Stock(
            ticker="TEST",
            company_name="Test Pharma",
            exchange="NASDAQ",
            market_cap=500_000_000,
            sic="2834",  # Pharmaceutical Preparations
        )
        assert filter.filter_stock(stock)

    def test_sic_based_filtering_fails_non_pharma(self, filter):
        """Test that stocks with non-biopharma SIC codes fail."""
        stock = Stock(
            ticker="TEST",
            company_name="Test Tech",
            exchange="NASDAQ",
            market_cap=500_000_000,
            sic="7372",  # Software
        )
        assert not filter.filter_stock(stock)

    def test_from_config(self):
        """Test creating filter from config."""
        config = {
            "exchanges": ["NASDAQ"],
            "market_cap": {"min": 0, "max": 1_000_000_000},
            "include_missing_market_cap": False,
        }
        filter = StockFilter.from_config(config)

        assert filter.exchanges == ["NASDAQ"]
        assert filter.max_market_cap == 1_000_000_000
        assert filter.include_missing_market_cap is False
