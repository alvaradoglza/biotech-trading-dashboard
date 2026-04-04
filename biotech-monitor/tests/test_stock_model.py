"""Tests for the Stock model."""

import pytest
from src.models.stock import Stock, MarketCapCategory, DataSource


class TestMarketCapCategory:
    """Tests for MarketCapCategory enum."""

    def test_from_market_cap_nano(self):
        """Test nano cap classification."""
        assert MarketCapCategory.from_market_cap(10_000_000) == MarketCapCategory.NANO
        assert MarketCapCategory.from_market_cap(49_999_999) == MarketCapCategory.NANO

    def test_from_market_cap_micro(self):
        """Test micro cap classification."""
        assert MarketCapCategory.from_market_cap(50_000_000) == MarketCapCategory.MICRO
        assert MarketCapCategory.from_market_cap(299_999_999) == MarketCapCategory.MICRO

    def test_from_market_cap_small(self):
        """Test small cap classification."""
        assert MarketCapCategory.from_market_cap(300_000_000) == MarketCapCategory.SMALL
        assert MarketCapCategory.from_market_cap(1_999_999_999) == MarketCapCategory.SMALL

    def test_from_market_cap_mid(self):
        """Test mid cap classification."""
        assert MarketCapCategory.from_market_cap(2_000_000_000) == MarketCapCategory.MID
        assert MarketCapCategory.from_market_cap(9_999_999_999) == MarketCapCategory.MID

    def test_from_market_cap_large(self):
        """Test large cap classification."""
        assert MarketCapCategory.from_market_cap(10_000_000_000) == MarketCapCategory.LARGE
        assert MarketCapCategory.from_market_cap(100_000_000_000) == MarketCapCategory.LARGE

    def test_from_market_cap_none(self):
        """Test None market cap."""
        assert MarketCapCategory.from_market_cap(None) is None


class TestDataSource:
    """Tests for DataSource enum."""

    def test_data_source_values(self):
        """Test DataSource enum values."""
        assert DataSource.EODHD_FUNDAMENTALS.value == "eodhd_fundamentals"
        assert DataSource.MANUAL.value == "manual"


class TestStock:
    """Tests for Stock dataclass."""

    def test_basic_creation(self):
        """Test basic stock creation."""
        stock = Stock(
            ticker="MRNA",
            company_name="Moderna Inc",
            exchange="NASDAQ",
        )
        assert stock.ticker == "MRNA"
        assert stock.company_name == "Moderna Inc"
        assert stock.exchange == "NASDAQ"

    def test_quality_score_full(self):
        """Test quality score with all data present."""
        stock = Stock(
            ticker="MRNA",
            company_name="Moderna Inc",
            exchange="NASDAQ",
            market_cap=15_000_000_000,
            industry="Biotechnology",
            cik="0001682852",
            website="https://www.modernatx.com",
            ir_url="https://investors.modernatx.com",
        )
        assert stock.data_quality_score == 100

    def test_quality_score_with_sic_instead_of_industry(self):
        """Test quality score when SIC is present but industry is not."""
        stock = Stock(
            ticker="MRNA",
            company_name="Moderna Inc",
            exchange="NASDAQ",
            market_cap=15_000_000_000,
            sic="2836",  # SIC instead of industry
            cik="0001682852",
            website="https://www.modernatx.com",
            ir_url="https://investors.modernatx.com",
        )
        assert stock.data_quality_score == 100  # SIC counts as industry

    def test_quality_score_missing_cik(self):
        """Test quality score deduction for missing CIK."""
        stock = Stock(
            ticker="TEST",
            company_name="Test Company",
            exchange="NASDAQ",
            market_cap=100_000_000,
            industry="Biotechnology",
            website="https://test.com",
            ir_url="https://test.com/investors",
        )
        assert stock.data_quality_score == 70  # -30 for missing CIK

    def test_quality_score_missing_ir_url(self):
        """Test quality score deduction for missing IR URL."""
        stock = Stock(
            ticker="TEST",
            company_name="Test Company",
            exchange="NASDAQ",
            market_cap=100_000_000,
            industry="Biotechnology",
            cik="0001234567",
            website="https://test.com",
        )
        assert stock.data_quality_score == 75  # -25 for missing IR URL

    def test_quality_score_missing_multiple(self):
        """Test quality score with multiple missing fields."""
        stock = Stock(
            ticker="TEST",
            company_name="Test Company",
            exchange="NASDAQ",
        )
        # -30 CIK, -25 IR URL, -20 market cap, -15 website, -10 industry/sic = 0
        assert stock.data_quality_score == 0

    def test_from_eodhd_fundamentals(self, sample_eodhd_fundamentals):
        """Test creating stock from EODHD fundamentals."""
        stock = Stock.from_eodhd_fundamentals(
            ticker="MRNA",
            exchange="NASDAQ",
            data=sample_eodhd_fundamentals,
        )

        assert stock.ticker == "MRNA"
        assert stock.company_name == "Moderna Inc"
        assert stock.exchange == "NASDAQ"
        assert stock.market_cap == 15_000_000_000
        assert stock.sector == "Healthcare"
        assert stock.industry == "Biotechnology"
        assert stock.cik == "0001682852"
        assert stock.website == "https://www.modernatx.com"
        assert stock.market_cap_category == MarketCapCategory.LARGE  # $15B is large cap
        assert stock.data_source == DataSource.EODHD_FUNDAMENTALS

    def test_from_eodhd_fundamentals_no_http(self):
        """Test that website URLs get http prefix added."""
        data = {
            "General": {
                "Name": "Test Co",
                "WebURL": "www.test.com",
            },
            "Highlights": {},
        }
        stock = Stock.from_eodhd_fundamentals("TEST", "NYSE", data)
        assert stock.website == "https://www.test.com"

    def test_to_dict(self, sample_eodhd_fundamentals):
        """Test converting stock to dictionary."""
        stock = Stock.from_eodhd_fundamentals(
            ticker="MRNA",
            exchange="NASDAQ",
            data=sample_eodhd_fundamentals,
        )
        d = stock.to_dict()

        assert d["ticker"] == "MRNA"
        assert d["company_name"] == "Moderna Inc"
        assert d["market_cap"] == 15_000_000_000
        assert d["market_cap_category"] == "large"  # $15B is large cap
        assert d["data_source"] == "eodhd_fundamentals"

    def test_to_dict_includes_new_fields(self):
        """Test that to_dict includes SIC and data_source fields."""
        stock = Stock(
            ticker="TEST",
            company_name="Test Co",
            exchange="NASDAQ",
            sic="2834",
            sic_description="Pharmaceutical Preparations",
            data_source=DataSource.EODHD_FUNDAMENTALS,
            shares_outstanding=100000000,
            last_price=10.50,
        )
        d = stock.to_dict()

        assert d["sic"] == "2834"
        assert d["sic_description"] == "Pharmaceutical Preparations"
        assert d["data_source"] == "eodhd_fundamentals"
        assert d["shares_outstanding"] == 100000000
        assert d["last_price"] == 10.50

    def test_from_dict(self):
        """Test creating stock from dictionary."""
        d = {
            "ticker": "MRNA",
            "company_name": "Moderna Inc",
            "exchange": "NASDAQ",
            "market_cap": 15_000_000_000,
            "market_cap_category": "large",
            "sector": "Healthcare",
            "industry": "Biotechnology",
            "cik": "0001682852",
            "data_source": "eodhd_fundamentals",
        }
        stock = Stock.from_dict(d)

        assert stock.ticker == "MRNA"
        assert stock.market_cap_category == MarketCapCategory.LARGE
        assert stock.data_source == DataSource.EODHD_FUNDAMENTALS

    def test_from_dict_with_sic(self):
        """Test creating stock from dictionary with SIC fields."""
        d = {
            "ticker": "TEST",
            "company_name": "Test Pharma",
            "exchange": "NYSE",
            "sic": "2834",
            "sic_description": "Pharmaceutical Preparations",
            "data_source": "eodhd_fundamentals",
        }
        stock = Stock.from_dict(d)

        assert stock.sic == "2834"
        assert stock.sic_description == "Pharmaceutical Preparations"
        assert stock.data_source == DataSource.EODHD_FUNDAMENTALS

    def test_get_quality_issues(self):
        """Test getting quality issues list."""
        stock = Stock(
            ticker="TEST",
            company_name="Test Company",
            exchange="NASDAQ",
        )
        issues = stock.get_quality_issues()

        assert len(issues) == 5
        assert any("CIK" in issue for issue in issues)
        assert any("IR URL" in issue for issue in issues)
        assert any("market cap" in issue for issue in issues)
        assert any("website" in issue for issue in issues)
        assert any("industry" in issue for issue in issues)

    def test_get_quality_issues_with_sic(self):
        """Test that SIC counts as industry for quality issues."""
        stock = Stock(
            ticker="TEST",
            company_name="Test Company",
            exchange="NASDAQ",
            sic="2834",
        )
        issues = stock.get_quality_issues()

        # Should not have industry issue since SIC is present
        assert not any("industry" in issue.lower() for issue in issues)
