"""Stock data model with quality scoring."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class MarketCapCategory(Enum):
    """Market capitalization categories."""

    NANO = "nano"  # < $50M
    MICRO = "micro"  # $50M - $300M
    SMALL = "small"  # $300M - $2B
    MID = "mid"  # $2B - $10B
    LARGE = "large"  # > $10B

    @classmethod
    def from_market_cap(cls, market_cap: Optional[float]) -> Optional["MarketCapCategory"]:
        """Determine category from market cap value.

        Args:
            market_cap: Market capitalization in USD

        Returns:
            MarketCapCategory or None if market_cap is None
        """
        if market_cap is None:
            return None

        if market_cap < 50_000_000:
            return cls.NANO
        elif market_cap < 300_000_000:
            return cls.MICRO
        elif market_cap < 2_000_000_000:
            return cls.SMALL
        elif market_cap < 10_000_000_000:
            return cls.MID
        else:
            return cls.LARGE


class DataSource(Enum):
    """Source of the stock data."""

    EODHD_FUNDAMENTALS = "eodhd_fundamentals"
    MANUAL = "manual"


@dataclass
class Stock:
    """Represents a stock in the biopharma universe.

    Attributes:
        ticker: Stock ticker symbol
        company_name: Full company name
        exchange: Exchange where the stock trades
        market_cap: Market capitalization in USD
        market_cap_category: Categorized market cap (micro/small/mid/etc)
        sector: Business sector
        industry: Specific industry classification
        sic: SEC Standard Industrial Classification code
        sic_description: SEC SIC code description
        cik: SEC Central Index Key
        website: Company website URL
        ir_url: Investor relations page URL
        data_quality_score: Quality score from 0-100
        data_source: Where the data came from
        created_at: When this record was created
        updated_at: When this record was last updated
    """

    ticker: str
    company_name: str
    exchange: str
    market_cap: Optional[float] = None
    market_cap_category: Optional[MarketCapCategory] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    sic: Optional[str] = None
    sic_description: Optional[str] = None
    cik: Optional[str] = None
    website: Optional[str] = None
    ir_url: Optional[str] = None
    data_quality_score: int = 0
    data_source: Optional[DataSource] = None
    isin: Optional[str] = None
    cusip: Optional[str] = None
    country: Optional[str] = None
    currency: Optional[str] = None
    shares_outstanding: Optional[int] = None
    last_price: Optional[float] = None
    created_at: Optional[datetime] = field(default=None)
    updated_at: Optional[datetime] = field(default=None)

    def __post_init__(self):
        """Calculate derived fields after initialization."""
        if self.market_cap is not None and self.market_cap_category is None:
            self.market_cap_category = MarketCapCategory.from_market_cap(self.market_cap)
        self.data_quality_score = self.calculate_quality_score()

    def calculate_quality_score(self) -> int:
        """Calculate data quality score based on completeness.

        Returns:
            Score from 0-100 where 100 is fully complete
        """
        score = 100

        if not self.cik:
            score -= 30

        if not self.ir_url:
            score -= 25

        if self.market_cap is None:
            score -= 20

        if not self.website:
            score -= 15

        # Check for industry or SIC
        if not self.industry and not self.sic:
            score -= 10

        return max(0, score)

    @classmethod
    def from_eodhd_fundamentals(
        cls,
        ticker: str,
        exchange: str,
        data: dict[str, Any],
    ) -> "Stock":
        """Create a Stock from EODHD fundamentals API response.

        Args:
            ticker: Stock ticker symbol
            exchange: Exchange code
            data: EODHD fundamentals API response

        Returns:
            Stock instance populated from the API data
        """
        general = data.get("General", {})
        highlights = data.get("Highlights", {})

        market_cap = highlights.get("MarketCapitalization")
        if market_cap is not None:
            try:
                market_cap = float(market_cap)
            except (ValueError, TypeError):
                market_cap = None

        website = general.get("WebURL")
        if website and not website.startswith(("http://", "https://")):
            website = f"https://{website}"

        cik = general.get("CIK")
        if cik:
            cik = str(cik).zfill(10)

        return cls(
            ticker=ticker,
            company_name=general.get("Name", ticker),
            exchange=exchange,
            market_cap=market_cap,
            sector=general.get("Sector"),
            industry=general.get("Industry"),
            cik=cik,
            website=website,
            isin=general.get("ISIN"),
            cusip=general.get("CUSIP"),
            country=general.get("CountryISO") or general.get("Country"),
            currency=general.get("CurrencyCode"),
            data_source=DataSource.EODHD_FUNDAMENTALS,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert stock to dictionary for serialization.

        Returns:
            Dictionary representation of the stock
        """
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "exchange": self.exchange,
            "market_cap": self.market_cap,
            "market_cap_category": (
                self.market_cap_category.value if self.market_cap_category else None
            ),
            "sector": self.sector,
            "industry": self.industry,
            "sic": self.sic,
            "sic_description": self.sic_description,
            "cik": self.cik,
            "website": self.website,
            "ir_url": self.ir_url,
            "data_quality_score": self.data_quality_score,
            "data_source": self.data_source.value if self.data_source else None,
            "isin": self.isin,
            "cusip": self.cusip,
            "country": self.country,
            "currency": self.currency,
            "shares_outstanding": self.shares_outstanding,
            "last_price": self.last_price,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Stock":
        """Create a Stock from a dictionary.

        Args:
            data: Dictionary with stock fields

        Returns:
            Stock instance
        """
        market_cap_cat = data.get("market_cap_category")
        if market_cap_cat and isinstance(market_cap_cat, str):
            market_cap_cat = MarketCapCategory(market_cap_cat)

        data_source = data.get("data_source")
        if data_source and isinstance(data_source, str):
            try:
                data_source = DataSource(data_source)
            except ValueError:
                data_source = None

        return cls(
            ticker=data["ticker"],
            company_name=data["company_name"],
            exchange=data["exchange"],
            market_cap=data.get("market_cap"),
            market_cap_category=market_cap_cat,
            sector=data.get("sector"),
            industry=data.get("industry"),
            sic=data.get("sic"),
            sic_description=data.get("sic_description"),
            cik=data.get("cik"),
            website=data.get("website"),
            ir_url=data.get("ir_url"),
            data_source=data_source,
            isin=data.get("isin"),
            cusip=data.get("cusip"),
            country=data.get("country"),
            currency=data.get("currency"),
            shares_outstanding=data.get("shares_outstanding"),
            last_price=data.get("last_price"),
        )

    def get_quality_issues(self) -> list[str]:
        """Get list of data quality issues.

        Returns:
            List of issue descriptions
        """
        issues = []

        if not self.cik:
            issues.append("Missing CIK")

        if not self.ir_url:
            issues.append("Missing IR URL (press releases unavailable)")

        if self.market_cap is None:
            issues.append("Missing market cap")

        if not self.website:
            issues.append("Missing website")

        if not self.industry and not self.sic:
            issues.append("Missing industry classification")

        return issues
