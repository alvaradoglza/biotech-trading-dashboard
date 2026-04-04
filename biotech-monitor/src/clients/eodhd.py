"""EODHD API client for stock data and fundamentals."""

import re
from typing import Any, Optional

from src.clients.base import BaseAPIClient, APIError, AuthenticationError
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EODHDAPIError(APIError):
    """EODHD-specific API error."""

    pass


class EODHDEntitlementError(EODHDAPIError):
    """EODHD plan does not include the requested endpoint.

    This is raised when the API returns 403 Forbidden, indicating
    the API key is valid but the subscription plan doesn't include
    access to the endpoint (e.g., Fundamentals requires separate plan).
    """

    pass


class EODHDClient(BaseAPIClient):
    """Client for the EODHD API.

    EODHD provides stock market data including:
    - Exchange symbol lists
    - Company fundamentals (requires Fundamentals plan)
    - Historical prices
    - Real-time quotes
    """

    BASE_URL = "https://eodhd.com/api"

    def __init__(
        self,
        api_key: str,
        requests_per_second: float = 5.0,
    ):
        """Initialize the EODHD client.

        Args:
            api_key: EODHD API key
            requests_per_second: Maximum request rate (default 5/sec)
        """
        super().__init__(
            base_url=self.BASE_URL,
            api_key=api_key,
            requests_per_second=requests_per_second,
            user_agent="BiopharmaMonitor/1.0",
        )
        self._fundamentals_available: Optional[bool] = None

    async def check_fundamentals_access(self) -> bool:
        """Check if the API key has access to the Fundamentals endpoint.

        Makes a test request to fundamentals/AAPL.US to verify access.
        This should be called at startup to determine if fallback mode is needed.

        Returns:
            True if fundamentals endpoint is accessible, False otherwise
        """
        if self._fundamentals_available is not None:
            return self._fundamentals_available

        try:
            await self.get_fundamentals_filtered("AAPL", "US")
            self._fundamentals_available = True
            logger.info("EODHD Fundamentals endpoint accessible")
            return True
        except (EODHDEntitlementError, AuthenticationError):
            self._fundamentals_available = False
            logger.warning(
                "EODHD Fundamentals endpoint not accessible - "
                "plan may not include Fundamentals. Will use SEC+EOD fallback."
            )
            return False
        except EODHDAPIError as e:
            # Other API errors (not auth related) - assume accessible but had transient issue
            logger.warning(f"EODHD Fundamentals check had error: {e}")
            return True

    @property
    def fundamentals_available(self) -> Optional[bool]:
        """Whether fundamentals endpoint is available (None if not checked yet)."""
        return self._fundamentals_available

    async def get_exchange_symbols(
        self,
        exchange: str = "US",
    ) -> list[dict[str, Any]]:
        """Get all symbols listed on an exchange.

        Args:
            exchange: Exchange code (default "US" for all US exchanges)

        Returns:
            List of symbol dictionaries with Code, Name, Country, Exchange, etc.
        """
        endpoint = f"exchange-symbol-list/{exchange}"
        params = {
            "api_token": self.api_key,
            "fmt": "json",
        }

        logger.info("Fetching exchange symbols", exchange=exchange)

        response = await self.get_json(endpoint, params=params)

        if isinstance(response, dict) and "error" in response:
            raise EODHDAPIError(f"EODHD API error: {response['error']}")

        logger.info(
            "Fetched exchange symbols",
            exchange=exchange,
            count=len(response),
        )

        return response

    async def get_fundamentals(
        self,
        ticker: str,
        exchange: str = "US",
    ) -> dict[str, Any]:
        """Get company fundamentals.

        Args:
            ticker: Stock ticker symbol
            exchange: Exchange code

        Returns:
            Fundamentals data including General, Highlights, Valuation, etc.

        Raises:
            EODHDEntitlementError: If plan doesn't include Fundamentals
            EODHDAPIError: For other API errors
        """
        symbol = f"{ticker}.{exchange}"
        endpoint = f"fundamentals/{symbol}"
        params = {
            "api_token": self.api_key,
            "fmt": "json",
        }

        try:
            response = await self.get_json(endpoint, params=params)
        except AuthenticationError as e:
            # 403 on fundamentals means plan doesn't include it
            raise EODHDEntitlementError(
                f"Fundamentals endpoint not available on this plan. "
                f"Upgrade to Fundamentals Data Feed or All-In-One plan. "
                f"Original error: {e}"
            ) from e

        if isinstance(response, dict) and "error" in response:
            raise EODHDAPIError(
                f"EODHD API error for {ticker}: {response['error']}"
            )

        return response

    async def get_fundamentals_filtered(
        self,
        ticker: str,
        exchange: str = "US",
    ) -> dict[str, Any]:
        """Get filtered company fundamentals (General + Highlights only).

        This reduces data transfer and API costs by only requesting
        the sections we need for stock filtering.

        Args:
            ticker: Stock ticker symbol
            exchange: Exchange code

        Returns:
            Filtered fundamentals with only General and Highlights sections

        Raises:
            EODHDEntitlementError: If plan doesn't include Fundamentals
            EODHDAPIError: For other API errors
        """
        symbol = f"{ticker}.{exchange}"
        endpoint = f"fundamentals/{symbol}"
        params = {
            "api_token": self.api_key,
            "fmt": "json",
            "filter": "General,Highlights",
        }

        logger.debug(
            "EODHD fundamentals request",
            ticker=ticker,
            endpoint=endpoint,
        )

        try:
            response = await self.get_json(endpoint, params=params)
        except AuthenticationError as e:
            # 403 on fundamentals means plan doesn't include it
            raise EODHDEntitlementError(
                f"Fundamentals endpoint not available on this plan for {ticker}. "
                f"Upgrade to Fundamentals Data Feed or All-In-One plan."
            ) from e

        if isinstance(response, dict) and "error" in response:
            raise EODHDAPIError(
                f"EODHD API error for {ticker}: {response['error']}"
            )

        return response

    async def get_last_close(
        self,
        ticker: str,
        exchange: str = "US",
    ) -> Optional[float]:
        """Get the last closing price for a stock.

        Uses the EOD endpoint with filter=last_close which is available
        on basic EOD plans (doesn't require Fundamentals subscription).

        Args:
            ticker: Stock ticker symbol
            exchange: Exchange code

        Returns:
            Last closing price as float, or None if unavailable
        """
        symbol = f"{ticker}.{exchange}"
        endpoint = f"eod/{symbol}"
        params = {
            "api_token": self.api_key,
            "fmt": "json",
            "filter": "last_close",
        }

        logger.debug("Fetching last close", ticker=ticker)

        try:
            response = await self.get_json(endpoint, params=params)

            if isinstance(response, (int, float)):
                return float(response)
            elif isinstance(response, dict):
                # Sometimes returns {"close": value}
                if "close" in response:
                    return float(response["close"])
                if "last_close" in response:
                    return float(response["last_close"])

            logger.warning(f"Unexpected last_close response format for {ticker}: {response}")
            return None

        except Exception as e:
            logger.warning(f"Failed to get last close for {ticker}: {e}")
            return None

    async def get_historical_prices(
        self,
        ticker: str,
        exchange: str = "US",
        days_back: Optional[int] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get historical OHLCV data for a stock.

        Args:
            ticker: Stock ticker symbol
            exchange: Exchange code (default "US")
            days_back: Number of days of history (optional)
            from_date: Start date in YYYY-MM-DD format (optional)
            to_date: End date in YYYY-MM-DD format (optional)

        Returns:
            List of OHLCV dictionaries with date, open, high, low, close, volume
        """
        from datetime import datetime, timedelta

        symbol = f"{ticker}.{exchange}"
        endpoint = f"eod/{symbol}"
        params = {
            "api_token": self.api_key,
            "fmt": "json",
            "order": "a",  # Ascending order (oldest first)
        }

        # Set date range
        if days_back:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            params["from"] = start_date.strftime("%Y-%m-%d")
            params["to"] = end_date.strftime("%Y-%m-%d")
        elif from_date:
            params["from"] = from_date
            if to_date:
                params["to"] = to_date

        logger.debug("Fetching historical prices", ticker=ticker, params=params)

        try:
            response = await self.get_json(endpoint, params=params)

            if isinstance(response, dict) and "error" in response:
                raise EODHDAPIError(f"EODHD API error for {ticker}: {response['error']}")

            if not isinstance(response, list):
                logger.warning(f"Unexpected response format for {ticker}: {type(response)}")
                return []

            logger.info(
                "Fetched historical prices",
                ticker=ticker,
                count=len(response),
            )

            return response

        except Exception as e:
            logger.error(f"Failed to fetch historical prices for {ticker}: {e}")
            raise

    async def get_eod_bulk(
        self,
        exchange: str = "US",
        date: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get bulk EOD data for all symbols on an exchange.

        This is more efficient than individual calls when you need
        prices for many symbols.

        Args:
            exchange: Exchange code
            date: Optional date in YYYY-MM-DD format (default: latest)

        Returns:
            List of EOD data dictionaries with code, close, volume, etc.
        """
        endpoint = f"eod-bulk-last-day/{exchange}"
        params = {
            "api_token": self.api_key,
            "fmt": "json",
        }
        if date:
            params["date"] = date

        logger.info("Fetching bulk EOD data", exchange=exchange)

        response = await self.get_json(endpoint, params=params)

        if isinstance(response, dict) and "error" in response:
            raise EODHDAPIError(f"EODHD API error: {response['error']}")

        logger.info(
            "Fetched bulk EOD data",
            exchange=exchange,
            count=len(response) if isinstance(response, list) else 0,
        )

        return response if isinstance(response, list) else []


class StockFilter:
    """Filter stocks based on industry and market cap criteria."""

    BIOPHARMA_INDUSTRIES = [
        "biotechnology",
        "pharmaceutical",
        "pharmaceuticals",
        "drug manufacturer",
        "drug manufacturers",
        "drug manufacturers - general",
        "drug manufacturers - specialty & generic",
        "drug manufacturers—general",
        "drug manufacturers—specialty & generic",
        "diagnostics & research",
        "medical devices",
        "medical instruments & supplies",
    ]

    BIOPHARMA_SECTORS = [
        "healthcare",
        "health care",
    ]

    COMMON_STOCK_TYPES = [
        "Common Stock",
        "common stock",
    ]

    # SIC codes for biopharma industry classification (SEC-based)
    BIOPHARMA_SIC_CODES = {
        "2833": "Medicinal Chemicals and Botanical Products",
        "2834": "Pharmaceutical Preparations",
        "2835": "In Vitro and In Vivo Diagnostic Substances",
        "2836": "Biological Products, Except Diagnostic Substances",
        "3826": "Laboratory Analytical Instruments",
        "3841": "Surgical and Medical Instruments and Apparatus",
        "3845": "Electromedical and Electrotherapeutic Apparatus",
        "8731": "Commercial Physical and Biological Research",
    }

    def __init__(
        self,
        max_market_cap: float = 2_000_000_000,
        min_market_cap: float = 0,
        exchanges: Optional[list[str]] = None,
        include_missing_market_cap: bool = True,
    ):
        """Initialize the stock filter.

        Args:
            max_market_cap: Maximum market cap in USD (default $2B)
            min_market_cap: Minimum market cap in USD (default $0)
            exchanges: List of allowed exchanges (None = all)
            include_missing_market_cap: Include stocks with no market cap data
        """
        self.max_market_cap = max_market_cap
        self.min_market_cap = min_market_cap
        self.exchanges = exchanges
        self.include_missing_market_cap = include_missing_market_cap

    def is_biopharma_industry(self, industry: Optional[str]) -> bool:
        """Check if industry is biopharma-related.

        Uses fuzzy matching to handle variations in industry names.

        Args:
            industry: Industry classification string

        Returns:
            True if the industry is biopharma-related
        """
        if not industry:
            return False

        industry_lower = industry.lower().strip()

        for biopharma in self.BIOPHARMA_INDUSTRIES:
            if biopharma in industry_lower or industry_lower in biopharma:
                return True

            industry_normalized = re.sub(r"[^a-z0-9]", "", industry_lower)
            biopharma_normalized = re.sub(r"[^a-z0-9]", "", biopharma)
            if (
                biopharma_normalized in industry_normalized
                or industry_normalized in biopharma_normalized
            ):
                return True

        return False

    def is_biopharma_sic(self, sic: Optional[str]) -> bool:
        """Check if SIC code indicates a biopharma company.

        Args:
            sic: SIC code (4-digit string)

        Returns:
            True if SIC code is biopharma-related
        """
        if not sic:
            return False
        return str(sic).strip() in self.BIOPHARMA_SIC_CODES

    def is_biopharma_sector(self, sector: Optional[str]) -> bool:
        """Check if sector is healthcare.

        Args:
            sector: Sector classification string

        Returns:
            True if the sector is healthcare-related
        """
        if not sector:
            return False

        sector_lower = sector.lower().strip()
        return any(s in sector_lower for s in self.BIOPHARMA_SECTORS)

    def passes_market_cap_filter(self, market_cap: Optional[float]) -> bool:
        """Check if market cap is within the allowed range.

        Args:
            market_cap: Market capitalization in USD

        Returns:
            True if market cap passes the filter
        """
        if market_cap is None:
            return self.include_missing_market_cap

        return self.min_market_cap <= market_cap <= self.max_market_cap

    def passes_exchange_filter(self, exchange: Optional[str]) -> bool:
        """Check if exchange is allowed.

        Args:
            exchange: Exchange code

        Returns:
            True if exchange is allowed
        """
        if self.exchanges is None:
            return True

        if not exchange:
            return False

        return exchange.upper() in [e.upper() for e in self.exchanges]

    def is_common_stock(self, symbol_type: Optional[str]) -> bool:
        """Check if symbol is a common stock.

        Args:
            symbol_type: Symbol type from exchange list

        Returns:
            True if it's a common stock
        """
        if not symbol_type:
            return False

        return symbol_type in self.COMMON_STOCK_TYPES

    def filter_symbol(self, symbol: dict[str, Any]) -> bool:
        """Pre-filter a symbol from the exchange list.

        This is a quick filter before fetching fundamentals.

        Args:
            symbol: Symbol dict from exchange-symbol-list endpoint

        Returns:
            True if the symbol should be considered
        """
        symbol_type = symbol.get("Type", "")
        if not self.is_common_stock(symbol_type):
            return False

        exchange = symbol.get("Exchange", "")
        if not self.passes_exchange_filter(exchange):
            return False

        return True

    def filter_stock(self, stock: "Stock") -> bool:
        """Apply full filtering to a stock.

        Args:
            stock: Stock object with fundamentals data

        Returns:
            True if the stock passes all filters
        """
        # Check industry (EODHD style) or SIC code (SEC style)
        industry_match = self.is_biopharma_industry(stock.industry)
        sic_match = self.is_biopharma_sic(getattr(stock, 'sic', None))
        sector_match = self.is_biopharma_sector(stock.sector)

        if not (industry_match or sic_match or sector_match):
            return False

        if not self.passes_market_cap_filter(stock.market_cap):
            return False

        if not self.passes_exchange_filter(stock.exchange):
            return False

        return True

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "StockFilter":
        """Create a StockFilter from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            StockFilter instance
        """
        market_cap_config = config.get("market_cap", {})
        return cls(
            max_market_cap=market_cap_config.get("max", 2_000_000_000),
            min_market_cap=market_cap_config.get("min", 0),
            exchanges=config.get("exchanges"),
            include_missing_market_cap=config.get("include_missing_market_cap", True),
        )


# Import Stock here to avoid circular import
from src.models.stock import Stock
