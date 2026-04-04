"""Stock data fetcher using EODHD."""

from typing import Any, Optional

from src.clients.eodhd import EODHDClient, StockFilter
from src.models.stock import Stock
from src.utils.logging import get_logger

logger = get_logger(__name__)


class StockDataFetcher:
    """Fetches and filters stock data from EODHD."""

    def __init__(
        self,
        eodhd_client: EODHDClient,
        stock_filter: StockFilter,
    ):
        """Initialize the stock data fetcher.

        Args:
            eodhd_client: EODHD API client
            stock_filter: Filter criteria for stocks
        """
        self.eodhd_client = eodhd_client
        self.stock_filter = stock_filter

    async def fetch_stock(
        self,
        ticker: str,
        exchange: str,
    ) -> Optional[Stock]:
        """Fetch stock data for a single ticker via EODHD fundamentals.

        Args:
            ticker: Stock ticker symbol
            exchange: Exchange code

        Returns:
            Stock object or None if not found/not matching filter
        """
        try:
            fundamentals = await self.eodhd_client.get_fundamentals_filtered(ticker, "US")

            if not fundamentals or fundamentals.get("General") is None:
                logger.debug(f"No fundamentals data for {ticker}")
                return None

            return Stock.from_eodhd_fundamentals(ticker, exchange, fundamentals)

        except Exception as e:
            logger.error(f"Error fetching EODHD data for {ticker}: {e}")
            return None

    async def fetch_filtered_stocks(
        self,
        symbols: list[dict[str, Any]],
        limit: Optional[int] = None,
        progress_callback: Optional[callable] = None,
    ) -> tuple[list[Stock], list[dict], dict[str, int]]:
        """Fetch and filter stocks from a list of symbols.

        Args:
            symbols: List of symbol dicts from EODHD exchange-symbol-list
            limit: Maximum number of symbols to process
            progress_callback: Optional callback(ticker, index, total)

        Returns:
            Tuple of (filtered_stocks, quality_issues, stats)
        """
        candidates = [s for s in symbols if self.stock_filter.filter_symbol(s)]
        logger.info(f"Pre-filtered to {len(candidates)} candidates")

        if limit:
            candidates = candidates[:limit]

        filtered_stocks: list[Stock] = []
        quality_issues: list[dict] = []
        stats = {
            "total": len(candidates),
            "processed": 0,
            "matched": 0,
            "skipped": 0,
            "errors": 0,
        }

        for i, symbol in enumerate(candidates):
            ticker = symbol.get("Code", "")
            exchange = symbol.get("Exchange", "")

            if progress_callback:
                progress_callback(ticker, i, len(candidates))

            try:
                stock = await self.fetch_stock(ticker, exchange)
                stats["processed"] += 1

                if stock is None:
                    stats["skipped"] += 1
                    continue

                if self.stock_filter.filter_stock(stock):
                    filtered_stocks.append(stock)
                    stats["matched"] += 1

                    issues = stock.get_quality_issues()
                    if issues:
                        quality_issues.append({
                            "ticker": stock.ticker,
                            "company_name": stock.company_name,
                            "quality_score": stock.data_quality_score,
                            "issues": "; ".join(issues),
                        })
                else:
                    stats["skipped"] += 1

            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                stats["errors"] += 1

        logger.info(
            f"Fetch complete: {stats['matched']} matched, "
            f"{stats['skipped']} skipped, {stats['errors']} errors"
        )

        return filtered_stocks, quality_issues, stats
