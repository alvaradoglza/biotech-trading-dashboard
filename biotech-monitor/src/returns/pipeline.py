"""Batch processing pipeline for return calculations."""

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from src.clients.eodhd import EODHDClient
from src.returns.price_lookup import PriceLookup
from src.returns.calculator import ReturnCalculator
from src.utils.logging import get_logger

logger = get_logger(__name__)

# CSV columns for returns
RETURN_COLUMNS = ["return_5d", "return_30d", "return_60d", "return_90d"]


class ReturnPipeline:
    """
    Pipeline for calculating returns across all announcements.

    Processes announcements in batches by ticker to minimize API calls.
    Each ticker requires only one price history fetch.
    """

    def __init__(
        self,
        eodhd_api_key: str,
        announcements_csv: str = "data/index/announcements.csv",
        days_back: int = 730,  # 2 years of price history
    ):
        """
        Initialize the return pipeline.

        Args:
            eodhd_api_key: API key for EODHD
            announcements_csv: Path to announcements CSV file
            days_back: Days of price history to fetch per ticker
        """
        self.eodhd_api_key = eodhd_api_key
        self.announcements_csv = Path(announcements_csv)
        self.days_back = days_back
        self._client: Optional[EODHDClient] = None

    async def __aenter__(self):
        self._client = EODHDClient(api_key=self.eodhd_api_key)
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)

    def _load_announcements(self) -> list[dict]:
        """Load all announcements from CSV."""
        if not self.announcements_csv.exists():
            logger.error("Announcements CSV not found", path=str(self.announcements_csv))
            return []

        announcements = []
        with open(self.announcements_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                announcements.append(dict(row))

        return announcements

    def _save_announcements(self, announcements: list[dict], fieldnames: list[str]):
        """Save announcements back to CSV."""
        # Ensure return columns are in fieldnames
        for col in RETURN_COLUMNS:
            if col not in fieldnames:
                fieldnames.append(col)

        with open(self.announcements_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(announcements)

    def _group_by_ticker(self, announcements: list[dict]) -> dict[str, list[dict]]:
        """Group announcements by ticker for batch processing."""
        grouped = {}
        for ann in announcements:
            ticker = ann.get("ticker", "")
            if ticker not in grouped:
                grouped[ticker] = []
            grouped[ticker].append(ann)
        return grouped

    def _parse_date(self, date_str: str) -> Optional[date]:
        """Parse date from various formats."""
        if not date_str:
            return None

        # Try ISO format first
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00")).date()
        except ValueError:
            pass

        # Try simple date format
        try:
            return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        except ValueError:
            pass

        return None

    def _needs_calculation(self, ann: dict, force: bool = False) -> bool:
        """Check if an announcement needs return calculation."""
        if force:
            return True

        # Only calculate for OK status
        if ann.get("parse_status") != "OK":
            return False

        # Check if any return column is missing
        for col in RETURN_COLUMNS:
            value = ann.get(col, "")
            if value == "" or value is None:
                return True

        return False

    async def _fetch_prices(self, ticker: str) -> Optional[PriceLookup]:
        """Fetch price history for a ticker."""
        if not self._client:
            raise RuntimeError("Pipeline not initialized. Use 'async with' context.")

        try:
            prices = await self._client.get_historical_prices(
                ticker=ticker,
                exchange="US",
                days_back=self.days_back,
            )

            if not prices:
                logger.warning("No price data returned", ticker=ticker)
                return None

            # Convert to list of (date, adjusted_close) tuples
            price_tuples = []
            for p in prices:
                try:
                    price_date = datetime.strptime(p["date"], "%Y-%m-%d").date()
                    # Use adjusted_close if available, otherwise close
                    price = p.get("adjusted_close", p.get("close", 0))
                    if price and price > 0:
                        price_tuples.append((price_date, float(price)))
                except (KeyError, ValueError) as e:
                    logger.debug("Skipping invalid price record", error=str(e))
                    continue

            if not price_tuples:
                logger.warning("No valid price data", ticker=ticker)
                return None

            return PriceLookup(price_tuples)

        except Exception as e:
            logger.error("Failed to fetch prices", ticker=ticker, error=str(e))
            return None

    async def process_ticker(
        self,
        ticker: str,
        announcements: list[dict],
        force: bool = False,
    ) -> dict[str, int]:
        """
        Process all announcements for a single ticker.

        Args:
            ticker: Stock ticker symbol
            announcements: List of announcement dicts for this ticker
            force: If True, recalculate even if values exist

        Returns:
            Stats dict with counts of calculated, skipped, failed
        """
        stats = {"calculated": 0, "skipped": 0, "failed": 0}

        # Filter to announcements that need calculation
        to_process = [a for a in announcements if self._needs_calculation(a, force)]

        if not to_process:
            stats["skipped"] = len(announcements)
            return stats

        # Fetch price history once for this ticker
        price_lookup = await self._fetch_prices(ticker)

        if price_lookup is None:
            # Mark all as failed
            stats["failed"] = len(to_process)
            stats["skipped"] = len(announcements) - len(to_process)
            logger.warning(
                "Skipping ticker - no price data",
                ticker=ticker,
                announcements=len(to_process),
            )
            return stats

        # Calculate returns for each announcement
        calculator = ReturnCalculator(price_lookup)

        for ann in to_process:
            date_str = ann.get("published_at", "")
            ann_date = self._parse_date(date_str)

            if ann_date is None:
                logger.warning(
                    "Invalid announcement date",
                    ticker=ticker,
                    date_str=date_str,
                )
                stats["failed"] += 1
                continue

            try:
                return_5d, return_30d, return_60d, return_90d = calculator.get_return_values(ann_date)

                # Update announcement dict (None becomes empty string for CSV)
                ann["return_5d"] = "" if return_5d is None else str(return_5d)
                ann["return_30d"] = "" if return_30d is None else str(return_30d)
                ann["return_60d"] = "" if return_60d is None else str(return_60d)
                ann["return_90d"] = "" if return_90d is None else str(return_90d)

                stats["calculated"] += 1

            except Exception as e:
                logger.error(
                    "Return calculation failed",
                    ticker=ticker,
                    date=str(ann_date),
                    error=str(e),
                )
                stats["failed"] += 1

        stats["skipped"] = len(announcements) - len(to_process)
        return stats

    async def run(self, force: bool = False) -> dict[str, int]:
        """
        Run return calculation for all announcements.

        Args:
            force: If True, recalculate all returns even if values exist

        Returns:
            Stats dict with total counts
        """
        logger.info("Starting return calculation pipeline")

        # Load announcements
        announcements = self._load_announcements()
        if not announcements:
            logger.warning("No announcements to process")
            return {"total": 0, "calculated": 0, "skipped": 0, "failed": 0}

        # Get fieldnames from first row
        fieldnames = list(announcements[0].keys())

        # Group by ticker
        by_ticker = self._group_by_ticker(announcements)
        total_tickers = len(by_ticker)

        logger.info(
            "Processing announcements",
            total=len(announcements),
            tickers=total_tickers,
        )

        totals = {"total": len(announcements), "calculated": 0, "skipped": 0, "failed": 0}

        for i, (ticker, ticker_anns) in enumerate(sorted(by_ticker.items()), 1):
            print(f"[{i}/{total_tickers}] {ticker} ({len(ticker_anns)} announcements)...", end=" ")

            stats = await self.process_ticker(ticker, ticker_anns, force)

            totals["calculated"] += stats["calculated"]
            totals["skipped"] += stats["skipped"]
            totals["failed"] += stats["failed"]

            if stats["calculated"] > 0:
                print(f"calculated {stats['calculated']}")
            elif stats["failed"] > 0:
                print(f"failed {stats['failed']}")
            else:
                print("skipped")

        # Save updated announcements
        self._save_announcements(announcements, fieldnames)

        logger.info(
            "Return calculation complete",
            calculated=totals["calculated"],
            skipped=totals["skipped"],
            failed=totals["failed"],
        )

        return totals

    def get_stats(self) -> dict:
        """Get statistics about current return data."""
        announcements = self._load_announcements()

        if not announcements:
            return {
                "total": 0,
                "with_returns": 0,
                "without_returns": 0,
                "by_status": {},
            }

        with_returns = 0
        without_returns = 0
        by_status = {}

        for ann in announcements:
            status = ann.get("parse_status", "UNKNOWN")
            by_status[status] = by_status.get(status, 0) + 1

            # Check if has any return value
            has_return = any(
                ann.get(col, "") not in ("", None)
                for col in RETURN_COLUMNS
            )

            if has_return:
                with_returns += 1
            else:
                without_returns += 1

        return {
            "total": len(announcements),
            "with_returns": with_returns,
            "without_returns": without_returns,
            "by_status": by_status,
        }
