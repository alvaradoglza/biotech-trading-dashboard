"""Tests for the StockDataFetcher pipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.clients.eodhd import EODHDClient, StockFilter
from src.models.stock import Stock, DataSource
from src.pipeline.stock_fetcher import StockDataFetcher


class TestStockDataFetcher:
    """Tests for StockDataFetcher."""

    @pytest.fixture
    def mock_eodhd_client(self):
        """Create a mock EODHD client."""
        client = MagicMock(spec=EODHDClient)
        client.get_fundamentals_filtered = AsyncMock()
        return client

    @pytest.fixture
    def stock_filter(self):
        """Create a stock filter."""
        return StockFilter(exchanges=["NASDAQ", "NYSE"])

    @pytest.mark.asyncio
    async def test_fetch_stock_via_eodhd(
        self, mock_eodhd_client, stock_filter, sample_eodhd_fundamentals
    ):
        """Test fetching stock via EODHD fundamentals."""
        mock_eodhd_client.get_fundamentals_filtered.return_value = sample_eodhd_fundamentals

        fetcher = StockDataFetcher(
            eodhd_client=mock_eodhd_client,
            stock_filter=stock_filter,
        )

        stock = await fetcher.fetch_stock("MRNA", "NASDAQ")

        assert stock is not None
        assert stock.ticker == "MRNA"
        assert stock.data_source == DataSource.EODHD_FUNDAMENTALS

    @pytest.mark.asyncio
    async def test_fetch_stock_returns_none_when_no_fundamentals(
        self, mock_eodhd_client, stock_filter
    ):
        """Test that missing fundamentals returns None."""
        mock_eodhd_client.get_fundamentals_filtered.return_value = None

        fetcher = StockDataFetcher(
            eodhd_client=mock_eodhd_client,
            stock_filter=stock_filter,
        )

        stock = await fetcher.fetch_stock("NOTREAL", "NASDAQ")

        assert stock is None

    @pytest.mark.asyncio
    async def test_fetch_stock_returns_none_on_error(
        self, mock_eodhd_client, stock_filter
    ):
        """Test that API errors return None."""
        mock_eodhd_client.get_fundamentals_filtered.side_effect = Exception("API error")

        fetcher = StockDataFetcher(
            eodhd_client=mock_eodhd_client,
            stock_filter=stock_filter,
        )

        stock = await fetcher.fetch_stock("MRNA", "NASDAQ")

        assert stock is None

    @pytest.mark.asyncio
    async def test_fetch_filtered_stocks(
        self, mock_eodhd_client, stock_filter, sample_eodhd_fundamentals
    ):
        """Test batch filtering of stocks."""
        mock_eodhd_client.get_fundamentals_filtered.return_value = sample_eodhd_fundamentals

        fetcher = StockDataFetcher(
            eodhd_client=mock_eodhd_client,
            stock_filter=stock_filter,
        )

        symbols = [
            {"Code": "MRNA", "Exchange": "NASDAQ", "Type": "Common Stock"},
        ]

        stocks, issues, stats = await fetcher.fetch_filtered_stocks(symbols)

        assert stats["processed"] >= 0
        assert "matched" in stats
        assert "skipped" in stats
        assert "errors" in stats
