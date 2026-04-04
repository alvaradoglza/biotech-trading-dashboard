"""Tests for the EODHD client."""

import pytest
from unittest.mock import AsyncMock, patch

from src.clients.base import AuthenticationError
from src.clients.eodhd import EODHDClient, EODHDAPIError, EODHDEntitlementError


class TestEODHDClient:
    """Tests for EODHDClient."""

    def test_init(self):
        """Test client initialization."""
        client = EODHDClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.base_url == "https://eodhd.com/api"

    def test_init_custom_rate_limit(self):
        """Test client with custom rate limit."""
        client = EODHDClient(api_key="test_key", requests_per_second=2.0)
        assert client.rate_limiter.requests_per_second == 2.0

    @pytest.mark.asyncio
    async def test_get_exchange_symbols(self, sample_eodhd_symbol):
        """Test fetching exchange symbols."""
        client = EODHDClient(api_key="test_key")

        with patch.object(client, 'get_json', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = [sample_eodhd_symbol]

            result = await client.get_exchange_symbols("US")

            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert "exchange-symbol-list/US" in call_args[0][0]
            assert result == [sample_eodhd_symbol]

        await client.close()

    @pytest.mark.asyncio
    async def test_get_exchange_symbols_error(self):
        """Test handling API error response."""
        client = EODHDClient(api_key="test_key")

        with patch.object(client, 'get_json', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"error": "Invalid API key"}

            with pytest.raises(EODHDAPIError) as exc_info:
                await client.get_exchange_symbols("US")

            assert "Invalid API key" in str(exc_info.value)

        await client.close()

    @pytest.mark.asyncio
    async def test_get_fundamentals(self, sample_eodhd_fundamentals):
        """Test fetching fundamentals."""
        client = EODHDClient(api_key="test_key")

        with patch.object(client, 'get_json', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = sample_eodhd_fundamentals

            result = await client.get_fundamentals("MRNA", "US")

            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert "fundamentals/MRNA.US" in call_args[0][0]
            assert result == sample_eodhd_fundamentals

        await client.close()

    @pytest.mark.asyncio
    async def test_get_fundamentals_filtered(self, sample_eodhd_fundamentals):
        """Test fetching filtered fundamentals."""
        client = EODHDClient(api_key="test_key")

        with patch.object(client, 'get_json', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = sample_eodhd_fundamentals

            result = await client.get_fundamentals_filtered("MRNA", "US")

            mock_get.assert_called_once()
            call_args = mock_get.call_args
            # Check filter parameter is included
            params = call_args[1].get("params", {})
            assert params.get("filter") == "General,Highlights"

        await client.close()

    @pytest.mark.asyncio
    async def test_get_fundamentals_403_raises_entitlement_error(self):
        """Test that 403 on fundamentals raises EODHDEntitlementError."""
        client = EODHDClient(api_key="test_key")

        with patch.object(client, 'get_json', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = AuthenticationError(
                "Forbidden. Please contact support@eodhistoricaldata.com",
                status_code=403,
            )

            with pytest.raises(EODHDEntitlementError) as exc_info:
                await client.get_fundamentals("MRNA", "US")

            assert "not available on this plan" in str(exc_info.value)

        await client.close()

    @pytest.mark.asyncio
    async def test_get_fundamentals_filtered_403_raises_entitlement_error(self):
        """Test that 403 on filtered fundamentals raises EODHDEntitlementError."""
        client = EODHDClient(api_key="test_key")

        with patch.object(client, 'get_json', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = AuthenticationError(
                "Forbidden",
                status_code=403,
            )

            with pytest.raises(EODHDEntitlementError) as exc_info:
                await client.get_fundamentals_filtered("AAPL", "US")

            assert "not available on this plan" in str(exc_info.value)

        await client.close()

    @pytest.mark.asyncio
    async def test_check_fundamentals_access_success(self, sample_eodhd_fundamentals):
        """Test check_fundamentals_access returns True when accessible."""
        client = EODHDClient(api_key="test_key")

        with patch.object(client, 'get_fundamentals_filtered', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = sample_eodhd_fundamentals

            result = await client.check_fundamentals_access()

            assert result is True
            assert client.fundamentals_available is True

        await client.close()

    @pytest.mark.asyncio
    async def test_check_fundamentals_access_403(self):
        """Test check_fundamentals_access returns False on 403."""
        client = EODHDClient(api_key="test_key")

        with patch.object(client, 'get_fundamentals_filtered', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = EODHDEntitlementError("Not available")

            result = await client.check_fundamentals_access()

            assert result is False
            assert client.fundamentals_available is False

        await client.close()

    @pytest.mark.asyncio
    async def test_check_fundamentals_access_cached(self):
        """Test that check_fundamentals_access caches the result."""
        client = EODHDClient(api_key="test_key")

        with patch.object(client, 'get_fundamentals_filtered', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {}

            # First call
            await client.check_fundamentals_access()
            # Second call should use cache
            await client.check_fundamentals_access()

            # Should only call the API once
            assert mock_get.call_count == 1

        await client.close()

    @pytest.mark.asyncio
    async def test_get_last_close_numeric_response(self):
        """Test get_last_close with numeric response."""
        client = EODHDClient(api_key="test_key")

        with patch.object(client, 'get_json', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = 45.50

            result = await client.get_last_close("MRNA", "US")

            assert result == 45.50

        await client.close()

    @pytest.mark.asyncio
    async def test_get_last_close_dict_response(self):
        """Test get_last_close with dict response."""
        client = EODHDClient(api_key="test_key")

        with patch.object(client, 'get_json', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"close": 45.50}

            result = await client.get_last_close("MRNA", "US")

            assert result == 45.50

        await client.close()

    @pytest.mark.asyncio
    async def test_get_last_close_error_returns_none(self):
        """Test get_last_close returns None on error."""
        client = EODHDClient(api_key="test_key")

        with patch.object(client, 'get_json', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Network error")

            result = await client.get_last_close("MRNA", "US")

            assert result is None

        await client.close()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with EODHDClient(api_key="test_key") as client:
            assert client.api_key == "test_key"


class TestStockFilterSIC:
    """Tests for SIC code filtering."""

    def test_is_biopharma_sic_2834(self):
        """Test SIC 2834 (Pharmaceutical Preparations) matches."""
        from src.clients.eodhd import StockFilter

        filter = StockFilter()
        assert filter.is_biopharma_sic("2834")

    def test_is_biopharma_sic_2836(self):
        """Test SIC 2836 (Biological Products) matches."""
        from src.clients.eodhd import StockFilter

        filter = StockFilter()
        assert filter.is_biopharma_sic("2836")

    def test_is_biopharma_sic_non_pharma(self):
        """Test non-pharma SIC codes don't match."""
        from src.clients.eodhd import StockFilter

        filter = StockFilter()
        assert not filter.is_biopharma_sic("3571")  # Computers
        assert not filter.is_biopharma_sic("7372")  # Software

    def test_is_biopharma_sic_none(self):
        """Test None SIC doesn't match."""
        from src.clients.eodhd import StockFilter

        filter = StockFilter()
        assert not filter.is_biopharma_sic(None)
        assert not filter.is_biopharma_sic("")
