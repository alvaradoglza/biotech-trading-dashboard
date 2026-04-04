"""
prices.py — Live price fetching from EODHD with 15-minute Streamlit cache.

Prices are fetched on-demand when the dashboard loads, not on a schedule.
The 15-minute TTL (900 seconds) prevents excessive API calls during a session.

IMPORTANT: tickers must be passed as a tuple (not list) for Streamlit cache
hashing to work correctly. Lists are not hashable by st.cache_data.
"""

import logging
import os
from typing import Any

import requests
import streamlit as st

logger = logging.getLogger(__name__)

EODHD_BASE_URL = "https://eodhd.com/api"
CACHE_TTL_SECONDS = 900  # 15 minutes


def get_live_prices(tickers: tuple[str, ...]) -> dict[str, float | None]:
    """Fetch current prices for the given tickers from EODHD.

    Cached for 15 minutes. Handles API errors gracefully — missing tickers
    return None rather than raising.

    Args:
        tickers: Tuple of ticker symbols (must be tuple for cache hashing).

    Returns:
        Dict mapping ticker → price (float) or None if unavailable.
    """
    return _cached_prices(tickers)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _cached_prices(tickers: tuple[str, ...]) -> dict[str, float | None]:
    """Internal cached implementation. Called via get_live_prices()."""
    if not tickers:
        return {}

    api_key = _get_api_key()
    if not api_key:
        return {t: None for t in tickers}

    prices = {}

    # EODHD supports bulk real-time quotes (up to 50 per request)
    # Split into batches of 50
    ticker_list = list(tickers)
    for batch_start in range(0, len(ticker_list), 50):
        batch = ticker_list[batch_start : batch_start + 50]
        batch_prices = _fetch_batch(batch, api_key)
        prices.update(batch_prices)

    return prices


def _fetch_batch(tickers: list[str], api_key: str) -> dict[str, float | None]:
    """Fetch prices for a batch of tickers using EODHD real-time endpoint."""
    # Format: TICKER1.US,TICKER2.US,...
    symbols = ",".join(f"{t}.US" for t in tickers)

    try:
        url = f"{EODHD_BASE_URL}/real-time/{tickers[0]}.US"
        params = {
            "api_token": api_key,
            "fmt": "json",
            "s": symbols,  # additional symbols
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.warning("EODHD batch price fetch failed: %s", e)
        return {t: None for t in tickers}
    except Exception as e:
        logger.warning("Unexpected error fetching prices: %s", e)
        return {t: None for t in tickers}

    # Parse response — can be dict (single ticker) or list (multiple)
    prices: dict[str, float | None] = {}

    if isinstance(data, dict):
        # Single ticker response
        ticker = tickers[0]
        prices[ticker] = _extract_price(data)
    elif isinstance(data, list):
        # Multiple tickers — map by code field
        by_code = {item.get("code", "").replace(".US", ""): item for item in data}
        for ticker in tickers:
            item = by_code.get(ticker) or by_code.get(f"{ticker}.US")
            prices[ticker] = _extract_price(item) if item else None
    else:
        # Unknown format
        for t in tickers:
            prices[t] = None

    return prices


def _extract_price(item: dict | None) -> float | None:
    """Extract the best available price from an EODHD quote item."""
    if not item:
        return None
    # Prefer 'close' (today's close), fallback to 'previousClose'
    for field in ["close", "previousClose", "open"]:
        val = item.get(field)
        if val is not None:
            try:
                price = float(val)
                if price > 0:
                    return price
            except (TypeError, ValueError):
                continue
    return None


def _get_api_key() -> str | None:
    """Get EODHD API key from Streamlit secrets or environment variables."""
    # Try Streamlit secrets first
    try:
        return st.secrets["EODHD_API_KEY"]
    except (AttributeError, KeyError):
        pass
    # Fall back to environment variable
    return os.environ.get("EODHD_API_KEY")


def enrich_positions_with_prices(
    positions_df,
    prices: dict[str, float | None],
) -> "pd.DataFrame":
    """Add live_price, live_market_value, and unrealized_pnl_live columns to positions DataFrame."""
    import pandas as pd

    if positions_df.empty:
        return positions_df

    df = positions_df.copy()
    df["live_price"] = df["ticker"].map(prices)
    df["live_market_value"] = df.apply(
        lambda r: r["quantity"] * r["live_price"] if r["live_price"] else None, axis=1
    )
    df["unrealized_pnl_live"] = df.apply(
        lambda r: (r["live_price"] - r["avg_cost"]) * r["quantity"] if r["live_price"] else None,
        axis=1,
    )
    df["unrealized_pnl_pct"] = df.apply(
        lambda r: ((r["live_price"] - r["avg_cost"]) / r["avg_cost"] * 100)
        if r["live_price"] and r["avg_cost"] else None,
        axis=1,
    )
    return df
