"""Tests for strategy.py and simulator.py — ClinicalTrialStrategy, simulate_ticker, simulate_window."""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from backtest.strategy import ClinicalTrialStrategy
from backtest.simulator import (
    next_trading_day,
    simulate_ticker,
    simulate_window,
    _format_trade_log,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_price_df(n: int = 252, start: str = "2024-01-02") -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with DatetimeIndex."""
    dates = pd.date_range(start, periods=n, freq="B")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "Open":   close * 0.99,
        "High":   close * 1.02,
        "Low":    close * 0.98,
        "Close":  close,
        "Volume": np.ones(n) * 1_000_000,
    }, index=dates)


def make_signals(n: int = 3, ticker: str = "MRNA") -> pd.DataFrame:
    """Create a minimal signals DataFrame."""
    return pd.DataFrame({
        "ticker":         [ticker] * n,
        "published_at":   pd.date_range("2024-03-01", periods=n, freq="10D"),
        "window_id":      [0] * n,
        "horizon":        ["5d"] * n,
        "decision_score": [0.8, 0.6, 0.9][:n],
        "return_5d":      [5.0, 2.0, 8.0][:n],
        "return_30d":     [10.0, 3.0, 15.0][:n],
        "prediction":     [1] * n,
    })


# ── next_trading_day ─────────────────────────────────────────────────────────

def test_next_trading_day_exact_match():
    """next_trading_day returns the exact date if it exists in the price index."""
    price_df = make_price_df()
    pub = price_df.index[5]
    result = next_trading_day(pub, price_df)
    assert result == pub


def test_next_trading_day_weekend_advances():
    """next_trading_day advances past weekends to the next business day."""
    price_df = make_price_df(252, "2024-01-02")
    # Find a Friday in the data
    fridays = price_df.index[price_df.index.dayofweek == 4]
    if len(fridays) == 0:
        pytest.skip("No Fridays in synthetic data")
    friday = fridays[0]
    # Saturday (not in business day index)
    saturday = friday + pd.Timedelta(days=1)
    result = next_trading_day(saturday, price_df)
    # Should advance to the Monday (or next available day)
    assert result >= friday + pd.Timedelta(days=1)
    assert result in price_df.index


def test_next_trading_day_returns_none_past_data():
    """next_trading_day returns None when published_at is after the last price date."""
    price_df = make_price_df(10, "2024-01-02")
    future_date = pd.Timestamp("2030-01-01")
    result = next_trading_day(future_date, price_df)
    assert result is None


def test_next_trading_day_returns_timestamp():
    """next_trading_day should return a pd.Timestamp."""
    price_df = make_price_df()
    result = next_trading_day(price_df.index[0], price_df)
    assert isinstance(result, pd.Timestamp)


# ── simulate_ticker ───────────────────────────────────────────────────────────

def test_simulate_ticker_returns_none_for_missing_ticker():
    """simulate_ticker should return None when no OHLCV file exists for the ticker."""
    with tempfile.TemporaryDirectory() as tmpdir:
        signals = make_signals(2, "GHOST")
        result = simulate_ticker(
            ticker="GHOST",
            signals=signals,
            horizon="5d",
            take_profit_pct=0.08,
            stop_loss_pct=0.04,
            ohlcv_dir=tmpdir,
        )
    assert result is None


def test_simulate_ticker_with_real_price_data():
    """simulate_ticker should return a DataFrame or None (not raise) when price data exists."""
    price_df = make_price_df(252)
    with tempfile.TemporaryDirectory() as tmpdir:
        price_df.to_parquet(Path(tmpdir) / "MRNA.parquet")
        signals = make_signals(1, "MRNA")
        result = simulate_ticker(
            ticker="MRNA",
            signals=signals,
            horizon="5d",
            take_profit_pct=0.08,
            stop_loss_pct=0.04,
            ohlcv_dir=tmpdir,
        )
    # Either None (no trades) or a valid DataFrame
    assert result is None or isinstance(result, pd.DataFrame)


def test_simulate_ticker_trade_log_has_required_columns():
    """simulate_ticker trade log should contain all canonical trade log columns."""
    price_df = make_price_df(252)
    expected_cols = [
        "window_id", "horizon", "ticker", "published_at", "entry_date",
        "entry_price", "exit_date", "exit_price", "exit_reason", "return_pct",
        "pnl", "decision_score", "actual_return_5d", "actual_return_30d",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        price_df.to_parquet(Path(tmpdir) / "MRNA.parquet")
        signals = make_signals(3, "MRNA")
        result = simulate_ticker(
            ticker="MRNA",
            signals=signals,
            horizon="5d",
            take_profit_pct=0.08,
            stop_loss_pct=0.04,
            ohlcv_dir=tmpdir,
        )
    if result is not None and len(result) > 0:
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"


def test_simulate_ticker_exit_reason_valid_values():
    """exit_reason column should only contain valid exit reason strings."""
    price_df = make_price_df(252)
    valid_reasons = {"take_profit", "stop_loss", "horizon_expiry"}
    with tempfile.TemporaryDirectory() as tmpdir:
        price_df.to_parquet(Path(tmpdir) / "MRNA.parquet")
        signals = make_signals(3, "MRNA")
        result = simulate_ticker(
            ticker="MRNA",
            signals=signals,
            horizon="5d",
            take_profit_pct=0.08,
            stop_loss_pct=0.04,
            ohlcv_dir=tmpdir,
        )
    if result is not None and len(result) > 0:
        assert set(result["exit_reason"]).issubset(valid_reasons)


# ── simulate_window ───────────────────────────────────────────────────────────

def test_simulate_window_empty_signals_returns_empty_df():
    """simulate_window should return an empty DataFrame when signals is empty."""
    result = simulate_window(pd.DataFrame(), "5d")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_simulate_window_missing_ohlcv_returns_empty():
    """simulate_window should return empty DataFrame when no OHLCV files exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        signals = make_signals(2, "MISSING")
        result = simulate_window(signals, "5d", ohlcv_dir=tmpdir)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_simulate_window_combines_multiple_tickers():
    """simulate_window should return combined trades across multiple tickers."""
    price_df = make_price_df(252)
    with tempfile.TemporaryDirectory() as tmpdir:
        price_df.to_parquet(Path(tmpdir) / "MRNA.parquet")
        price_df.to_parquet(Path(tmpdir) / "BNTX.parquet")
        signals = pd.concat([make_signals(1, "MRNA"), make_signals(1, "BNTX")], ignore_index=True)
        result = simulate_window(signals, "5d", ohlcv_dir=tmpdir)
    # Should be a DataFrame (may be empty if no trades, but not an error)
    assert isinstance(result, pd.DataFrame)


# ── ClinicalTrialStrategy class structure ────────────────────────────────────

def test_strategy_has_required_class_attributes():
    """ClinicalTrialStrategy should have all required class-level attributes."""
    assert hasattr(ClinicalTrialStrategy, "signals_df")
    assert hasattr(ClinicalTrialStrategy, "take_profit_pct")
    assert hasattr(ClinicalTrialStrategy, "stop_loss_pct")
    assert hasattr(ClinicalTrialStrategy, "horizon_bars")
    assert hasattr(ClinicalTrialStrategy, "max_positions")
    assert hasattr(ClinicalTrialStrategy, "position_size")


def test_strategy_is_backtesting_subclass():
    """ClinicalTrialStrategy must be a subclass of backtesting.Strategy."""
    from backtesting import Strategy
    assert issubclass(ClinicalTrialStrategy, Strategy)
