"""Tests for metrics.py — compute_metrics, compute_benchmark_metrics, compare_horizons."""

import numpy as np
import pandas as pd
import pytest

from backtest.metrics import (
    compute_metrics,
    compute_benchmark_metrics,
    compare_horizons,
    _build_equity_curve,
    _compute_max_drawdown,
)
from backtest.config import INITIAL_CAPITAL


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_trade_log(n: int = 10, win_pct: float = 0.6) -> pd.DataFrame:
    """Create a synthetic trade log for metrics testing."""
    np.random.seed(42)
    n_wins = int(n * win_pct)
    returns = [5.0] * n_wins + [-3.0] * (n - n_wins)
    pnl = [r / 100 * 10_000 for r in returns]
    entry_dates = pd.date_range("2024-03-01", periods=n, freq="7D")
    exit_dates  = entry_dates + pd.Timedelta(days=5)
    reasons = ["take_profit"] * n_wins + ["stop_loss"] * (n - n_wins)
    return pd.DataFrame({
        "ticker":       [f"T{i}" for i in range(n)],
        "return_pct":   returns,
        "pnl":          pnl,
        "entry_date":   entry_dates,
        "exit_date":    exit_dates,
        "exit_reason":  reasons,
    })


def make_benchmark_df(n: int = 252, start: str = "2024-01-02") -> pd.DataFrame:
    """Create a synthetic benchmark OHLCV DataFrame."""
    dates = pd.date_range(start, periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "Open":   close * 0.99,
        "High":   close * 1.02,
        "Low":    close * 0.98,
        "Close":  close,
        "Volume": np.ones(n) * 1_000_000,
    }, index=dates)
    return df


# ── _build_equity_curve ───────────────────────────────────────────────────────

def test_build_equity_curve_starts_at_initial_capital():
    """Equity curve first value should equal initial_capital."""
    tl = make_trade_log(5)
    curve = _build_equity_curve(tl, INITIAL_CAPITAL)
    assert curve[0] == INITIAL_CAPITAL


def test_build_equity_curve_length():
    """Equity curve should have n+1 elements (initial + one per trade)."""
    tl = make_trade_log(5)
    curve = _build_equity_curve(tl, INITIAL_CAPITAL)
    assert len(curve) == 6


# ── _compute_max_drawdown ─────────────────────────────────────────────────────

def test_compute_max_drawdown_zero_for_monotone_growth():
    """Max drawdown should be 0% for a monotonically increasing equity curve."""
    curve = np.array([100, 105, 110, 115, 120], dtype=float)
    assert _compute_max_drawdown(curve) == 0.0


def test_compute_max_drawdown_detects_drop():
    """Max drawdown should detect a 50% drop from peak."""
    curve = np.array([100, 150, 75, 120], dtype=float)
    dd = _compute_max_drawdown(curve)
    assert abs(dd - 50.0) < 1.0  # ~50% drawdown


# ── compute_metrics ───────────────────────────────────────────────────────────

def test_compute_metrics_empty_returns_zero_dict():
    """compute_metrics should return an all-zero/NaN dict for an empty trade log."""
    result = compute_metrics(pd.DataFrame())
    assert result["total_trades"] == 0
    assert result["total_return_pct"] == 0.0


def test_compute_metrics_returns_all_keys():
    """compute_metrics should return all required metric keys."""
    required_keys = [
        "total_return_pct", "annualized_return_pct", "sharpe_ratio",
        "max_drawdown_pct", "win_rate_pct", "profit_factor", "total_trades",
        "avg_return_per_trade", "avg_holding_days", "exposure_pct",
    ]
    tl = make_trade_log(10)
    result = compute_metrics(tl)
    for key in required_keys:
        assert key in result, f"Missing metric: {key}"


def test_compute_metrics_total_trades():
    """compute_metrics total_trades should match the number of rows in trade_log."""
    tl = make_trade_log(7)
    result = compute_metrics(tl)
    assert result["total_trades"] == 7


def test_compute_metrics_win_rate():
    """compute_metrics win_rate_pct should reflect the actual win fraction."""
    tl = make_trade_log(10, win_pct=0.7)
    result = compute_metrics(tl)
    assert abs(result["win_rate_pct"] - 70.0) < 1.0


def test_compute_metrics_positive_total_return_for_net_profit():
    """total_return_pct should be positive when total PnL is positive."""
    tl = make_trade_log(10, win_pct=1.0)  # all wins
    result = compute_metrics(tl)
    assert result["total_return_pct"] > 0


def test_compute_metrics_max_drawdown_nonnegative():
    """max_drawdown_pct should always be >= 0."""
    tl = make_trade_log(10)
    result = compute_metrics(tl)
    assert result["max_drawdown_pct"] >= 0


def test_compute_metrics_profit_factor_positive():
    """profit_factor should be positive when there are both wins and losses."""
    tl = make_trade_log(10, win_pct=0.6)
    result = compute_metrics(tl)
    assert result["profit_factor"] > 0


def test_compute_metrics_avg_holding_days():
    """avg_holding_days should be approximately the actual holding period."""
    tl = make_trade_log(10)
    result = compute_metrics(tl)
    assert abs(result["avg_holding_days"] - 5.0) < 1.0


# ── compute_benchmark_metrics ─────────────────────────────────────────────────

def test_compute_benchmark_metrics_returns_all_keys():
    """compute_benchmark_metrics should return all required metric keys."""
    bench = make_benchmark_df(252)
    result = compute_benchmark_metrics(bench, "2024-01-01", "2024-12-31")
    required = [
        "total_return_pct", "annualized_return_pct", "sharpe_ratio",
        "max_drawdown_pct", "win_rate_pct",
    ]
    for k in required:
        assert k in result, f"Missing: {k}"


def test_compute_benchmark_metrics_exposure_is_100():
    """Benchmark exposure_pct should be 100% (always invested)."""
    bench = make_benchmark_df()
    result = compute_benchmark_metrics(bench, "2024-01-01", "2024-12-31")
    assert result["exposure_pct"] == 100.0


def test_compute_benchmark_metrics_empty_range_returns_zeros():
    """compute_benchmark_metrics should return zero dict for out-of-range dates."""
    bench = make_benchmark_df(10, "2024-01-02")
    result = compute_benchmark_metrics(bench, "2030-01-01", "2030-12-31")
    assert result["total_trades"] == 0


# ── compare_horizons ──────────────────────────────────────────────────────────

def test_compare_horizons_returns_dataframe():
    """compare_horizons should return a DataFrame."""
    m5  = compute_metrics(make_trade_log(5))
    m30 = compute_metrics(make_trade_log(5))
    result = compare_horizons(m5, m30)
    assert isinstance(result, pd.DataFrame)


def test_compare_horizons_has_both_rows():
    """compare_horizons DataFrame should have rows for both horizons."""
    m5  = compute_metrics(make_trade_log(5))
    m30 = compute_metrics(make_trade_log(5))
    result = compare_horizons(m5, m30)
    assert "5d_strategy" in result.index
    assert "30d_strategy" in result.index


def test_compare_horizons_with_benchmark():
    """compare_horizons with benchmark should add a third row."""
    m5   = compute_metrics(make_trade_log(5))
    m30  = compute_metrics(make_trade_log(5))
    bench = compute_benchmark_metrics(make_benchmark_df(), "2024-01-01", "2024-12-31")
    result = compare_horizons(m5, m30, bench)
    assert "benchmark" in result.index
    assert result.shape[0] == 3
