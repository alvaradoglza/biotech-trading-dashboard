"""Tests for sweep.py — run_sweep and best_params."""

import pandas as pd
import pytest

from backtest.sweep import run_sweep, best_params


# ── Helpers ───────────────────────────────────────────────────────────────────

def mock_backtest_fn(tp: float, sl: float) -> dict:
    """A mock backtest function that returns metrics based on tp - sl."""
    return {
        "total_return_pct":      (tp - sl) * 100,
        "sharpe_ratio":          tp / max(sl, 0.001),
        "max_drawdown_pct":      sl * 50,
        "win_rate_pct":          60.0,
        "total_trades":          20,
    }


# ── run_sweep ─────────────────────────────────────────────────────────────────

def test_run_sweep_returns_dataframe():
    """run_sweep should return a DataFrame."""
    result = run_sweep(mock_backtest_fn, tp_grid=[0.05, 0.10], sl_grid=[0.03, 0.05])
    assert isinstance(result, pd.DataFrame)


def test_run_sweep_row_count():
    """run_sweep should have len(tp_grid) * len(sl_grid) rows."""
    tp = [0.05, 0.10, 0.15]
    sl = [0.03, 0.05]
    result = run_sweep(mock_backtest_fn, tp_grid=tp, sl_grid=sl)
    assert len(result) == 6


def test_run_sweep_has_tp_sl_columns():
    """run_sweep result should contain take_profit_pct and stop_loss_pct columns."""
    result = run_sweep(mock_backtest_fn, tp_grid=[0.08], sl_grid=[0.04])
    assert "take_profit_pct" in result.columns
    assert "stop_loss_pct" in result.columns


def test_run_sweep_sorted_by_total_return():
    """run_sweep should return rows sorted by total_return_pct descending."""
    result = run_sweep(mock_backtest_fn, tp_grid=[0.05, 0.10, 0.15], sl_grid=[0.03, 0.05])
    returns = result["total_return_pct"].values
    assert list(returns) == sorted(returns, reverse=True)


def test_run_sweep_metric_values_match_mock():
    """run_sweep total_trades should equal mock_backtest_fn's return value."""
    result = run_sweep(mock_backtest_fn, tp_grid=[0.10], sl_grid=[0.05])
    assert result["total_trades"].iloc[0] == 20


# ── best_params ───────────────────────────────────────────────────────────────

def test_best_params_returns_dict():
    """best_params should return a dict."""
    sweep_df = run_sweep(mock_backtest_fn, tp_grid=[0.08, 0.15], sl_grid=[0.03, 0.05])
    result = best_params(sweep_df)
    assert isinstance(result, dict)


def test_best_params_has_tp_sl_keys():
    """best_params dict should have take_profit_pct and stop_loss_pct keys."""
    sweep_df = run_sweep(mock_backtest_fn, tp_grid=[0.08, 0.15], sl_grid=[0.03, 0.05])
    result = best_params(sweep_df)
    assert "take_profit_pct" in result
    assert "stop_loss_pct" in result


def test_best_params_returns_highest_return_row():
    """best_params should select the row with highest total_return_pct."""
    sweep_df = run_sweep(mock_backtest_fn, tp_grid=[0.05, 0.15], sl_grid=[0.03])
    result = best_params(sweep_df, "total_return_pct")
    # Higher TP leads to higher return in mock
    assert result["take_profit_pct"] == 0.15


def test_best_params_raises_on_empty_df():
    """best_params should raise ValueError for an empty DataFrame."""
    with pytest.raises(ValueError, match="empty"):
        best_params(pd.DataFrame())


def test_best_params_raises_on_unknown_metric():
    """best_params should raise ValueError for a non-existent metric column."""
    sweep_df = run_sweep(mock_backtest_fn, tp_grid=[0.08], sl_grid=[0.04])
    with pytest.raises(ValueError, match="Metric"):
        best_params(sweep_df, "nonexistent_metric")
