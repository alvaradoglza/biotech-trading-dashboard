"""
metrics.py — Aggregate performance metrics, benchmark comparison, and 5d vs 30d comparison.
All logic is pure functions; no classes.
"""

import numpy as np
import pandas as pd

from backtest.config import INITIAL_CAPITAL


# ── Core metric computation ────────────────────────────────────────────────────

def compute_metrics(
    trade_log: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    equity_curve: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Compute aggregate performance metrics from a trade log DataFrame.

    Takes a trade log with columns [pnl, return_pct, entry_date, exit_date,
    exit_reason], an optional daily equity_curve DataFrame (DatetimeIndex,
    'equity' column) from the portfolio simulator. Returns a dict of named
    metrics. Returns all-zero/NaN dict if trade_log is empty.

    When equity_curve is provided: Sharpe is computed from daily portfolio
    returns (correct) and drawdown from the chronological equity series.
    Annualised return uses 365 calendar days for consistency with the
    benchmark calculation.
    """
    if len(trade_log) == 0:
        return _empty_metrics()

    pnl = trade_log["pnl"].dropna()
    returns = trade_log["return_pct"].dropna()

    total_pnl = pnl.sum()
    total_return_pct = (total_pnl / initial_capital) * 100

    # Annualized return — 365 calendar days (matches benchmark convention)
    n_days = _backtest_duration_days(trade_log)
    if n_days > 0:
        annualized_return_pct = ((1 + total_pnl / initial_capital) ** (365 / n_days) - 1) * 100
    else:
        annualized_return_pct = 0.0

    # Sharpe ratio — use daily equity curve returns when available
    if equity_curve is not None and len(equity_curve) > 1:
        daily_returns = equity_curve["equity"].pct_change().dropna()
        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = np.nan
    else:
        # Fallback: per-trade returns (less accurate, kept for compatibility)
        trade_daily_returns = returns / 100
        if len(trade_daily_returns) > 1 and trade_daily_returns.std() > 0:
            sharpe = (trade_daily_returns.mean() / trade_daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = np.nan

    # Max drawdown — use chronological equity curve when available
    if equity_curve is not None and len(equity_curve) > 1:
        max_drawdown_pct = _compute_max_drawdown(equity_curve["equity"].values)
    else:
        eq = _build_equity_curve(trade_log, initial_capital)
        max_drawdown_pct = _compute_max_drawdown(eq)

    # Win rate
    n_trades = len(returns)
    n_wins = (returns > 0).sum()
    win_rate = (n_wins / n_trades) * 100 if n_trades > 0 else 0.0

    # Profit factor
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf

    # Average holding period
    avg_holding_days = _avg_holding_days(trade_log)

    # Exposure time (fraction of backtest duration with open positions)
    exposure_pct = _compute_exposure(trade_log, n_days)

    return {
        "total_return_pct":       round(total_return_pct, 4),
        "annualized_return_pct":  round(annualized_return_pct, 4),
        "sharpe_ratio":           round(float(sharpe), 4) if not np.isnan(sharpe) else np.nan,
        "max_drawdown_pct":       round(max_drawdown_pct, 4),
        "win_rate_pct":           round(win_rate, 4),
        "profit_factor":          round(float(profit_factor), 4),
        "total_trades":           int(n_trades),
        "avg_return_per_trade":   round(float(returns.mean()), 4) if n_trades > 0 else 0.0,
        "avg_holding_days":       round(avg_holding_days, 2),
        "exposure_pct":           round(exposure_pct, 4),
    }


def _empty_metrics() -> dict[str, float]:
    """Return a zeroed metrics dict for empty trade logs."""
    return {
        "total_return_pct":      0.0,
        "annualized_return_pct": 0.0,
        "sharpe_ratio":          np.nan,
        "max_drawdown_pct":      0.0,
        "win_rate_pct":          0.0,
        "profit_factor":         np.nan,
        "total_trades":          0,
        "avg_return_per_trade":  0.0,
        "avg_holding_days":      0.0,
        "exposure_pct":          0.0,
    }


def _backtest_duration_days(trade_log: pd.DataFrame) -> float:
    """Compute the span in calendar days from first entry to last exit."""
    try:
        first_entry = pd.to_datetime(trade_log["entry_date"]).min()
        last_exit   = pd.to_datetime(trade_log["exit_date"]).max()
        if pd.isna(first_entry) or pd.isna(last_exit):
            return 0.0
        return max((last_exit - first_entry).days, 1)
    except Exception:
        return 0.0


def _build_equity_curve(trade_log: pd.DataFrame, initial_capital: float) -> np.ndarray:
    """Build a cumulative equity curve from the trade PnL series."""
    pnl = trade_log["pnl"].fillna(0).values
    equity = np.cumsum(pnl) + initial_capital
    return np.concatenate([[initial_capital], equity])


def _compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """Compute maximum drawdown percentage from a cumulative equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return float(abs(drawdown.min()) * 100)


def _avg_holding_days(trade_log: pd.DataFrame) -> float:
    """Compute average holding period in calendar days."""
    try:
        entry = pd.to_datetime(trade_log["entry_date"])
        exit_ = pd.to_datetime(trade_log["exit_date"])
        durations = (exit_ - entry).dt.days.dropna()
        return float(durations.mean()) if len(durations) > 0 else 0.0
    except Exception:
        return 0.0


def _compute_exposure(trade_log: pd.DataFrame, total_days: float) -> float:
    """Compute true time-in-market by merging overlapping trade date intervals.

    Sorts all (entry, exit) pairs, merges overlapping intervals, and sums unique
    calendar days with at least one open position. Returns percent of total_days.
    """
    if total_days <= 0 or len(trade_log) == 0:
        return 0.0
    try:
        entry_dates = pd.to_datetime(trade_log["entry_date"])
        exit_dates = pd.to_datetime(trade_log["exit_date"])
        intervals = sorted(zip(entry_dates, exit_dates))
        merged_days = 0
        cur_start, cur_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
            else:
                merged_days += (cur_end - cur_start).days + 1
                cur_start, cur_end = start, end
        merged_days += (cur_end - cur_start).days + 1
        return min(merged_days / total_days * 100, 100.0)
    except Exception:
        return 0.0


# ── Benchmark comparison ──────────────────────────────────────────────────────

def compute_benchmark_metrics(
    benchmark_df: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date:   str | pd.Timestamp,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict[str, float]:
    """Compute buy-and-hold performance metrics for a benchmark (e.g. SPY).

    Takes an OHLCV DataFrame (DatetimeIndex, Close column), date range, and capital.
    Returns a metrics dict in the same format as compute_metrics().
    """
    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(end_date)
    df = benchmark_df[(benchmark_df.index >= start) & (benchmark_df.index <= end)].copy()

    if len(df) < 2:
        return _empty_metrics()

    start_price = df["Close"].iloc[0]
    end_price   = df["Close"].iloc[-1]
    total_return_pct = ((end_price - start_price) / start_price) * 100

    n_days = (df.index[-1] - df.index[0]).days
    if n_days > 0:
        annualized_return_pct = ((1 + total_return_pct / 100) ** (365 / n_days) - 1) * 100
    else:
        annualized_return_pct = 0.0

    daily_returns = df["Close"].pct_change().dropna()
    if daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = np.nan

    equity = (df["Close"] / start_price) * initial_capital
    max_drawdown_pct = _compute_max_drawdown(equity.values)

    return {
        "total_return_pct":      round(total_return_pct, 4),
        "annualized_return_pct": round(annualized_return_pct, 4),
        "sharpe_ratio":          round(float(sharpe), 4) if not np.isnan(sharpe) else np.nan,
        "max_drawdown_pct":      round(max_drawdown_pct, 4),
        "win_rate_pct":          round(float((daily_returns > 0).mean() * 100), 4),
        "profit_factor":         np.nan,
        "total_trades":          1,
        "avg_return_per_trade":  round(total_return_pct, 4),
        "avg_holding_days":      float(n_days),
        "exposure_pct":          100.0,
    }


# ── Side-by-side comparison ───────────────────────────────────────────────────

def compare_horizons(
    metrics_5d: dict[str, float],
    metrics_30d: dict[str, float],
    benchmark: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build a side-by-side comparison DataFrame of 5d, 30d, and optionally benchmark metrics.

    Takes metric dicts and returns a DataFrame with metric names as rows and
    strategy/benchmark as columns.
    """
    data = {"5d_strategy": metrics_5d, "30d_strategy": metrics_30d}
    if benchmark is not None:
        data["benchmark"] = benchmark
    return pd.DataFrame(data).T
