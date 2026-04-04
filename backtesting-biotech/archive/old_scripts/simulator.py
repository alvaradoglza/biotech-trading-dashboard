"""
simulator.py — Run backtesting.py per ticker per window and collect trade results.
All logic is pure functions; the Strategy class is imported from strategy.py.
"""

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from backtesting import Backtest

from backtest.config import (
    OHLCV_DIR,
    INITIAL_CAPITAL,
    MAX_OPEN_POSITIONS,
    COMMISSION_PCT,
    SLIPPAGE_PCT,
    TP_SL_CONFIG,
)
from backtest.data_loader import load_ohlcv
from backtest.strategy import ClinicalTrialStrategy


# ── Entry date resolution ─────────────────────────────────────────────────────

def next_trading_day(
    published_at: pd.Timestamp,
    price_df: pd.DataFrame,
) -> pd.Timestamp | None:
    """Find the next available trading day at or after published_at in price_df.

    Takes an announcement timestamp and the OHLCV DataFrame (DatetimeIndex).
    Returns the next trading day Timestamp, or None if no future date exists in price_df.
    """
    pub_date = published_at.normalize()
    future_dates = price_df.index[price_df.index >= pub_date]
    if len(future_dates) == 0:
        return None
    return future_dates[0]


# ── Single ticker simulation ──────────────────────────────────────────────────

def simulate_ticker(
    ticker: str,
    signals: pd.DataFrame,
    horizon: str,
    take_profit_pct: float,
    stop_loss_pct: float,
    ohlcv_dir: str | Path = OHLCV_DIR,
    initial_capital: float = INITIAL_CAPITAL,
    max_positions: int = MAX_OPEN_POSITIONS,
    commission_pct: float = COMMISSION_PCT,
    slippage_pct: float = SLIPPAGE_PCT,
) -> pd.DataFrame | None:
    """Simulate trades for one ticker using backtesting.py.

    Takes a ticker string, a signals DataFrame filtered to this ticker, a horizon,
    TP/SL percentages, and config. Returns a trade log DataFrame, or None if the
    ticker has no OHLCV data or no valid entry dates.
    """
    try:
        price_df = load_ohlcv(ticker, ohlcv_dir)
    except FileNotFoundError:
        warnings.warn(f"Ticker {ticker} has no OHLCV data, skipping.")
        return None

    horizon_bars = 5 if horizon == "5d" else 30

    # Resolve entry dates and precompute TP/SL anchored to entry Open + slippage
    entry_records = []
    for _, row in signals.iterrows():
        entry_date = next_trading_day(row["published_at"], price_df)
        if entry_date is None or entry_date not in price_df.index:
            continue
        entry_open = float(price_df.at[entry_date, "Open"])
        entry_fill = entry_open * (1.0 + slippage_pct)   # slippage on entry
        entry_records.append({
            "entry_date":     entry_date,
            "window_id":      row.get("window_id", -1),
            "horizon":        horizon,
            "ticker":         ticker,
            "published_at":   row["published_at"],
            "decision_score": row.get("decision_score", np.nan),
            "actual_return_5d":  row.get("return_5d", np.nan),
            "actual_return_30d": row.get("return_30d", np.nan),
            "tp_price":       entry_fill * (1.0 + take_profit_pct),
            "sl_price":       entry_fill * (1.0 - stop_loss_pct),
        })

    if not entry_records:
        return None

    signals_df_clean = pd.DataFrame(entry_records)
    position_size = 1.0 / max_positions

    # Configure strategy class attributes
    ClinicalTrialStrategy.signals_df      = signals_df_clean
    ClinicalTrialStrategy.take_profit_pct = take_profit_pct
    ClinicalTrialStrategy.stop_loss_pct   = stop_loss_pct
    ClinicalTrialStrategy.horizon_bars    = horizon_bars
    ClinicalTrialStrategy.max_positions   = max_positions
    ClinicalTrialStrategy.position_size   = position_size

    bt = Backtest(
        price_df,
        ClinicalTrialStrategy,
        cash=initial_capital,
        commission=commission_pct,
        exclusive_orders=False,
    )

    try:
        stats = bt.run()
    except Exception as e:
        warnings.warn(f"Backtest failed for {ticker}: {e}")
        return None

    # Extract trade log from backtesting.py stats
    trades_raw = stats.get("_trades", None)
    if trades_raw is None or len(trades_raw) == 0:
        return None

    return _format_trade_log(trades_raw, signals_df_clean, ticker, horizon)


# ── Trade log formatting ──────────────────────────────────────────────────────

def _format_trade_log(
    trades_raw: pd.DataFrame,
    signals_df: pd.DataFrame,
    ticker: str,
    horizon: str,
) -> pd.DataFrame:
    """Map raw backtesting.py trade output to the canonical trade log schema.

    Takes the _trades DataFrame from backtesting.py stats, the signals DataFrame,
    and metadata. Returns a trade log DataFrame conforming to the spec schema.
    """
    records = []
    # backtesting.py _trades columns: EntryBar, ExitBar, EntryPrice, ExitPrice,
    # PnL, ReturnPct, EntryTime, ExitTime, Tag, Duration
    for _, trade in trades_raw.iterrows():
        entry_time = pd.Timestamp(trade.get("EntryTime", pd.NaT))

        # Match to signal record by nearest entry date
        matched = signals_df[signals_df["entry_date"] == entry_time.normalize()]
        if len(matched) == 0:
            # Fallback: use first signal for this ticker
            matched = signals_df.iloc[:1]

        sig = matched.iloc[0]

        exit_price = float(trade.get("ExitPrice", np.nan))
        entry_price = float(trade.get("EntryPrice", np.nan))
        return_pct = float(trade.get("ReturnPct", np.nan)) * 100  # convert to %
        pnl = float(trade.get("PnL", np.nan))

        # Infer exit reason from return
        tp_level = entry_price * (1 + ClinicalTrialStrategy.take_profit_pct)
        sl_level = entry_price * (1 - ClinicalTrialStrategy.stop_loss_pct)
        if not np.isnan(exit_price):
            if exit_price >= tp_level * 0.999:
                exit_reason = "take_profit"
            elif exit_price <= sl_level * 1.001:
                exit_reason = "stop_loss"
            else:
                exit_reason = "horizon_expiry"
        else:
            exit_reason = "horizon_expiry"

        records.append({
            "window_id":         int(sig.get("window_id", -1)),
            "horizon":           horizon,
            "ticker":            ticker,
            "published_at":      sig["published_at"],
            "entry_date":        entry_time,
            "entry_price":       entry_price,
            "exit_date":         pd.Timestamp(trade.get("ExitTime", pd.NaT)),
            "exit_price":        exit_price,
            "exit_reason":       exit_reason,
            "return_pct":        return_pct,
            "pnl":               pnl,
            "decision_score":    float(sig.get("decision_score", np.nan)),
            "actual_return_5d":  float(sig.get("actual_return_5d", np.nan)),
            "actual_return_30d": float(sig.get("actual_return_30d", np.nan)),
        })

    return pd.DataFrame(records)


# ── Window-level simulation ───────────────────────────────────────────────────

def simulate_window(
    signals: pd.DataFrame,
    horizon: str,
    take_profit_pct: float | None = None,
    stop_loss_pct:   float | None = None,
    ohlcv_dir: str | Path = OHLCV_DIR,
    initial_capital: float = INITIAL_CAPITAL,
    max_positions:   int   = MAX_OPEN_POSITIONS,
    commission_pct:  float = COMMISSION_PCT,
    slippage_pct:    float = SLIPPAGE_PCT,
) -> pd.DataFrame:
    """Simulate all signals from one rolling window across all tickers.

    Takes a signals DataFrame (may contain multiple tickers) and returns a
    combined trade log DataFrame. Returns empty DataFrame if no trades executed.
    """
    if take_profit_pct is None:
        take_profit_pct = TP_SL_CONFIG[horizon]["take_profit"]
    if stop_loss_pct is None:
        stop_loss_pct = TP_SL_CONFIG[horizon]["stop_loss"]

    if len(signals) == 0:
        return pd.DataFrame()

    trade_frames = []
    for ticker in signals["ticker"].unique():
        ticker_signals = signals[signals["ticker"] == ticker].copy()
        trades = simulate_ticker(
            ticker=ticker,
            signals=ticker_signals,
            horizon=horizon,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            ohlcv_dir=ohlcv_dir,
            initial_capital=initial_capital,
            max_positions=max_positions,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
        )
        if trades is not None and len(trades) > 0:
            trade_frames.append(trades)

    if not trade_frames:
        return pd.DataFrame()

    return pd.concat(trade_frames, ignore_index=True)
