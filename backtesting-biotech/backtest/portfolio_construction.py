"""
portfolio_construction.py — Portfolio-weighting strategies for the active buy list.

Each public function takes:
  rows           — list of signal dicts for one rolling window (from _resolve_entry_dates)
  initial_capital — total capital available for this window's allocation
  ohlcv_cache    — dict of ticker → OHLCV DataFrame
  **kwargs       — strategy-specific parameters (passed via functools.partial)

Each function sets row["target_dollars"] in-place — the intended dollar allocation
per position.  A value of 0.0 means the signal is excluded this window.

All historical lookbacks use data strictly before the earliest entry_date in the
window to prevent look-ahead bias.
"""

import warnings
import numpy as np
import pandas as pd


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _ref_date(rows: list[dict]) -> pd.Timestamp:
    """Return the earliest entry_date in the window — used as lookback cutoff."""
    return min(pd.Timestamp(r["entry_date"]) for r in rows)


def _close_returns(ohlcv: pd.DataFrame, cutoff: pd.Timestamp, window: int) -> pd.Series:
    """Daily close-to-close returns for up to `window` bars strictly before cutoff."""
    hist = ohlcv[ohlcv.index < cutoff].tail(window)["Close"]
    return hist.pct_change().dropna()


def _apply_capped_weights(
    rows: list[dict],
    raw_weights: np.ndarray,
    initial_capital: float,
    max_weight: float,
) -> None:
    """Normalise raw_weights → cap each at max_weight → set target_dollars.

    No renormalisation after capping: weight above max_weight stays as uninvested
    cash. This preserves the intent of max_weight=0.10 — no single position can
    exceed 10% of initial_capital regardless of how few signals there are.
    """
    total = raw_weights.sum()
    w = raw_weights / total if total > 0 else np.ones(len(rows)) / len(rows)
    w = np.minimum(w, max_weight)
    for row, wi in zip(rows, w):
        row["target_dollars"] = float(initial_capital * wi)


# ── Strategy 1: Equal-weight ───────────────────────────────────────────────────

def allocate_equal_weight(
    rows: list[dict],
    initial_capital: float,
    ohlcv_cache: dict,
    *,
    max_weight: float = 0.10,
    **_,
) -> None:
    """Equal-weight all signals in the window, each capped at max_weight.

    Takes all active buy signals in a rolling window. Assigns weight 1/N to each,
    then caps at max_weight and renormalises. Unused cash stays in the pool.
    """
    if not rows:
        return
    raw = np.ones(len(rows))
    _apply_capped_weights(rows, raw, initial_capital, max_weight)


# ── Strategy 2: Inverse-volatility weighting ─────────────────────────────────

def allocate_inverse_volatility(
    rows: list[dict],
    initial_capital: float,
    ohlcv_cache: dict,
    *,
    vol_window: int = 60,
    max_weight: float = 0.10,
    **_,
) -> None:
    """Weight each signal proportional to 1/annualised_volatility, capped at max_weight.

    Uses vol_window bars of close-to-close returns before the window entry date.
    Tickers with insufficient history get the median inverse-vol as fallback.
    """
    if not rows:
        return
    cutoff = _ref_date(rows)
    inv_vols = []
    for row in rows:
        ohlcv = ohlcv_cache.get(row["ticker"])
        if ohlcv is None:
            inv_vols.append(np.nan)
            continue
        rets = _close_returns(ohlcv, cutoff, vol_window)
        if len(rets) < 10:
            inv_vols.append(np.nan)
            continue
        vol = float(rets.std() * np.sqrt(252))
        inv_vols.append(1.0 / vol if vol > 0 else np.nan)

    arr = np.array(inv_vols, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        allocate_equal_weight(rows, initial_capital, ohlcv_cache, max_weight=max_weight)
        return
    arr = np.where(np.isnan(arr), float(np.median(valid)), arr)
    _apply_capped_weights(rows, arr, initial_capital, max_weight)


