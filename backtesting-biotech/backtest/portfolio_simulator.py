"""
portfolio_simulator.py — True single-pool portfolio simulation.

Enforces a shared capital pool, correct horizon bar counting, TP/SL anchored
to actual fill price, and real slippage + commission on both legs.
Position sizing is delegated to a weight_fn (default: equal-weight).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.config import (
    OHLCV_DIR,
    INITIAL_CAPITAL,
    MAX_OPEN_POSITIONS,
    COMMISSION_PCT,
    SLIPPAGE_PCT,
    TP_SL_CONFIG,
)
from backtest.data_loader import load_ohlcv


def simulate_portfolio(
    signals: pd.DataFrame,
    horizon: str,
    take_profit_pct: float | None = None,
    stop_loss_pct: float | None = None,
    ohlcv_dir: str | Path = OHLCV_DIR,
    initial_capital: float = INITIAL_CAPITAL,
    max_positions: int = MAX_OPEN_POSITIONS,
    commission_pct: float = COMMISSION_PCT,
    slippage_pct: float = SLIPPAGE_PCT,
    weight_fn: object | None = None,
    horizon_bars: int | None = None,
    _ohlcv_cache: dict | None = None,
    tiered_slippage: bool = False,
    adv_cap_pct: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a long-only portfolio from pre-computed signals.

    Takes a signals DataFrame (output of run_rolling_loop, all windows concatenated),
    a horizon string, exit parameters, and execution costs. Returns (trade_log, equity_curve).

    Position sizing: if weight_fn is provided it is called per window to set
    target_dollars on each signal row. Otherwise equal allocation
    (initial_capital / max_positions) is used.

    Exit logic per bar (evaluated in priority order):
      1. Open gaps up through TP  → fill at Open
      2. High reaches TP intraday → fill at TP price
      3. Open gaps down through SL → fill at Open
      4. Low reaches SL intraday  → fill at SL price
      5. bars_held >= horizon_bars → fill at Close

    TP/SL anchored to entry_fill (Open * (1+slippage)), not prior close.
    Commission applied on notional of both entry and exit legs.
    """
    if take_profit_pct is None:
        take_profit_pct = TP_SL_CONFIG[horizon]["take_profit"]
    if stop_loss_pct is None:
        stop_loss_pct = TP_SL_CONFIG[horizon]["stop_loss"]

    horizon_bars = horizon_bars if horizon_bars is not None else (5 if horizon == "5d" else 30)

    if len(signals) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # ── Load price data ───────────────────────────────────────────────────────
    ohlcv_cache = (
        _ohlcv_cache
        if _ohlcv_cache is not None
        else _load_ohlcv_cache(signals["ticker"].unique(), ohlcv_dir)
    )

    # ── Resolve entry dates ───────────────────────────────────────────────────
    signal_rows = _resolve_entry_dates(signals, ohlcv_cache)
    n_skipped_data = len(signals) - len(signal_rows)
    print(f"  Resolved {len(signal_rows)} signals ({n_skipped_data} skipped: no price data).")

    if not signal_rows:
        return pd.DataFrame(), pd.DataFrame()

    # ── Pre-compute position sizes ────────────────────────────────────────────
    if weight_fn is not None:
        _apply_weight_fn(signal_rows, weight_fn, ohlcv_cache, initial_capital)
        n_windows = len({r["window_id"] for r in signal_rows})
        print(f"  Custom weight_fn sizing: {len(signal_rows)} signals across {n_windows} windows.")
    else:
        fixed_dollars = initial_capital / max_positions
        for row in signal_rows:
            row["target_dollars"] = fixed_dollars

    # ── Build trading calendar ────────────────────────────────────────────────
    all_dates = _build_trading_calendar(ohlcv_cache, signal_rows)

    # Group signals by entry date for O(1) daily lookup
    signals_by_date: dict[pd.Timestamp, list[dict]] = {}
    for row in signal_rows:
        signals_by_date.setdefault(row["entry_date"], []).append(row)

    # ── Main simulation loop ──────────────────────────────────────────────────
    cash: float = float(initial_capital)
    open_positions: list[dict] = []
    completed_trades: list[dict] = []
    equity_rows: list[dict] = []
    skipped_cash = 0
    skipped_cap = 0
    min_position = initial_capital * 0.001   # skip positions < 0.1% of capital

    for date in all_dates:

        # 1. Open new positions at today's open
        if date in signals_by_date:
            candidates = sorted(
                signals_by_date[date],
                key=lambda x: x["decision_score"],
                reverse=True,
            )
            for sig in candidates:
                # Enforce max_positions cap regardless of sizing mode
                if len(open_positions) >= max_positions:
                    skipped_cap += 1
                    continue

                # Deploy the pre-computed target, capped by available cash
                target = sig["target_dollars"]
                position_dollars = min(target, cash)

                if position_dollars < min_position:
                    skipped_cash += 1
                    continue

                ticker = sig["ticker"]
                ohlcv = ohlcv_cache.get(ticker)
                if ohlcv is None or date not in ohlcv.index:
                    continue

                entry_open = float(ohlcv.at[date, "Open"])
                _entry_slip = _tiered_slip(entry_open, slippage_pct) if tiered_slippage else slippage_pct
                entry_fill = entry_open * (1.0 + _entry_slip)

                # Cap position by average daily dollar volume
                if adv_cap_pct is not None:
                    adv = _compute_adv(ohlcv, date)
                    position_dollars = min(position_dollars, adv * adv_cap_pct)
                    if position_dollars < min_position:
                        skipped_cash += 1
                        continue

                cost_per_share = entry_fill * (1.0 + commission_pct)
                shares = position_dollars / cost_per_share

                _tp = entry_fill * (1.0 + take_profit_pct)
                _sl = entry_fill * (1.0 - stop_loss_pct)

                cash -= position_dollars
                open_positions.append({
                    "ticker":            ticker,
                    "entry_date":        date,
                    "entry_fill":        entry_fill,
                    "shares":            shares,
                    "position_dollars":  position_dollars,
                    "tp_price":          _tp,
                    "sl_price":          _sl,
                    "bars_held":         0,
                    "window_id":         sig["window_id"],
                    "horizon":           sig["horizon"],
                    "published_at":      sig["published_at"],
                    "decision_score":    sig["decision_score"],
                    "actual_return_5d":  sig.get("return_5d", float("nan")),
                    "actual_return_30d": sig.get("return_30d", float("nan")),
                })

        # 2. Check exits for all open positions (including those opened today)
        for pos in list(open_positions):
            ohlcv = ohlcv_cache.get(pos["ticker"])
            if ohlcv is None or date not in ohlcv.index:
                continue

            bar = ohlcv.loc[date]
            # Don't count entry day as a held bar — bars_held tracks days
            # elapsed *after* entry so horizon fires on the correct close.
            if date != pos["entry_date"]:
                pos["bars_held"] += 1

            o = float(bar["Open"])
            h = float(bar["High"])
            l = float(bar["Low"])
            c = float(bar["Close"])

            tp = pos["tp_price"]
            sl = pos["sl_price"]

            raw_exit: float | None = None
            exit_reason: str | None = None

            if o >= tp:
                raw_exit, exit_reason = o, "take_profit"
            elif h >= tp:
                raw_exit, exit_reason = tp, "take_profit"
            elif o <= sl:
                raw_exit, exit_reason = o, "stop_loss"
            elif l <= sl:
                raw_exit, exit_reason = sl, "stop_loss"

            # Horizon expiry always applies
            if raw_exit is None and pos["bars_held"] >= horizon_bars:
                raw_exit, exit_reason = c, "horizon_expiry"

            if raw_exit is not None:
                _exit_slip = _tiered_slip(raw_exit, slippage_pct) if tiered_slippage else slippage_pct
                exit_fill = raw_exit * (1.0 - _exit_slip)
                proceeds = pos["shares"] * exit_fill * (1.0 - commission_pct)
                pnl = proceeds - pos["position_dollars"]
                cash += proceeds

                completed_trades.append({
                    "window_id":         pos["window_id"],
                    "horizon":           pos["horizon"],
                    "ticker":            pos["ticker"],
                    "published_at":      pos["published_at"],
                    "entry_date":        pos["entry_date"],
                    "entry_price":       pos["entry_fill"],
                    "exit_date":         date,
                    "exit_price":        exit_fill,
                    "exit_reason":       exit_reason,
                    "return_pct":        pnl / pos["position_dollars"] * 100.0,
                    "pnl":               pnl,
                    "decision_score":    pos["decision_score"],
                    "actual_return_5d":  pos["actual_return_5d"],
                    "actual_return_30d": pos["actual_return_30d"],
                })
                open_positions.remove(pos)

        # 3. Mark-to-market equity snapshot
        mtm = cash
        for pos in open_positions:
            ohlcv = ohlcv_cache.get(pos["ticker"])
            if ohlcv is not None and date in ohlcv.index:
                mtm += pos["shares"] * float(ohlcv.at[date, "Close"])
            else:
                mtm += pos["position_dollars"]
        equity_rows.append({"date": date, "equity": mtm})

    # Close any positions still open after the last calendar date
    if open_positions:
        _close_remaining(
            open_positions, completed_trades, ohlcv_cache,
            all_dates[-1], slippage_pct, commission_pct,
            tiered_slippage=tiered_slippage,
        )

    if skipped_cap > 0:
        print(f"  {skipped_cap} signals skipped (max {max_positions} concurrent positions).")
    if skipped_cash > 0:
        print(f"  {skipped_cash} signals skipped (insufficient cash).")

    trade_log = pd.DataFrame(completed_trades) if completed_trades else pd.DataFrame()
    equity_df = (
        pd.DataFrame(equity_rows).set_index("date")
        if equity_rows
        else pd.DataFrame(columns=["equity"])
    )
    return trade_log, equity_df


# ── Private helpers ───────────────────────────────────────────────────────────

def _apply_weight_fn(
    signal_rows: list[dict],
    weight_fn: object,
    ohlcv_cache: dict,
    initial_capital: float,
) -> None:
    """Group signal_rows by window_id and call weight_fn on each group in-place.

    weight_fn signature: (rows, initial_capital, ohlcv_cache, **kwargs) -> None
    It must set row["target_dollars"] on each row in the group.
    """
    windows: dict[int, list[dict]] = {}
    for row in signal_rows:
        windows.setdefault(row["window_id"], []).append(row)
    for rows in windows.values():
        weight_fn(rows, initial_capital, ohlcv_cache)


def _tiered_slip(price: float, base: float) -> float:
    """Return slippage rate scaled to price level — wider for penny stocks."""
    if price < 2.0:
        return 0.05
    if price < 5.0:
        return 0.02
    return base


def _compute_adv(ohlcv: pd.DataFrame, date: pd.Timestamp, lookback: int = 20) -> float:
    """Compute average daily dollar volume over `lookback` bars before `date`."""
    hist = ohlcv[ohlcv.index < date].tail(lookback)
    if len(hist) < 5:
        return float("inf")
    return float((hist["Close"] * hist["Volume"]).mean())


def _load_ohlcv_cache(tickers: list[str], ohlcv_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load OHLCV data for every ticker; warn and skip missing files.

    Takes an iterable of ticker strings and a directory path. Returns a dict
    mapping ticker → normalized OHLCV DataFrame.
    """
    cache = {}
    for ticker in tickers:
        try:
            cache[ticker] = load_ohlcv(ticker, ohlcv_dir)
        except FileNotFoundError:
            warnings.warn(f"No OHLCV file for {ticker} — signals will be skipped.")
    return cache


def _resolve_entry_dates(
    signals: pd.DataFrame,
    ohlcv_cache: dict[str, pd.DataFrame],
) -> list[dict]:
    """Find the first available trading day on or after each signal's published_at.

    Takes the signals DataFrame and a loaded OHLCV cache. Returns a list of
    signal dicts with an added 'entry_date' key. Rows with no future trading day
    are silently dropped.
    """
    rows = []
    for _, sig in signals.iterrows():
        ticker = sig["ticker"]
        ohlcv = ohlcv_cache.get(ticker)
        if ohlcv is None:
            continue
        pub = pd.Timestamp(sig["published_at"]).normalize()
        # Enter NEXT trading day AFTER published_at — we cannot trade on the
        # announcement day itself without knowing intraday release time.
        future = ohlcv.index[ohlcv.index > pub]
        if len(future) == 0:
            continue
        row = sig.to_dict()
        row["entry_date"] = future[0]
        rows.append(row)
    return rows


def _build_trading_calendar(
    ohlcv_cache: dict[str, pd.DataFrame],
    signal_rows: list[dict],
) -> list[pd.Timestamp]:
    """Build a sorted list of all trading dates from the earliest signal entry onward.

    Takes the OHLCV cache and resolved signal rows. Returns sorted Timestamps
    covering the simulation period across all loaded tickers.
    """
    min_date = min(r["entry_date"] for r in signal_rows)
    all_dates: set[pd.Timestamp] = set()
    for ohlcv in ohlcv_cache.values():
        all_dates.update(ohlcv.index[ohlcv.index >= min_date])
    return sorted(all_dates)


def _close_remaining(
    open_positions: list[dict],
    completed_trades: list[dict],
    ohlcv_cache: dict[str, pd.DataFrame],
    last_date: pd.Timestamp,
    slippage_pct: float,
    commission_pct: float,
    tiered_slippage: bool = False,
) -> None:
    """Force-close any positions still open at end of simulation at last available price.

    Mutates completed_trades in-place. Uses the last available Close for each
    ticker on or before last_date, falling back to entry_fill if no price found.
    """
    for pos in open_positions:
        ohlcv = ohlcv_cache.get(pos["ticker"])
        if ohlcv is not None:
            past = ohlcv[ohlcv.index <= last_date]
            close_price = float(past.iloc[-1]["Close"]) if len(past) > 0 else pos["entry_fill"]
        else:
            close_price = pos["entry_fill"]

        _slip = _tiered_slip(close_price, slippage_pct) if tiered_slippage else slippage_pct
        exit_fill = close_price * (1.0 - _slip)
        proceeds = pos["shares"] * exit_fill * (1.0 - commission_pct)
        pnl = proceeds - pos["position_dollars"]

        completed_trades.append({
            "window_id":         pos["window_id"],
            "horizon":           pos["horizon"],
            "ticker":            pos["ticker"],
            "published_at":      pos["published_at"],
            "entry_date":        pos["entry_date"],
            "entry_price":       pos["entry_fill"],
            "exit_date":         last_date,
            "exit_price":        exit_fill,
            "exit_reason":       "end_of_backtest",
            "return_pct":        pnl / pos["position_dollars"] * 100.0,
            "pnl":               pnl,
            "decision_score":    pos["decision_score"],
            "actual_return_5d":  pos["actual_return_5d"],
            "actual_return_30d": pos["actual_return_30d"],
        })
