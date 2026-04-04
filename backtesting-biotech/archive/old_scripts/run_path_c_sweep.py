"""
run_path_c_sweep.py — Exhaustive Path C parameter exploration with robustness checks.

Phases:
  1. Position sizing + capacity (max_weight × max_positions × sizing)
  2. Take profit level
  3. Horizon bars
  4. ADV cap
  5. Grace period
  6. Price tier analysis (robustness only — NOT parameter selection)
  7. Trailing stops

At each phase, run validation: exclusion test, year split, sensitivity smoothness.

Usage: python -m backtest.run_path_c_sweep
"""

import warnings
from functools import partial
from itertools import product

import numpy as np
import pandas as pd

from backtest.config import (
    ANNOUNCEMENTS_PATH, OHLCV_DIR, OUTPUT_DIR,
    START_DATE, END_DATE, TRAIN_MONTHS, PREDICT_WEEKS, STEP_WEEKS,
    INITIAL_CAPITAL, COMMISSION_PCT, SLIPPAGE_PCT,
    TIERED_SLIPPAGE, ADV_CAP_PCT,
)
from backtest.data_loader import load_announcements
from backtest.features import fit_ohe
from backtest.model import generate_windows
from backtest.portfolio_simulator import simulate_portfolio, _load_ohlcv_cache
from backtest.portfolio_construction import allocate_equal_weight, allocate_inverse_volatility
from backtest.metrics import compute_metrics


# ── Signal generation (no model) ──────────────────────────────────────────────

def build_all_signals(announcements: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Build signals from ALL announcements in each prediction window — no model."""
    windows = generate_windows(
        start_date=START_DATE, end_date=END_DATE,
        train_months=TRAIN_MONTHS, predict_weeks=PREDICT_WEEKS,
        step_weeks=STEP_WEEKS,
    )
    all_signals = []
    for window in windows:
        pred_df = announcements[
            (announcements["published_at"] >= window.pred_start) &
            (announcements["published_at"] < window.pred_end)
        ].copy()
        if len(pred_df) == 0:
            continue
        pred_df["prediction"] = 1
        pred_df["decision_score"] = 1.0
        pred_df["window_id"] = window.window_id
        pred_df["horizon"] = horizon
        pred_df = pred_df.sort_values("published_at", ascending=True)
        pred_df = pred_df.drop_duplicates(subset=["ticker"], keep="first")
        all_signals.append(pred_df)
    if not all_signals:
        return pd.DataFrame()
    return pd.concat(all_signals, ignore_index=True)


# ── Robustness metrics ────────────────────────────────────────────────────────

def robustness_metrics(trade_log: pd.DataFrame, initial_capital: float) -> dict:
    """Compute exclusion test + year split + concentration for a trade log."""
    if len(trade_log) == 0:
        return {"ex_top1_ret": 0, "ex_top5_ret": 0, "top1_pct": 0, "top5_pct": 0,
                "y1_ret": 0, "y2_ret": 0, "penny_pnl": 0, "normal_pnl": 0}

    total_pnl = trade_log["pnl"].sum()
    ticker_pnl = trade_log.groupby("ticker")["pnl"].sum().sort_values(ascending=False)

    # Exclusion test
    top1 = ticker_pnl.iloc[0] if len(ticker_pnl) > 0 else 0
    top5 = ticker_pnl.head(5).sum() if len(ticker_pnl) >= 5 else total_pnl
    ex_top1 = (total_pnl - top1) / initial_capital * 100
    ex_top5 = (total_pnl - top5) / initial_capital * 100

    # Concentration
    top1_pct = top1 / total_pnl * 100 if abs(total_pnl) > 0 else 0
    top5_pct = top5 / total_pnl * 100 if abs(total_pnl) > 0 else 0

    # Year split
    tl = trade_log.copy()
    tl["entry_year"] = pd.to_datetime(tl["entry_date"]).dt.year
    y1_pnl = tl[tl["entry_year"] == 2024]["pnl"].sum()
    y2_pnl = tl[tl["entry_year"] == 2025]["pnl"].sum()

    # Price tier
    penny_pnl = trade_log[trade_log["entry_price"] < 2.0]["pnl"].sum()
    normal_pnl = trade_log[trade_log["entry_price"] >= 5.0]["pnl"].sum()

    return {
        "ex_top1_ret": round(ex_top1, 2),
        "ex_top5_ret": round(ex_top5, 2),
        "top1_pct": round(top1_pct, 1),
        "top5_pct": round(top5_pct, 1),
        "top1_ticker": ticker_pnl.index[0] if len(ticker_pnl) > 0 else "",
        "y1_ret": round(y1_pnl / initial_capital * 100, 2),
        "y2_ret": round(y2_pnl / initial_capital * 100, 2),
        "penny_pnl": round(penny_pnl),
        "normal_pnl": round(normal_pnl),
    }


# ── Sweep runner ──────────────────────────────────────────────────────────────

def run_single(
    signals: pd.DataFrame,
    ohlcv_cache: dict,
    weight_fn,
    max_positions: int,
    take_profit_pct: float = 0.35,
    stop_loss_pct: float = 1.00,
    horizon_bars: int = 30,
    grace_period_bars: int = 0,
    use_trailing_stop: bool = False,
    adv_cap_pct: float | None = ADV_CAP_PCT,
) -> tuple[dict, pd.DataFrame]:
    """Run one Path C configuration, return (metrics_dict, trade_log)."""
    trade_log, equity_curve = simulate_portfolio(
        signals, "30d",
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        ohlcv_dir=OHLCV_DIR,
        initial_capital=INITIAL_CAPITAL,
        max_positions=max_positions,
        commission_pct=COMMISSION_PCT,
        slippage_pct=SLIPPAGE_PCT,
        weight_fn=weight_fn,
        _ohlcv_cache=ohlcv_cache,
        tiered_slippage=TIERED_SLIPPAGE,
        adv_cap_pct=adv_cap_pct,
        horizon_bars=horizon_bars,
        grace_period_bars=grace_period_bars,
        use_trailing_stop=use_trailing_stop,
    )
    m = compute_metrics(trade_log, INITIAL_CAPITAL, equity_curve=equity_curve)
    r = robustness_metrics(trade_log, INITIAL_CAPITAL)
    m.update(r)
    return m, trade_log


def print_table(rows: list[dict], title: str, sort_by: str = "sharpe_ratio") -> None:
    """Print a formatted comparison table sorted by sort_by."""
    df = pd.DataFrame(rows).sort_values(sort_by, ascending=False)
    print(f"\n{'='*100}")
    print(title)
    print(f"{'='*100}")
    cols = ["label", "total_trades", "total_return_pct", "sharpe_ratio",
            "max_drawdown_pct", "win_rate_pct", "profit_factor",
            "ex_top1_ret", "ex_top5_ret", "top1_pct", "top1_ticker", "y1_ret", "y2_ret"]
    header = f"{'Label':<35} {'Tr':>4} {'Ret%':>8} {'Sharpe':>7} {'DD%':>7} {'WR%':>6} {'PF':>6} {'exT1%':>7} {'exT5%':>7} {'T1%':>5} {'T1':>6} {'Y1%':>7} {'Y2%':>7}"
    print(header)
    print("-" * len(header))
    for _, r in df.iterrows():
        print(f"{r.get('label',''):<35} {r.get('total_trades',0):>4} "
              f"{r.get('total_return_pct',0):>+7.1f}% {r.get('sharpe_ratio',0):>7.2f} "
              f"{r.get('max_drawdown_pct',0):>6.1f}% {r.get('win_rate_pct',0):>5.1f}% "
              f"{r.get('profit_factor',0):>6.2f} {r.get('ex_top1_ret',0):>+6.1f}% "
              f"{r.get('ex_top5_ret',0):>+6.1f}% {r.get('top1_pct',0):>4.0f}% "
              f"{str(r.get('top1_ticker',''))[:6]:>6} {r.get('y1_ret',0):>+6.1f}% "
              f"{r.get('y2_ret',0):>+6.1f}%")
    return df


# ── Phase runners ─────────────────────────────────────────────────────────────

def phase_1_sizing(signals, ohlcv_cache):
    """Phase 1: Position sizing × max_positions × sizing approach."""
    print("\n" + "#" * 100)
    print("PHASE 1: POSITION SIZING + CAPACITY")
    print("#" * 100)

    max_weights = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03]
    max_positions_grid = [50, 75, 100, 150, 200]
    sizing_fns = [
        ("eq", allocate_equal_weight),
        ("iv", allocate_inverse_volatility),
    ]

    rows = []
    total = len(max_weights) * len(max_positions_grid) * len(sizing_fns)
    i = 0
    for mw, mp, (sz_name, sz_fn) in product(max_weights, max_positions_grid, sizing_fns):
        i += 1
        if sz_name == "iv":
            wf = partial(sz_fn, vol_window=60, max_weight=mw)
        else:
            wf = partial(sz_fn, max_weight=mw)
        label = f"{sz_name}_w{mw:.3f}_mp{mp}"
        print(f"  [{i:>3}/{total}] {label}", end="\r", flush=True)
        m, _ = run_single(signals, ohlcv_cache, wf, mp)
        m["label"] = label
        m["sizing"] = sz_name
        m["max_weight"] = mw
        m["max_positions"] = mp
        rows.append(m)

    df = print_table(rows, "PHASE 1 RESULTS (sorted by Sharpe)")

    # Save full results
    df.to_csv(OUTPUT_DIR / "path_c_phase1.csv", index=False)
    print(f"\nSaved: output/path_c_phase1.csv ({len(df)} combos)")

    # Find best by Sharpe with ex_top5 > 0 (profitable even without outliers)
    robust = df[df["ex_top5_ret"] > 0]
    if len(robust) > 0:
        best = robust.sort_values("sharpe_ratio", ascending=False).iloc[0]
        print(f"\nBest robust (Sharpe, ex-top5 > 0): {best['label']}")
    else:
        best = df.sort_values("sharpe_ratio", ascending=False).iloc[0]
        print(f"\nBest (Sharpe, no robust filter): {best['label']}")
    print(f"  Return={best['total_return_pct']:+.1f}%, Sharpe={best['sharpe_ratio']:.2f}, "
          f"DD={best['max_drawdown_pct']:.1f}%, ex-top5={best['ex_top5_ret']:+.1f}%")

    return best["sizing"], best["max_weight"], int(best["max_positions"]), df


def phase_2_tp(signals, ohlcv_cache, sizing, max_weight, max_positions):
    """Phase 2: Take profit level sweep."""
    print("\n" + "#" * 100)
    print(f"PHASE 2: TAKE PROFIT LEVEL (sizing={sizing}, w={max_weight}, mp={max_positions})")
    print("#" * 100)

    tp_grid = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.75, 1.00, 5.00]

    if sizing == "iv":
        wf = partial(allocate_inverse_volatility, vol_window=60, max_weight=max_weight)
    else:
        wf = partial(allocate_equal_weight, max_weight=max_weight)

    rows = []
    for tp in tp_grid:
        label = f"TP={tp:.0%}" if tp <= 1.0 else "TP=none"
        print(f"  {label}", end="\r", flush=True)
        m, tl = run_single(signals, ohlcv_cache, wf, max_positions, take_profit_pct=tp)
        m["label"] = label
        m["tp"] = tp
        # TP exit count
        if len(tl) > 0:
            m["tp_exits"] = int((tl["exit_reason"] == "take_profit").sum())
            m["hor_exits"] = int((tl["exit_reason"] == "horizon_expiry").sum())
        else:
            m["tp_exits"] = 0
            m["hor_exits"] = 0
        rows.append(m)

    df = print_table(rows, "PHASE 2 RESULTS (sorted by Sharpe)")
    df.to_csv(OUTPUT_DIR / "path_c_phase2.csv", index=False)

    best = df.sort_values("sharpe_ratio", ascending=False).iloc[0]
    print(f"\nBest TP: {best['label']} → Sharpe={best['sharpe_ratio']:.2f}, "
          f"Return={best['total_return_pct']:+.1f}%, ex-top5={best['ex_top5_ret']:+.1f}%")
    return float(best["tp"]), df


def phase_3_horizon(signals, ohlcv_cache, sizing, max_weight, max_positions, tp):
    """Phase 3: Horizon bars sweep."""
    print("\n" + "#" * 100)
    print(f"PHASE 3: HORIZON BARS (TP={tp:.0%}, sizing={sizing}, w={max_weight}, mp={max_positions})")
    print("#" * 100)

    horizon_grid = [10, 15, 20, 25, 30, 40, 50]

    if sizing == "iv":
        wf = partial(allocate_inverse_volatility, vol_window=60, max_weight=max_weight)
    else:
        wf = partial(allocate_equal_weight, max_weight=max_weight)

    rows = []
    for hb in horizon_grid:
        label = f"horizon={hb}bars"
        print(f"  {label}", end="\r", flush=True)
        m, _ = run_single(signals, ohlcv_cache, wf, max_positions,
                          take_profit_pct=tp, horizon_bars=hb)
        m["label"] = label
        m["horizon_bars"] = hb
        rows.append(m)

    df = print_table(rows, "PHASE 3 RESULTS (sorted by Sharpe)")
    df.to_csv(OUTPUT_DIR / "path_c_phase3.csv", index=False)

    best = df.sort_values("sharpe_ratio", ascending=False).iloc[0]
    print(f"\nBest horizon: {best['label']} → Sharpe={best['sharpe_ratio']:.2f}, "
          f"Return={best['total_return_pct']:+.1f}%, ex-top5={best['ex_top5_ret']:+.1f}%")
    return int(best["horizon_bars"]), df


def phase_4_adv(signals, ohlcv_cache, sizing, max_weight, max_positions, tp, hb):
    """Phase 4: ADV cap sweep."""
    print("\n" + "#" * 100)
    print(f"PHASE 4: ADV CAP (TP={tp:.0%}, horizon={hb}, sizing={sizing})")
    print("#" * 100)

    adv_grid = [0.05, 0.10, 0.15, 0.20, None]

    if sizing == "iv":
        wf = partial(allocate_inverse_volatility, vol_window=60, max_weight=max_weight)
    else:
        wf = partial(allocate_equal_weight, max_weight=max_weight)

    rows = []
    for adv in adv_grid:
        label = f"ADV={adv:.0%}" if adv is not None else "ADV=none"
        print(f"  {label}", end="\r", flush=True)
        m, _ = run_single(signals, ohlcv_cache, wf, max_positions,
                          take_profit_pct=tp, horizon_bars=hb, adv_cap_pct=adv)
        m["label"] = label
        m["adv_cap"] = adv if adv is not None else 999
        rows.append(m)

    df = print_table(rows, "PHASE 4 RESULTS (sorted by Sharpe)")
    df.to_csv(OUTPUT_DIR / "path_c_phase4.csv", index=False)

    best = df.sort_values("sharpe_ratio", ascending=False).iloc[0]
    adv_val = None if best["adv_cap"] == 999 else best["adv_cap"]
    print(f"\nBest ADV: {best['label']} → Sharpe={best['sharpe_ratio']:.2f}, "
          f"Return={best['total_return_pct']:+.1f}%, ex-top5={best['ex_top5_ret']:+.1f}%")
    return adv_val, df


def phase_5_grace(signals, ohlcv_cache, sizing, max_weight, max_positions, tp, hb, adv):
    """Phase 5: Grace period sweep."""
    print("\n" + "#" * 100)
    print(f"PHASE 5: GRACE PERIOD (TP={tp:.0%}, horizon={hb}, ADV={'none' if adv is None else f'{adv:.0%}'})")
    print("#" * 100)

    grace_grid = [0, 2, 3, 5]

    if sizing == "iv":
        wf = partial(allocate_inverse_volatility, vol_window=60, max_weight=max_weight)
    else:
        wf = partial(allocate_equal_weight, max_weight=max_weight)

    rows = []
    for g in grace_grid:
        label = f"grace={g}bars"
        print(f"  {label}", end="\r", flush=True)
        m, _ = run_single(signals, ohlcv_cache, wf, max_positions,
                          take_profit_pct=tp, horizon_bars=hb, adv_cap_pct=adv,
                          grace_period_bars=g)
        m["label"] = label
        m["grace"] = g
        rows.append(m)

    df = print_table(rows, "PHASE 5 RESULTS (sorted by Sharpe)")
    df.to_csv(OUTPUT_DIR / "path_c_phase5.csv", index=False)

    best = df.sort_values("sharpe_ratio", ascending=False).iloc[0]
    print(f"\nBest grace: {best['label']} → Sharpe={best['sharpe_ratio']:.2f}, "
          f"Return={best['total_return_pct']:+.1f}%, ex-top5={best['ex_top5_ret']:+.1f}%")
    return int(best["grace"]), df


def phase_6_price_tier(signals, ohlcv_cache, sizing, max_weight, max_positions, tp, hb, adv, grace):
    """Phase 6: Price tier ANALYSIS (robustness test, not parameter selection)."""
    print("\n" + "#" * 100)
    print("PHASE 6: PRICE TIER ANALYSIS (robustness test — NOT for parameter selection)")
    print("#" * 100)

    if sizing == "iv":
        wf = partial(allocate_inverse_volatility, vol_window=60, max_weight=max_weight)
    else:
        wf = partial(allocate_equal_weight, max_weight=max_weight)

    # Run baseline first
    m_all, tl_all = run_single(signals, ohlcv_cache, wf, max_positions,
                                take_profit_pct=tp, horizon_bars=hb, adv_cap_pct=adv,
                                grace_period_bars=grace)
    m_all["label"] = "ALL (baseline)"

    # Filter signals by entry price tier (approximate using historical close)
    rows = [m_all]
    tiers = [
        ("exclude_penny (>=$2)", lambda s: s[s["ticker"].map(
            lambda t: _approx_price(ohlcv_cache, t) >= 2.0)]),
        ("exclude_sub5 (>=$5)", lambda s: s[s["ticker"].map(
            lambda t: _approx_price(ohlcv_cache, t) >= 5.0)]),
        ("only_penny (<$2)", lambda s: s[s["ticker"].map(
            lambda t: _approx_price(ohlcv_cache, t) < 2.0)]),
        ("mid_cap ($2-$10)", lambda s: s[s["ticker"].map(
            lambda t: 2.0 <= _approx_price(ohlcv_cache, t) < 10.0)]),
    ]

    for tier_name, filter_fn in tiers:
        filtered = filter_fn(signals)
        if len(filtered) == 0:
            continue
        label = tier_name
        print(f"  {label} ({len(filtered)} signals)", end="\r", flush=True)
        m, _ = run_single(filtered, ohlcv_cache, wf, max_positions,
                          take_profit_pct=tp, horizon_bars=hb, adv_cap_pct=adv,
                          grace_period_bars=grace)
        m["label"] = label
        rows.append(m)

    df = print_table(rows, "PHASE 6 RESULTS (price tier robustness)")
    df.to_csv(OUTPUT_DIR / "path_c_phase6.csv", index=False)
    return df


def phase_7_trailing(signals, ohlcv_cache, sizing, max_weight, max_positions, tp, hb, adv, grace):
    """Phase 7: Trailing stop variants."""
    print("\n" + "#" * 100)
    print("PHASE 7: TRAILING STOPS")
    print("#" * 100)

    if sizing == "iv":
        wf = partial(allocate_inverse_volatility, vol_window=60, max_weight=max_weight)
    else:
        wf = partial(allocate_equal_weight, max_weight=max_weight)

    rows = []
    # Baseline: no trailing, no SL
    m, _ = run_single(signals, ohlcv_cache, wf, max_positions,
                      take_profit_pct=tp, horizon_bars=hb, adv_cap_pct=adv,
                      grace_period_bars=grace)
    m["label"] = "no_trail_no_SL (baseline)"
    rows.append(m)

    # Trailing stop variants
    trail_grid = [0.10, 0.15, 0.20, 0.25, 0.30]
    for ts in trail_grid:
        label = f"trail_{ts:.0%}_TP{tp:.0%}"
        m, _ = run_single(signals, ohlcv_cache, wf, max_positions,
                          take_profit_pct=tp, stop_loss_pct=ts, horizon_bars=hb,
                          adv_cap_pct=adv, grace_period_bars=grace,
                          use_trailing_stop=True)
        m["label"] = label
        rows.append(m)

    # Also test with wider TPs + trailing
    for ts, tp2 in [(0.15, 0.50), (0.20, 0.75), (0.25, 5.0)]:
        label = f"trail_{ts:.0%}_TP{'none' if tp2 >= 5.0 else f'{tp2:.0%}'}"
        m, _ = run_single(signals, ohlcv_cache, wf, max_positions,
                          take_profit_pct=tp2, stop_loss_pct=ts, horizon_bars=hb,
                          adv_cap_pct=adv, grace_period_bars=grace,
                          use_trailing_stop=True)
        m["label"] = label
        rows.append(m)

    df = print_table(rows, "PHASE 7 RESULTS (sorted by Sharpe)")
    df.to_csv(OUTPUT_DIR / "path_c_phase7.csv", index=False)
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _approx_price(ohlcv_cache: dict, ticker: str) -> float:
    """Get approximate recent price for a ticker (median close)."""
    ohlcv = ohlcv_cache.get(ticker)
    if ohlcv is None or len(ohlcv) == 0:
        return 999.0  # treat as expensive → won't be filtered into "penny"
    return float(ohlcv["Close"].median())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore")
    print("=" * 100)
    print("PATH C EXHAUSTIVE SWEEP — with robustness validation at each phase")
    print("=" * 100)

    # Load data once
    print("\nLoading data...")
    announcements = load_announcements(ANNOUNCEMENTS_PATH)
    signals = build_all_signals(announcements, "30d")
    print(f"  {len(announcements):,} announcements → {len(signals)} signals (no model filter)")

    all_tickers = signals["ticker"].unique()
    print(f"  Loading OHLCV for {len(all_tickers)} tickers...")
    ohlcv_cache = _load_ohlcv_cache(all_tickers, OHLCV_DIR)
    print(f"  Loaded {len(ohlcv_cache)} tickers with price data.")

    # Phase 1: Sizing
    sizing, max_weight, max_positions, p1_df = phase_1_sizing(signals, ohlcv_cache)

    # Phase 2: TP
    tp, p2_df = phase_2_tp(signals, ohlcv_cache, sizing, max_weight, max_positions)

    # Phase 3: Horizon
    hb, p3_df = phase_3_horizon(signals, ohlcv_cache, sizing, max_weight, max_positions, tp)

    # Phase 4: ADV cap
    adv, p4_df = phase_4_adv(signals, ohlcv_cache, sizing, max_weight, max_positions, tp, hb)

    # Phase 5: Grace period
    grace, p5_df = phase_5_grace(signals, ohlcv_cache, sizing, max_weight, max_positions, tp, hb, adv)

    # Phase 6: Price tier analysis
    p6_df = phase_6_price_tier(signals, ohlcv_cache, sizing, max_weight, max_positions, tp, hb, adv, grace)

    # Phase 7: Trailing stops
    p7_df = phase_7_trailing(signals, ohlcv_cache, sizing, max_weight, max_positions, tp, hb, adv, grace)

    # Final summary
    print("\n" + "=" * 100)
    print("FINAL OPTIMIZED CONFIGURATION")
    print("=" * 100)
    print(f"  Sizing:         {sizing}")
    print(f"  Max weight:     {max_weight:.2%}")
    print(f"  Max positions:  {max_positions}")
    print(f"  Take profit:    {tp:.0%}" if tp <= 1.0 else f"  Take profit:    none")
    print(f"  Horizon bars:   {hb}")
    print(f"  ADV cap:        {'none' if adv is None else f'{adv:.0%}'}")
    print(f"  Grace period:   {grace} bars")
    print()

    # Run final config and save trade log
    if sizing == "iv":
        wf = partial(allocate_inverse_volatility, vol_window=60, max_weight=max_weight)
    else:
        wf = partial(allocate_equal_weight, max_weight=max_weight)
    m_final, tl_final = run_single(signals, ohlcv_cache, wf, max_positions,
                                    take_profit_pct=tp, horizon_bars=hb,
                                    adv_cap_pct=adv, grace_period_bars=grace)
    print(f"  Return:         {m_final['total_return_pct']:+.1f}%")
    print(f"  Sharpe:         {m_final['sharpe_ratio']:.2f}")
    print(f"  Max DD:         {m_final['max_drawdown_pct']:.1f}%")
    print(f"  Win rate:       {m_final['win_rate_pct']:.1f}%")
    print(f"  Trades:         {m_final['total_trades']}")
    print(f"  Ex-top1 return: {m_final['ex_top1_ret']:+.1f}%")
    print(f"  Ex-top5 return: {m_final['ex_top5_ret']:+.1f}%")
    print(f"  Y1 return:      {m_final['y1_ret']:+.1f}%")
    print(f"  Y2 return:      {m_final['y2_ret']:+.1f}%")

    if len(tl_final) > 0:
        tl_final.to_csv(OUTPUT_DIR / "trade_log_path_c_final.csv", index=False)
        print(f"\n  Trade log: output/trade_log_path_c_final.csv")


if __name__ == "__main__":
    main()
