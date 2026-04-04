"""
run_path_c.py — Test "buy every catalyst announcement" with no model filtering.

Path C hypothesis: biotech catalyst announcements are inherently asymmetric.
A strategy that enters EVERY announcement with small position sizes and
relies on the tail (1-2 extreme outliers per year) may outperform a model
that tries (and fails) to select winners.

Variants tested:
  C1: All announcements, 1% equal weight, max_pos=100
  C2: All announcements, 0.5% equal weight, max_pos=200
  C3: All announcements, inverse-vol 1% cap, max_pos=100
  C4: Model-filtered only (current approach), 10% cap, max_pos=20  [baseline]

Usage: python -m backtest.run_path_c
"""

import warnings
from functools import partial

import pandas as pd
import numpy as np

from backtest.config import (
    ANNOUNCEMENTS_PATH, OHLCV_DIR, OUTPUT_DIR,
    START_DATE, END_DATE, TRAIN_MONTHS, PREDICT_WEEKS, STEP_WEEKS,
    INITIAL_CAPITAL, COMMISSION_PCT, SLIPPAGE_PCT,
    TIERED_SLIPPAGE, ADV_CAP_PCT,
)
from backtest.data_loader import load_announcements
from backtest.features import fit_ohe
from backtest.model import generate_windows, run_rolling_loop
from backtest.portfolio_simulator import simulate_portfolio, _load_ohlcv_cache
from backtest.portfolio_construction import allocate_inverse_volatility, allocate_equal_weight
from backtest.metrics import compute_metrics


def _build_all_signals(announcements: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Build signals from ALL announcements in each prediction window — no model filtering.

    Every announcement becomes a 'buy' signal with a dummy decision_score=1.0.
    This tests whether the universe itself has alpha, independent of the model.
    """
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
        pred_df["decision_score"] = 1.0  # uniform — no model preference
        pred_df["window_id"] = window.window_id
        pred_df["horizon"] = horizon

        # Deduplicate: keep first announcement per ticker per window
        pred_df = pred_df.sort_values("published_at", ascending=True)
        pred_df = pred_df.drop_duplicates(subset=["ticker"], keep="first")

        all_signals.append(pred_df)

    if not all_signals:
        return pd.DataFrame()
    return pd.concat(all_signals, ignore_index=True)


def run_variant(
    name: str,
    signals: pd.DataFrame,
    ohlcv_cache: dict,
    weight_fn,
    max_positions: int,
) -> dict:
    """Run one Path C variant and return metrics + trade log."""
    print(f"\n  [{name}] {len(signals)} signals, max_pos={max_positions} ...", end=" ", flush=True)

    trade_log, equity_curve = simulate_portfolio(
        signals, "30d",
        take_profit_pct=0.35,
        stop_loss_pct=1.00,       # no stop loss
        ohlcv_dir=OHLCV_DIR,
        initial_capital=INITIAL_CAPITAL,
        max_positions=max_positions,
        commission_pct=COMMISSION_PCT,
        slippage_pct=SLIPPAGE_PCT,
        weight_fn=weight_fn,
        _ohlcv_cache=ohlcv_cache,
        tiered_slippage=TIERED_SLIPPAGE,
        adv_cap_pct=ADV_CAP_PCT,
    )

    metrics = compute_metrics(trade_log, INITIAL_CAPITAL, equity_curve=equity_curve)
    print(f"trades={metrics['total_trades']}, return={metrics['total_return_pct']:+.1f}%, "
          f"sharpe={metrics['sharpe_ratio']:.2f}, DD={metrics['max_drawdown_pct']:.1f}%")
    return {"name": name, "metrics": metrics, "trade_log": trade_log, "equity_curve": equity_curve}


def main():
    warnings.filterwarnings("ignore")
    print("=" * 70)
    print("PATH C TEST: Buy every catalyst announcement (no model)")
    print("=" * 70)

    # Load data
    print("\nLoading announcements...")
    announcements = load_announcements(ANNOUNCEMENTS_PATH)
    print(f"  {len(announcements):,} announcements loaded.")

    # Build ALL signals (no model)
    all_signals = _build_all_signals(announcements, "30d")
    print(f"  All-announcement signals: {len(all_signals)}")

    # Build model-filtered signals (current approach)
    print("\nGenerating model-filtered signals for baseline...")
    ohe = fit_ohe(announcements)
    model_frames, _ = run_rolling_loop(announcements, ohe, "30d")
    model_signals = pd.concat(model_frames, ignore_index=True) if model_frames else pd.DataFrame()
    print(f"  Model-filtered signals: {len(model_signals)}")

    # Load OHLCV
    all_tickers = set(all_signals["ticker"].unique())
    if len(model_signals) > 0:
        all_tickers.update(model_signals["ticker"].unique())
    print(f"\nLoading OHLCV for {len(all_tickers)} tickers...")
    ohlcv_cache = _load_ohlcv_cache(list(all_tickers), OHLCV_DIR)

    # Define variants
    results = []

    # C1: All announcements, 1% equal weight
    wf_eq_1pct = partial(allocate_equal_weight, max_weight=0.01)
    results.append(run_variant("C1_all_eq_1pct", all_signals, ohlcv_cache, wf_eq_1pct, 100))

    # C2: All announcements, 0.5% equal weight
    wf_eq_half = partial(allocate_equal_weight, max_weight=0.005)
    results.append(run_variant("C2_all_eq_0.5pct", all_signals, ohlcv_cache, wf_eq_half, 200))

    # C3: All announcements, inverse-vol 1% cap
    wf_iv_1pct = partial(allocate_inverse_volatility, vol_window=60, max_weight=0.01)
    results.append(run_variant("C3_all_invvol_1pct", all_signals, ohlcv_cache, wf_iv_1pct, 100))

    # C4: All announcements, inverse-vol 2% cap
    wf_iv_2pct = partial(allocate_inverse_volatility, vol_window=60, max_weight=0.02)
    results.append(run_variant("C4_all_invvol_2pct", all_signals, ohlcv_cache, wf_iv_2pct, 50))

    # C5: Model-filtered, 10% cap (current production approach)
    if len(model_signals) > 0:
        wf_iv_10pct = partial(allocate_inverse_volatility, vol_window=60, max_weight=0.10)
        results.append(run_variant("C5_model_invvol_10pct", model_signals, ohlcv_cache, wf_iv_10pct, 20))

    # Comparison table
    print("\n" + "=" * 70)
    print("PATH C COMPARISON")
    print("=" * 70)
    header = f"{'Variant':<28} {'Trades':>7} {'Return':>10} {'Ann.Ret':>10} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'PF':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        m = r["metrics"]
        print(f"{r['name']:<28} {m['total_trades']:>7} {m['total_return_pct']:>+9.1f}% "
              f"{m['annualized_return_pct']:>+9.1f}% {m['sharpe_ratio']:>8.2f} "
              f"{m['max_drawdown_pct']:>7.1f}% {m['win_rate_pct']:>7.1f}% {m['profit_factor']:>6.2f}")
    print(f"{'SPY (buy & hold)':<28} {'1':>7} {'  +44.3':>9}% {'  +20.1':>9}% {'1.21':>8} {'19.0':>7}% {'—':>7} {'—':>6}")

    # Concentration analysis for each variant
    print("\n" + "=" * 70)
    print("CONCENTRATION ANALYSIS")
    print("=" * 70)
    for r in results:
        tl = r["trade_log"]
        if len(tl) == 0:
            continue
        total_pnl = tl["pnl"].sum()
        if abs(total_pnl) < 1:
            continue
        top_ticker = tl.groupby("ticker")["pnl"].sum().sort_values(ascending=False)
        top1_pnl = top_ticker.iloc[0]
        top1_name = top_ticker.index[0]
        top5_pnl = top_ticker.head(5).sum()

        penny = tl[tl["entry_price"] < 2.0]["pnl"].sum()
        normal = tl[tl["entry_price"] >= 5.0]["pnl"].sum()

        print(f"\n  {r['name']}:")
        print(f"    Top ticker: {top1_name} = ${top1_pnl:,.0f} ({top1_pnl/total_pnl*100:.0f}% of PnL)")
        print(f"    Top 5 tickers: ${top5_pnl:,.0f} ({top5_pnl/total_pnl*100:.0f}% of PnL)")
        print(f"    Penny (<$2): ${penny:,.0f} | Normal (>=$5): ${normal:,.0f}")

    # Save best variant's trade log
    best = max(results, key=lambda r: r["metrics"]["total_return_pct"])
    if len(best["trade_log"]) > 0:
        best["trade_log"].to_csv(OUTPUT_DIR / "trade_log_path_c_best.csv", index=False)
        print(f"\nBest variant trade log saved: output/trade_log_path_c_best.csv")

    # Save comparison
    comp_rows = []
    for r in results:
        row = {"variant": r["name"]}
        row.update(r["metrics"])
        comp_rows.append(row)
    pd.DataFrame(comp_rows).to_csv(OUTPUT_DIR / "path_c_comparison.csv", index=False)
    print(f"Comparison saved: output/path_c_comparison.csv")


if __name__ == "__main__":
    main()
