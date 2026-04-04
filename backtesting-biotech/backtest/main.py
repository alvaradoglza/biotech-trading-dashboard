"""
main.py — CLI entry point: orchestrate the full rolling-window backtest pipeline.
Run: python -m backtest.main [--horizon 5d|30d|both] [--sweep]
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

from backtest.config import (
    ANNOUNCEMENTS_PATH, OHLCV_DIR, BENCHMARK_PATH, OUTPUT_DIR,
    START_DATE, END_DATE, TRAIN_MONTHS, PREDICT_WEEKS, STEP_WEEKS,
    INITIAL_CAPITAL, MAX_OPEN_POSITIONS, MAX_WEIGHT, VOL_WINDOW,
    COMMISSION_PCT, SLIPPAGE_PCT,
    TP_SL_CONFIG, TP_GRID, SL_GRID,
    TIERED_SLIPPAGE, ADV_CAP_PCT,
)
from backtest.data_loader import load_announcements, load_benchmark
from backtest.features import fit_ohe
from backtest.model import run_rolling_loop
from backtest.portfolio_simulator import simulate_portfolio, _load_ohlcv_cache
from backtest.portfolio_construction import allocate_inverse_volatility
from backtest.metrics import compute_metrics, compute_benchmark_metrics, compare_horizons
from backtest.sweep import run_sweep, best_params
from functools import partial


def run_backtest(
    horizon: str,
    take_profit_pct: float | None = None,
    stop_loss_pct: float | None = None,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Run the complete backtest pipeline for one horizon.

    Loads data, fits OHE, runs rolling windows, simulates trades, and computes metrics.
    Returns (trade_log, metrics_dict, equity_curve).
    """
    print(f"\n{'='*60}")
    print(f"Running {horizon} backtest: {start_date} → {end_date}")
    print(f"{'='*60}")

    # Load data
    print("Loading announcements...")
    announcements = load_announcements(ANNOUNCEMENTS_PATH)
    print(f"  {len(announcements):,} announcements loaded (after EDGAR filter).")

    # Fit global OHE once on full dataset
    print("Fitting global OneHotEncoder...")
    ohe = fit_ohe(announcements)
    n_ohe_dims = ohe.transform(announcements[["source", "event_type"]]).shape[1]
    print(f"  OHE dimensions: {n_ohe_dims}")

    # Rolling loop — generate signals
    print(f"\nRunning rolling windows ({horizon})...")
    signal_frames, active_windows = run_rolling_loop(
        announcements, ohe, horizon,
        start_date=start_date,
        end_date=end_date,
        train_months=TRAIN_MONTHS,
        predict_weeks=PREDICT_WEEKS,
        step_weeks=STEP_WEEKS,
    )
    n_with_signals = sum(1 for f in signal_frames if len(f) > 0)
    print(f"  {len(active_windows)} windows ran, {n_with_signals} produced signals.")

    if not signal_frames:
        print("  No signals generated. Returning empty results.")
        return pd.DataFrame(), compute_metrics(pd.DataFrame())

    all_signals = pd.concat(signal_frames, ignore_index=True)
    print(f"  Total signals across all windows: {len(all_signals)}")

    # Resolve TP/SL/horizon_bars from config
    tp = take_profit_pct if take_profit_pct is not None else TP_SL_CONFIG[horizon]["take_profit"]
    sl = stop_loss_pct   if stop_loss_pct   is not None else TP_SL_CONFIG[horizon]["stop_loss"]
    hb = TP_SL_CONFIG[horizon].get("horizon_bars", 50)

    weight_fn = partial(allocate_inverse_volatility, vol_window=VOL_WINDOW, max_weight=MAX_WEIGHT)
    sl_label = "none" if sl >= 1.0 else f"{sl:.0%}"
    print(f"\nSimulating portfolio (TP={tp:.0%}, SL={sl_label}, horizon={hb} bars, inverse-vol mw={MAX_WEIGHT:.0%}, tiered_slip={TIERED_SLIPPAGE}, ADV_cap={ADV_CAP_PCT})...")
    trade_log, equity_curve = simulate_portfolio(
        all_signals, horizon,
        take_profit_pct=tp,
        stop_loss_pct=sl,
        ohlcv_dir=OHLCV_DIR,
        initial_capital=INITIAL_CAPITAL,
        commission_pct=COMMISSION_PCT,
        slippage_pct=SLIPPAGE_PCT,
        weight_fn=weight_fn,
        tiered_slippage=TIERED_SLIPPAGE,
        adv_cap_pct=ADV_CAP_PCT,
        horizon_bars=hb,
    )

    if len(trade_log) == 0:
        print("No trades executed.")
        return pd.DataFrame(), compute_metrics(pd.DataFrame()), pd.DataFrame()

    print(f"Total trades: {len(trade_log)}")

    # Report in-period vs out-of-period trades
    trade_log["exit_date"] = pd.to_datetime(trade_log["exit_date"])
    in_period = trade_log[trade_log["exit_date"] <= pd.Timestamp(end_date)]
    out_period = trade_log[trade_log["exit_date"] > pd.Timestamp(end_date)]
    if len(out_period) > 0:
        print(f"  {len(in_period)} trades exit within {start_date}–{end_date}")
        print(f"  {len(out_period)} trades exit after {end_date} (included in total, flagged here)")
        print(f"  In-period PnL: ${in_period['pnl'].sum():+,.2f} ({in_period['pnl'].sum()/INITIAL_CAPITAL*100:+.2f}%)")
        print(f"  Post-period PnL: ${out_period['pnl'].sum():+,.2f} ({out_period['pnl'].sum()/INITIAL_CAPITAL*100:+.2f}%)")

    # Compute metrics
    metrics = compute_metrics(trade_log, INITIAL_CAPITAL, equity_curve=equity_curve)
    print(f"\n{horizon} Strategy Metrics (all trades):")
    for k, v in metrics.items():
        print(f"  {k:30s}: {v}")

    # Also compute in-period-only metrics
    if len(out_period) > 0 and len(in_period) > 0:
        # Build in-period equity curve
        in_period_eq = equity_curve[equity_curve.index <= pd.Timestamp(end_date)] if equity_curve is not None else None
        metrics_in_period = compute_metrics(in_period, INITIAL_CAPITAL, equity_curve=in_period_eq)
        print(f"\n{horizon} Strategy Metrics (in-period only, exits through {end_date}):")
        for k, v in metrics_in_period.items():
            print(f"  {k:30s}: {v}")

    return trade_log, metrics, equity_curve


def save_outputs(trade_log: pd.DataFrame, metrics: dict, horizon: str,
                  equity_curve: pd.DataFrame | None = None) -> None:
    """Save trade log, metrics, and equity curve to the output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if len(trade_log) > 0:
        trade_log.to_csv(OUTPUT_DIR / f"trade_log_{horizon}.csv", index=False)
        print(f"  Trade log saved: output/trade_log_{horizon}.csv")
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(OUTPUT_DIR / f"metrics_{horizon}.csv", index=False)
    print(f"  Metrics saved: output/metrics_{horizon}.csv")
    if equity_curve is not None and len(equity_curve) > 0:
        equity_curve.to_csv(OUTPUT_DIR / f"equity_curve_{horizon}.csv")
        print(f"  Equity curve saved: output/equity_curve_{horizon}.csv")


def main() -> None:
    """Parse CLI arguments and run the backtest pipeline."""
    parser = argparse.ArgumentParser(description="Clinical Trial Announcement Backtester")
    parser.add_argument("--horizon", choices=["5d", "30d", "both"], default="both",
                        help="Which horizon to run (default: both)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run TP/SL grid sweep instead of single backtest")
    parser.add_argument("--start", default=START_DATE, help="Backtest start date")
    parser.add_argument("--end",   default=END_DATE,   help="Backtest end date")
    args = parser.parse_args()

    horizons = ["5d", "30d"] if args.horizon == "both" else [args.horizon]
    results = {}

    for horizon in horizons:
        if args.sweep:
            print(f"\nRunning TP/SL sweep for {horizon}...")
            print("(Generating signals once, then sweeping TP/SL only...)")

            # Generate signals once — reused for every TP/SL combination
            announcements = load_announcements(ANNOUNCEMENTS_PATH)
            ohe = fit_ohe(announcements)
            signal_frames, _ = run_rolling_loop(
                announcements, ohe, horizon,
                start_date=args.start, end_date=args.end,
                train_months=TRAIN_MONTHS, predict_weeks=PREDICT_WEEKS,
                step_weeks=STEP_WEEKS,
            )
            if not signal_frames:
                print("  No signals generated, skipping sweep.")
                results[horizon] = compute_metrics(pd.DataFrame())
                continue

            sweep_signals = pd.concat(signal_frames, ignore_index=True)
            ohlcv_cache = _load_ohlcv_cache(sweep_signals["ticker"].unique(), OHLCV_DIR)
            sweep_weight_fn = partial(allocate_inverse_volatility, vol_window=VOL_WINDOW, max_weight=MAX_WEIGHT)

            def _run_for_params(tp: float, sl: float) -> dict:
                tl, eq = simulate_portfolio(
                    sweep_signals, horizon, tp, sl,
                    ohlcv_dir=OHLCV_DIR,
                    initial_capital=INITIAL_CAPITAL,
                    commission_pct=COMMISSION_PCT,
                    slippage_pct=SLIPPAGE_PCT,
                    weight_fn=sweep_weight_fn,
                    _ohlcv_cache=ohlcv_cache,
                    tiered_slippage=TIERED_SLIPPAGE,
                    adv_cap_pct=ADV_CAP_PCT,
                )
                return compute_metrics(tl, INITIAL_CAPITAL, equity_curve=eq)

            sweep_df = run_sweep(_run_for_params, TP_GRID, SL_GRID)
            sweep_df.to_csv(OUTPUT_DIR / f"sweep_{horizon}.csv", index=False)
            print(f"\nSweep results saved: output/sweep_{horizon}.csv")
            print(f"Top 5 TP/SL combinations:\n{sweep_df.head()}")
            best = best_params(sweep_df)
            print(f"\nBest params: {best}")
            results[horizon] = compute_metrics(pd.DataFrame())  # placeholder
        else:
            trade_log, metrics, equity_curve = run_backtest(horizon, start_date=args.start, end_date=args.end)
            save_outputs(trade_log, metrics, horizon, equity_curve)
            results[horizon] = metrics

    # Benchmark comparison
    if not args.sweep:
        try:
            benchmark_df = load_benchmark(BENCHMARK_PATH)
            bench_metrics = compute_benchmark_metrics(benchmark_df, args.start, args.end)
        except FileNotFoundError:
            print("\nNo benchmark file found — skipping benchmark comparison.")
            bench_metrics = None

        # Always regenerate comparison.csv to prevent stale data
        if len(results) == 2:
            comparison = compare_horizons(results["5d"], results["30d"], bench_metrics)
        elif len(results) == 1 and bench_metrics is not None:
            horizon_key = list(results.keys())[0]
            data = {f"{horizon_key}_strategy": results[horizon_key], "benchmark": bench_metrics}
            comparison = pd.DataFrame(data).T
        else:
            comparison = None

        if comparison is not None:
            print(f"\n{'='*60}")
            print("Strategy vs Benchmark:")
            print(comparison.to_string())
            comparison.to_csv(OUTPUT_DIR / "comparison.csv")
            print(f"  Comparison saved: output/comparison.csv")


if __name__ == "__main__":
    main()
