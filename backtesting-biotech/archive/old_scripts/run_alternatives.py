"""
run_alternatives.py — Compare 6 portfolio construction strategies × 3 TP/SL combos.

Generates signals once, loads OHLCV once, then runs 18 simulations and prints
a comparison table sorted by Sharpe ratio.

Usage: python -m backtest.run_alternatives [--horizon 30d]
"""

import argparse
import warnings
from functools import partial

import pandas as pd

from backtest.config import (
    ANNOUNCEMENTS_PATH, OHLCV_DIR, BENCHMARK_PATH, OUTPUT_DIR,
    START_DATE, END_DATE, TRAIN_MONTHS, PREDICT_WEEKS, STEP_WEEKS,
    INITIAL_CAPITAL, COMMISSION_PCT, SLIPPAGE_PCT,
)
from backtest.data_loader import load_announcements, load_benchmark
from backtest.features import fit_ohe
from backtest.model import run_rolling_loop
from backtest.portfolio_simulator import simulate_portfolio, _load_ohlcv_cache
from backtest.portfolio_construction import (
    allocate_equal_weight,
    allocate_low_spy_correlation,
    allocate_inverse_volatility,
    allocate_risk_parity,
    allocate_minimum_variance,
    allocate_maximum_decorrelation,
)
from backtest.metrics import compute_metrics


# Top 3 TP/SL combos from the corrected sweep (best risk-adjusted)
TOP3_TP_SL = [
    (0.25, 0.10),
    (0.30, 0.10),
    (0.35, 0.10),
]

METRIC_COLS = [
    "strategy", "take_profit", "stop_loss",
    "total_return_pct", "annualized_return_pct", "sharpe_ratio",
    "max_drawdown_pct", "win_rate_pct", "profit_factor",
    "total_trades", "avg_return_per_trade", "avg_holding_days",
]


def run_alternatives(horizon: str = "30d") -> pd.DataFrame:
    """Run all 6 strategies × 3 TP/SL combos and return a comparison DataFrame.

    Generates signals once and loads OHLCV once to avoid redundant I/O.
    Returns a DataFrame with one row per (strategy, TP, SL) combination,
    sorted by Sharpe ratio descending.
    """
    # ── Generate signals once ────────────────────────────────────────────────
    print("Loading announcements...")
    announcements = load_announcements(ANNOUNCEMENTS_PATH)
    print(f"  {len(announcements):,} announcements loaded.")

    print("Fitting OHE...")
    ohe = fit_ohe(announcements)

    print(f"Running rolling windows ({horizon})...")
    signal_frames, _ = run_rolling_loop(
        announcements, ohe, horizon,
        start_date=START_DATE, end_date=END_DATE,
        train_months=TRAIN_MONTHS, predict_weeks=PREDICT_WEEKS,
        step_weeks=STEP_WEEKS,
    )
    if not signal_frames:
        print("No signals generated.")
        return pd.DataFrame()

    all_signals = pd.concat(signal_frames, ignore_index=True)
    print(f"  {len(all_signals)} signals across {len(signal_frames)} windows.\n")

    # ── Load OHLCV and SPY once ──────────────────────────────────────────────
    print("Loading OHLCV cache...")
    ohlcv_cache = _load_ohlcv_cache(all_signals["ticker"].unique(), OHLCV_DIR)

    print("Loading SPY benchmark...")
    try:
        spy_df = load_benchmark(BENCHMARK_PATH)
    except FileNotFoundError:
        warnings.warn("SPY benchmark not found — strategy 2 (low SPY corr) will use fallback.")
        spy_df = pd.DataFrame(columns=["Close"])

    # ── Define the 6 strategies ──────────────────────────────────────────────
    strategies = {
        "1_equal_weight":        partial(allocate_equal_weight,
                                         max_weight=0.10),
        "2_low_spy_corr":        partial(allocate_low_spy_correlation,
                                         spy_df=spy_df, top_k=15,
                                         corr_window=126, max_weight=0.08),
        "3_inverse_vol":         partial(allocate_inverse_volatility,
                                         vol_window=60, max_weight=0.10),
        "4_risk_parity":         partial(allocate_risk_parity,
                                         vol_window=60, max_weight=0.10),
        "5_min_variance":        partial(allocate_minimum_variance,
                                         vol_window=60, max_weight=0.10),
        "6_max_decorrelation":   partial(allocate_maximum_decorrelation,
                                         corr_window=60, max_weight=0.10,
                                         corr_threshold=0.60),
    }

    # ── Run 18 combinations ──────────────────────────────────────────────────
    results = []
    total = len(strategies) * len(TOP3_TP_SL)
    run_n = 0

    for strat_name, weight_fn in strategies.items():
        for tp, sl in TOP3_TP_SL:
            run_n += 1
            print(f"  [{run_n:2d}/{total}] {strat_name}  TP={tp:.0%}  SL={sl:.0%} ...", end=" ", flush=True)

            trade_log, equity_curve = simulate_portfolio(
                all_signals, horizon,
                take_profit_pct=tp,
                stop_loss_pct=sl,
                initial_capital=INITIAL_CAPITAL,
                commission_pct=COMMISSION_PCT,
                slippage_pct=SLIPPAGE_PCT,
                weight_fn=weight_fn,
                _ohlcv_cache=ohlcv_cache,
            )

            metrics = compute_metrics(trade_log, INITIAL_CAPITAL, equity_curve=equity_curve)
            print(f"return={metrics['total_return_pct']:+.1f}%  sharpe={metrics['sharpe_ratio']:.2f}")

            results.append({
                "strategy":             strat_name,
                "take_profit":          f"{tp:.0%}",
                "stop_loss":            f"{sl:.0%}",
                **metrics,
            })

    df = pd.DataFrame(results)
    df = df.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)
    return df


def main() -> None:
    """Parse CLI args, run alternatives, print and save results."""
    parser = argparse.ArgumentParser(description="Portfolio construction alternatives comparison")
    parser.add_argument("--horizon", choices=["5d", "30d"], default="30d")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Portfolio Construction Alternatives — {args.horizon} horizon")
    print(f"Initial capital: ${INITIAL_CAPITAL:,.0f}  |  {len(TOP3_TP_SL)} TP/SL combos  |  6 strategies")
    print(f"{'='*70}\n")

    results = run_alternatives(args.horizon)

    if len(results) == 0:
        print("No results.")
        return

    print(f"\n{'='*70}")
    print("RESULTS — sorted by Sharpe ratio")
    print(f"{'='*70}")

    display_cols = [
        "strategy", "take_profit", "stop_loss",
        "total_return_pct", "annualized_return_pct", "sharpe_ratio",
        "max_drawdown_pct", "win_rate_pct", "profit_factor", "total_trades",
    ]
    print(results[display_cols].to_string(index=True))

    # Best per strategy
    print(f"\n{'='*70}")
    print("BEST TP/SL PER STRATEGY (by Sharpe)")
    print(f"{'='*70}")
    best = results.loc[results.groupby("strategy")["sharpe_ratio"].idxmax()]
    print(best[display_cols].to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / f"alternatives_{args.horizon}.csv", index=False)
    print(f"\nFull results saved: output/alternatives_{args.horizon}.csv")


if __name__ == "__main__":
    main()
