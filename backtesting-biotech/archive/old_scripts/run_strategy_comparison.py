"""
run_strategy_comparison.py — Compare exit strategies for the catalyst model.

Tests different stop-loss approaches (none, fixed, ATR-adaptive, trailing),
grace periods, and take-profit levels. All with inverse-volatility sizing.

Usage: python -m backtest.run_strategy_comparison [--horizon 30d]
"""

import argparse
import warnings
from functools import partial

import pandas as pd

from backtest.config import (
    ANNOUNCEMENTS_PATH, OHLCV_DIR, OUTPUT_DIR,
    START_DATE, END_DATE, TRAIN_MONTHS, PREDICT_WEEKS, STEP_WEEKS,
    INITIAL_CAPITAL, COMMISSION_PCT, SLIPPAGE_PCT,
    TIERED_SLIPPAGE, ADV_CAP_PCT,
)
from backtest.data_loader import load_announcements
from backtest.features import fit_ohe
from backtest.model import run_rolling_loop
from backtest.portfolio_simulator import simulate_portfolio, _load_ohlcv_cache
from backtest.portfolio_construction import allocate_inverse_volatility
from backtest.metrics import compute_metrics


# ── Strategy definitions ─────────────────────────────────────────────────────
# Each dict: name + kwargs passed directly to simulate_portfolio

STRATEGIES = [
    # A: Baseline (current approach)
    {"name": "A_baseline_TP25_SL5",
     "kwargs": {"take_profit_pct": 0.25, "stop_loss_pct": 0.05}},

    # B: No stop loss — test whether removing stops captures leaked alpha
    {"name": "B1_horizon_only",
     "kwargs": {"take_profit_pct": 10.0, "stop_loss_pct": 1.0}},
    {"name": "B2_TP25_no_SL",
     "kwargs": {"take_profit_pct": 0.25, "stop_loss_pct": 1.0}},
    {"name": "B3_TP35_no_SL",
     "kwargs": {"take_profit_pct": 0.35, "stop_loss_pct": 1.0}},
    {"name": "B4_TP50_no_SL",
     "kwargs": {"take_profit_pct": 0.50, "stop_loss_pct": 1.0}},

    # C: Grace period — protect first N days from stops
    {"name": "C1_grace3_SL10_TP25",
     "kwargs": {"take_profit_pct": 0.25, "stop_loss_pct": 0.10, "grace_period_bars": 3}},
    {"name": "C2_grace3_SL10_TP35",
     "kwargs": {"take_profit_pct": 0.35, "stop_loss_pct": 0.10, "grace_period_bars": 3}},
    {"name": "C3_grace3_SL15_TP35",
     "kwargs": {"take_profit_pct": 0.35, "stop_loss_pct": 0.15, "grace_period_bars": 3}},
    {"name": "C4_grace5_SL10_TP35",
     "kwargs": {"take_profit_pct": 0.35, "stop_loss_pct": 0.10, "grace_period_bars": 5}},
    {"name": "C5_grace5_SL15_TP50",
     "kwargs": {"take_profit_pct": 0.50, "stop_loss_pct": 0.15, "grace_period_bars": 5}},
    {"name": "C6_grace3_SL10_TP50",
     "kwargs": {"take_profit_pct": 0.50, "stop_loss_pct": 0.10, "grace_period_bars": 3}},

    # D: Trailing stops — let winners run, protect gains
    {"name": "D1_trail15_horizon",
     "kwargs": {"take_profit_pct": 10.0, "stop_loss_pct": 0.15, "use_trailing_stop": True}},
    {"name": "D2_trail20_horizon",
     "kwargs": {"take_profit_pct": 10.0, "stop_loss_pct": 0.20, "use_trailing_stop": True}},
    {"name": "D3_trail25_horizon",
     "kwargs": {"take_profit_pct": 10.0, "stop_loss_pct": 0.25, "use_trailing_stop": True}},
    {"name": "D4_trail10_TP50",
     "kwargs": {"take_profit_pct": 0.50, "stop_loss_pct": 0.10, "use_trailing_stop": True}},
    {"name": "D5_trail15_TP50",
     "kwargs": {"take_profit_pct": 0.50, "stop_loss_pct": 0.15, "use_trailing_stop": True}},

    # E: ATR-based adaptive stops — wider for volatile names
    {"name": "E1_ATR2x_horizon",
     "kwargs": {"take_profit_pct": 10.0, "use_atr_stops": True, "atr_stop_multiple": 2.0}},
    {"name": "E2_ATR2.5x_horizon",
     "kwargs": {"take_profit_pct": 10.0, "use_atr_stops": True, "atr_stop_multiple": 2.5}},
    {"name": "E3_ATR3x_horizon",
     "kwargs": {"take_profit_pct": 10.0, "use_atr_stops": True, "atr_stop_multiple": 3.0}},
    {"name": "E4_ATR2x_TP35",
     "kwargs": {"take_profit_pct": 0.35, "use_atr_stops": True, "atr_stop_multiple": 2.0}},
    {"name": "E5_ATR2.5x_TP50",
     "kwargs": {"take_profit_pct": 0.50, "use_atr_stops": True, "atr_stop_multiple": 2.5}},

    # F: Wider fixed stops (no grace, no trail)
    {"name": "F1_SL15_TP35",
     "kwargs": {"take_profit_pct": 0.35, "stop_loss_pct": 0.15}},
    {"name": "F2_SL20_TP50",
     "kwargs": {"take_profit_pct": 0.50, "stop_loss_pct": 0.20}},
    {"name": "F3_SL25_TP50",
     "kwargs": {"take_profit_pct": 0.50, "stop_loss_pct": 0.25}},
    {"name": "F4_SL15_TP50",
     "kwargs": {"take_profit_pct": 0.50, "stop_loss_pct": 0.15}},

    # G: Combinations — grace + trailing, grace + ATR
    {"name": "G1_grace3_trail15_TP50",
     "kwargs": {"take_profit_pct": 0.50, "stop_loss_pct": 0.15,
                "grace_period_bars": 3, "use_trailing_stop": True}},
    {"name": "G2_grace3_trail20_horizon",
     "kwargs": {"take_profit_pct": 10.0, "stop_loss_pct": 0.20,
                "grace_period_bars": 3, "use_trailing_stop": True}},
    {"name": "G3_grace5_ATR2.5x_horizon",
     "kwargs": {"take_profit_pct": 10.0, "grace_period_bars": 5,
                "use_atr_stops": True, "atr_stop_multiple": 2.5}},
    {"name": "G4_grace3_ATR2x_TP50",
     "kwargs": {"take_profit_pct": 0.50, "grace_period_bars": 3,
                "use_atr_stops": True, "atr_stop_multiple": 2.0}},
]


def run_comparison(horizon: str = "30d") -> pd.DataFrame:
    """Run all exit strategies and return a comparison DataFrame."""
    # ── Generate signals once ─────────────────────────────────────────────
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

    # ── Load OHLCV once ───────────────────────────────────────────────────
    print("Loading OHLCV cache...")
    ohlcv_cache = _load_ohlcv_cache(all_signals["ticker"].unique(), OHLCV_DIR)

    weight_fn = partial(allocate_inverse_volatility, vol_window=60, max_weight=0.10)

    # ── Run all strategies ────────────────────────────────────────────────
    results = []
    total = len(STRATEGIES)

    for i, strat in enumerate(STRATEGIES, 1):
        name = strat["name"]
        kw = strat["kwargs"]
        print(f"  [{i:2d}/{total}] {name} ...", end=" ", flush=True)

        trade_log, equity_curve = simulate_portfolio(
            all_signals, horizon,
            initial_capital=INITIAL_CAPITAL,
            commission_pct=COMMISSION_PCT,
            slippage_pct=SLIPPAGE_PCT,
            weight_fn=weight_fn,
            _ohlcv_cache=ohlcv_cache,
            tiered_slippage=TIERED_SLIPPAGE,
            adv_cap_pct=ADV_CAP_PCT,
            **kw,
        )

        metrics = compute_metrics(trade_log, INITIAL_CAPITAL, equity_curve=equity_curve)

        # Derive Calmar ratio
        ann_ret = metrics.get("annualized_return_pct", 0)
        max_dd = metrics.get("max_drawdown_pct", 0)
        calmar = ann_ret / max_dd if max_dd > 0 else 0.0

        print(f"return={metrics['total_return_pct']:+.1f}%  "
              f"sharpe={metrics['sharpe_ratio']:.2f}  "
              f"DD={metrics['max_drawdown_pct']:.1f}%  "
              f"calmar={calmar:.2f}")

        results.append({
            "strategy":             name,
            "total_return_pct":     metrics["total_return_pct"],
            "annualized_return_pct": metrics["annualized_return_pct"],
            "sharpe_ratio":         metrics["sharpe_ratio"],
            "max_drawdown_pct":     metrics["max_drawdown_pct"],
            "calmar_ratio":         round(calmar, 3),
            "win_rate_pct":         metrics["win_rate_pct"],
            "profit_factor":        metrics["profit_factor"],
            "total_trades":         metrics["total_trades"],
            "avg_return_per_trade": metrics["avg_return_per_trade"],
            "avg_holding_days":     metrics["avg_holding_days"],
        })

    df = pd.DataFrame(results)
    return df


def main() -> None:
    """Parse CLI args, run comparison, print and save results."""
    parser = argparse.ArgumentParser(description="Exit strategy comparison")
    parser.add_argument("--horizon", choices=["5d", "30d"], default="30d")
    parser.add_argument("--sort", default="total_return_pct",
                        help="Column to sort by (default: total_return_pct)")
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"Exit Strategy Comparison — {args.horizon} horizon, inverse-vol sizing, ${INITIAL_CAPITAL:,.0f}")
    print(f"{'='*80}\n")

    results = run_comparison(args.horizon)

    if len(results) == 0:
        print("No results.")
        return

    # Sort by chosen metric
    sort_col = args.sort
    if sort_col in results.columns:
        results = results.sort_values(sort_col, ascending=False).reset_index(drop=True)

    print(f"\n{'='*80}")
    print(f"RESULTS — sorted by {sort_col}")
    print(f"{'='*80}")

    display_cols = [
        "strategy", "total_return_pct", "annualized_return_pct",
        "sharpe_ratio", "max_drawdown_pct", "calmar_ratio",
        "win_rate_pct", "profit_factor", "total_trades", "avg_holding_days",
    ]
    print(results[display_cols].to_string(index=True))

    # Top 5 by different metrics
    for metric in ["total_return_pct", "sharpe_ratio", "calmar_ratio", "profit_factor"]:
        top = results.nlargest(5, metric)
        print(f"\n--- Top 5 by {metric} ---")
        print(top[["strategy", metric, "total_return_pct", "sharpe_ratio", "max_drawdown_pct"]].to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / f"strategy_comparison_{args.horizon}.csv", index=False)
    print(f"\nFull results saved: output/strategy_comparison_{args.horizon}.csv")


if __name__ == "__main__":
    main()
