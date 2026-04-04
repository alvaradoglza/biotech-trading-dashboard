"""
run_standout_backtest.py — Full production backtest of the audited standout strategy.

Strategy: t9_200d4_all_thp85_WC_mw7_tp30_h50
  - GBM(n_estimators=200, max_depth=4, lr=0.05)
  - 9-month training window, 4-week predict/step
  - All event types, P85 threshold (7.2225%), drop sf_word_count
  - Inverse-vol sizing (vol_window=60, max_weight=0.07)
  - TP=30%, no stop loss, 50-bar horizon
  - Tiered slippage, 5% ADV cap
  - Period: 2024-01-01 to 2025-12-31, $1M initial capital

Outputs:
  - output/standout_trade_log.csv
  - output/standout_metrics.csv
  - output/standout_equity_curve.csv
  - output/standout_report.txt
  - output/standout_vs_benchmark.csv
  - output/standout_window_details.csv
  - output/standout_ticker_summary.csv
  - output/standout_monthly_pnl.csv
"""

import warnings
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from backtest.config import (
    ANNOUNCEMENTS_PATH, OHLCV_DIR, BENCHMARK_PATH, OUTPUT_DIR,
    START_DATE, END_DATE, PREDICT_WEEKS, STEP_WEEKS,
    INITIAL_CAPITAL, COMMISSION_PCT, SLIPPAGE_PCT,
    RANDOM_STATE, MIN_TRAIN_SAMPLES, P85_30D, SF_COLS,
    TIERED_SLIPPAGE,
)
from backtest.data_loader import load_announcements, load_benchmark
from backtest.features import fit_ohe, build_feature_matrix
from backtest.model import generate_windows
from backtest.portfolio_simulator import simulate_portfolio, _load_ohlcv_cache
from backtest.portfolio_construction import allocate_inverse_volatility
from backtest.metrics import compute_metrics, compute_benchmark_metrics

warnings.filterwarnings("ignore")

# ── Strategy constants ───────────────────────────────────────────────────────
STRATEGY_NAME = "t9_200d4_all_thp85_WC_mw7_tp30_h50"
TRAIN_MONTHS = 9
N_ESTIMATORS = 200
MAX_DEPTH = 4
LEARNING_RATE = 0.05
THRESHOLD = P85_30D          # 7.2225
MAX_WEIGHT = 0.07            # 7% max per position
TP_PCT = 0.30                # 30% take-profit
SL_PCT = 1.00                # no stop loss (100% = effectively disabled)
HORIZON_BARS = 50            # 50 trading days max hold
ADV_CAP = 0.05               # 5% of 20-day average daily dollar volume
DROP_FEATURES = ["sf_word_count"]
VOL_WINDOW = 60


def build_signals(announcements: pd.DataFrame, ohe) -> tuple[pd.DataFrame, list[dict]]:
    """Generate signals using the standout config's exact parameters.

    Returns (signals_df, window_details_list) for full transparency.
    """
    windows = generate_windows(
        start_date=START_DATE, end_date=END_DATE,
        train_months=TRAIN_MONTHS, predict_weeks=PREDICT_WEEKS,
        step_weeks=STEP_WEEKS,
    )

    # Determine feature columns to drop
    drop_idxs = [SF_COLS.index(f) for f in DROP_FEATURES if f in SF_COLS]

    all_signals = []
    window_details = []

    for w in windows:
        # Training data with 30-day leakage prevention
        clean_train_end = w.train_end - pd.DateOffset(days=30)
        train_df = announcements[
            (announcements["published_at"] >= w.train_start) &
            (announcements["published_at"] < clean_train_end)
        ].copy()

        # Prediction data
        pred_df = announcements[
            (announcements["published_at"] >= w.pred_start) &
            (announcements["published_at"] < w.pred_end)
        ].copy()

        if len(train_df) < MIN_TRAIN_SAMPLES or len(pred_df) == 0:
            continue

        # Build features
        X_train = build_feature_matrix(ohe, train_df)
        y_train = (train_df["return_30d"] >= THRESHOLD).astype(int).values

        X_pred = build_feature_matrix(ohe, pred_df)

        # Drop sf_word_count columns
        if drop_idxs:
            n_ohe = X_train.shape[1] - len(SF_COLS)
            cols_to_drop = [n_ohe + i for i in drop_idxs]
            X_train = np.delete(X_train, cols_to_drop, axis=1)
            X_pred = np.delete(X_pred, cols_to_drop, axis=1)

        # Skip single-class windows
        if len(np.unique(y_train)) < 2:
            continue

        # Train GBM
        model = GradientBoostingClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_pred)
        scores = model.predict_proba(X_pred)[:, 1]

        pred_df = pred_df.copy()
        pred_df["prediction"] = preds
        pred_df["decision_score"] = scores

        # Filter positive predictions
        signals = pred_df[pred_df["prediction"] == 1].copy()

        if len(signals) == 0:
            window_details.append({
                "window_id": w.window_id,
                "train_start": w.train_start.strftime("%Y-%m-%d"),
                "train_end": clean_train_end.strftime("%Y-%m-%d"),
                "pred_start": w.pred_start.strftime("%Y-%m-%d"),
                "pred_end": w.pred_end.strftime("%Y-%m-%d"),
                "train_samples": len(train_df),
                "pos_rate": float(y_train.mean()),
                "pred_samples": len(pred_df),
                "signals": 0,
            })
            continue

        # Dedup: keep first per ticker per window
        signals = signals.sort_values("published_at", ascending=True)
        signals = signals.drop_duplicates(subset=["ticker"], keep="first")

        signals["window_id"] = w.window_id
        signals["horizon"] = "30d"

        window_details.append({
            "window_id": w.window_id,
            "train_start": w.train_start.strftime("%Y-%m-%d"),
            "train_end": clean_train_end.strftime("%Y-%m-%d"),
            "pred_start": w.pred_start.strftime("%Y-%m-%d"),
            "pred_end": w.pred_end.strftime("%Y-%m-%d"),
            "train_samples": len(train_df),
            "pos_rate": float(y_train.mean()),
            "pred_samples": len(pred_df),
            "signals": len(signals),
        })

        all_signals.append(signals)

    if not all_signals:
        return pd.DataFrame(), window_details

    return pd.concat(all_signals, ignore_index=True), window_details


def build_ticker_summary(trade_log: pd.DataFrame) -> pd.DataFrame:
    """Build per-ticker PnL summary from trade log."""
    if len(trade_log) == 0:
        return pd.DataFrame()

    summary = trade_log.groupby("ticker").agg(
        trades=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        avg_return_pct=("return_pct", "mean"),
        win_rate=("return_pct", lambda x: (x > 0).mean() * 100),
        avg_hold_days=("entry_date", lambda x: 0),  # placeholder
    ).reset_index()

    # Compute actual avg hold days
    hold_days = []
    for ticker in summary["ticker"]:
        t_trades = trade_log[trade_log["ticker"] == ticker]
        durations = (pd.to_datetime(t_trades["exit_date"]) - pd.to_datetime(t_trades["entry_date"])).dt.days
        hold_days.append(durations.mean())
    summary["avg_hold_days"] = hold_days

    summary = summary.sort_values("total_pnl", ascending=False).reset_index(drop=True)
    summary["pct_of_total_pnl"] = (summary["total_pnl"] / summary["total_pnl"].sum() * 100).round(2)

    return summary


def build_monthly_pnl(trade_log: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    """Build monthly PnL breakdown from trade log."""
    if len(trade_log) == 0:
        return pd.DataFrame()

    tl = trade_log.copy()
    tl["exit_date"] = pd.to_datetime(tl["exit_date"])
    tl["month"] = tl["exit_date"].dt.to_period("M")

    monthly = tl.groupby("month").agg(
        trades=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        avg_return_pct=("return_pct", "mean"),
        win_rate=("return_pct", lambda x: (x > 0).mean() * 100),
        winners=("return_pct", lambda x: (x > 0).sum()),
        losers=("return_pct", lambda x: (x <= 0).sum()),
    ).reset_index()

    monthly["cumulative_pnl"] = monthly["total_pnl"].cumsum()
    monthly["cumulative_return_pct"] = (monthly["cumulative_pnl"] / initial_capital * 100).round(2)
    monthly["month"] = monthly["month"].astype(str)

    return monthly


def build_report(
    metrics: dict,
    bench_metrics: dict,
    trade_log: pd.DataFrame,
    signals: pd.DataFrame,
    window_details: list[dict],
    ticker_summary: pd.DataFrame,
    monthly_pnl: pd.DataFrame,
) -> str:
    """Generate the full text report."""
    m = metrics
    b = bench_metrics

    n_windows_with_signals = sum(1 for w in window_details if w["signals"] > 0)
    n_total_windows = len(window_details)

    # Exit type breakdown
    exit_counts = trade_log["exit_reason"].value_counts()
    tp_trades = exit_counts.get("take_profit", 0)
    horizon_trades = exit_counts.get("horizon_expiry", 0)
    eob_trades = exit_counts.get("end_of_backtest", 0)

    # Yearly breakdown
    tl = trade_log.copy()
    tl["exit_year"] = pd.to_datetime(tl["exit_date"]).dt.year
    y1 = tl[tl["exit_year"] == 2024]["pnl"].sum()
    y2 = tl[tl["exit_year"] == 2025]["pnl"].sum()
    y3 = tl[tl["exit_year"] == 2026]["pnl"].sum()

    # Top and bottom tickers
    top5 = ticker_summary.head(5)
    bot5 = ticker_summary.tail(5).iloc[::-1]

    report = f"""
{'='*80}
BACKTEST REPORT: {STRATEGY_NAME}
{'='*80}

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Period: {START_DATE} to {END_DATE}
Initial Capital: ${INITIAL_CAPITAL:,.0f}

{'─'*80}
STRATEGY CONFIGURATION
{'─'*80}

  Model:            GradientBoostingClassifier
  n_estimators:     {N_ESTIMATORS}
  max_depth:        {MAX_DEPTH}
  learning_rate:    {LEARNING_RATE}
  Training window:  {TRAIN_MONTHS} months
  Predict window:   {PREDICT_WEEKS} weeks
  Step size:        {STEP_WEEKS} weeks
  Label threshold:  P85_30D = {THRESHOLD}% (30-day return >= 85th percentile)
  Event filter:     All event types
  Dropped features: {DROP_FEATURES}

  Position sizing:  Inverse-volatility (vol_window={VOL_WINDOW})
  Max weight:       {MAX_WEIGHT:.0%} per position
  Take profit:      {TP_PCT:.0%}
  Stop loss:        None (disabled)
  Horizon:          {HORIZON_BARS} trading days
  ADV cap:          {ADV_CAP:.0%} of 20-day avg daily dollar volume
  Tiered slippage:  {TIERED_SLIPPAGE} (<$2→5%, $2-$5→2%, >=$5→0.1%)
  Commission:       {COMMISSION_PCT:.1%}

{'─'*80}
PERFORMANCE SUMMARY
{'─'*80}

  {'Metric':<30s}  {'Strategy':>12s}  {'SPY (B&H)':>12s}  {'Alpha':>12s}
  {'─'*30}  {'─'*12}  {'─'*12}  {'─'*12}
  {'Total Return':<30s}  {m['total_return_pct']:>+11.1f}%  {b['total_return_pct']:>+11.1f}%  {m['total_return_pct']-b['total_return_pct']:>+11.1f}%
  {'Annualized Return':<30s}  {m['annualized_return_pct']:>+11.1f}%  {b['annualized_return_pct']:>+11.1f}%  {m['annualized_return_pct']-b['annualized_return_pct']:>+11.1f}%
  {'Sharpe Ratio':<30s}  {m['sharpe_ratio']:>12.2f}  {b['sharpe_ratio']:>12.2f}  {m['sharpe_ratio']-b['sharpe_ratio']:>+11.2f}
  {'Max Drawdown':<30s}  {m['max_drawdown_pct']:>11.1f}%  {b['max_drawdown_pct']:>11.1f}%
  {'Win Rate':<30s}  {m['win_rate_pct']:>11.1f}%
  {'Profit Factor':<30s}  {m['profit_factor']:>12.2f}
  {'Total Trades':<30s}  {m['total_trades']:>12d}
  {'Avg Return/Trade':<30s}  {m['avg_return_per_trade']:>+11.2f}%
  {'Avg Holding Period':<30s}  {m['avg_holding_days']:>10.1f}d
  {'Exposure':<30s}  {m['exposure_pct']:>11.1f}%

{'─'*80}
SIGNAL GENERATION
{'─'*80}

  Rolling windows:     {n_total_windows}
  Windows with signals: {n_windows_with_signals}
  Total signals:        {len(signals)}
  Signals executed:     {m['total_trades']}
  Signal utilization:   {m['total_trades']/len(signals)*100:.1f}%

{'─'*80}
EXIT ANALYSIS
{'─'*80}

  Take-profit hits:    {tp_trades:>4d}  ({tp_trades/m['total_trades']*100:.1f}%)
  Horizon expiry:      {horizon_trades:>4d}  ({horizon_trades/m['total_trades']*100:.1f}%)
  End-of-backtest:     {eob_trades:>4d}  ({eob_trades/m['total_trades']*100:.1f}%)

  TP trades are 100% winners by definition.
  Horizon expiry win rate: {(tl[tl['exit_reason']=='horizon_expiry']['return_pct'] > 0).mean()*100:.1f}%

{'─'*80}
YEARLY BREAKDOWN
{'─'*80}

  2024:  ${y1:>+12,.0f}  ({y1/INITIAL_CAPITAL*100:>+6.1f}%)
  2025:  ${y2:>+12,.0f}  ({y2/INITIAL_CAPITAL*100:>+6.1f}%)
  2026:  ${y3:>+12,.0f}  ({y3/INITIAL_CAPITAL*100:>+6.1f}%)  (partial — trades exiting in Jan 2026)
  Total: ${y1+y2+y3:>+12,.0f}  ({(y1+y2+y3)/INITIAL_CAPITAL*100:>+6.1f}%)

{'─'*80}
MONTHLY PnL
{'─'*80}

  {'Month':<10s}  {'Trades':>6s}  {'PnL':>12s}  {'Cumul PnL':>12s}  {'Cumul %':>8s}  {'WR':>6s}
  {'─'*10}  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*6}
"""
    for _, row in monthly_pnl.iterrows():
        report += f"  {row['month']:<10s}  {row['trades']:>6d}  ${row['total_pnl']:>+11,.0f}  ${row['cumulative_pnl']:>+11,.0f}  {row['cumulative_return_pct']:>+7.1f}%  {row['win_rate']:>5.0f}%\n"

    report += f"""
{'─'*80}
TOP 15 TICKERS BY PnL
{'─'*80}

  {'Ticker':<8s}  {'Trades':>6s}  {'Total PnL':>12s}  {'% of Total':>10s}  {'Avg Ret':>8s}  {'WR':>6s}
  {'─'*8}  {'─'*6}  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*6}
"""
    for _, row in ticker_summary.head(15).iterrows():
        report += f"  {row['ticker']:<8s}  {row['trades']:>6.0f}  ${row['total_pnl']:>+11,.0f}  {row['pct_of_total_pnl']:>9.1f}%  {row['avg_return_pct']:>+7.1f}%  {row['win_rate']:>5.0f}%\n"

    report += f"""
{'─'*80}
BOTTOM 5 TICKERS BY PnL
{'─'*80}

  {'Ticker':<8s}  {'Trades':>6s}  {'Total PnL':>12s}  {'Avg Ret':>8s}
  {'─'*8}  {'─'*6}  {'─'*12}  {'─'*8}
"""
    for _, row in ticker_summary.tail(5).iloc[::-1].iterrows():
        report += f"  {row['ticker']:<8s}  {row['trades']:>6.0f}  ${row['total_pnl']:>+11,.0f}  {row['avg_return_pct']:>+7.1f}%\n"

    report += f"""
{'─'*80}
CONCENTRATION ANALYSIS
{'─'*80}

  Unique tickers:       {ticker_summary['ticker'].nunique()}
  Net positive tickers: {(ticker_summary['total_pnl'] > 0).sum()}
  Net negative tickers: {(ticker_summary['total_pnl'] <= 0).sum()}

  Top-1 ticker PnL:     {ticker_summary.iloc[0]['pct_of_total_pnl']:.1f}%
  Top-3 tickers PnL:    {ticker_summary.head(3)['pct_of_total_pnl'].sum():.1f}%
  Top-5 tickers PnL:    {ticker_summary.head(5)['pct_of_total_pnl'].sum():.1f}%
  Top-10 tickers PnL:   {ticker_summary.head(10)['pct_of_total_pnl'].sum():.1f}%

{'─'*80}
AUDIT STATUS (from forensic audit, 11 checks)
{'─'*80}

  Look-ahead bias:       PASS (30-day gap between train/predict)
  Price verification:    PASS (all entry prices match OHLCV opens)
  Ticker concentration:  PASS (92 tickers, not single-name dependent)
  Window stability:      PASS (20/23 windows positive)
  Randomized labels:     PASS (z=6.5, p=0.000)
  Multiple testing:      PASS (p=0.000 after 1034 configs)
  Liquidity:             MITIGATED (5% ADV cap applied)
  Leave-one-ticker-out:  PASS (ex-OLMA +96.6%, ex-top10 +61.6%)
  Phantom returns:       PASS (1 gap-through verified as legitimate)
  Exit analysis:         PASS (balanced TP/horizon mix)
  Top trades:            PASS (diverse events, reasonable hold periods)

  Conservative estimate (5% ADV + TP-capped): +73.7% vs SPY +53%

{'='*80}
"""
    return report


def main():
    """Run the full backtest and generate all outputs."""
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"FULL BACKTEST: {STRATEGY_NAME}")
    print(f"{'='*60}")

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    announcements = load_announcements(ANNOUNCEMENTS_PATH)
    print(f"  {len(announcements):,} announcements loaded")

    ohe = fit_ohe(announcements)
    n_ohe = ohe.transform(announcements[["source", "event_type"]]).shape[1]
    print(f"  OHE fitted: {n_ohe} categories")

    # ── Generate signals ─────────────────────────────────────────────────────
    print("\n[2/6] Generating signals (9-month GBM rolling windows)...")
    signals, window_details = build_signals(announcements, ohe)
    print(f"  {len(window_details)} windows processed")
    print(f"  {len(signals)} signals generated across {sum(1 for w in window_details if w['signals'] > 0)} active windows")

    if len(signals) == 0:
        print("  ERROR: No signals generated. Aborting.")
        return

    # ── Simulate portfolio ───────────────────────────────────────────────────
    print(f"\n[3/6] Simulating portfolio...")
    print(f"  TP={TP_PCT:.0%}, SL=none, Horizon={HORIZON_BARS} bars")
    print(f"  Inverse-vol sizing, max_weight={MAX_WEIGHT:.0%}, ADV_cap={ADV_CAP:.0%}")
    print(f"  Tiered slippage={TIERED_SLIPPAGE}, Commission={COMMISSION_PCT:.1%}")

    weight_fn = partial(allocate_inverse_volatility, vol_window=VOL_WINDOW, max_weight=MAX_WEIGHT)

    trade_log, equity_curve = simulate_portfolio(
        signals, "30d",
        take_profit_pct=TP_PCT,
        stop_loss_pct=SL_PCT,
        ohlcv_dir=OHLCV_DIR,
        initial_capital=INITIAL_CAPITAL,
        commission_pct=COMMISSION_PCT,
        slippage_pct=SLIPPAGE_PCT,
        weight_fn=weight_fn,
        tiered_slippage=TIERED_SLIPPAGE,
        adv_cap_pct=ADV_CAP,
        horizon_bars=HORIZON_BARS,
    )
    print(f"  {len(trade_log)} trades executed")

    # ── Compute metrics ──────────────────────────────────────────────────────
    print("\n[4/6] Computing metrics...")
    metrics = compute_metrics(trade_log, INITIAL_CAPITAL, equity_curve=equity_curve)

    benchmark_df = load_benchmark(BENCHMARK_PATH)
    bench_metrics = compute_benchmark_metrics(benchmark_df, START_DATE, END_DATE)

    for k, v in metrics.items():
        print(f"  {k:30s}: {v}")
    print(f"\n  SPY benchmark: {bench_metrics['total_return_pct']:+.1f}%, Sharpe {bench_metrics['sharpe_ratio']:.2f}")

    # ── Build summaries ──────────────────────────────────────────────────────
    print("\n[5/6] Building summaries...")
    ticker_summary = build_ticker_summary(trade_log)
    monthly_pnl = build_monthly_pnl(trade_log, INITIAL_CAPITAL)

    # ── Save outputs ─────────────────────────────────────────────────────────
    print("\n[6/6] Saving outputs...")

    # Trade log
    trade_log.to_csv(OUTPUT_DIR / "standout_trade_log.csv", index=False)
    print(f"  -> output/standout_trade_log.csv ({len(trade_log)} trades)")

    # Metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(OUTPUT_DIR / "standout_metrics.csv", index=False)
    print(f"  -> output/standout_metrics.csv")

    # Equity curve
    equity_curve.to_csv(OUTPUT_DIR / "standout_equity_curve.csv")
    print(f"  -> output/standout_equity_curve.csv ({len(equity_curve)} days)")

    # Benchmark comparison
    comparison = pd.DataFrame({
        "strategy": metrics,
        "SPY_benchmark": bench_metrics,
    }).T
    comparison.to_csv(OUTPUT_DIR / "standout_vs_benchmark.csv")
    print(f"  -> output/standout_vs_benchmark.csv")

    # Window details
    pd.DataFrame(window_details).to_csv(OUTPUT_DIR / "standout_window_details.csv", index=False)
    print(f"  -> output/standout_window_details.csv ({len(window_details)} windows)")

    # Ticker summary
    ticker_summary.to_csv(OUTPUT_DIR / "standout_ticker_summary.csv", index=False)
    print(f"  -> output/standout_ticker_summary.csv ({len(ticker_summary)} tickers)")

    # Monthly PnL
    monthly_pnl.to_csv(OUTPUT_DIR / "standout_monthly_pnl.csv", index=False)
    print(f"  -> output/standout_monthly_pnl.csv ({len(monthly_pnl)} months)")

    # Full report
    report = build_report(metrics, bench_metrics, trade_log, signals, window_details, ticker_summary, monthly_pnl)
    with open(OUTPUT_DIR / "standout_report.txt", "w") as f:
        f.write(report)
    print(f"  -> output/standout_report.txt")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.1f}s — all outputs in output/")
    print(f"{'='*60}")

    # Print the report to console
    print(report)


if __name__ == "__main__":
    main()
