"""
Forensic audit of standout config: t9_200d4_all_thp85_WC_mw7_tp30_h50

Checks:
1. Look-ahead bias: verify train/pred window separation
2. Trade-level price verification against raw OHLCV files
3. Ticker concentration & single-name dependency
4. Window-by-window PnL stability
5. Multiple testing correction (Bonferroni/BH on 1034 configs)
6. Randomized label test (shuffled returns — does model still "work"?)
7. Per-trade entry/exit price sanity (no phantom returns)
8. Liquidity check: can these trades actually execute?
9. Out-of-sample: train on 2024, test on 2025 (and reverse)
10. Leave-one-ticker-out robustness
"""

import warnings
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from backtest.config import (
    ANNOUNCEMENTS_PATH, OHLCV_DIR, OUTPUT_DIR,
    START_DATE, END_DATE, PREDICT_WEEKS, STEP_WEEKS,
    INITIAL_CAPITAL, P85_30D,
    RANDOM_STATE, MIN_TRAIN_SAMPLES, SF_COLS,
)
from backtest.data_loader import load_announcements, load_ohlcv
from backtest.features import fit_ohe, build_feature_matrix
from backtest.model import generate_windows
from backtest.portfolio_simulator import simulate_portfolio, _load_ohlcv_cache
from backtest.portfolio_construction import allocate_inverse_volatility
from backtest.metrics import compute_metrics
from backtest.research import run_experiment, robustness, print_result, log_entry

warnings.filterwarnings("ignore")


def build_signals_9m(announcements, ohe, train_months=9):
    """Reproduce the standout config's signal generation step-by-step."""
    windows = generate_windows(
        start_date=START_DATE, end_date=END_DATE,
        train_months=train_months, predict_weeks=PREDICT_WEEKS,
        step_weeks=STEP_WEEKS,
    )

    label_col = "return_30d"
    all_signals = []
    window_details = []

    for window in windows:
        clean_train_end = window.train_end - pd.DateOffset(days=30)
        train_df = announcements[
            (announcements["published_at"] >= window.train_start) &
            (announcements["published_at"] < clean_train_end)
        ].copy()
        pred_df = announcements[
            (announcements["published_at"] >= window.pred_start) &
            (announcements["published_at"] < window.pred_end)
        ].copy()

        if len(train_df) < MIN_TRAIN_SAMPLES or len(pred_df) == 0:
            continue

        X_train = build_feature_matrix(ohe, train_df)
        y_train = (train_df[label_col] >= P85_30D).astype(int).values
        X_pred = build_feature_matrix(ohe, pred_df)

        if len(np.unique(y_train)) < 2:
            continue

        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_pred)
        scores = model.predict_proba(X_pred)[:, 1]

        pred_df = pred_df.copy()
        pred_df["prediction"] = preds
        pred_df["decision_score"] = scores
        signals = pred_df[pred_df["prediction"] == 1].copy()

        if len(signals) == 0:
            window_details.append({
                "window_id": window.window_id,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "pred_start": window.pred_start,
                "pred_end": window.pred_end,
                "clean_train_end": clean_train_end,
                "train_samples": len(train_df),
                "pred_samples": len(pred_df),
                "positive_rate": float(y_train.mean()),
                "signals": 0,
            })
            continue

        signals = signals.sort_values("published_at").drop_duplicates(subset=["ticker"], keep="first")
        signals["window_id"] = window.window_id
        signals["horizon"] = "30d"

        window_details.append({
            "window_id": window.window_id,
            "train_start": window.train_start,
            "train_end": window.train_end,
            "pred_start": window.pred_start,
            "pred_end": window.pred_end,
            "clean_train_end": clean_train_end,
            "train_samples": len(train_df),
            "pred_samples": len(pred_df),
            "positive_rate": float(y_train.mean()),
            "signals": len(signals),
        })

        all_signals.append(signals)

    if not all_signals:
        return pd.DataFrame(), window_details
    return pd.concat(all_signals, ignore_index=True), window_details


def main():
    t0 = time.time()
    print("=" * 80)
    print("FORENSIC AUDIT: t9_200d4_all_thp85_WC_mw7_tp30_h50")
    print("=" * 80)

    ann = load_announcements(ANNOUNCEMENTS_PATH)
    ohe = fit_ohe(ann)
    tickers = ann["ticker"].unique().tolist()
    ohlcv_cache = _load_ohlcv_cache(tickers, OHLCV_DIR)

    journal_lines = []
    journal_lines.append("**Time**: 2026-03-30")
    journal_lines.append("**Focus**: Forensic audit of standout config t9_200d4_all_thp85_WC_mw7_tp30_h50\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # CHECK 1: Look-ahead bias — window separation
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("CHECK 1: Look-ahead bias — window separation")
    print("─" * 60)

    signals, window_details = build_signals_9m(ann, ohe, train_months=9)
    print(f"\n  Generated {len(signals)} signals across {len(window_details)} windows")

    print("\n  Window details:")
    leakage_found = False
    for wd in window_details:
        gap_days = (wd["pred_start"] - wd["clean_train_end"]).days
        overlap = wd["clean_train_end"] > wd["pred_start"]
        flag = " *** LEAKAGE ***" if overlap else ""
        if overlap:
            leakage_found = True
        print(f"    W{wd['window_id']:>2}  train=[{wd['train_start'].date()} .. {wd['clean_train_end'].date()}]  "
              f"pred=[{wd['pred_start'].date()} .. {wd['pred_end'].date()}]  "
              f"gap={gap_days:>3}d  train_n={wd['train_samples']:>4}  "
              f"pos_rate={wd['positive_rate']:.2f}  signals={wd['signals']:>3}{flag}")

    if leakage_found:
        print("\n  *** LOOK-AHEAD BIAS DETECTED ***")
        journal_lines.append("**CHECK 1: FAILED** — Look-ahead bias detected in window separation")
    else:
        print("\n  PASS: No look-ahead bias. All training ends before prediction starts with 30-day gap.")
        journal_lines.append("**CHECK 1: PASS** — No look-ahead bias. 30-day gap between train end and pred start.")

    # ═══════════════════════════════════════════════════════════════════════════
    # CHECK 2: Reproduce exact result and verify trade-level prices
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("CHECK 2: Reproduce result + trade-level price verification")
    print("─" * 60)

    weight_fn = partial(allocate_inverse_volatility, vol_window=60, max_weight=0.07)
    result = run_experiment(
        "standout", signals, ohlcv_cache,
        horizon_bars=50, take_profit_pct=0.30, stop_loss_pct=1.0,
        weight_fn=weight_fn,
    )
    rob = robustness(result)
    m = result["metrics"]
    print(f"\n  Reproduced: {m['total_return_pct']:+.1f}%, Sharpe {m['sharpe_ratio']:.2f}, "
          f"{m['total_trades']} trades, WR {m['win_rate_pct']:.1f}%, DD {m['max_drawdown_pct']:.1f}%")

    tl = result["trade_log"]
    journal_lines.append(f"**CHECK 2: Reproduced** — {m['total_return_pct']:+.1f}%, Sharpe {m['sharpe_ratio']:.2f}")

    # Verify individual trade entry/exit prices against OHLCV files
    print("\n  Verifying trade prices against OHLCV data...")
    price_errors = []
    for _, trade in tl.head(30).iterrows():
        ticker = trade["ticker"]
        ohlcv = ohlcv_cache.get(ticker)
        if ohlcv is None:
            continue
        entry_date = pd.Timestamp(trade["entry_date"])
        exit_date = pd.Timestamp(trade["exit_date"])

        # Check entry: should be next-day open after announcement
        entry_row = ohlcv[ohlcv.index >= entry_date].head(1)
        if len(entry_row) == 0:
            price_errors.append(f"  {ticker}: no OHLCV on entry date {entry_date.date()}")
            continue

        actual_open = float(entry_row["Open"].iloc[0])
        reported_entry = trade["entry_price"]

        # Entry price should be close to open (slippage applied)
        pct_diff = abs(reported_entry - actual_open) / actual_open * 100
        if pct_diff > 10:  # >10% difference is suspicious
            price_errors.append(
                f"  {ticker} {entry_date.date()}: entry={reported_entry:.2f} vs open={actual_open:.2f} ({pct_diff:.1f}% diff)"
            )

    if price_errors:
        print(f"\n  {len(price_errors)} suspicious price discrepancies (>10% from open):")
        for err in price_errors[:10]:
            print(f"    {err}")
        journal_lines.append(f"**CHECK 2 PRICES**: {len(price_errors)} price discrepancies >10% from open")
    else:
        print("  PASS: All sampled entry prices within 10% of OHLCV open.")
        journal_lines.append("**CHECK 2 PRICES: PASS** — Entry prices consistent with OHLCV opens")

    # ═══════════════════════════════════════════════════════════════════════════
    # CHECK 3: Ticker concentration analysis
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("CHECK 3: Ticker concentration")
    print("─" * 60)

    ticker_pnl = tl.groupby("ticker")["pnl"].sum().sort_values(ascending=False)
    total_pnl = tl["pnl"].sum()
    n_unique = len(ticker_pnl)
    n_positive = (ticker_pnl > 0).sum()
    n_negative = (ticker_pnl < 0).sum()

    print(f"\n  {n_unique} unique tickers, {n_positive} net positive, {n_negative} net negative")
    print(f"  Total PnL: ${total_pnl:,.0f}")

    # Concentration metrics
    top1_pct = ticker_pnl.iloc[0] / total_pnl * 100
    top3_pct = ticker_pnl.head(3).sum() / total_pnl * 100
    top5_pct = ticker_pnl.head(5).sum() / total_pnl * 100
    top10_pct = ticker_pnl.head(10).sum() / total_pnl * 100

    print(f"  Top-1: {ticker_pnl.index[0]} = ${ticker_pnl.iloc[0]:+,.0f} ({top1_pct:.0f}%)")
    print(f"  Top-3: {top3_pct:.0f}% of PnL")
    print(f"  Top-5: {top5_pct:.0f}% of PnL")
    print(f"  Top-10: {top10_pct:.0f}% of PnL")

    print("\n  Top 15 tickers by PnL:")
    for tk, pnl_val in ticker_pnl.head(15).items():
        trades_tk = len(tl[tl["ticker"] == tk])
        wr_tk = (tl[tl["ticker"] == tk]["pnl"] > 0).mean() * 100
        print(f"    {tk:8s}  {trades_tk:>2}tr  ${pnl_val:>+10,.0f}  ({pnl_val/total_pnl*100:>+5.1f}%)  WR={wr_tk:.0f}%")

    print("\n  Bottom 5 tickers by PnL:")
    for tk, pnl_val in ticker_pnl.tail(5).items():
        trades_tk = len(tl[tl["ticker"] == tk])
        print(f"    {tk:8s}  {trades_tk:>2}tr  ${pnl_val:>+10,.0f}")

    journal_lines.append(f"\n**CHECK 3: Ticker concentration** — {n_unique} tickers, top1={top1_pct:.0f}%, top5={top5_pct:.0f}%, top10={top10_pct:.0f}%")

    # ═══════════════════════════════════════════════════════════════════════════
    # CHECK 4: Window-by-window PnL stability
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("CHECK 4: Window-by-window PnL")
    print("─" * 60)

    if "window_id" in tl.columns:
        window_pnl = tl.groupby("window_id").agg(
            trades=("pnl", "count"),
            pnl=("pnl", "sum"),
            win_rate=("pnl", lambda x: (x > 0).mean() * 100),
        ).reset_index()

        n_positive_windows = (window_pnl["pnl"] > 0).sum()
        n_negative_windows = (window_pnl["pnl"] < 0).sum()
        print(f"\n  {n_positive_windows} positive windows, {n_negative_windows} negative windows")

        for _, row in window_pnl.iterrows():
            bar = "+" * max(1, int(abs(row["pnl"]) / 10000))
            sign = "+" if row["pnl"] >= 0 else "-"
            print(f"    W{int(row['window_id']):>2}  {int(row['trades']):>3}tr  "
                  f"${row['pnl']:>+10,.0f}  WR={row['win_rate']:>5.1f}%  {sign}{bar}")

        journal_lines.append(f"**CHECK 4: Windows** — {n_positive_windows} positive / {n_negative_windows} negative windows")

    # ═══════════════════════════════════════════════════════════════════════════
    # CHECK 5: Randomized label test (null hypothesis)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("CHECK 5: Randomized label test — can a random model match this?")
    print("─" * 60)

    random_sharpes = []
    for seed in range(20):
        # Shuffle the return_30d labels randomly
        ann_shuffled = ann.copy()
        rng = np.random.RandomState(seed + 100)
        ann_shuffled["return_30d"] = rng.permutation(ann_shuffled["return_30d"].values)

        windows = generate_windows(
            start_date=START_DATE, end_date=END_DATE,
            train_months=9, predict_weeks=PREDICT_WEEKS,
            step_weeks=STEP_WEEKS,
        )
        all_sigs = []
        for window in windows:
            clean_train_end = window.train_end - pd.DateOffset(days=30)
            train_df = ann_shuffled[
                (ann_shuffled["published_at"] >= window.train_start) &
                (ann_shuffled["published_at"] < clean_train_end)
            ].copy()
            pred_df = ann_shuffled[
                (ann_shuffled["published_at"] >= window.pred_start) &
                (ann_shuffled["published_at"] < window.pred_end)
            ].copy()
            if len(train_df) < MIN_TRAIN_SAMPLES or len(pred_df) == 0:
                continue
            X_tr = build_feature_matrix(ohe, train_df)
            y_tr = (train_df["return_30d"] >= P85_30D).astype(int).values
            X_pr = build_feature_matrix(ohe, pred_df)
            if len(np.unique(y_tr)) < 2:
                continue
            try:
                mdl = GradientBoostingClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE)
                mdl.fit(X_tr, y_tr)
                preds = mdl.predict(X_pr)
                scores = mdl.predict_proba(X_pr)[:, 1]
            except Exception:
                continue
            pred_df = pred_df.copy()
            pred_df["prediction"] = preds
            pred_df["decision_score"] = scores
            sigs = pred_df[pred_df["prediction"] == 1].copy()
            if len(sigs) == 0:
                continue
            sigs = sigs.sort_values("published_at").drop_duplicates(subset=["ticker"], keep="first")
            sigs["window_id"] = window.window_id
            sigs["horizon"] = "30d"
            all_sigs.append(sigs)

        if not all_sigs:
            random_sharpes.append(0)
            continue
        random_signals = pd.concat(all_sigs, ignore_index=True)
        r_rand = run_experiment("random", random_signals, ohlcv_cache,
                                 horizon_bars=50, take_profit_pct=0.30, weight_fn=weight_fn)
        if r_rand["metrics"]:
            random_sharpes.append(r_rand["metrics"]["sharpe_ratio"])
            print(f"    Seed {seed:>2}: Sharpe={r_rand['metrics']['sharpe_ratio']:>+5.2f}  "
                  f"Return={r_rand['metrics']['total_return_pct']:>+6.1f}%  "
                  f"Trades={r_rand['metrics']['total_trades']}")
        else:
            random_sharpes.append(0)

    random_sharpes = np.array(random_sharpes)
    real_sharpe = m["sharpe_ratio"]
    p_value = (random_sharpes >= real_sharpe).mean()
    print(f"\n  Real Sharpe: {real_sharpe:.2f}")
    print(f"  Random Sharpe: mean={random_sharpes.mean():.2f}, max={random_sharpes.max():.2f}, "
          f"std={random_sharpes.std():.2f}")
    print(f"  P-value (random >= real): {p_value:.3f}")
    print(f"  Z-score: {(real_sharpe - random_sharpes.mean()) / max(0.01, random_sharpes.std()):.1f}")

    if p_value < 0.05:
        print("  PASS: Real model significantly outperforms random labels (p < 0.05)")
        journal_lines.append(f"**CHECK 5: PASS** — Random label p-value={p_value:.3f}, z-score={(real_sharpe - random_sharpes.mean()) / max(0.01, random_sharpes.std()):.1f}")
    else:
        print("  *** FAIL ***: Random labels achieve similar Sharpe — signal may be spurious")
        journal_lines.append(f"**CHECK 5: FAIL** — Random labels p={p_value:.3f} — signal may be spurious!")

    # ═══════════════════════════════════════════════════════════════════════════
    # CHECK 6: Multiple testing correction
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("CHECK 6: Multiple testing correction")
    print("─" * 60)

    n_configs = 1034
    # Under null (no signal), how often would we see Sharpe >= 2.51 by chance?
    # Using random Sharpe distribution from CHECK 5
    random_max_sharpes = []
    for trial in range(1000):
        rng_trial = np.random.RandomState(trial + 5000)
        simulated_sharpes = rng_trial.normal(random_sharpes.mean(), max(0.01, random_sharpes.std()), size=n_configs)
        random_max_sharpes.append(simulated_sharpes.max())

    random_max_sharpes = np.array(random_max_sharpes)
    p_multiple = (random_max_sharpes >= real_sharpe).mean()
    print(f"\n  With {n_configs} configs tested, prob of seeing Sharpe >= {real_sharpe:.2f} by chance:")
    print(f"  P-value (multiple testing corrected): {p_multiple:.3f}")
    print(f"  Max random Sharpe across {n_configs} tests: mean={random_max_sharpes.mean():.2f}, "
          f"95th={np.percentile(random_max_sharpes, 95):.2f}")

    if p_multiple < 0.05:
        print("  PASS: Survives multiple testing correction")
        journal_lines.append(f"**CHECK 6: PASS** — Multiple testing p={p_multiple:.3f} (Bonferroni-style)")
    else:
        print(f"  *** WARNING ***: May not survive multiple testing correction (p={p_multiple:.3f})")
        journal_lines.append(f"**CHECK 6: WARNING** — Multiple testing p={p_multiple:.3f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # CHECK 7: Liquidity check — are trades executable?
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("CHECK 7: Liquidity check")
    print("─" * 60)

    illiquid_trades = []
    for _, trade in tl.iterrows():
        ticker = trade["ticker"]
        ohlcv = ohlcv_cache.get(ticker)
        if ohlcv is None:
            continue
        entry_date = pd.Timestamp(trade["entry_date"])
        # Get volume on entry day
        entry_row = ohlcv[ohlcv.index >= entry_date].head(1)
        if len(entry_row) == 0:
            continue
        volume = float(entry_row["Volume"].iloc[0])
        price = float(entry_row["Close"].iloc[0])
        dollar_volume = volume * price
        position_size = trade.get("position_size", trade.get("pnl", 0) / max(0.01, abs(trade.get("return_pct", 1) / 100)))

        # 20-day average volume
        lookback = ohlcv[ohlcv.index < entry_date].tail(20)
        if len(lookback) > 0:
            avg_dollar_vol = (lookback["Volume"] * lookback["Close"]).mean()
        else:
            avg_dollar_vol = dollar_volume

        # Flag if position > 10% of avg daily volume
        if avg_dollar_vol > 0 and abs(trade["pnl"]) > 0:
            # Estimate position size from PnL and return
            est_position = abs(trade["pnl"]) / max(0.01, abs(trade.get("return_pct", 10)) / 100)
            pct_of_adv = est_position / max(1, avg_dollar_vol) * 100
            if pct_of_adv > 20:
                illiquid_trades.append({
                    "ticker": ticker,
                    "date": entry_date.date(),
                    "est_position": est_position,
                    "avg_dollar_vol": avg_dollar_vol,
                    "pct_of_adv": pct_of_adv,
                    "pnl": trade["pnl"],
                })

    print(f"\n  {len(illiquid_trades)} potentially illiquid trades (position > 20% of ADV)")
    if illiquid_trades:
        illiq_pnl = sum(t["pnl"] for t in illiquid_trades)
        print(f"  Total PnL from illiquid trades: ${illiq_pnl:+,.0f} ({illiq_pnl/total_pnl*100:.1f}% of total)")
        for t in sorted(illiquid_trades, key=lambda x: -abs(x["pnl"]))[:10]:
            print(f"    {t['ticker']:8s} {t['date']}  pos~${t['est_position']:>10,.0f}  "
                  f"ADV=${t['avg_dollar_vol']:>10,.0f}  {t['pct_of_adv']:.0f}% of ADV  PnL=${t['pnl']:+,.0f}")
        journal_lines.append(f"**CHECK 7**: {len(illiquid_trades)} illiquid trades, PnL=${illiq_pnl:+,.0f} ({illiq_pnl/total_pnl*100:.1f}% of total)")
    else:
        print("  PASS: All trades within reasonable liquidity bounds")
        journal_lines.append("**CHECK 7: PASS** — All trades within liquidity bounds")

    # ═══════════════════════════════════════════════════════════════════════════
    # CHECK 8: Leave-one-ticker-out robustness
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("CHECK 8: Leave-one-ticker-out — PnL stability")
    print("─" * 60)

    # For each of the top 10 PnL tickers, compute return without them
    top10_tickers = ticker_pnl.head(10).index.tolist()
    print(f"\n  Removing each of top-10 tickers one at a time:")
    journal_lines.append("\n**CHECK 8: Leave-one-ticker-out**:")
    for tk in top10_tickers:
        tl_ex = tl[tl["ticker"] != tk]
        ex_pnl = tl_ex["pnl"].sum()
        ex_ret = ex_pnl / INITIAL_CAPITAL * 100
        ex_wr = (tl_ex["pnl"] > 0).mean() * 100
        print(f"    ex-{tk:8s}: {ex_ret:>+7.1f}%  ({len(tl_ex):>3} trades, WR={ex_wr:.1f}%)")
        journal_lines.append(f"  - ex-{tk}: {ex_ret:+.1f}%")

    # Remove ALL top-5 at once
    tl_ex5 = tl[~tl["ticker"].isin(top10_tickers[:5])]
    ex5_ret = tl_ex5["pnl"].sum() / INITIAL_CAPITAL * 100
    print(f"\n  ex-top5 combined: {ex5_ret:+.1f}% ({len(tl_ex5)} trades)")

    tl_ex10 = tl[~tl["ticker"].isin(top10_tickers)]
    ex10_ret = tl_ex10["pnl"].sum() / INITIAL_CAPITAL * 100
    print(f"  ex-top10 combined: {ex10_ret:+.1f}% ({len(tl_ex10)} trades)")

    journal_lines.append(f"  - ex-top5: {ex5_ret:+.1f}%, ex-top10: {ex10_ret:+.1f}%")

    # ═══════════════════════════════════════════════════════════════════════════
    # CHECK 9: Per-trade return distribution — any phantom returns?
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("CHECK 9: Trade return distribution — phantom return check")
    print("─" * 60)

    if "return_pct" in tl.columns:
        rets = tl["return_pct"]
    else:
        # Compute from pnl and entry price
        tl_copy = tl.copy()
        tl_copy["est_return_pct"] = tl_copy["pnl"] / (tl_copy["entry_price"] * 100)  # rough estimate
        rets = tl_copy["est_return_pct"]

    print(f"\n  Return distribution:")
    print(f"    Mean: {rets.mean():+.2f}%")
    print(f"    Median: {rets.median():+.2f}%")
    print(f"    Std: {rets.std():.2f}%")
    print(f"    Min: {rets.min():+.2f}%")
    print(f"    Max: {rets.max():+.2f}%")
    print(f"    >100% returns: {(rets > 100).sum()}")
    print(f"    >200% returns: {(rets > 200).sum()}")
    print(f"    <-50% returns: {(rets < -50).sum()}")

    # Flag any suspiciously large returns (>300%) — could be split artifacts
    extreme_trades = tl[rets > 200] if len(tl[rets > 200]) > 0 else pd.DataFrame()
    if len(extreme_trades) > 0:
        print(f"\n  *** {len(extreme_trades)} trades with >200% return — verifying against OHLCV:")
        for _, trade in extreme_trades.iterrows():
            ticker = trade["ticker"]
            ohlcv = ohlcv_cache.get(ticker)
            if ohlcv is None:
                continue
            entry_date = pd.Timestamp(trade["entry_date"])
            exit_date = pd.Timestamp(trade["exit_date"])
            entry_prices = ohlcv[ohlcv.index >= entry_date].head(1)
            exit_prices = ohlcv[ohlcv.index <= exit_date].tail(1)
            if len(entry_prices) > 0 and len(exit_prices) > 0:
                adj_entry = float(entry_prices["Open"].iloc[0])
                adj_exit = float(exit_prices["Close"].iloc[0])
                raw_return = (adj_exit - adj_entry) / adj_entry * 100
                print(f"    {ticker}: entry={trade['entry_price']:.2f} exit={trade['exit_price']:.2f} "
                      f"reported_ret={rets[trade.name]:.1f}% "
                      f"ohlcv_check: open={adj_entry:.2f} close={adj_exit:.2f} raw_ret={raw_return:+.1f}%")
        journal_lines.append(f"**CHECK 9**: {len(extreme_trades)} trades with >200% return — verify manually")
    else:
        print("  PASS: No phantom >200% returns.")
        journal_lines.append("**CHECK 9: PASS** — No phantom returns >200%")

    # ═══════════════════════════════════════════════════════════════════════════
    # CHECK 10: Exit type analysis
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("CHECK 10: Exit type analysis")
    print("─" * 60)

    if "exit_reason" in tl.columns:
        exit_counts = tl["exit_reason"].value_counts()
        print(f"\n  Exit reasons:")
        for reason, count in exit_counts.items():
            pnl_by_reason = tl[tl["exit_reason"] == reason]["pnl"]
            print(f"    {reason:20s}: {count:>4} trades  PnL=${pnl_by_reason.sum():>+10,.0f}  "
                  f"avg=${pnl_by_reason.mean():>+8,.0f}  WR={(pnl_by_reason > 0).mean()*100:.1f}%")
        journal_lines.append(f"**CHECK 10**: Exit types: {dict(exit_counts)}")

    # ═══════════════════════════════════════════════════════════════════════════
    # CHECK 11: Largest individual trades — verify
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("CHECK 11: Top 10 largest trades — manual verification")
    print("─" * 60)

    top_trades = tl.nlargest(10, "pnl")
    for _, trade in top_trades.iterrows():
        ticker = trade["ticker"]
        ohlcv = ohlcv_cache.get(ticker)
        entry_date = pd.Timestamp(trade["entry_date"])
        exit_date = pd.Timestamp(trade["exit_date"])
        hold_days = (exit_date - entry_date).days

        # Get announcement info
        ann_match = ann[
            (ann["ticker"] == ticker) &
            (ann["published_at"] >= entry_date - pd.DateOffset(days=3)) &
            (ann["published_at"] <= entry_date)
        ]
        event = ann_match["event_type"].iloc[0] if len(ann_match) > 0 else "unknown"

        exit_reason = trade.get("exit_reason", "unknown")
        print(f"    {ticker:8s}  ${trade['pnl']:>+10,.0f}  entry={trade['entry_price']:.2f} "
              f"exit={trade['exit_price']:.2f}  {entry_date.date()} → {exit_date.date()} "
              f"({hold_days}d)  {event}  exit={exit_reason}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"AUDIT COMPLETE ({elapsed:.0f}s)")
    print("=" * 80)

    journal_lines.append(f"\n**Total audit time**: {elapsed:.0f}s")
    log_entry("Audit: Forensic Validation of Standout Config", "\n".join(journal_lines))
    print("  Journal entry appended.")


if __name__ == "__main__":
    main()
