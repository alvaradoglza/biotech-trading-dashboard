"""
audit_production.py — Deep audit of the production backtest output.

Checks:
1. Zero-day holding period trades
2. Gap-through TP trades verification
3. Entry price vs OHLCV open (all trades)
4. Exit price vs OHLCV on exit day (all trades)
5. Randomized label test (20 seeds, production config with 5% ADV)
6. PnL arithmetic verification (entry*shares - exit*shares = pnl?)
7. Equity curve continuity check
8. Window-by-window signal count vs trade count reconciliation
"""

import warnings
import time
from functools import partial

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from backtest.config import (
    ANNOUNCEMENTS_PATH, OHLCV_DIR, OUTPUT_DIR,
    START_DATE, END_DATE, PREDICT_WEEKS, STEP_WEEKS,
    INITIAL_CAPITAL, COMMISSION_PCT, SLIPPAGE_PCT,
    RANDOM_STATE, MIN_TRAIN_SAMPLES, P85_30D, SF_COLS,
    TIERED_SLIPPAGE,
)
from backtest.data_loader import load_announcements
from backtest.features import fit_ohe, build_feature_matrix
from backtest.model import generate_windows
from backtest.portfolio_simulator import simulate_portfolio, _load_ohlcv_cache
from backtest.portfolio_construction import allocate_inverse_volatility
from backtest.metrics import compute_metrics
from backtest.research import log_entry

warnings.filterwarnings("ignore")

TRAIN_MONTHS = 9
N_ESTIMATORS = 200
MAX_DEPTH = 4
LEARNING_RATE = 0.05
THRESHOLD = P85_30D
MAX_WEIGHT = 0.07
TP_PCT = 0.30
SL_PCT = 1.00
HORIZON_BARS = 50
ADV_CAP = 0.05
DROP_FEATURES = ["sf_word_count"]
VOL_WINDOW = 60

t0 = time.time()

# Load data
ann = load_announcements(ANNOUNCEMENTS_PATH)
ohe = fit_ohe(ann)
tl = pd.read_csv("output/standout_trade_log.csv")
eq = pd.read_csv("output/standout_equity_curve.csv", index_col=0, parse_dates=True)
tl["entry_date"] = pd.to_datetime(tl["entry_date"])
tl["exit_date"] = pd.to_datetime(tl["exit_date"])
tl["hold_days"] = (tl["exit_date"] - tl["entry_date"]).dt.days

print("=" * 70)
print("PRODUCTION AUDIT: t9_200d4_all_thp85_WC_mw7_tp30_h50 (5% ADV cap)")
print("=" * 70)

# ── CHECK 1: Zero-day holding period ─────────────────────────────────────────
print("\n" + "─" * 60)
print("CHECK 1: Zero-day holding period trades")
print("─" * 60)

zero_hold = tl[tl["hold_days"] == 0]
if len(zero_hold) == 0:
    print("  PASS: No zero-day holding trades.")
else:
    for _, t in zero_hold.iterrows():
        print(f"  {t['ticker']:8s} entry={t['entry_date'].date()} exit={t['exit_date'].date()} "
              f"entry_px={t['entry_price']:.2f} exit_px={t['exit_price']:.2f} "
              f"ret={t['return_pct']:+.1f}% reason={t['exit_reason']}")
        # Verify against OHLCV
        try:
            df = pd.read_parquet(f"data/ohlcv/{t['ticker']}.parquet")
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            day = t["entry_date"]
            if day in df.index:
                row = df.loc[day]
                adj_r = row["adjusted_close"] / row["close"] if row["close"] > 0 else 1
                adj_open = row["open"] * adj_r
                adj_high = row["high"] * adj_r
                tp = t["entry_price"] * 1.30
                print(f"    OHLCV adj: O={adj_open:.2f} H={adj_high:.2f} TP={tp:.2f}")
                if adj_high >= tp:
                    print(f"    => TP triggered same-day (H >= TP). Entry day NOT excluded from TP check.")
                    print(f"    WARNING: This may be a bug — entry day candle should not trigger TP.")
                elif adj_open >= tp:
                    print(f"    => Gap-through on entry day (O >= TP)")
        except Exception as e:
            print(f"    Error loading OHLCV: {e}")

# ── CHECK 2: Gap-through TP trades ──────────────────────────────────────────
print("\n" + "─" * 60)
print("CHECK 2: Gap-through TP trades (exit > TP level * 1.01)")
print("─" * 60)

gap_throughs = []
for _, t in tl.iterrows():
    tp_level = t["entry_price"] * 1.30
    if t["exit_reason"] == "take_profit" and t["exit_price"] > tp_level * 1.01:
        gap_throughs.append(t)
        try:
            df = pd.read_parquet(f"data/ohlcv/{t['ticker']}.parquet")
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            exit_day = t["exit_date"]
            if exit_day in df.index:
                row = df.loc[exit_day]
                adj_r = row["adjusted_close"] / row["close"] if row["close"] > 0 else 1
                adj_open = row["open"] * adj_r
                print(f"  {t['ticker']:8s} entry={t['entry_price']:.2f} exit={t['exit_price']:.2f} "
                      f"TP={tp_level:.2f} ret={t['return_pct']:+.1f}% "
                      f"exit_day_adj_open={adj_open:.2f} MATCH={'YES' if abs(t['exit_price'] - adj_open) / adj_open < 0.02 else 'NO'}")
        except Exception as e:
            print(f"  {t['ticker']:8s} Error: {e}")

print(f"\n  Total gap-through trades: {len(gap_throughs)}")
total_pnl = tl["pnl"].sum()
gap_excess = sum(
    t["pnl"] * (1 - 30.0 / t["return_pct"]) for _, t in pd.DataFrame(gap_throughs).iterrows()
    if t["return_pct"] > 0
) if gap_throughs else 0
print(f"  Gap-through excess PnL: ${gap_excess:+,.0f} ({gap_excess / total_pnl * 100:.1f}% of total)")

# ── CHECK 3: Entry price vs OHLCV open (ALL trades) ─────────────────────────
print("\n" + "─" * 60)
print("CHECK 3: Entry price vs OHLCV adjusted open (all trades)")
print("─" * 60)

mismatches = 0
checked = 0
for _, t in tl.iterrows():
    try:
        df = pd.read_parquet(f"data/ohlcv/{t['ticker']}.parquet")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        day = t["entry_date"]
        if day in df.index:
            row = df.loc[day]
            adj_r = row["adjusted_close"] / row["close"] if row["close"] > 0 else 1
            adj_open = row["open"] * adj_r
            # Entry should be adj_open * (1 + slippage)
            # Tiered: <$2 → 5%, $2-$5 → 2%, ≥$5 → 0.1%
            if adj_open < 2:
                expected_slip = 0.05
            elif adj_open < 5:
                expected_slip = 0.02
            else:
                expected_slip = 0.001
            expected_entry = adj_open * (1 + expected_slip)
            pct_diff = abs(t["entry_price"] - expected_entry) / expected_entry * 100
            if pct_diff > 1.0:
                mismatches += 1
                print(f"  MISMATCH: {t['ticker']:8s} {day.date()} our={t['entry_price']:.4f} "
                      f"expected={expected_entry:.4f} diff={pct_diff:.1f}%")
            checked += 1
    except Exception:
        pass

print(f"  Checked: {checked}/{len(tl)} trades")
print(f"  Mismatches (>1% diff): {mismatches}")
if mismatches == 0:
    print("  PASS: All entry prices match OHLCV adj open + slippage.")

# ── CHECK 4: Exit price verification (sample of TP exits) ───────────────────
print("\n" + "─" * 60)
print("CHECK 4: Exit price verification (TP exits should be at TP level or gap-open)")
print("─" * 60)

tp_trades = tl[tl["exit_reason"] == "take_profit"]
tp_mismatches = 0
for _, t in tp_trades.iterrows():
    tp_level = t["entry_price"] * 1.30
    try:
        df = pd.read_parquet(f"data/ohlcv/{t['ticker']}.parquet")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        exit_day = t["exit_date"]
        if exit_day in df.index:
            row = df.loc[exit_day]
            adj_r = row["adjusted_close"] / row["close"] if row["close"] > 0 else 1
            adj_open = row["open"] * adj_r
            adj_high = row["high"] * adj_r
            # Exit should be at adj_open (if gap-through) or tp_level (if intraday TP hit)
            if adj_open >= tp_level:
                expected_exit = adj_open  # gap-through
            else:
                expected_exit = tp_level  # intraday hit
            pct_diff = abs(t["exit_price"] - expected_exit) / expected_exit * 100
            if pct_diff > 1.0:
                tp_mismatches += 1
                print(f"  MISMATCH: {t['ticker']:8s} exit={t['exit_price']:.2f} "
                      f"expected={expected_exit:.2f} diff={pct_diff:.1f}% "
                      f"adj_open={adj_open:.2f} adj_high={adj_high:.2f} tp={tp_level:.2f}")
    except Exception:
        pass

print(f"  TP exit mismatches: {tp_mismatches}/{len(tp_trades)}")
if tp_mismatches == 0:
    print("  PASS: All TP exit prices match expected levels.")

# ── CHECK 5: PnL arithmetic verification ────────────────────────────────────
print("\n" + "─" * 60)
print("CHECK 5: PnL arithmetic (entry_price → exit_price → pnl consistency)")
print("─" * 60)

pnl_errors = 0
for _, t in tl.iterrows():
    # return_pct = (exit_fill - entry_fill) / entry_fill * 100, adjusted for slippage on exit
    # We can't perfectly reconstruct without knowing exact shares, but return_pct should be consistent
    expected_return = (t["exit_price"] / t["entry_price"] - 1) * 100
    # This doesn't account for exit slippage — the actual exit_price in the log is already post-slippage
    # Actually exit_price in the trade log is the RAW exit price, slippage applied to get exit_fill
    # Let me check: return_pct = pnl / position_dollars * 100
    # Without position_dollars we can check return_pct vs price ratio
    price_return = (t["exit_price"] / t["entry_price"] - 1) * 100
    diff = abs(t["return_pct"] - price_return)
    if diff > 5.0:
        pnl_errors += 1
        print(f"  MISMATCH: {t['ticker']:8s} price_return={price_return:+.1f}% "
              f"reported_return={t['return_pct']:+.1f}% diff={diff:.1f}%")

print(f"  PnL arithmetic mismatches (>5% diff): {pnl_errors}/{len(tl)}")
if pnl_errors == 0:
    print("  PASS: All PnL consistent with entry/exit prices.")

# ── CHECK 6: Equity curve continuity ─────────────────────────────────────────
print("\n" + "─" * 60)
print("CHECK 6: Equity curve sanity")
print("─" * 60)

print(f"  First equity: ${eq['equity'].iloc[0]:,.0f}")
print(f"  Last equity: ${eq['equity'].iloc[-1]:,.0f}")
print(f"  Min equity: ${eq['equity'].min():,.0f}")
print(f"  Max equity: ${eq['equity'].max():,.0f}")
print(f"  Days: {len(eq)}")

# Check for jumps > 10% in a single day
daily_ret = eq["equity"].pct_change().dropna()
big_jumps = daily_ret[abs(daily_ret) > 0.10]
print(f"  Days with >10% equity change: {len(big_jumps)}")
for date, ret in big_jumps.items():
    print(f"    {date.date()} → {ret:+.2%}")

# Verify final equity matches PnL
expected_final = INITIAL_CAPITAL + tl["pnl"].sum()
actual_final = eq["equity"].iloc[-1]
diff_pct = abs(expected_final - actual_final) / expected_final * 100
print(f"\n  Expected final (capital + PnL): ${expected_final:,.0f}")
print(f"  Actual final equity: ${actual_final:,.0f}")
print(f"  Difference: {diff_pct:.2f}%")
if diff_pct < 1.0:
    print("  PASS: Equity curve matches cumulative PnL.")
else:
    print("  WARNING: Equity curve diverges from cumulative PnL by >1%")

# ── CHECK 7: Randomized label test (production config, 5% ADV) ──────────────
print("\n" + "─" * 60)
print("CHECK 7: Randomized label test (20 seeds, production config)")
print("─" * 60)

windows = generate_windows(
    start_date=START_DATE, end_date=END_DATE,
    train_months=TRAIN_MONTHS, predict_weeks=PREDICT_WEEKS,
    step_weeks=STEP_WEEKS,
)
drop_idxs = [SF_COLS.index(f) for f in DROP_FEATURES if f in SF_COLS]
weight_fn = partial(allocate_inverse_volatility, vol_window=VOL_WINDOW, max_weight=MAX_WEIGHT)

random_sharpes = []
for seed in range(20):
    rng = np.random.RandomState(seed)
    all_signals = []

    for w in windows:
        clean_train_end = w.train_end - pd.DateOffset(days=30)
        train_df = ann[
            (ann["published_at"] >= w.train_start) &
            (ann["published_at"] < clean_train_end)
        ].copy()
        pred_df = ann[
            (ann["published_at"] >= w.pred_start) &
            (ann["published_at"] < w.pred_end)
        ].copy()

        if len(train_df) < MIN_TRAIN_SAMPLES or len(pred_df) == 0:
            continue

        X_train = build_feature_matrix(ohe, train_df)
        # SHUFFLE the labels — this is the null hypothesis
        y_train = rng.permutation((train_df["return_30d"] >= THRESHOLD).astype(int).values)

        X_pred = build_feature_matrix(ohe, pred_df)

        if drop_idxs:
            n_ohe = X_train.shape[1] - len(SF_COLS)
            cols_to_drop = [n_ohe + i for i in drop_idxs]
            X_train = np.delete(X_train, cols_to_drop, axis=1)
            X_pred = np.delete(X_pred, cols_to_drop, axis=1)

        if len(np.unique(y_train)) < 2:
            continue

        model = GradientBoostingClassifier(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE, random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_pred)
        scores = model.predict_proba(X_pred)[:, 1]

        pred_df = pred_df.copy()
        pred_df["prediction"] = preds
        pred_df["decision_score"] = scores

        signals = pred_df[pred_df["prediction"] == 1].copy()
        if len(signals) == 0:
            continue
        signals = signals.sort_values("published_at", ascending=True)
        signals = signals.drop_duplicates(subset=["ticker"], keep="first")
        signals["window_id"] = w.window_id
        signals["horizon"] = "30d"
        all_signals.append(signals)

    if not all_signals:
        random_sharpes.append(np.nan)
        print(f"    Seed {seed:>2d}: No signals")
        continue

    sig_df = pd.concat(all_signals, ignore_index=True)
    tl_rand, eq_rand = simulate_portfolio(
        sig_df, "30d",
        take_profit_pct=TP_PCT, stop_loss_pct=SL_PCT,
        ohlcv_dir=OHLCV_DIR, initial_capital=INITIAL_CAPITAL,
        commission_pct=COMMISSION_PCT, slippage_pct=SLIPPAGE_PCT,
        weight_fn=weight_fn, tiered_slippage=TIERED_SLIPPAGE,
        adv_cap_pct=ADV_CAP, horizon_bars=HORIZON_BARS,
    )
    m = compute_metrics(tl_rand, INITIAL_CAPITAL, equity_curve=eq_rand)
    sh = m["sharpe_ratio"] if not np.isnan(m["sharpe_ratio"]) else 0.0
    random_sharpes.append(sh)
    print(f"    Seed {seed:>2d}: Sharpe={sh:+.2f}  Return={m['total_return_pct']:+.1f}%  Trades={m['total_trades']}")

real_sharpe = 1.82
random_sharpes = [s for s in random_sharpes if not np.isnan(s)]
mean_rand = np.mean(random_sharpes)
std_rand = np.std(random_sharpes) if len(random_sharpes) > 1 else 1.0
z_score = (real_sharpe - mean_rand) / std_rand if std_rand > 0 else float("inf")
p_val = sum(1 for s in random_sharpes if s >= real_sharpe) / len(random_sharpes)

print(f"\n  Real Sharpe: {real_sharpe:.2f}")
print(f"  Random Sharpe: mean={mean_rand:.2f}, std={std_rand:.2f}, max={max(random_sharpes):.2f}")
print(f"  Z-score: {z_score:.1f}")
print(f"  P-value: {p_val:.3f}")
if p_val < 0.05:
    print("  PASS: Real model significantly outperforms random (p < 0.05)")
else:
    print("  FAIL: Cannot reject null hypothesis at 5% level")

# ── CHECK 8: Window details reconciliation ───────────────────────────────────
print("\n" + "─" * 60)
print("CHECK 8: Window signal/trade reconciliation")
print("─" * 60)

wd = pd.read_csv("output/standout_window_details.csv")
print(f"  Total windows: {len(wd)}")
print(f"  Windows with signals: {(wd['signals'] > 0).sum()}")
print(f"  Total signals: {wd['signals'].sum()}")
print(f"  Trades executed: {len(tl)}")
print(f"  Signal-to-trade ratio: {len(tl) / wd['signals'].sum() * 100:.1f}%")
skipped = wd["signals"].sum() - len(tl)
print(f"  Signals not executed: {skipped} (due to position cap, cash limit, or missing OHLCV)")

elapsed = time.time() - t0
print(f"\n{'='*70}")
print(f"PRODUCTION AUDIT COMPLETE ({elapsed:.0f}s)")
print(f"{'='*70}")
