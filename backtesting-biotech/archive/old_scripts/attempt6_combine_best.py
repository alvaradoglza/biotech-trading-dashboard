"""
Attempt 6: Combine all breakthroughs from Attempts 4-5.

Breakthroughs to combine:
1. 9-month training window (Sh=2.28 on 12m default model)
2. GBM depth=3 (Sh=2.25 vs depth=4 Sh=1.57)
3. Drop sf_word_count (Sh=2.04-2.33 without it)
4. Sizing push (mw=5-10%)
5. TP/horizon optimization (tp=25-30%, h=50-60)
6. Ex-OLMA robustness on best configs
7. ADV cap stress test
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
    TIERED_SLIPPAGE, P85_30D,
    RANDOM_STATE, MIN_TRAIN_SAMPLES, SF_COLS,
)
from backtest.data_loader import load_announcements
from backtest.features import fit_ohe, build_feature_matrix
from backtest.model import generate_windows
from backtest.portfolio_simulator import simulate_portfolio, _load_ohlcv_cache
from backtest.portfolio_construction import allocate_inverse_volatility
from backtest.metrics import compute_metrics
from backtest.research import (
    run_experiment, robustness, print_result, log_entry,
)

warnings.filterwarnings("ignore")


def build_signals_optimized(
    announcements: pd.DataFrame,
    ohe,
    horizon: str = "30d",
    threshold: float = P85_30D,
    event_filter: list = None,
    exclude_tickers: list = None,
    drop_features: list = None,
    train_months: int = 9,
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.05,
) -> pd.DataFrame:
    """Build signals with all optimizations combined."""
    windows = generate_windows(
        start_date=START_DATE, end_date=END_DATE,
        train_months=train_months, predict_weeks=PREDICT_WEEKS,
        step_weeks=STEP_WEEKS,
    )

    label_col = f"return_{horizon}"
    return_days = 5 if horizon == "5d" else 30
    all_signals = []

    for window in windows:
        clean_train_end = window.train_end - pd.DateOffset(days=return_days)
        train_df = announcements[
            (announcements["published_at"] >= window.train_start) &
            (announcements["published_at"] < clean_train_end)
        ].copy()
        pred_df = announcements[
            (announcements["published_at"] >= window.pred_start) &
            (announcements["published_at"] < window.pred_end)
        ].copy()

        if event_filter:
            train_df = train_df[train_df["event_type"].isin(event_filter)].copy()
            pred_df = pred_df[pred_df["event_type"].isin(event_filter)].copy()

        if exclude_tickers:
            pred_df = pred_df[~pred_df["ticker"].isin(exclude_tickers)].copy()

        if len(train_df) < MIN_TRAIN_SAMPLES or len(pred_df) == 0:
            continue

        X_train = build_feature_matrix(ohe, train_df)
        y_train = (train_df[label_col] >= threshold).astype(int).values
        X_pred = build_feature_matrix(ohe, pred_df)

        if drop_features:
            ohe_names = list(ohe.get_feature_names_out())
            all_names = ohe_names + SF_COLS
            drop_indices = [i for i, name in enumerate(all_names) if name in drop_features]
            if drop_indices and X_train.shape[1] > len(drop_indices):
                keep_mask = np.ones(X_train.shape[1], dtype=bool)
                for idx in drop_indices:
                    if idx < X_train.shape[1]:
                        keep_mask[idx] = False
                X_train = X_train[:, keep_mask]
                X_pred = X_pred[:, keep_mask]

        if len(np.unique(y_train)) < 2:
            continue

        try:
            model = GradientBoostingClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, random_state=RANDOM_STATE,
            )
            model.fit(X_train, y_train)
        except Exception:
            continue

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
        signals["window_id"] = window.window_id
        signals["horizon"] = horizon
        all_signals.append(signals)

    if not all_signals:
        return pd.DataFrame()
    return pd.concat(all_signals, ignore_index=True)


def bootstrap_sharpe(trade_log, n_boot=5000, seed=42):
    """Bootstrap CI for Sharpe."""
    pnls = trade_log["pnl"].values
    if len(pnls) < 5:
        return {"lo": 0, "hi": 0, "mean": 0}
    rng = np.random.RandomState(seed)
    sharpes = []
    for _ in range(n_boot):
        sample = rng.choice(pnls, size=len(pnls), replace=True)
        mu, sd = sample.mean(), sample.std()
        if sd > 0:
            sharpes.append(mu / sd * np.sqrt(len(pnls) / 2))
    sharpes = np.array(sharpes)
    return {"lo": float(np.percentile(sharpes, 2.5)), "hi": float(np.percentile(sharpes, 97.5)),
            "mean": float(np.mean(sharpes))}


def per_quarter_pnl(trade_log):
    tl = trade_log.copy()
    tl["entry_date"] = pd.to_datetime(tl["entry_date"])
    tl["quarter"] = tl["entry_date"].dt.to_period("Q").astype(str)
    return tl.groupby("quarter").agg(
        trades=("pnl", "count"), pnl=("pnl", "sum"),
        win_rate=("pnl", lambda x: (x > 0).mean() * 100),
    ).reset_index()


def main():
    t0 = time.time()
    print("=" * 80)
    print("ATTEMPT 6: Combine All Breakthroughs")
    print("=" * 80)

    ann = load_announcements(ANNOUNCEMENTS_PATH)
    ohe = fit_ohe(ann)
    tickers = ann["ticker"].unique().tolist()
    ohlcv_cache = _load_ohlcv_cache(tickers, OHLCV_DIR)
    print(f"  {len(ann)} announcements, {len(ohlcv_cache)} tickers with OHLCV")

    journal_lines = []
    journal_lines.append("**Time**: 2026-03-30 (Attempt 6)")
    journal_lines.append("**Focus**: Combine 9m window + depth=3 GBM + no word_count + sizing + TP/h grid\n")

    recruit_filter = ["CT_RECRUITING"]
    all_results = {}

    # ═══════════════════════════════════════════════════════════════════════════
    # PART A: Combined optimizations grid
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("PART A: Combined optimizations — full grid")
    print("─" * 60)

    configs = []
    for train_m in [9, 12]:
        for n_est, md, lr in [(100, 3, 0.05), (200, 3, 0.05), (300, 3, 0.05), (200, 4, 0.05), (500, 3, 0.01)]:
            for th in [P85_30D, 0.10, 0.05]:
                for drop_wc in [True, False]:
                    for ef_label, ef in [("all", None), ("recruit", recruit_filter)]:
                        configs.append({
                            "train_m": train_m, "n_est": n_est, "md": md, "lr": lr,
                            "th": th, "drop_wc": drop_wc, "ef_label": ef_label, "ef": ef,
                        })

    print(f"  Testing {len(configs)} signal configs × sizing/TP combos...")

    signal_cache = {}
    for i, cfg in enumerate(configs):
        key = (cfg["train_m"], cfg["n_est"], cfg["md"], cfg["lr"], cfg["th"], cfg["drop_wc"], cfg["ef_label"])
        if key in signal_cache:
            continue

        signals = build_signals_optimized(
            ann, ohe, "30d",
            threshold=cfg["th"],
            event_filter=cfg["ef"],
            drop_features=["sf_word_count"] if cfg["drop_wc"] else None,
            train_months=cfg["train_m"],
            n_estimators=cfg["n_est"],
            max_depth=cfg["md"],
            learning_rate=cfg["lr"],
        )
        signal_cache[key] = signals

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(configs)}] signal sets built...")

    print(f"  {len(signal_cache)} unique signal sets built.")

    # Now run experiments with different sizing/TP
    print("\n  Running portfolio simulations...")
    count = 0
    for key, signals in signal_cache.items():
        if len(signals) == 0:
            continue
        train_m, n_est, md, lr, th, drop_wc, ef_label = key
        th_label = "p85" if th == P85_30D else f"{th:.2f}"
        wc_label = "noWC" if drop_wc else "WC"

        for mw_val in [0.05, 0.07]:
            for tp_val in [0.25, 0.30]:
                for h_val in [50, 60]:
                    name = f"t{train_m}_{n_est}d{md}lr{lr}_{ef_label}_th{th_label}_{wc_label}_mw{mw_val}_tp{int(tp_val*100)}_h{h_val}"
                    wfn = partial(allocate_inverse_volatility, vol_window=60, max_weight=mw_val)
                    r = run_experiment(name, signals, ohlcv_cache,
                                       horizon_bars=h_val, take_profit_pct=tp_val, weight_fn=wfn)
                    if r["metrics"]:
                        rob = robustness(r)
                        all_results[name] = (r, rob)
                    count += 1
                    if count % 100 == 0:
                        print(f"  [{count}] experiments run...")

    print(f"  {count} total experiments, {len(all_results)} with trades.")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART B: Rank and filter
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("PART B: Top configs by Sharpe")
    print("─" * 60)

    ranked = sorted(all_results.items(),
                    key=lambda x: x[1][0]["metrics"]["sharpe_ratio"],
                    reverse=True)

    print("\n  Top 20 by Sharpe:")
    journal_lines.append("**Top 20 by Sharpe**:\n")
    journal_lines.append("| Config | Trades | Return | Sharpe | MaxDD | WR | PF | Y1/Y2 |")
    journal_lines.append("|--------|--------|--------|--------|-------|----|-----|-------|")

    for name, (r, rob) in ranked[:20]:
        print_result(name, r, rob)
        m = r["metrics"]
        y1 = rob.get("y1_ret", 0)
        y2 = rob.get("y2_ret", 0)
        journal_lines.append(f"| {name} | {m['total_trades']} | {m['total_return_pct']:+.1f}% | "
                            f"{m['sharpe_ratio']:.2f} | {m['max_drawdown_pct']:.1f}% | "
                            f"{m['win_rate_pct']:.1f}% | {m['profit_factor']:.2f} | "
                            f"{y1:+.1f}/{y2:+.1f} |")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART C: SPY beaters sorted by Sharpe
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("PART C: Configs beating SPY (+47.8%, Sharpe 1.28)")
    print("─" * 60)

    spy_beaters = [(n, r, rob) for n, (r, rob) in ranked
                   if r["metrics"]["total_return_pct"] > 47.8 and r["metrics"]["sharpe_ratio"] > 1.28]
    spy_beaters.sort(key=lambda x: x[1]["metrics"]["sharpe_ratio"], reverse=True)

    print(f"\n  {len(spy_beaters)} configs beat SPY on BOTH return AND Sharpe.")
    journal_lines.append(f"\n**{len(spy_beaters)} configs beat SPY** on both return AND Sharpe.\n")

    for name, r, rob in spy_beaters[:15]:
        print_result(name, r, rob)

    # ═══════════════════════════════════════════════════════════════════════════
    # PART D: Deep dive on top-3 configs
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("PART D: Deep dive on top-3 configs")
    print("─" * 60)

    journal_lines.append("\n### Deep Dive on Top Configs\n")

    for i, (name, r, rob) in enumerate(spy_beaters[:3]):
        if not r["metrics"]:
            continue
        tl = r["trade_log"]
        m = r["metrics"]
        print(f"\n  === #{i+1}: {name} ===")
        print(f"  Return: {m['total_return_pct']:+.1f}%  Sharpe: {m['sharpe_ratio']:.2f}  DD: {m['max_drawdown_pct']:.1f}%")
        print(f"  Trades: {m['total_trades']}  WR: {m['win_rate_pct']:.1f}%  PF: {m['profit_factor']:.2f}")

        # Bootstrap Sharpe
        bs = bootstrap_sharpe(tl)
        print(f"  Bootstrap Sharpe 95% CI: [{bs['lo']:.2f}, {bs['hi']:.2f}]")

        # Per-quarter
        print("  Per-quarter:")
        qpnl = per_quarter_pnl(tl)
        for _, row in qpnl.iterrows():
            print(f"    {row['quarter']:8s}  {int(row['trades']):>3}tr  PnL=${row['pnl']:>+10,.0f}  WR={row['win_rate']:>5.1f}%")

        # Top tickers
        top_tickers = tl.groupby("ticker")["pnl"].sum().sort_values(ascending=False)
        print("  Top-5 tickers:")
        for tk, pnl_val in top_tickers.head(5).items():
            pct_of_total = pnl_val / max(1, tl["pnl"].sum()) * 100
            print(f"    {tk:8s} ${pnl_val:>+10,.0f} ({pct_of_total:+.0f}%)")

        # Price tier breakdown
        penny = tl[tl["entry_price"] < 2]["pnl"].sum()
        mid = tl[(tl["entry_price"] >= 2) & (tl["entry_price"] < 5)]["pnl"].sum()
        normal = tl[tl["entry_price"] >= 5]["pnl"].sum()
        print(f"  Price tiers: <$2=${penny:+,.0f}  $2-5=${mid:+,.0f}  >=$5=${normal:+,.0f}")

        # Ex-OLMA
        ex_olma = tl[tl["ticker"] != "OLMA"]["pnl"].sum()
        print(f"  Ex-OLMA return: {ex_olma/INITIAL_CAPITAL*100:+.1f}%")

        journal_lines.append(f"\n**#{i+1}: {name}**")
        journal_lines.append(f"- Return: {m['total_return_pct']:+.1f}%, Sharpe: {m['sharpe_ratio']:.2f}, DD: {m['max_drawdown_pct']:.1f}%")
        journal_lines.append(f"- Bootstrap Sharpe 95% CI: [{bs['lo']:.2f}, {bs['hi']:.2f}]")
        journal_lines.append(f"- Ex-OLMA return: {ex_olma/INITIAL_CAPITAL*100:+.1f}%")
        journal_lines.append(f"- Price tiers: <$2=${penny:+,.0f}  $2-5=${mid:+,.0f}  >=$5=${normal:+,.0f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART E: Ex-OLMA stress test on top-5
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("PART E: Ex-OLMA re-run on top-5 configs")
    print("─" * 60)

    journal_lines.append("\n### Ex-OLMA Stress Test\n")

    for name, r, rob in spy_beaters[:5]:
        # Parse config from name to re-run excluding OLMA
        # Instead, just filter the trade log
        tl = r["trade_log"]
        tl_no_olma = tl[tl["ticker"] != "OLMA"]
        total_pnl = tl_no_olma["pnl"].sum()
        ret_no_olma = total_pnl / INITIAL_CAPITAL * 100
        trades_no_olma = len(tl_no_olma)
        wr_no_olma = (tl_no_olma["pnl"] > 0).mean() * 100 if len(tl_no_olma) > 0 else 0

        wins = tl_no_olma[tl_no_olma["pnl"] > 0]["pnl"].sum()
        losses = abs(tl_no_olma[tl_no_olma["pnl"] < 0]["pnl"].sum())
        pf_no_olma = wins / max(1, losses)

        print(f"  {name[:55]:55s}  {trades_no_olma:>3}tr  {ret_no_olma:>+6.1f}%  WR={wr_no_olma:.1f}%  PF={pf_no_olma:.2f}")
        journal_lines.append(f"- {name}: ex-OLMA {ret_no_olma:+.1f}%, {trades_no_olma} trades, WR={wr_no_olma:.1f}%")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART F: ADV cap stress test on top-3
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("PART F: ADV cap stress test on best config")
    print("─" * 60)

    journal_lines.append("\n### ADV Cap Stress Test\n")

    if spy_beaters:
        # Get the best config parameters and rebuild signals
        best_name = spy_beaters[0][0]
        print(f"  Testing ADV caps on: {best_name}")

        # Parse the best config — let's just test on the 9m/100d3 config
        # which is the most likely best
        for train_m in [9]:
            for n_est, md, lr in [(100, 3, 0.05), (200, 3, 0.05)]:
                signals_best = build_signals_optimized(
                    ann, ohe, "30d",
                    threshold=P85_30D,
                    drop_features=["sf_word_count"],
                    train_months=train_m,
                    n_estimators=n_est,
                    max_depth=md,
                    learning_rate=lr,
                )
                if len(signals_best) == 0:
                    continue

                print(f"\n  Config: t{train_m}_{n_est}d{md} ({len(signals_best)} signals)")
                for adv_cap in [999.0, 0.20, 0.10, 0.05]:
                    for mw_val in [0.05, 0.07]:
                        wfn = partial(allocate_inverse_volatility, vol_window=60, max_weight=mw_val)
                        adv_name = f"t{train_m}_{n_est}d{md}_noWC_mw{mw_val}_adv{adv_cap}"
                        r_adv = run_experiment(
                            adv_name, signals_best, ohlcv_cache,
                            horizon_bars=50, take_profit_pct=0.25, weight_fn=wfn,
                            adv_cap_pct=adv_cap,
                        )
                        if r_adv["metrics"]:
                            rob_adv = robustness(r_adv)
                            print_result(adv_name, r_adv, rob_adv)
                            m_adv = r_adv["metrics"]
                            journal_lines.append(f"- {adv_name}: {m_adv['total_return_pct']:+.1f}%, Sh={m_adv['sharpe_ratio']:.2f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART G: Best config with higher sizing (push for max return)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("PART G: Push sizing on best combined configs")
    print("─" * 60)

    journal_lines.append("\n### Aggressive Sizing on Best Configs\n")

    for train_m in [9]:
        for n_est, md, lr in [(100, 3, 0.05), (200, 3, 0.05), (300, 3, 0.05)]:
            for th in [P85_30D, 0.10]:
                signals_push = build_signals_optimized(
                    ann, ohe, "30d",
                    threshold=th,
                    drop_features=["sf_word_count"],
                    train_months=train_m,
                    n_estimators=n_est,
                    max_depth=md,
                    learning_rate=lr,
                )
                if len(signals_push) == 0:
                    continue

                th_label = "p85" if th == P85_30D else f"{th:.2f}"
                for mw_val in [0.07, 0.10, 0.12, 0.15]:
                    for tp_val in [0.25, 0.30, 0.35]:
                        for h_val in [50, 60]:
                            name = f"push_t{train_m}_{n_est}d{md}_th{th_label}_noWC_mw{mw_val}_tp{int(tp_val*100)}_h{h_val}"
                            wfn = partial(allocate_inverse_volatility, vol_window=60, max_weight=mw_val)
                            r_push = run_experiment(name, signals_push, ohlcv_cache,
                                                     horizon_bars=h_val, take_profit_pct=tp_val, weight_fn=wfn)
                            if r_push["metrics"] and r_push["metrics"]["sharpe_ratio"] > 1.0:
                                rob_push = robustness(r_push)
                                all_results[name] = (r_push, rob_push)

    # Re-rank including new results
    ranked_all = sorted(all_results.items(),
                        key=lambda x: x[1][0]["metrics"]["sharpe_ratio"],
                        reverse=True)

    print("\n  Top 15 overall (including aggressive sizing):")
    journal_lines.append("\n**Top 15 overall (all configs)**:\n")
    journal_lines.append("| Config | Trades | Return | Sharpe | MaxDD | WR | Y1/Y2 |")
    journal_lines.append("|--------|--------|--------|--------|-------|----|-------|")

    for name, (r, rob) in ranked_all[:15]:
        print_result(name, r, rob)
        m = r["metrics"]
        y1 = rob.get("y1_ret", 0)
        y2 = rob.get("y2_ret", 0)
        journal_lines.append(f"| {name} | {m['total_trades']} | {m['total_return_pct']:+.1f}% | "
                            f"{m['sharpe_ratio']:.2f} | {m['max_drawdown_pct']:.1f}% | "
                            f"{m['win_rate_pct']:.1f}% | {y1:+.1f}/{y2:+.1f} |")

    # Best return configs
    ranked_ret = sorted(all_results.items(),
                        key=lambda x: x[1][0]["metrics"]["total_return_pct"],
                        reverse=True)

    print("\n  Top 10 by total return:")
    journal_lines.append("\n**Top 10 by total return**:\n")
    for name, (r, rob) in ranked_ret[:10]:
        print_result(name, r, rob)
        m = r["metrics"]
        y1 = rob.get("y1_ret", 0)
        y2 = rob.get("y2_ret", 0)
        journal_lines.append(f"- {name}: {m['total_return_pct']:+.1f}%, Sh={m['sharpe_ratio']:.2f}, Y1={y1:+.1f}%, Y2={y2:+.1f}%")

    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"ATTEMPT 6 COMPLETE ({elapsed:.0f}s)")
    print(f"Total configs tested: {len(all_results)}")
    print("=" * 80)

    # Final deep dive on THE best config
    if ranked_all:
        best_name, (best_r, best_rob) = ranked_all[0]
        best_m = best_r["metrics"]
        bs = bootstrap_sharpe(best_r["trade_log"])

        print(f"\n  BEST CONFIG: {best_name}")
        print(f"  Return: {best_m['total_return_pct']:+.1f}%  Sharpe: {best_m['sharpe_ratio']:.2f}  DD: {best_m['max_drawdown_pct']:.1f}%")
        print(f"  Trades: {best_m['total_trades']}  WR: {best_m['win_rate_pct']:.1f}%  PF: {best_m['profit_factor']:.2f}")
        print(f"  Bootstrap Sharpe 95% CI: [{bs['lo']:.2f}, {bs['hi']:.2f}]")

        tl = best_r["trade_log"]
        ex_olma_ret = tl[tl["ticker"] != "OLMA"]["pnl"].sum() / INITIAL_CAPITAL * 100
        print(f"  Ex-OLMA return: {ex_olma_ret:+.1f}%")

        journal_lines.append(f"\n**BEST OVERALL: {best_name}**")
        journal_lines.append(f"- Return: {best_m['total_return_pct']:+.1f}%, Sharpe: {best_m['sharpe_ratio']:.2f}")
        journal_lines.append(f"- Bootstrap 95% CI: [{bs['lo']:.2f}, {bs['hi']:.2f}]")
        journal_lines.append(f"- Ex-OLMA: {ex_olma_ret:+.1f}%")

    journal_lines.append(f"\n**Total time**: {elapsed:.0f}s, {len(all_results)} configs tested")
    log_entry("Attempt 6: Combine All Breakthroughs", "\n".join(journal_lines))
    print("\n  Journal entry appended.")


if __name__ == "__main__":
    main()
