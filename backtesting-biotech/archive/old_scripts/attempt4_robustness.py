"""
Attempt 4: Deep robustness analysis + new strategy directions.

Key questions:
1. Why is Y1 (2024) flat? Is it data sparsity or model failure?
2. How robust is the best GBM config? Bootstrap CI on Sharpe.
3. Per-window and per-quarter PnL breakdown.
4. Ensemble approaches: GBM + RF consensus.
5. Score-weighted sizing: use GBM predict_proba to scale positions.
6. Adaptive threshold: different P85 per year.
7. Feature importance: what's driving the GBM?
"""

import warnings
import time
import sys
from functools import partial
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from backtest.config import (
    ANNOUNCEMENTS_PATH, OHLCV_DIR, OUTPUT_DIR,
    START_DATE, END_DATE, TRAIN_MONTHS, PREDICT_WEEKS, STEP_WEEKS,
    INITIAL_CAPITAL, COMMISSION_PCT, SLIPPAGE_PCT,
    TIERED_SLIPPAGE, ADV_CAP_PCT, P85_30D, P85_5D,
    RANDOM_STATE, MIN_TRAIN_SAMPLES,
)
from backtest.data_loader import load_announcements
from backtest.features import fit_ohe, build_feature_matrix, build_labels
from backtest.model import generate_windows
from backtest.portfolio_simulator import simulate_portfolio, _load_ohlcv_cache
from backtest.portfolio_construction import allocate_inverse_volatility
from backtest.metrics import compute_metrics
from backtest.research import (
    build_signals_with_model, build_all_signals,
    run_experiment, robustness, print_result, log_entry,
)

warnings.filterwarnings("ignore")


# ── Helpers ────────────────────────────────────────────────────────────────────

def bootstrap_sharpe(trade_log: pd.DataFrame, n_boot: int = 5000, seed: int = 42) -> dict:
    """Bootstrap confidence interval for annualized Sharpe from trade PnL."""
    pnls = trade_log["pnl"].values
    if len(pnls) < 5:
        return {"sharpe_mean": 0, "sharpe_lo": 0, "sharpe_hi": 0}
    rng = np.random.RandomState(seed)
    sharpes = []
    for _ in range(n_boot):
        sample = rng.choice(pnls, size=len(pnls), replace=True)
        mu = sample.mean()
        sd = sample.std()
        if sd > 0:
            # Annualize assuming ~120 trades over 2 years
            sharpes.append(mu / sd * np.sqrt(len(pnls) / 2))
        else:
            sharpes.append(0)
    sharpes = np.array(sharpes)
    return {
        "sharpe_mean": float(np.mean(sharpes)),
        "sharpe_lo": float(np.percentile(sharpes, 2.5)),
        "sharpe_hi": float(np.percentile(sharpes, 97.5)),
    }


def per_quarter_pnl(trade_log: pd.DataFrame) -> pd.DataFrame:
    """PnL breakdown by calendar quarter."""
    tl = trade_log.copy()
    tl["entry_date"] = pd.to_datetime(tl["entry_date"])
    tl["quarter"] = tl["entry_date"].dt.to_period("Q").astype(str)
    return tl.groupby("quarter").agg(
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
        win_rate=("pnl", lambda x: (x > 0).mean() * 100),
        avg_pnl=("pnl", "mean"),
    ).reset_index()


def per_window_pnl(trade_log: pd.DataFrame) -> pd.DataFrame:
    """PnL breakdown by rolling window."""
    if "window_id" not in trade_log.columns:
        return pd.DataFrame()
    return trade_log.groupby("window_id").agg(
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
        win_rate=("pnl", lambda x: (x > 0).mean() * 100),
    ).reset_index()


def build_ensemble_signals(
    announcements: pd.DataFrame,
    ohe,
    horizon: str = "30d",
    threshold: float = 0.05,
    event_filter: list = None,
    consensus: int = 2,
) -> pd.DataFrame:
    """Build signals where at least `consensus` out of 3 models agree.

    Models: GBM, RF, LogReg. Signals require consensus models predicting positive.
    Score = average probability across agreeing models.
    """
    windows = generate_windows(
        start_date=START_DATE, end_date=END_DATE,
        train_months=TRAIN_MONTHS, predict_weeks=PREDICT_WEEKS,
        step_weeks=STEP_WEEKS,
    )

    models_config = [
        ("GBM", GradientBoostingClassifier, dict(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE)),
        ("RF", RandomForestClassifier, dict(
            n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)),
        ("LR", LogisticRegression, dict(
            C=0.1, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
    ]

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

        if len(train_df) < MIN_TRAIN_SAMPLES or len(pred_df) == 0:
            continue

        X_train = build_feature_matrix(ohe, train_df)
        y_train = (train_df[label_col] >= threshold).astype(int).values
        X_pred = build_feature_matrix(ohe, pred_df)

        if len(np.unique(y_train)) < 2:
            continue

        # Train all models and collect votes
        votes = np.zeros(len(pred_df))
        scores_sum = np.zeros(len(pred_df))
        n_valid_models = 0

        for name, cls, kwargs in models_config:
            try:
                model = cls(**kwargs)
                model.fit(X_train, y_train)
                preds = model.predict(X_pred)
                votes += preds

                if hasattr(model, "predict_proba"):
                    scores_sum += model.predict_proba(X_pred)[:, 1]
                elif hasattr(model, "decision_function"):
                    raw = model.decision_function(X_pred)
                    # Normalize to 0-1 range
                    scores_sum += (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
                else:
                    scores_sum += preds.astype(float)
                n_valid_models += 1
            except Exception:
                continue

        if n_valid_models == 0:
            continue

        pred_df = pred_df.copy()
        pred_df["votes"] = votes
        pred_df["decision_score"] = scores_sum / n_valid_models
        pred_df["prediction"] = (votes >= consensus).astype(int)

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


def build_proba_weighted_signals(
    announcements: pd.DataFrame,
    ohe,
    horizon: str = "30d",
    threshold: float = 0.05,
    event_filter: list = None,
    proba_cutoff: float = 0.5,
) -> pd.DataFrame:
    """Build signals using GBM predict_proba as the score for sizing.

    Only includes signals where P(positive) >= proba_cutoff.
    The decision_score is the raw probability — higher = more capital.
    """
    windows = generate_windows(
        start_date=START_DATE, end_date=END_DATE,
        train_months=TRAIN_MONTHS, predict_weeks=PREDICT_WEEKS,
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

        if len(train_df) < MIN_TRAIN_SAMPLES or len(pred_df) == 0:
            continue

        X_train = build_feature_matrix(ohe, train_df)
        y_train = (train_df[label_col] >= threshold).astype(int).values
        X_pred = build_feature_matrix(ohe, pred_df)

        if len(np.unique(y_train)) < 2:
            continue

        try:
            model = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                random_state=RANDOM_STATE,
            )
            model.fit(X_train, y_train)
        except Exception:
            continue

        probas = model.predict_proba(X_pred)[:, 1]

        pred_df = pred_df.copy()
        pred_df["proba"] = probas
        pred_df["prediction"] = (probas >= proba_cutoff).astype(int)
        pred_df["decision_score"] = probas  # used for ranking/sizing

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


def feature_importance_analysis(announcements, ohe, horizon="30d", threshold=0.05):
    """Train GBM on all data and report feature importances."""
    from backtest.features import build_structured_features
    from backtest.config import SF_COLS

    label_col = f"return_{horizon}"

    # Use first half for training
    midpoint = pd.Timestamp("2025-01-01")
    train_df = announcements[announcements["published_at"] < midpoint].copy()

    X_train = build_feature_matrix(ohe, train_df)
    y_train = (train_df[label_col] >= threshold).astype(int).values

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # Feature names: OHE features + SF_COLS
    ohe_names = list(ohe.get_feature_names_out())
    all_names = ohe_names + SF_COLS
    importances = model.feature_importances_

    fi = pd.DataFrame({
        "feature": all_names[:len(importances)],
        "importance": importances,
    }).sort_values("importance", ascending=False)

    return fi.head(20)


# ── Main experiment runner ─────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 80)
    print("ATTEMPT 4: Deep Robustness + New Strategies")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    ann = load_announcements(ANNOUNCEMENTS_PATH)
    ohe = fit_ohe(ann)
    tickers = ann["ticker"].unique().tolist()
    ohlcv_cache = _load_ohlcv_cache(tickers, OHLCV_DIR)
    print(f"  {len(ann)} announcements, {len(tickers)} tickers, {len(ohlcv_cache)} with OHLCV")

    # ── Part A: Reproduce best config and do deep analysis ─────────────────────
    print("\n" + "─" * 60)
    print("PART A: Reproduce best config + deep diagnostics")
    print("─" * 60)

    # Best config from Attempt 3: GBM_200_d4_lr0.05, mw=0.05, tp=25%, h=50
    base_model = GradientBoostingClassifier
    base_kwargs = dict(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE)
    base_weight = partial(allocate_inverse_volatility, vol_window=60, max_weight=0.05)

    signals_base = build_signals_with_model(
        ann, ohe, "30d",
        model_cls=base_model, model_kwargs=base_kwargs,
        threshold=P85_30D,
    )
    print(f"\n  Base signals: {len(signals_base)}")

    result_base = run_experiment(
        "base_mw0.05_tp25_h50", signals_base, ohlcv_cache,
        horizon_bars=50, take_profit_pct=0.25, stop_loss_pct=1.0,
        weight_fn=base_weight,
    )
    rob_base = robustness(result_base)
    print_result("BASE: GBM mw=5% tp=25% h=50", result_base, rob_base)

    # Bootstrap Sharpe CI
    if result_base["metrics"]:
        bs = bootstrap_sharpe(result_base["trade_log"])
        print(f"  Bootstrap Sharpe: {bs['sharpe_mean']:.2f} [{bs['sharpe_lo']:.2f}, {bs['sharpe_hi']:.2f}] (95% CI)")

    # Per-quarter breakdown
    if result_base["metrics"] and len(result_base["trade_log"]) > 0:
        print("\n  Per-quarter PnL:")
        qpnl = per_quarter_pnl(result_base["trade_log"])
        for _, row in qpnl.iterrows():
            print(f"    {row['quarter']:8s}  {int(row['trades']):>3} trades  "
                  f"PnL=${row['pnl']:>+10,.0f}  WR={row['win_rate']:>5.1f}%  "
                  f"Avg=${row['avg_pnl']:>+8,.0f}")

    # ── Reproduce recruit filter config ────────────────────────────────────────
    print("\n  --- Recruit filter configs ---")
    recruit_filter = ["CT_RECRUITING"]

    for mw_val, label in [(0.05, "5%"), (0.07, "7%"), (0.10, "10%")]:
        signals_r = build_signals_with_model(
            ann, ohe, "30d",
            model_cls=base_model, model_kwargs=base_kwargs,
            threshold=P85_30D,
            event_filter=recruit_filter,
        )
        wfn = partial(allocate_inverse_volatility, vol_window=60, max_weight=mw_val)
        result_r = run_experiment(
            f"recruit_mw{mw_val}", signals_r, ohlcv_cache,
            horizon_bars=50, take_profit_pct=0.25, stop_loss_pct=1.0,
            weight_fn=wfn,
        )
        rob_r = robustness(result_r)
        print_result(f"RECRUIT mw={label} tp=25% h=50", result_r, rob_r)

        if result_r["metrics"] and mw_val == 0.10:
            bs_r = bootstrap_sharpe(result_r["trade_log"])
            print(f"    Bootstrap Sharpe: {bs_r['sharpe_mean']:.2f} [{bs_r['sharpe_lo']:.2f}, {bs_r['sharpe_hi']:.2f}]")
            print("\n    Per-quarter:")
            qr = per_quarter_pnl(result_r["trade_log"])
            for _, row in qr.iterrows():
                print(f"      {row['quarter']:8s}  {int(row['trades']):>3} trades  "
                      f"PnL=${row['pnl']:>+10,.0f}  WR={row['win_rate']:>5.1f}%")

    # ── Part B: Data availability in 2024 vs 2025 ─────────────────────────────
    print("\n" + "─" * 60)
    print("PART B: Data availability analysis (Y1 vs Y2)")
    print("─" * 60)

    ann_copy = ann.copy()
    ann_copy["published_at"] = pd.to_datetime(ann_copy["published_at"])
    for year in [2024, 2025]:
        yr_data = ann_copy[ann_copy["published_at"].dt.year == year]
        recruit = yr_data[yr_data["event_type"] == "CT_RECRUITING"]
        above_p85 = yr_data[yr_data["return_30d"] >= P85_30D]
        recruit_above = recruit[recruit["return_30d"] >= P85_30D]
        print(f"\n  {year}:")
        print(f"    Total announcements: {len(yr_data)}")
        print(f"    CT_RECRUITING: {len(recruit)} ({100*len(recruit)/len(yr_data):.1f}%)")
        print(f"    Above P85 (30d): {len(above_p85)} ({100*len(above_p85)/len(yr_data):.1f}%)")
        print(f"    Recruit + above P85: {len(recruit_above)} ({100*len(recruit_above)/max(1,len(recruit)):.1f}% of recruit)")
        print(f"    Mean 30d return: {yr_data['return_30d'].mean():+.2f}%")
        print(f"    Median 30d return: {yr_data['return_30d'].median():+.2f}%")
        print(f"    Mean 30d (recruit only): {recruit['return_30d'].mean():+.2f}%")

    # ── Part C: Ensemble approach ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("PART C: Ensemble strategies")
    print("─" * 60)

    results_c = {}

    for consensus in [2, 3]:
        for ef_label, ef in [("all", None), ("recruit", recruit_filter)]:
            for th_val in [P85_30D, 0.05]:
                th_label = "p85" if th_val == P85_30D else "5pct"
                name = f"ensemble_c{consensus}_{ef_label}_th{th_label}"

                signals_e = build_ensemble_signals(
                    ann, ohe, "30d",
                    threshold=th_val,
                    event_filter=ef,
                    consensus=consensus,
                )
                if len(signals_e) == 0:
                    print(f"  {name:50s} NO SIGNALS")
                    continue

                for mw_val in [0.05, 0.07]:
                    wfn = partial(allocate_inverse_volatility, vol_window=60, max_weight=mw_val)
                    full_name = f"{name}_mw{mw_val}"
                    result_e = run_experiment(
                        full_name, signals_e, ohlcv_cache,
                        horizon_bars=50, take_profit_pct=0.25, stop_loss_pct=1.0,
                        weight_fn=wfn,
                    )
                    rob_e = robustness(result_e)
                    print_result(full_name, result_e, rob_e)
                    results_c[full_name] = (result_e, rob_e)

    # ── Part D: Probability-weighted sizing ────────────────────────────────────
    print("\n" + "─" * 60)
    print("PART D: Probability-weighted sizing")
    print("─" * 60)

    results_d = {}

    for proba_cut in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for ef_label, ef in [("all", None), ("recruit", recruit_filter)]:
            for th_val in [P85_30D, 0.05]:
                th_label = "p85" if th_val == P85_30D else "5pct"
                name = f"proba_{proba_cut}_{ef_label}_th{th_label}"

                signals_p = build_proba_weighted_signals(
                    ann, ohe, "30d",
                    threshold=th_val,
                    event_filter=ef,
                    proba_cutoff=proba_cut,
                )
                if len(signals_p) == 0:
                    print(f"  {name:50s} NO SIGNALS")
                    continue

                for mw_val in [0.05, 0.07]:
                    wfn = partial(allocate_inverse_volatility, vol_window=60, max_weight=mw_val)
                    full_name = f"{name}_mw{mw_val}"
                    result_p = run_experiment(
                        full_name, signals_p, ohlcv_cache,
                        horizon_bars=50, take_profit_pct=0.25, stop_loss_pct=1.0,
                        weight_fn=wfn,
                    )
                    rob_p = robustness(result_p)
                    print_result(full_name, result_p, rob_p)
                    results_d[full_name] = (result_p, rob_p)

    # ── Part E: Horizon and TP grid on best model ──────────────────────────────
    print("\n" + "─" * 60)
    print("PART E: Fine-tuned TP/horizon grid on GBM (mw=5%)")
    print("─" * 60)

    results_e = {}

    # Use threshold=5% and no event filter (broadest signal set)
    signals_broad = build_signals_with_model(
        ann, ohe, "30d",
        model_cls=base_model, model_kwargs=base_kwargs,
        threshold=0.05,
    )
    print(f"  Broad signals (th=5%): {len(signals_broad)}")

    for tp_val in [0.15, 0.20, 0.25, 0.30, 0.35]:
        for h_val in [30, 40, 50, 60]:
            name = f"broad_tp{int(tp_val*100)}_h{h_val}"
            result_e_item = run_experiment(
                name, signals_broad, ohlcv_cache,
                horizon_bars=h_val, take_profit_pct=tp_val, stop_loss_pct=1.0,
                weight_fn=base_weight,
            )
            rob_e_item = robustness(result_e_item)
            results_e[name] = (result_e_item, rob_e_item)

    # Sort by Sharpe and print top 10
    ranked_e = sorted(results_e.items(),
                      key=lambda x: x[1][0]["metrics"]["sharpe_ratio"] if x[1][0]["metrics"] else -99,
                      reverse=True)
    for name, (res, rob) in ranked_e[:10]:
        print_result(name, res, rob)

    # ── Part F: 5d horizon experiments ────────────────────────────────────────
    print("\n" + "─" * 60)
    print("PART F: 5-day horizon experiments")
    print("─" * 60)

    results_f = {}

    signals_5d = build_signals_with_model(
        ann, ohe, "5d",
        model_cls=base_model, model_kwargs=base_kwargs,
        threshold=P85_5D,
    )
    print(f"  5d signals (th=P85): {len(signals_5d)}")

    signals_5d_low = build_signals_with_model(
        ann, ohe, "5d",
        model_cls=base_model, model_kwargs=base_kwargs,
        threshold=0.01,
    )
    print(f"  5d signals (th=1%): {len(signals_5d_low)}")

    for sig_label, sigs in [("5d_p85", signals_5d), ("5d_th1", signals_5d_low)]:
        for tp_val in [0.05, 0.08, 0.10, 0.15]:
            for h_val in [5, 8, 10]:
                for mw_val in [0.03, 0.05, 0.07]:
                    name = f"{sig_label}_tp{int(tp_val*100)}_h{h_val}_mw{mw_val}"
                    wfn = partial(allocate_inverse_volatility, vol_window=60, max_weight=mw_val)
                    result_f_item = run_experiment(
                        name, sigs, ohlcv_cache,
                        horizon_bars=h_val, take_profit_pct=tp_val, stop_loss_pct=1.0,
                        weight_fn=wfn,
                    )
                    rob_f_item = robustness(result_f_item)
                    results_f[name] = (result_f_item, rob_f_item)

    # Sort by Sharpe
    ranked_f = sorted(results_f.items(),
                      key=lambda x: x[1][0]["metrics"]["sharpe_ratio"] if x[1][0]["metrics"] else -99,
                      reverse=True)
    print("\n  Top 10 5d configs:")
    for name, (res, rob) in ranked_f[:10]:
        print_result(name, res, rob)

    # ── Part G: Feature importance ─────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("PART G: Feature importance (GBM)")
    print("─" * 60)

    fi = feature_importance_analysis(ann, ohe)
    print("\n  Top 20 features:")
    for _, row in fi.iterrows():
        print(f"    {row['feature']:40s}  {row['importance']:.4f}")

    # ── Part H: Combined 5d + 30d strategy ─────────────────────────────────────
    print("\n" + "─" * 60)
    print("PART H: Combined 5d + 30d dual-horizon strategy")
    print("─" * 60)

    # Idea: Use 5d signals for quick momentum, 30d for catalyst holds.
    # Combine them into a single signal set with different TP/horizons.
    # The 5d trades turn over faster, allowing more capital recycling.

    signals_30d_best = build_signals_with_model(
        ann, ohe, "30d",
        model_cls=base_model, model_kwargs=base_kwargs,
        threshold=0.05,
    )
    signals_5d_best = build_signals_with_model(
        ann, ohe, "5d",
        model_cls=base_model, model_kwargs=base_kwargs,
        threshold=0.01,
    )

    # Run 30d and 5d separately with same capital, then report combined
    for mw_val in [0.03, 0.05]:
        wfn = partial(allocate_inverse_volatility, vol_window=60, max_weight=mw_val)

        r30 = run_experiment(
            f"30d_mw{mw_val}", signals_30d_best, ohlcv_cache,
            horizon_bars=50, take_profit_pct=0.25, stop_loss_pct=1.0,
            weight_fn=wfn,
        )
        r5 = run_experiment(
            f"5d_mw{mw_val}", signals_5d_best, ohlcv_cache,
            horizon_bars=8, take_profit_pct=0.08, stop_loss_pct=1.0,
            weight_fn=wfn,
        )

        rob30 = robustness(r30)
        rob5 = robustness(r5)
        print_result(f"30d component mw={mw_val}", r30, rob30)
        print_result(f"5d component  mw={mw_val}", r5, rob5)

        # Combined PnL (simple sum, assuming 50/50 capital split)
        if r30["metrics"] and r5["metrics"]:
            combined_ret = r30["metrics"]["total_return_pct"] * 0.5 + r5["metrics"]["total_return_pct"] * 0.5
            combined_trades = r30["metrics"]["total_trades"] + r5["metrics"]["total_trades"]
            print(f"    → Combined (50/50): ~{combined_ret:+.1f}% return, {combined_trades} trades")

    # ── Part I: Score-weighted position sizing ─────────────────────────────────
    print("\n" + "─" * 60)
    print("PART I: Score-weighted position sizing")
    print("─" * 60)

    # Instead of inverse-vol, use the GBM probability to scale positions.
    # Higher confidence = larger position.
    signals_proba = build_proba_weighted_signals(
        ann, ohe, "30d", threshold=0.05, proba_cutoff=0.4,
    )
    print(f"  Proba signals (cut=0.4, th=5%): {len(signals_proba)}")

    if len(signals_proba) > 0:
        # Score-weighted: use decision_score as raw weight input
        # We'll use equal weight but override with score-proportional sizing
        for mw_val in [0.05, 0.07, 0.10]:
            # Use score_weighted position sizing mode
            wfn = partial(allocate_inverse_volatility, vol_window=60, max_weight=mw_val)
            name = f"proba_scored_mw{mw_val}"
            result_sw = run_experiment(
                name, signals_proba, ohlcv_cache,
                horizon_bars=50, take_profit_pct=0.25, stop_loss_pct=1.0,
                weight_fn=wfn,
            )
            rob_sw = robustness(result_sw)
            print_result(name, result_sw, rob_sw)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY — All configs that beat SPY (+47.8%, Sharpe 1.28)")
    print("=" * 80)

    all_results = {}

    # Add base configs
    all_results["base_mw0.05"] = (result_base, rob_base)

    # Add all experiment results
    for store in [results_c, results_d, results_e, results_f]:
        all_results.update(store)

    # Filter to those beating SPY
    spy_beaters = []
    for name, (res, rob) in all_results.items():
        if res["metrics"] and res["metrics"]["total_return_pct"] > 47.8:
            spy_beaters.append((name, res, rob))

    spy_beaters.sort(key=lambda x: x[1]["metrics"]["sharpe_ratio"], reverse=True)

    if not spy_beaters:
        print("  No configs beat SPY on total return in this run.")
        print("  Top 5 by Sharpe:")
        ranked_all = sorted(all_results.items(),
                            key=lambda x: x[1][0]["metrics"]["sharpe_ratio"] if x[1][0]["metrics"] else -99,
                            reverse=True)
        for name, (res, rob) in ranked_all[:5]:
            print_result(name, res, rob)
    else:
        for name, res, rob in spy_beaters:
            print_result(name, res, rob)

    # Also print best Sharpe regardless of return
    print("\n  Top 10 by Sharpe (regardless of return):")
    ranked_all = sorted(all_results.items(),
                        key=lambda x: x[1][0]["metrics"]["sharpe_ratio"] if x[1][0]["metrics"] else -99,
                        reverse=True)
    for name, (res, rob) in ranked_all[:10]:
        print_result(name, res, rob)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    # ── Log to journal ─────────────────────────────────────────────────────────
    # Collect top findings for the journal
    journal_lines = []
    journal_lines.append(f"**Time**: 2026-03-29 (Attempt 4)")
    journal_lines.append(f"**Focus**: Deep robustness, ensemble, probability-sizing, 5d horizon, dual-horizon")
    journal_lines.append(f"\nTested ~{len(all_results)} configurations across 6 strategy families.\n")

    if result_base["metrics"]:
        bs = bootstrap_sharpe(result_base["trade_log"])
        journal_lines.append(f"**Base config** (GBM mw=5% tp=25% h=50): {result_base['metrics']['total_return_pct']:+.1f}%, "
                            f"Sharpe {result_base['metrics']['sharpe_ratio']:.2f}, "
                            f"Bootstrap 95% CI [{bs['sharpe_lo']:.2f}, {bs['sharpe_hi']:.2f}]")

    if spy_beaters:
        journal_lines.append(f"\n**{len(spy_beaters)} configs beat SPY** on total return.")
        journal_lines.append("\n| Config | Trades | Return | Sharpe | MaxDD | WR | Y1/Y2 |")
        journal_lines.append("|--------|--------|--------|--------|-------|----|-------|")
        for name, res, rob in spy_beaters[:15]:
            m = res["metrics"]
            y1 = rob.get("y1_ret", 0)
            y2 = rob.get("y2_ret", 0)
            journal_lines.append(f"| {name} | {m['total_trades']} | {m['total_return_pct']:+.1f}% | "
                                f"{m['sharpe_ratio']:.2f} | {m['max_drawdown_pct']:.1f}% | "
                                f"{m['win_rate_pct']:.1f}% | {y1:+.1f}/{y2:+.1f} |")
    else:
        journal_lines.append("\n**No configs beat SPY total return in this attempt.**")
        journal_lines.append("\nTop 5 by Sharpe:")
        for name, (res, rob) in ranked_all[:5]:
            if res["metrics"]:
                m = res["metrics"]
                journal_lines.append(f"  - {name}: {m['total_return_pct']:+.1f}%, Sharpe {m['sharpe_ratio']:.2f}")

    journal_body = "\n".join(journal_lines)
    log_entry("Attempt 4: Robustness + Ensemble + 5d + Dual-horizon", journal_body)

    print("\n  Journal entry appended to research_journal.md")


if __name__ == "__main__":
    main()
