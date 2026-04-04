"""
Attempt 5: Stress-test robustness + address red flags from Attempt 4.

Red flags to address:
1. OLMA concentration — remove OLMA and re-test all best configs
2. sf_word_count dominance (46% feature importance) — test without it
3. Y1=0% on all recruit configs — find signals that work in 2024 too
4. Try non-recruit event types that have 2024 coverage
5. Test ex-top-N ticker robustness (remove top 1, 3, 5 tickers)
"""

import warnings
import time
from functools import partial
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from backtest.config import (
    ANNOUNCEMENTS_PATH, OHLCV_DIR, OUTPUT_DIR,
    START_DATE, END_DATE, TRAIN_MONTHS, PREDICT_WEEKS, STEP_WEEKS,
    INITIAL_CAPITAL, COMMISSION_PCT, SLIPPAGE_PCT,
    TIERED_SLIPPAGE, P85_30D, P85_5D,
    RANDOM_STATE, MIN_TRAIN_SAMPLES, SF_COLS,
)
from backtest.data_loader import load_announcements
from backtest.features import fit_ohe, build_feature_matrix
from backtest.model import generate_windows
from backtest.portfolio_simulator import simulate_portfolio, _load_ohlcv_cache
from backtest.portfolio_construction import allocate_inverse_volatility
from backtest.metrics import compute_metrics
from backtest.research import (
    build_signals_with_model, run_experiment, robustness, print_result, log_entry,
)

warnings.filterwarnings("ignore")


def run_ex_ticker(result: dict, exclude_tickers: list, ohlcv_cache: dict,
                  name: str, horizon_bars=50, tp=0.25, mw=0.05) -> dict:
    """Re-run experiment excluding specific tickers from signals."""
    tl = result.get("trade_log")
    if tl is None or len(tl) == 0:
        return {"name": name, "metrics": None, "trade_log": pd.DataFrame()}

    # Get original signals and remove excluded tickers
    # We can't easily get original signals, so we filter the trade log
    # and recompute metrics from the filtered trade log
    filtered_tl = tl[~tl["ticker"].isin(exclude_tickers)].copy()
    if len(filtered_tl) == 0:
        return {"name": name, "metrics": None, "trade_log": pd.DataFrame()}

    # Recompute equity curve from filtered trade log
    total_pnl = filtered_tl["pnl"].sum()
    metrics = {
        "total_return_pct": total_pnl / INITIAL_CAPITAL * 100,
        "total_trades": len(filtered_tl),
        "win_rate_pct": (filtered_tl["pnl"] > 0).mean() * 100,
        "profit_factor": abs(filtered_tl[filtered_tl["pnl"] > 0]["pnl"].sum()) / max(1, abs(filtered_tl[filtered_tl["pnl"] < 0]["pnl"].sum())),
        "max_drawdown_pct": result["metrics"]["max_drawdown_pct"],  # approximate
        "sharpe_ratio": 0,  # need equity curve for this
    }
    # Approximate sharpe from trade pnl
    pnls = filtered_tl["pnl"].values
    if len(pnls) > 1 and pnls.std() > 0:
        metrics["sharpe_ratio"] = (pnls.mean() / pnls.std()) * np.sqrt(len(pnls) / 2)

    return {"name": name, "metrics": metrics, "trade_log": filtered_tl}


def build_signals_no_wordcount(
    announcements: pd.DataFrame,
    ohe,
    horizon: str = "30d",
    threshold: float = 0.05,
    event_filter: list = None,
    drop_features: list = None,
) -> pd.DataFrame:
    """Build signals with GBM but drop specific features (e.g., sf_word_count)."""
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

        # Drop specified feature columns
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
                n_estimators=200, max_depth=4, learning_rate=0.05,
                random_state=RANDOM_STATE,
            )
            model.fit(X_train, y_train)
        except Exception:
            continue

        preds = model.predict(X_pred)
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_pred)[:, 1]
        else:
            scores = preds.astype(float)

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


def build_signals_exclude_tickers(
    announcements: pd.DataFrame,
    ohe,
    exclude_tickers: list,
    horizon: str = "30d",
    threshold: float = None,
    event_filter: list = None,
) -> pd.DataFrame:
    """Build signals using GBM but exclude specific tickers from prediction."""
    if threshold is None:
        threshold = P85_30D
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

        # Exclude tickers from prediction (but keep in training)
        pred_df = pred_df[~pred_df["ticker"].isin(exclude_tickers)].copy()

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

        preds = model.predict(X_pred)
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_pred)[:, 1]
        else:
            scores = preds.astype(float)

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


def main():
    t0 = time.time()
    print("=" * 80)
    print("ATTEMPT 5: Stress-Test Robustness")
    print("=" * 80)

    ann = load_announcements(ANNOUNCEMENTS_PATH)
    ohe = fit_ohe(ann)
    tickers = ann["ticker"].unique().tolist()
    ohlcv_cache = _load_ohlcv_cache(tickers, OHLCV_DIR)
    print(f"  {len(ann)} announcements, {len(tickers)} tickers, {len(ohlcv_cache)} with OHLCV")

    recruit_filter = ["CT_RECRUITING"]
    base_weight_5 = partial(allocate_inverse_volatility, vol_window=60, max_weight=0.05)
    base_weight_7 = partial(allocate_inverse_volatility, vol_window=60, max_weight=0.07)

    journal_lines = []
    journal_lines.append("**Time**: 2026-03-29 (Attempt 5)")
    journal_lines.append("**Focus**: OLMA dependency, word_count feature, 2024 event coverage, ex-ticker robustness\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART A: OLMA dependency test
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("PART A: OLMA dependency — exclude OLMA from signals")
    print("─" * 60)

    journal_lines.append("### Part A: OLMA Dependency Test\n")

    # First run baseline WITH OLMA for comparison
    signals_recruit = build_signals_with_model(
        ann, ohe, "30d",
        model_cls=GradientBoostingClassifier,
        model_kwargs=dict(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE),
        threshold=P85_30D,
        event_filter=recruit_filter,
    )

    r_with = run_experiment("recruit_with_OLMA_mw7", signals_recruit, ohlcv_cache,
                            horizon_bars=50, take_profit_pct=0.25, weight_fn=base_weight_7)
    rob_with = robustness(r_with)
    print_result("WITH OLMA (baseline)", r_with, rob_with)

    # Now exclude OLMA
    signals_no_olma = build_signals_exclude_tickers(
        ann, ohe, exclude_tickers=["OLMA"],
        threshold=P85_30D,
        event_filter=recruit_filter,
    )
    r_no_olma = run_experiment("recruit_no_OLMA_mw7", signals_no_olma, ohlcv_cache,
                               horizon_bars=50, take_profit_pct=0.25, weight_fn=base_weight_7)
    rob_no_olma = robustness(r_no_olma)
    print_result("WITHOUT OLMA", r_no_olma, rob_no_olma)

    if r_with["metrics"] and r_no_olma["metrics"]:
        delta_ret = r_no_olma["metrics"]["total_return_pct"] - r_with["metrics"]["total_return_pct"]
        journal_lines.append(f"- With OLMA: {r_with['metrics']['total_return_pct']:+.1f}%, Sh={r_with['metrics']['sharpe_ratio']:.2f}")
        journal_lines.append(f"- Without OLMA: {r_no_olma['metrics']['total_return_pct']:+.1f}%, Sh={r_no_olma['metrics']['sharpe_ratio']:.2f}")
        journal_lines.append(f"- Delta: {delta_ret:+.1f}pp return\n")

    # Also test on best proba config (recruit, th=5%, mw=7%)
    print("\n  --- Proba configs ex-OLMA ---")
    signals_proba_no_olma = build_signals_exclude_tickers(
        ann, ohe, exclude_tickers=["OLMA"],
        threshold=0.05,
        event_filter=recruit_filter,
    )

    for mw_val, wfn in [(0.05, base_weight_5), (0.07, base_weight_7)]:
        r_pn = run_experiment(f"proba_recruit_th5_noOLMA_mw{mw_val}", signals_proba_no_olma, ohlcv_cache,
                              horizon_bars=50, take_profit_pct=0.25, weight_fn=wfn)
        rob_pn = robustness(r_pn)
        print_result(f"recruit th=5% ex-OLMA mw={mw_val}", r_pn, rob_pn)
        if r_pn["metrics"]:
            journal_lines.append(f"- recruit th=5% ex-OLMA mw={mw_val}: {r_pn['metrics']['total_return_pct']:+.1f}%, Sh={r_pn['metrics']['sharpe_ratio']:.2f}")

    # Exclude top-5 tickers
    print("\n  --- Exclude top-5 PnL tickers ---")
    if r_with["metrics"] and len(r_with["trade_log"]) > 0:
        top5 = r_with["trade_log"].groupby("ticker")["pnl"].sum().nlargest(5).index.tolist()
        print(f"  Top-5 tickers by PnL: {top5}")
        signals_no_top5 = build_signals_exclude_tickers(
            ann, ohe, exclude_tickers=top5,
            threshold=P85_30D,
            event_filter=recruit_filter,
        )
        r_no5 = run_experiment("recruit_noTop5_mw7", signals_no_top5, ohlcv_cache,
                               horizon_bars=50, take_profit_pct=0.25, weight_fn=base_weight_7)
        rob_no5 = robustness(r_no5)
        print_result("EXCLUDE TOP-5 tickers", r_no5, rob_no5)
        if r_no5["metrics"]:
            journal_lines.append(f"- Exclude top-5 ({', '.join(top5)}): {r_no5['metrics']['total_return_pct']:+.1f}%, Sh={r_no5['metrics']['sharpe_ratio']:.2f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART B: Feature ablation — drop word_count
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("PART B: Feature ablation — drop sf_word_count")
    print("─" * 60)

    journal_lines.append("\n### Part B: Feature Ablation (drop sf_word_count)\n")

    signals_nowc = build_signals_no_wordcount(
        ann, ohe, "30d", threshold=P85_30D,
        drop_features=["sf_word_count"],
    )
    print(f"  Signals (no word_count, all events): {len(signals_nowc)}")

    for mw_val, wfn in [(0.05, base_weight_5), (0.07, base_weight_7)]:
        r_nowc = run_experiment(f"no_wordcount_all_mw{mw_val}", signals_nowc, ohlcv_cache,
                                horizon_bars=50, take_profit_pct=0.25, weight_fn=wfn)
        rob_nowc = robustness(r_nowc)
        print_result(f"no_wordcount all mw={mw_val}", r_nowc, rob_nowc)
        if r_nowc["metrics"]:
            journal_lines.append(f"- No word_count, all events, mw={mw_val}: {r_nowc['metrics']['total_return_pct']:+.1f}%, Sh={r_nowc['metrics']['sharpe_ratio']:.2f}")

    # Drop word_count + recruit filter
    signals_nowc_recruit = build_signals_no_wordcount(
        ann, ohe, "30d", threshold=P85_30D,
        event_filter=recruit_filter,
        drop_features=["sf_word_count"],
    )
    print(f"  Signals (no word_count, recruit): {len(signals_nowc_recruit)}")

    for mw_val, wfn in [(0.05, base_weight_5), (0.07, base_weight_7)]:
        r_nowcr = run_experiment(f"no_wordcount_recruit_mw{mw_val}", signals_nowc_recruit, ohlcv_cache,
                                 horizon_bars=50, take_profit_pct=0.25, weight_fn=wfn)
        rob_nowcr = robustness(r_nowcr)
        print_result(f"no_wordcount recruit mw={mw_val}", r_nowcr, rob_nowcr)
        if r_nowcr["metrics"]:
            journal_lines.append(f"- No word_count, recruit, mw={mw_val}: {r_nowcr['metrics']['total_return_pct']:+.1f}%, Sh={r_nowcr['metrics']['sharpe_ratio']:.2f}")

    # Also try dropping word_count + n_patients + phase (top 3 structured features)
    signals_no3 = build_signals_no_wordcount(
        ann, ohe, "30d", threshold=P85_30D,
        drop_features=["sf_word_count", "sf_n_patients", "sf_phase"],
    )
    print(f"  Signals (no wc/npatients/phase): {len(signals_no3)}")

    r_no3 = run_experiment("no_top3sf_all_mw5", signals_no3, ohlcv_cache,
                            horizon_bars=50, take_profit_pct=0.25, weight_fn=base_weight_5)
    rob_no3 = robustness(r_no3)
    print_result("no top-3 SF features mw=5%", r_no3, rob_no3)
    if r_no3["metrics"]:
        journal_lines.append(f"- No top-3 SF features: {r_no3['metrics']['total_return_pct']:+.1f}%, Sh={r_no3['metrics']['sharpe_ratio']:.2f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART C: 2024-active event types
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("PART C: Event types with 2024 coverage")
    print("─" * 60)

    journal_lines.append("\n### Part C: Event Types with 2024 Coverage\n")

    ann_copy = ann.copy()
    ann_copy["published_at"] = pd.to_datetime(ann_copy["published_at"])

    # Analyze all event types by year
    print("\n  Event type distribution by year:")
    for et in sorted(ann_copy["event_type"].unique()):
        y24 = len(ann_copy[(ann_copy["published_at"].dt.year == 2024) & (ann_copy["event_type"] == et)])
        y25 = len(ann_copy[(ann_copy["published_at"].dt.year == 2025) & (ann_copy["event_type"] == et)])
        ret24 = ann_copy[(ann_copy["published_at"].dt.year == 2024) & (ann_copy["event_type"] == et)]["return_30d"].mean()
        ret25 = ann_copy[(ann_copy["published_at"].dt.year == 2025) & (ann_copy["event_type"] == et)]["return_30d"].mean()
        if y24 + y25 >= 10:
            print(f"    {et:40s}  2024:{y24:>4}  2025:{y25:>4}  ret24={ret24:+6.2f}%  ret25={ret25:+6.2f}%")

    # Test event types that have reasonable 2024 coverage
    # Focus on types with >50 events in 2024
    event_combos = [
        ("active_not_recruiting", ["CT_ACTIVE_NOT_RECRUITING"]),
        ("completed", ["CT_COMPLETED"]),
        ("active+completed", ["CT_ACTIVE_NOT_RECRUITING", "CT_COMPLETED"]),
        ("active+completed+recruit", ["CT_ACTIVE_NOT_RECRUITING", "CT_COMPLETED", "CT_RECRUITING"]),
        ("not_yet_recruiting", ["CT_NOT_YET_RECRUITING"]),
        ("all_positive", ["CT_RECRUITING", "CT_ACTIVE_NOT_RECRUITING", "CT_COMPLETED", "CT_AVAILABLE"]),
        ("terminated", ["CT_TERMINATED"]),  # negative signal test
        ("approved", ["CT_APPROVED_FOR_MARKETING"]),
    ]

    for label, evt_list in event_combos:
        signals_evt = build_signals_with_model(
            ann, ohe, "30d",
            model_cls=GradientBoostingClassifier,
            model_kwargs=dict(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE),
            threshold=P85_30D,
            event_filter=evt_list,
        )
        if len(signals_evt) == 0:
            print(f"  {label:40s} NO SIGNALS")
            journal_lines.append(f"- {label}: NO SIGNALS")
            continue

        for mw_val, wfn in [(0.05, base_weight_5), (0.07, base_weight_7)]:
            r_evt = run_experiment(f"{label}_mw{mw_val}", signals_evt, ohlcv_cache,
                                   horizon_bars=50, take_profit_pct=0.25, weight_fn=wfn)
            rob_evt = robustness(r_evt)
            print_result(f"{label} mw={mw_val}", r_evt, rob_evt)
            if r_evt["metrics"]:
                m = r_evt["metrics"]
                y1 = rob_evt.get("y1_ret", 0)
                y2 = rob_evt.get("y2_ret", 0)
                journal_lines.append(f"- {label} mw={mw_val}: {m['total_return_pct']:+.1f}%, Sh={m['sharpe_ratio']:.2f}, Y1={y1:+.1f}%, Y2={y2:+.1f}%")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART D: All-events GBM with different thresholds (best Y1/Y2 balance)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("PART D: All-events GBM — optimize for Y1/Y2 balance")
    print("─" * 60)

    journal_lines.append("\n### Part D: All-Events GBM (Y1/Y2 Balance)\n")

    for th_val in [P85_30D, 0.10, 0.05, 0.03]:
        th_label = f"th={th_val:.2f}" if th_val != P85_30D else "th=P85"
        signals_all = build_signals_with_model(
            ann, ohe, "30d",
            model_cls=GradientBoostingClassifier,
            model_kwargs=dict(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE),
            threshold=th_val,
        )
        if len(signals_all) == 0:
            continue

        for mw_val, wfn in [(0.05, base_weight_5)]:
            for tp_val in [0.20, 0.25, 0.30]:
                for h_val in [40, 50, 60]:
                    name = f"all_{th_label}_mw{mw_val}_tp{int(tp_val*100)}_h{h_val}"
                    r_all = run_experiment(name, signals_all, ohlcv_cache,
                                           horizon_bars=h_val, take_profit_pct=tp_val, weight_fn=wfn)
                    rob_all = robustness(r_all)

                    if r_all["metrics"] and r_all["metrics"]["sharpe_ratio"] > 1.0:
                        y1 = rob_all.get("y1_ret", 0)
                        y2 = rob_all.get("y2_ret", 0)
                        # Prioritize configs where Y1 > 0
                        if y1 > 0:
                            print_result(name, r_all, rob_all)
                            journal_lines.append(f"- {name}: {r_all['metrics']['total_return_pct']:+.1f}%, Sh={r_all['metrics']['sharpe_ratio']:.2f}, Y1={y1:+.1f}%, Y2={y2:+.1f}%")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART E: Longer training window (try 18-month, 24-month)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("PART E: Training window length sweep")
    print("─" * 60)

    journal_lines.append("\n### Part E: Training Window Length\n")

    for train_m in [6, 9, 12, 18, 24]:
        windows = generate_windows(
            start_date=START_DATE, end_date=END_DATE,
            train_months=train_m, predict_weeks=PREDICT_WEEKS,
            step_weeks=STEP_WEEKS,
        )

        label_col = "return_30d"
        all_signals = []
        for window in windows:
            clean_train_end = window.train_end - pd.DateOffset(days=30)
            train_df = ann[
                (ann["published_at"] >= window.train_start) &
                (ann["published_at"] < clean_train_end)
            ].copy()
            pred_df = ann[
                (ann["published_at"] >= window.pred_start) &
                (ann["published_at"] < window.pred_end)
            ].copy()

            if len(train_df) < MIN_TRAIN_SAMPLES or len(pred_df) == 0:
                continue

            X_train = build_feature_matrix(ohe, train_df)
            y_train = (train_df[label_col] >= P85_30D).astype(int).values
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

            preds = model.predict(X_pred)
            scores = model.predict_proba(X_pred)[:, 1]

            pred_df = pred_df.copy()
            pred_df["prediction"] = preds
            pred_df["decision_score"] = scores
            signals = pred_df[pred_df["prediction"] == 1].copy()
            if len(signals) == 0:
                continue
            signals = signals.sort_values("published_at").drop_duplicates(subset=["ticker"], keep="first")
            signals["window_id"] = window.window_id
            signals["horizon"] = "30d"
            all_signals.append(signals)

        if not all_signals:
            print(f"  train={train_m}m: NO SIGNALS")
            continue
        signals_tw = pd.concat(all_signals, ignore_index=True)

        r_tw = run_experiment(f"train{train_m}m", signals_tw, ohlcv_cache,
                               horizon_bars=50, take_profit_pct=0.25, weight_fn=base_weight_5)
        rob_tw = robustness(r_tw)
        print_result(f"train_window={train_m}m", r_tw, rob_tw)
        if r_tw["metrics"]:
            y1 = rob_tw.get("y1_ret", 0)
            y2 = rob_tw.get("y2_ret", 0)
            journal_lines.append(f"- train={train_m}m: {r_tw['metrics']['total_return_pct']:+.1f}%, Sh={r_tw['metrics']['sharpe_ratio']:.2f}, {r_tw['metrics']['total_trades']} trades, Y1={y1:+.1f}%, Y2={y2:+.1f}%")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART F: GBM hyperparameter sensitivity
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("PART F: GBM hyperparameter sensitivity (all events, mw=5%)")
    print("─" * 60)

    journal_lines.append("\n### Part F: GBM Hyperparameter Sensitivity\n")

    hyper_results = {}
    for n_est in [100, 200, 300, 500]:
        for md in [2, 3, 4, 6]:
            for lr in [0.01, 0.03, 0.05, 0.1]:
                name = f"gbm_{n_est}_d{md}_lr{lr}"
                signals_h = build_signals_with_model(
                    ann, ohe, "30d",
                    model_cls=GradientBoostingClassifier,
                    model_kwargs=dict(n_estimators=n_est, max_depth=md, learning_rate=lr, random_state=RANDOM_STATE),
                    threshold=P85_30D,
                )
                if len(signals_h) == 0:
                    continue
                r_h = run_experiment(name, signals_h, ohlcv_cache,
                                     horizon_bars=50, take_profit_pct=0.25, weight_fn=base_weight_5)
                if r_h["metrics"]:
                    rob_h = robustness(r_h)
                    hyper_results[name] = (r_h, rob_h)

    # Print top 15 by Sharpe
    ranked_h = sorted(hyper_results.items(),
                      key=lambda x: x[1][0]["metrics"]["sharpe_ratio"],
                      reverse=True)
    print(f"\n  Tested {len(hyper_results)} GBM configs. Top 15:")
    for name, (r, rob) in ranked_h[:15]:
        print_result(name, r, rob)
        if name == ranked_h[0][0]:
            m = r["metrics"]
            y1 = rob.get("y1_ret", 0)
            y2 = rob.get("y2_ret", 0)
            journal_lines.append(f"- Best GBM: {name}, {m['total_return_pct']:+.1f}%, Sh={m['sharpe_ratio']:.2f}, Y1={y1:+.1f}%, Y2={y2:+.1f}%")

    # Log top 5
    journal_lines.append("\nTop 5 GBM configs:")
    for name, (r, rob) in ranked_h[:5]:
        m = r["metrics"]
        y1 = rob.get("y1_ret", 0)
        y2 = rob.get("y2_ret", 0)
        journal_lines.append(f"- {name}: {m['total_return_pct']:+.1f}%, Sh={m['sharpe_ratio']:.2f}, Y1={y1:+.1f}%, Y2={y2:+.1f}%")

    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"ATTEMPT 5 COMPLETE ({elapsed:.0f}s)")
    print("=" * 80)

    journal_lines.append(f"\n**Total time**: {elapsed:.0f}s")

    log_entry("Attempt 5: Stress-Test Robustness", "\n".join(journal_lines))
    print("  Journal entry appended.")


if __name__ == "__main__":
    main()
