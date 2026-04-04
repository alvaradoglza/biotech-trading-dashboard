"""
research.py — Rapid iteration framework for strategy experiments.

Usage: python -m backtest.research
"""

import warnings
import time
from functools import partial
from itertools import product

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from backtest.config import (
    ANNOUNCEMENTS_PATH, OHLCV_DIR, OUTPUT_DIR,
    START_DATE, END_DATE, TRAIN_MONTHS, PREDICT_WEEKS, STEP_WEEKS,
    INITIAL_CAPITAL, COMMISSION_PCT, SLIPPAGE_PCT,
    TIERED_SLIPPAGE, ADV_CAP_PCT, P85_30D, P85_5D,
    SVC_C, SVC_MAX_ITER, RANDOM_STATE, MIN_TRAIN_SAMPLES,
    SF_COLS,
)
from backtest.data_loader import load_announcements
from backtest.features import (
    fit_ohe, build_feature_matrix, build_labels, build_structured_features,
)
from backtest.model import generate_windows
from backtest.portfolio_simulator import simulate_portfolio, _load_ohlcv_cache  # noqa: F401
from backtest.portfolio_construction import (
    allocate_equal_weight, allocate_inverse_volatility,
)
from backtest.metrics import compute_metrics


JOURNAL_PATH = OUTPUT_DIR / "research_journal.md"


def log_entry(title: str, body: str):
    """Append an entry to the research journal."""
    with open(JOURNAL_PATH, "a") as f:
        f.write(f"\n---\n\n### {title}\n\n{body}\n")


def build_signals_with_model(
    announcements: pd.DataFrame,
    ohe,
    horizon: str,
    model_cls=None,
    model_kwargs=None,
    threshold: float = None,
    event_filter: list = None,
    min_price: float = None,
    label_col: str = None,
    top_n_per_window: int = None,
) -> pd.DataFrame:
    """Build signals using a configurable model and filters.

    This is a flexible version of run_rolling_loop that lets us swap models,
    thresholds, filters, and ranking strategies.
    """
    if threshold is None:
        threshold = P85_30D if horizon == "30d" else P85_5D
    if label_col is None:
        label_col = f"return_{horizon}"
    if model_cls is None:
        model_cls = LinearSVC
    if model_kwargs is None:
        model_kwargs = dict(C=SVC_C, class_weight="balanced", max_iter=SVC_MAX_ITER, random_state=RANDOM_STATE)

    windows = generate_windows(
        start_date=START_DATE, end_date=END_DATE,
        train_months=TRAIN_MONTHS, predict_weeks=PREDICT_WEEKS,
        step_weeks=STEP_WEEKS,
    )

    return_days = 5 if horizon == "5d" else 30
    all_signals = []

    for window in windows:
        # Training data with leakage prevention
        clean_train_end = window.train_end - pd.DateOffset(days=return_days)
        train_df = announcements[
            (announcements["published_at"] >= window.train_start) &
            (announcements["published_at"] < clean_train_end)
        ].copy()

        # Prediction data
        pred_df = announcements[
            (announcements["published_at"] >= window.pred_start) &
            (announcements["published_at"] < window.pred_end)
        ].copy()

        if len(train_df) < MIN_TRAIN_SAMPLES or len(pred_df) == 0:
            continue

        # Apply event filter to training data too
        if event_filter:
            train_df = train_df[train_df["event_type"].isin(event_filter)].copy()
            pred_df = pred_df[pred_df["event_type"].isin(event_filter)].copy()
            if len(train_df) < MIN_TRAIN_SAMPLES or len(pred_df) == 0:
                continue

        # Build features
        X_train = build_feature_matrix(ohe, train_df)
        y_train = (train_df[label_col] >= threshold).astype(int).values

        X_pred = build_feature_matrix(ohe, pred_df)

        # Skip if only one class
        if len(np.unique(y_train)) < 2:
            continue

        # Train model
        try:
            model = model_cls(**model_kwargs)
            model.fit(X_train, y_train)
        except Exception:
            continue

        # Predict
        preds = model.predict(X_pred)

        # Get scores for ranking
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_pred)
        elif hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_pred)[:, 1]
        else:
            scores = preds.astype(float)

        pred_df = pred_df.copy()
        pred_df["prediction"] = preds
        pred_df["decision_score"] = scores

        # Filter to positive predictions
        signals = pred_df[pred_df["prediction"] == 1].copy()

        if len(signals) == 0:
            continue

        # Apply min price filter
        if min_price is not None:
            # We'll filter later when we have price data; store the filter
            pass

        # Dedup: keep first per ticker
        signals = signals.sort_values("published_at", ascending=True)
        signals = signals.drop_duplicates(subset=["ticker"], keep="first")

        # If top_n, keep only highest-scoring
        if top_n_per_window and len(signals) > top_n_per_window:
            signals = signals.nlargest(top_n_per_window, "decision_score")

        signals["window_id"] = window.window_id
        signals["horizon"] = horizon
        all_signals.append(signals)

    if not all_signals:
        return pd.DataFrame()
    return pd.concat(all_signals, ignore_index=True)


def build_all_signals(
    announcements: pd.DataFrame,
    horizon: str,
    event_filter: list = None,
) -> pd.DataFrame:
    """Build signals from ALL announcements (no model), optionally filtered by event type."""
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

        if event_filter:
            pred_df = pred_df[pred_df["event_type"].isin(event_filter)].copy()

        if len(pred_df) == 0:
            continue

        pred_df["prediction"] = 1
        pred_df["decision_score"] = 1.0
        pred_df["window_id"] = window.window_id
        pred_df["horizon"] = horizon
        pred_df = pred_df.sort_values("published_at", ascending=True)
        pred_df = pred_df.drop_duplicates(subset=["ticker"], keep="first")
        all_signals.append(pred_df)

    if not all_signals:
        return pd.DataFrame()
    return pd.concat(all_signals, ignore_index=True)


def run_experiment(
    name: str,
    signals: pd.DataFrame,
    ohlcv_cache: dict,
    horizon_bars: int = 50,
    take_profit_pct: float = 0.35,
    stop_loss_pct: float = 1.00,
    weight_fn=None,
    max_positions: int = 100,
    adv_cap_pct: float = None,
    min_price: float = None,
) -> dict:
    """Run a single experiment and return metrics + trade log."""
    if len(signals) == 0:
        return {"name": name, "metrics": None, "trade_log": pd.DataFrame()}

    if weight_fn is None:
        weight_fn = partial(allocate_inverse_volatility, vol_window=60, max_weight=0.0075)

    if adv_cap_pct is None:
        adv_cap_pct = 999.0  # uncapped

    trade_log, equity_curve = simulate_portfolio(
        signals, "30d",
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        ohlcv_dir=OHLCV_DIR,
        initial_capital=INITIAL_CAPITAL,
        max_positions=max_positions,
        commission_pct=COMMISSION_PCT,
        slippage_pct=SLIPPAGE_PCT,
        weight_fn=weight_fn,
        _ohlcv_cache=ohlcv_cache,
        tiered_slippage=TIERED_SLIPPAGE,
        adv_cap_pct=adv_cap_pct,
        horizon_bars=horizon_bars,
    )

    # Filter by min_price if specified
    if min_price is not None and len(trade_log) > 0:
        # We can't easily filter pre-entry, so we note this
        pass

    metrics = compute_metrics(trade_log, INITIAL_CAPITAL, equity_curve=equity_curve)
    return {"name": name, "metrics": metrics, "trade_log": trade_log, "equity_curve": equity_curve}


def robustness(result: dict) -> dict:
    """Compute robustness metrics for an experiment result."""
    tl = result["trade_log"]
    m = result["metrics"]
    if m is None or len(tl) == 0:
        return {}

    total_pnl = tl["pnl"].sum()
    if abs(total_pnl) < 1:
        return {"total_pnl": 0}

    top_ticker = tl.groupby("ticker")["pnl"].sum().sort_values(ascending=False)

    r = {
        "return": m["total_return_pct"],
        "sharpe": m["sharpe_ratio"],
        "dd": m["max_drawdown_pct"],
        "trades": m["total_trades"],
        "wr": m["win_rate_pct"],
        "pf": m["profit_factor"],
        "top1": top_ticker.index[0] if len(top_ticker) > 0 else "",
        "top1_pct": top_ticker.iloc[0] / total_pnl * 100 if len(top_ticker) > 0 else 0,
    }

    # Ex-top1, ex-top5
    if len(top_ticker) > 0:
        ex1_pnl = total_pnl - top_ticker.iloc[0]
        r["ex_top1_ret"] = ex1_pnl / INITIAL_CAPITAL * 100
    if len(top_ticker) >= 5:
        ex5_pnl = total_pnl - top_ticker.head(5).sum()
        r["ex_top5_ret"] = ex5_pnl / INITIAL_CAPITAL * 100

    # Year split
    tl_copy = tl.copy()
    tl_copy["entry_date"] = pd.to_datetime(tl_copy["entry_date"])
    y1 = tl_copy[tl_copy["entry_date"].dt.year == 2024]["pnl"].sum()
    y2 = tl_copy[tl_copy["entry_date"].dt.year == 2025]["pnl"].sum()
    r["y1_ret"] = y1 / INITIAL_CAPITAL * 100
    r["y2_ret"] = y2 / INITIAL_CAPITAL * 100

    # Penny vs normal
    penny = tl[tl["entry_price"] < 2]["pnl"].sum()
    normal = tl[tl["entry_price"] >= 5]["pnl"].sum()
    r["penny_pnl"] = penny
    r["normal_pnl"] = normal

    return r


def print_result(name: str, result: dict, rob: dict):
    """Print a single experiment result."""
    if result["metrics"] is None:
        print(f"  {name:45s} NO TRADES")
        return
    m = result["metrics"]
    top1 = rob.get("top1", "")
    top1_pct = rob.get("top1_pct", 0)
    ex5 = rob.get("ex_top5_ret", 0)
    y1 = rob.get("y1_ret", 0)
    y2 = rob.get("y2_ret", 0)
    print(f"  {name:45s} {m['total_trades']:>4}tr  {m['total_return_pct']:>+7.1f}%  "
          f"Sh={m['sharpe_ratio']:>5.2f}  DD={m['max_drawdown_pct']:>5.1f}%  "
          f"WR={m['win_rate_pct']:>4.1f}%  PF={m['profit_factor']:>4.2f}  "
          f"top1={top1}({top1_pct:+.0f}%)  exT5={ex5:+.1f}%  "
          f"Y1={y1:+.1f}% Y2={y2:+.1f}%")
