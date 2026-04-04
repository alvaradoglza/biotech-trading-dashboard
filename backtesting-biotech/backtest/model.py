"""
model.py — Train GBM per rolling window, generate predictions, orchestrate rolling loop.
All logic is pure functions; no classes.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder

from backtest.config import (
    P85_5D, P85_30D,
    TRAIN_MONTHS, PREDICT_WEEKS, STEP_WEEKS,
    MIN_TRAIN_SAMPLES, RANDOM_STATE,
    GBM_N_ESTIMATORS, GBM_MAX_DEPTH, GBM_LEARNING_RATE,
    DROP_FEATURES, SF_COLS,
    START_DATE, END_DATE,
)
from backtest.features import build_feature_matrix, build_labels


# ── Window descriptor ─────────────────────────────────────────────────────────

@dataclass
class Window:
    """Immutable descriptor for one rolling backtest window."""
    window_id:    int
    train_start:  pd.Timestamp
    train_end:    pd.Timestamp
    pred_start:   pd.Timestamp
    pred_end:     pd.Timestamp


# ── Window generation ─────────────────────────────────────────────────────────

def generate_windows(
    start_date: str | pd.Timestamp = START_DATE,
    end_date:   str | pd.Timestamp = END_DATE,
    train_months: int = TRAIN_MONTHS,
    predict_weeks: int = PREDICT_WEEKS,
    step_weeks:    int = STEP_WEEKS,
) -> list[Window]:
    """Generate the sequence of rolling windows for the backtest.

    Each window has a fixed-length training period ending at pred_start, and a
    prediction period of predict_weeks. Windows advance by step_weeks each iteration.
    Returns a list of Window objects covering [start_date, end_date].
    """
    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(end_date)

    windows = []
    i = 0
    while True:
        pred_start = start + pd.DateOffset(weeks=i * step_weeks)
        pred_end   = pred_start + pd.DateOffset(weeks=predict_weeks)
        if pred_start >= end:
            break
        train_end   = pred_start
        train_start = pred_start - pd.DateOffset(months=train_months)
        windows.append(Window(
            window_id=i,
            train_start=train_start,
            train_end=train_end,
            pred_start=pred_start,
            pred_end=min(pred_end, end + pd.DateOffset(days=1)),
        ))
        i += 1

    return windows


# ── Feature dropping ──────────────────────────────────────────────────────────

def _drop_feature_columns(X: np.ndarray, n_ohe: int) -> np.ndarray:
    """Drop configured features (e.g. sf_word_count) from the feature matrix.

    Takes the full feature matrix and number of OHE columns. Returns the matrix
    with DROP_FEATURES columns removed. If DROP_FEATURES is empty, returns X unchanged.
    """
    if not DROP_FEATURES:
        return X
    drop_idxs = []
    for feat in DROP_FEATURES:
        if feat in SF_COLS:
            drop_idxs.append(n_ohe + SF_COLS.index(feat))
    if not drop_idxs:
        return X
    return np.delete(X, drop_idxs, axis=1)


# ── Model training ────────────────────────────────────────────────────────────

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> GradientBoostingClassifier:
    """Train a GradientBoostingClassifier on the provided feature matrix and labels.

    Takes float32 feature array X_train and integer label array y_train.
    Returns a fitted GradientBoostingClassifier.
    Raises ValueError if y_train has fewer than 2 classes.
    """
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        raise ValueError(
            f"Training labels contain only one class: {unique_classes}. "
            "Cannot train a classifier."
        )

    clf = GradientBoostingClassifier(
        n_estimators=GBM_N_ESTIMATORS,
        max_depth=GBM_MAX_DEPTH,
        learning_rate=GBM_LEARNING_RATE,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    return clf


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(clf, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate binary predictions and decision scores from a fitted classifier.

    Takes a fitted classifier and feature array X.
    Returns (predictions, decision_scores) — both 1D arrays of length len(X).
    """
    preds = clf.predict(X)
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X)[:, 1]
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
    else:
        scores = preds.astype(float)
    return preds, scores


# ── Per-window signal generation ─────────────────────────────────────────────

def run_window(
    window: Window,
    announcements: pd.DataFrame,
    ohe: OneHotEncoder,
    horizon: str,
) -> pd.DataFrame | None:
    """Run one rolling window: train model, predict on prediction window, return signals.

    Takes a Window descriptor, the full filtered announcements DataFrame, a globally
    fitted OneHotEncoder, and a horizon ('5d' or '30d'). Returns a DataFrame of
    signal rows (prediction == 1) augmented with decision_score, or None if the
    window is skipped due to insufficient training data.
    """
    threshold = P85_5D if horizon == "5d" else P85_30D

    # Slice training and prediction data.
    # Exclude rows whose return label window extends past pred_start: a row published
    # on date D has return_30d realized at D+30, so including it in training when
    # pred_start < D+30 leaks future prices. Safe cutoff: pred_start - horizon_days.
    return_days = 5 if horizon == "5d" else 30
    clean_train_end = window.train_end - pd.DateOffset(days=return_days)
    train_df = announcements[
        (announcements["published_at"] >= window.train_start) &
        (announcements["published_at"] <  clean_train_end)
    ].copy()

    pred_df = announcements[
        (announcements["published_at"] >= window.pred_start) &
        (announcements["published_at"] <  window.pred_end)
    ].copy()

    n_train = len(train_df)

    if n_train < MIN_TRAIN_SAMPLES:
        warnings.warn(
            f"Window {window.window_id} skipped: only {n_train} announcements in "
            f"training period [{window.train_start.date()} – {window.train_end.date()}] "
            f"(min: {MIN_TRAIN_SAMPLES})"
        )
        return None

    # Build features and labels for training
    X_train = build_feature_matrix(ohe, train_df)
    y_train = build_labels(train_df, horizon, threshold)

    X_pred = build_feature_matrix(ohe, pred_df)

    # Drop configured features (e.g. sf_word_count)
    n_ohe = X_train.shape[1] - len(SF_COLS)
    X_train = _drop_feature_columns(X_train, n_ohe)
    X_pred = _drop_feature_columns(X_pred, n_ohe)

    # Skip if only one class in training labels (can't train)
    if len(np.unique(y_train)) < 2:
        warnings.warn(
            f"Window {window.window_id} skipped: training labels contain only one class."
        )
        return None

    clf = train_model(X_train, y_train)

    n_pos_train = int(y_train.sum())
    print(
        f"Window {window.window_id}: "
        f"training on {n_train} announcements ({n_pos_train} positives), "
        f"prediction window has {len(pred_df)} announcements."
    )

    if len(pred_df) == 0:
        return pd.DataFrame()  # empty — no signals, but window ran

    preds, scores = predict(clf, X_pred)

    # Annotate prediction DataFrame
    pred_df = pred_df.copy()
    pred_df["prediction"]     = preds
    pred_df["decision_score"] = scores
    pred_df["window_id"]      = window.window_id
    pred_df["horizon"]        = horizon

    # Return only positive signals (prediction == 1)
    signals = pred_df[pred_df["prediction"] == 1].copy()

    if len(signals) == 0:
        return None  # no signals — window produced nothing

    # Deduplicate: one position per ticker per window — keep FIRST announcement
    # (lowest published_at) to avoid look-ahead from later-window scores.
    signals = signals.sort_values("published_at", ascending=True)
    signals = signals.drop_duplicates(subset=["ticker"], keep="first")

    return signals.reset_index(drop=True)


# ── Rolling loop orchestrator ─────────────────────────────────────────────────

def run_rolling_loop(
    announcements: pd.DataFrame,
    ohe: OneHotEncoder,
    horizon: str,
    start_date: str | pd.Timestamp = START_DATE,
    end_date:   str | pd.Timestamp = END_DATE,
    train_months:  int = TRAIN_MONTHS,
    predict_weeks: int = PREDICT_WEEKS,
    step_weeks:    int = STEP_WEEKS,
) -> tuple[list[pd.DataFrame], list[Window]]:
    """Run the full rolling window loop for one horizon.

    Takes the filtered announcements DataFrame, a globally fitted OneHotEncoder,
    and horizon ('5d' or '30d'). Returns (signal_frames, windows) where signal_frames
    is a list of per-window signal DataFrames (None entries excluded).
    """
    windows = generate_windows(
        start_date=start_date,
        end_date=end_date,
        train_months=train_months,
        predict_weeks=predict_weeks,
        step_weeks=step_weeks,
    )
    signal_frames: list[pd.DataFrame] = []
    active_windows: list[Window] = []

    for window in windows:
        signals = run_window(window, announcements, ohe, horizon)
        if signals is not None:
            signal_frames.append(signals)
            active_windows.append(window)

    return signal_frames, active_windows
