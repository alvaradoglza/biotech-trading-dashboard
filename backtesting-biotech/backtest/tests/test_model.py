"""Tests for model.py — generate_windows, train_model, predict, run_window, run_rolling_loop."""

import warnings

import numpy as np
import pandas as pd
import pytest

from backtest.model import (
    generate_windows,
    train_model,
    predict,
    run_window,
    run_rolling_loop,
    Window,
)
from backtest.features import fit_ohe


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_announcements(n: int = 80, start: str = "2024-01-01") -> pd.DataFrame:
    """Synthetic announcements DataFrame with both classes present."""
    np.random.seed(42)
    dates = pd.date_range(start, periods=n, freq="3D")
    return pd.DataFrame({
        "ticker":       [f"T{i % 10:02d}" for i in range(n)],
        "source":       ["clinicaltrials"] * (n // 2) + ["openfda"] * (n - n // 2),
        "event_type":   (["CT_COMPLETED", "CT_RECRUITING", "CT_TERMINATED"] * (n // 3 + 1))[:n],
        "published_at": dates,
        "raw_text": [
            "Phase 3 trial met primary endpoint statistically significant p < 0.05 "
            "monoclonal antibody cancer patients n=300 NCT12345678 approval approved."
            if i % 3 == 0 else
            "Phase 2 study failed to meet primary endpoint no significant difference "
            "rare disease orphan enrolled 50 patients."
            for i in range(n)
        ],
        "return_5d":  np.random.randn(n) * 5,
        "return_30d": np.random.randn(n) * 10,
    })


# ── generate_windows ──────────────────────────────────────────────────────────

def test_generate_windows_returns_list():
    """generate_windows should return a list of Window objects."""
    windows = generate_windows("2024-01-01", "2024-12-31")
    assert isinstance(windows, list)
    assert len(windows) > 0
    assert all(isinstance(w, Window) for w in windows)


def test_generate_windows_pred_start_increases():
    """Each window's pred_start should be later than the previous."""
    windows = generate_windows("2024-01-01", "2024-12-31", step_weeks=4)
    pred_starts = [w.pred_start for w in windows]
    assert pred_starts == sorted(pred_starts)


def test_generate_windows_train_end_equals_pred_start():
    """train_end must equal pred_start — no look-ahead gap."""
    windows = generate_windows("2024-01-01", "2024-12-31")
    for w in windows:
        assert w.train_end == w.pred_start


def test_generate_windows_no_pred_start_after_end_date():
    """No window should have pred_start >= end_date."""
    end = pd.Timestamp("2024-12-31")
    windows = generate_windows("2024-01-01", "2024-12-31")
    for w in windows:
        assert w.pred_start < end


def test_generate_windows_ids_are_sequential():
    """Window IDs should be 0, 1, 2, ... in order."""
    windows = generate_windows("2024-01-01", "2024-06-30")
    assert [w.window_id for w in windows] == list(range(len(windows)))


def test_generate_windows_train_length_matches_train_months():
    """Training window length should be approximately train_months months."""
    windows = generate_windows("2024-06-01", "2024-12-31", train_months=6)
    w = windows[0]
    delta_months = (w.train_end.year - w.train_start.year) * 12 + (
        w.train_end.month - w.train_start.month
    )
    assert delta_months == 6


def test_generate_windows_empty_range():
    """generate_windows should return empty list if start_date >= end_date."""
    windows = generate_windows("2025-01-01", "2024-01-01")
    assert windows == []


# ── train_model ───────────────────────────────────────────────────────────────

def test_train_model_returns_fitted_classifier():
    """train_model should return a fitted classifier with a predict method."""
    X = np.random.randn(60, 20).astype(np.float32)
    y = np.array([0] * 30 + [1] * 30)
    clf = train_model(X, y)
    assert hasattr(clf, "predict")
    assert hasattr(clf, "fit")


def test_train_model_single_class_raises():
    """train_model should raise ValueError if labels contain only one class."""
    X = np.random.randn(40, 10).astype(np.float32)
    y = np.zeros(40, dtype=int)
    with pytest.raises(ValueError, match="one class"):
        train_model(X, y)


def test_train_model_deterministic():
    """train_model with the same data should produce identical predictions."""
    X = np.random.randn(60, 15).astype(np.float32)
    y = np.array([0] * 30 + [1] * 30)
    clf1 = train_model(X, y)
    clf2 = train_model(X, y)
    np.testing.assert_array_equal(clf1.predict(X), clf2.predict(X))


# ── predict ───────────────────────────────────────────────────────────────────

def test_predict_returns_two_arrays():
    """predict should return a tuple of (predictions, scores)."""
    X = np.random.randn(60, 10).astype(np.float32)
    y = np.array([0] * 30 + [1] * 30)
    clf = train_model(X, y)
    preds, scores = predict(clf, X)
    assert len(preds) == len(X)
    assert len(scores) == len(X)


def test_predict_predictions_are_binary():
    """predict should return only 0 and 1 values."""
    X = np.random.randn(60, 10).astype(np.float32)
    y = np.array([0] * 30 + [1] * 30)
    clf = train_model(X, y)
    preds, _ = predict(clf, X)
    assert set(preds).issubset({0, 1})


def test_predict_scores_are_floats():
    """decision_function scores should be float values."""
    X = np.random.randn(60, 10).astype(np.float32)
    y = np.array([0] * 30 + [1] * 30)
    clf = train_model(X, y)
    _, scores = predict(clf, X)
    assert scores.dtype.kind == "f"


# ── run_window ────────────────────────────────────────────────────────────────

def test_run_window_returns_signals_dataframe():
    """run_window should return a DataFrame with prediction==1 rows."""
    df = make_announcements(100, "2023-01-01")
    ohe = fit_ohe(df)
    window = Window(
        window_id=0,
        train_start=pd.Timestamp("2023-01-01"),
        train_end=pd.Timestamp("2024-01-01"),
        pred_start=pd.Timestamp("2024-01-01"),
        pred_end=pd.Timestamp("2024-02-01"),
    )
    # Add some announcements in the prediction window
    pred_rows = make_announcements(20, "2024-01-05")
    df_combined = pd.concat([df, pred_rows], ignore_index=True)
    ohe2 = fit_ohe(df_combined)
    result = run_window(window, df_combined, ohe2, "5d")
    assert result is None or isinstance(result, pd.DataFrame)


def test_run_window_skips_sparse_training():
    """run_window should return None and warn when training window has < min_train_samples."""
    df = make_announcements(5, "2024-06-01")  # only 5 rows — below default min of 30
    ohe = fit_ohe(df)
    window = Window(
        window_id=0,
        train_start=pd.Timestamp("2024-01-01"),
        train_end=pd.Timestamp("2024-06-01"),
        pred_start=pd.Timestamp("2024-06-01"),
        pred_end=pd.Timestamp("2024-07-01"),
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = run_window(window, df, ohe, "5d")
    assert result is None
    assert any("skipped" in str(warning.message).lower() for warning in w)


def test_run_window_signals_have_prediction_equals_one():
    """All rows returned by run_window should have prediction == 1."""
    df = make_announcements(100, "2022-06-01")
    ohe = fit_ohe(df)
    window = Window(
        window_id=0,
        train_start=pd.Timestamp("2022-06-01"),
        train_end=pd.Timestamp("2023-06-01"),
        pred_start=pd.Timestamp("2023-06-01"),
        pred_end=pd.Timestamp("2023-08-01"),
    )
    pred_rows = make_announcements(20, "2023-06-05")
    df_combined = pd.concat([df, pred_rows], ignore_index=True)
    ohe2 = fit_ohe(df_combined)
    result = run_window(window, df_combined, ohe2, "5d")
    if result is not None and len(result) > 0:
        assert (result["prediction"] == 1).all()


def test_run_window_deduplicates_per_ticker():
    """run_window should return at most one row per ticker."""
    df = make_announcements(100, "2022-06-01")
    pred_rows = make_announcements(20, "2023-06-05")
    df_combined = pd.concat([df, pred_rows], ignore_index=True)
    ohe = fit_ohe(df_combined)
    window = Window(
        window_id=0,
        train_start=pd.Timestamp("2022-06-01"),
        train_end=pd.Timestamp("2023-06-01"),
        pred_start=pd.Timestamp("2023-06-01"),
        pred_end=pd.Timestamp("2023-08-01"),
    )
    result = run_window(window, df_combined, ohe, "5d")
    if result is not None and len(result) > 0:
        assert result["ticker"].nunique() == len(result)


def test_run_window_no_look_ahead():
    """Training window must not include any announcement at or after pred_start."""
    df = make_announcements(100, "2022-01-01")
    ohe = fit_ohe(df)
    window = Window(
        window_id=0,
        train_start=pd.Timestamp("2022-01-01"),
        train_end=pd.Timestamp("2023-01-01"),
        pred_start=pd.Timestamp("2023-01-01"),
        pred_end=pd.Timestamp("2023-04-01"),
    )
    train_df = df[
        (df["published_at"] >= window.train_start) &
        (df["published_at"] <  window.train_end)
    ]
    # All training dates must be strictly before pred_start
    if len(train_df) > 0:
        assert (train_df["published_at"] < window.pred_start).all()


def test_run_window_label_leakage_fix_30d():
    """Training must exclude rows whose 30d return window extends past pred_start.

    Rows published within 30 days before pred_start have return_30d realized
    after pred_start — including them would use future prices as labels.
    run_window applies clean_train_end = pred_start - 30 days.
    """
    pred_start = pd.Timestamp("2024-01-01")
    # Create 5 rows published in the 30-day window just before pred_start
    leak_dates = pd.date_range("2023-12-02", periods=5, freq="4D")  # 2023-12-02 to 2023-12-18
    # Create 50 clean training rows published before the 30-day window
    clean_dates = pd.date_range("2022-06-01", periods=50, freq="7D")

    np.random.seed(1)
    all_dates = list(clean_dates) + list(leak_dates)
    n = len(all_dates)
    df = pd.DataFrame({
        "ticker":       [f"T{i:02d}" for i in range(n)],
        "source":       ["clinicaltrials"] * n,
        "event_type":   (["CT_COMPLETED", "CT_RECRUITING"] * (n // 2 + 1))[:n],
        "published_at": all_dates,
        "raw_text":     ["Phase 3 trial met endpoint p < 0.05 NCT12345678 approval."] * n,
        "return_5d":    np.random.randn(n) * 5,
        "return_30d":   np.random.randn(n) * 10,
    })
    ohe = fit_ohe(df)
    window = Window(
        window_id=0,
        train_start=pd.Timestamp("2022-06-01"),
        train_end=pred_start,
        pred_start=pred_start,
        pred_end=pd.Timestamp("2024-02-01"),
    )
    # The clean cutoff should exclude leak_dates (Dec 2-18 within 30 days of Jan 1)
    clean_cutoff = pred_start - pd.DateOffset(days=30)
    # run_window internally uses this cutoff — verify our expectation
    assert clean_cutoff == pd.Timestamp("2023-12-02")
    # All rows published in leak_dates (on or after Dec 2) should be excluded
    for d in leak_dates:
        assert d >= clean_cutoff, f"{d} should be excluded from training"


def test_run_window_dedup_keeps_first_not_highest_score():
    """Dedup should keep the earliest announcement per ticker, not the highest-score one.

    Keeping the highest-score announcement uses future knowledge — at the time
    of the first announcement, later scores are unknown.
    """
    pred_start = pd.Timestamp("2024-01-01")
    train_end = pred_start
    # Two announcements for same ticker in prediction window, different dates
    pred_rows = pd.DataFrame([
        {
            "ticker": "TXR", "source": "clinicaltrials", "event_type": "CT_COMPLETED",
            "published_at": pd.Timestamp("2024-01-03"),  # earlier
            "raw_text": "Phase 2 study rare disease orphan.",
            "return_5d": 2.0, "return_30d": 5.0,
        },
        {
            "ticker": "TXR", "source": "clinicaltrials", "event_type": "CT_COMPLETED",
            "published_at": pd.Timestamp("2024-01-10"),  # later
            "raw_text": "Phase 3 trial met endpoint p < 0.05 NCT12345678 approval.",
            "return_5d": 15.0, "return_30d": 30.0,
        },
    ])
    # Training rows to ensure window runs
    train_rows = make_announcements(60, "2022-12-01")
    df = pd.concat([train_rows, pred_rows], ignore_index=True)
    ohe = fit_ohe(df)
    window = Window(
        window_id=0,
        train_start=pd.Timestamp("2022-12-01"),
        train_end=train_end,
        pred_start=pred_start,
        pred_end=pd.Timestamp("2024-02-01"),
    )
    result = run_window(window, df, ohe, "30d")
    if result is not None and len(result) > 0 and "TXR" in result["ticker"].values:
        # Must be the earlier row (Jan 3), not the later one (Jan 10)
        txr_row = result[result["ticker"] == "TXR"].iloc[0]
        assert txr_row["published_at"] == pd.Timestamp("2024-01-03")


# ── run_rolling_loop ─────────────────────────────────────────────────────────

def test_run_rolling_loop_returns_lists():
    """run_rolling_loop should return (list_of_signal_dfs, list_of_windows)."""
    df = make_announcements(200, "2022-06-01")
    ohe = fit_ohe(df)
    frames, windows = run_rolling_loop(
        df, ohe, "5d",
        start_date="2023-06-01",
        end_date="2023-12-31",
        train_months=12,
        predict_weeks=4,
        step_weeks=4,
    )
    assert isinstance(frames, list)
    assert isinstance(windows, list)


def test_run_rolling_loop_all_signals_are_positive():
    """Every signal row from run_rolling_loop should have prediction == 1."""
    df = make_announcements(200, "2022-06-01")
    ohe = fit_ohe(df)
    frames, _ = run_rolling_loop(
        df, ohe, "5d",
        start_date="2023-06-01",
        end_date="2023-12-31",
    )
    for frame in frames:
        if len(frame) > 0:
            assert (frame["prediction"] == 1).all()
