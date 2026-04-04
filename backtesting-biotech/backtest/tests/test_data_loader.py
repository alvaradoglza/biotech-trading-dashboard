"""Tests for data_loader.py — load_announcements, filter_window, load_ohlcv, _normalize_ohlcv."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from backtest.data_loader import (
    load_announcements,
    filter_window,
    load_ohlcv,
    load_benchmark,
    _normalize_ohlcv,
)
from backtest.config import ANNOUNCEMENTS_PATH


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_ohlcv_df(n: int = 10, start: str = "2024-01-02") -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame for testing."""
    dates = pd.date_range(start, periods=n, freq="B")
    return pd.DataFrame({
        "date": dates,
        "open":   np.linspace(10, 20, n),
        "high":   np.linspace(11, 21, n),
        "low":    np.linspace(9,  19, n),
        "close":  np.linspace(10, 20, n),
        "volume": np.ones(n) * 1_000_000,
    })


def make_announcements_df(n: int = 5) -> pd.DataFrame:
    """Create a minimal announcements DataFrame for testing."""
    return pd.DataFrame({
        "ticker":       ["MRNA"] * n,
        "source":       ["clinicaltrials"] * n,
        "event_type":   ["CT_COMPLETED"] * n,
        "published_at": pd.date_range("2024-01-01", periods=n, freq="30D"),
        "raw_text":     ["Phase 3 trial met primary endpoint statistically significant"] * n,
        "return_5d":    [5.0] * n,
        "return_30d":   [10.0] * n,
        "parse_status": ["OK"] * n,
    })


# ── _normalize_ohlcv ─────────────────────────────────────────────────────────

def test_normalize_ohlcv_title_cases_columns():
    """_normalize_ohlcv should produce Open/High/Low/Close/Volume columns."""
    raw = make_ohlcv_df()
    result = _normalize_ohlcv(raw)
    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_normalize_ohlcv_sets_datetime_index():
    """_normalize_ohlcv should set a DatetimeIndex."""
    raw = make_ohlcv_df()
    result = _normalize_ohlcv(raw)
    assert isinstance(result.index, pd.DatetimeIndex)


def test_normalize_ohlcv_sorted_ascending():
    """_normalize_ohlcv should return rows sorted by date ascending."""
    raw = make_ohlcv_df(5).iloc[::-1].reset_index(drop=True)  # reverse order
    result = _normalize_ohlcv(raw)
    assert result.index.is_monotonic_increasing


def test_normalize_ohlcv_drops_null_close():
    """_normalize_ohlcv should drop rows where Close is NaN."""
    raw = make_ohlcv_df(5)
    raw.loc[2, "close"] = np.nan
    result = _normalize_ohlcv(raw)
    assert len(result) == 4


def test_normalize_ohlcv_raises_on_missing_column():
    """_normalize_ohlcv should raise ValueError when a required column is missing."""
    raw = make_ohlcv_df().drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing columns"):
        _normalize_ohlcv(raw)


# ── load_ohlcv ────────────────────────────────────────────────────────────────

def test_load_ohlcv_raises_if_file_missing():
    """load_ohlcv should raise FileNotFoundError for unknown tickers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            load_ohlcv("UNKNOWN_TICKER_XYZ", ohlcv_dir=tmpdir)


def test_load_ohlcv_reads_parquet():
    """load_ohlcv should load a parquet file and return a normalized OHLCV DataFrame."""
    raw = make_ohlcv_df(20)
    with tempfile.TemporaryDirectory() as tmpdir:
        norm = _normalize_ohlcv(raw)
        norm.to_parquet(Path(tmpdir) / "TEST.parquet")
        result = load_ohlcv("TEST", ohlcv_dir=tmpdir)
    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert isinstance(result.index, pd.DatetimeIndex)
    assert len(result) == 20


def test_load_ohlcv_reads_csv():
    """load_ohlcv should load a CSV file and return normalized OHLCV DataFrame."""
    raw = make_ohlcv_df(10)
    with tempfile.TemporaryDirectory() as tmpdir:
        raw.to_csv(Path(tmpdir) / "TEST2.csv", index=False)
        result = load_ohlcv("TEST2", ohlcv_dir=tmpdir)
    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]


# ── load_announcements ────────────────────────────────────────────────────────

def test_load_announcements_filters_edgar():
    """load_announcements should exclude rows where source contains 'edgar'."""
    df = load_announcements(ANNOUNCEMENTS_PATH)
    assert not df["source"].str.contains("edgar", case=False).any()


def test_load_announcements_filters_parse_status():
    """load_announcements should only return rows with parse_status == 'OK'."""
    df = load_announcements(ANNOUNCEMENTS_PATH)
    # If parse_status column survived, all should be OK
    if "parse_status" in df.columns:
        assert (df["parse_status"] == "OK").all()


def test_load_announcements_published_at_is_datetime():
    """published_at column should be datetime64 after loading."""
    df = load_announcements(ANNOUNCEMENTS_PATH)
    assert pd.api.types.is_datetime64_any_dtype(df["published_at"])


def test_load_announcements_has_required_columns():
    """load_announcements should return a DataFrame containing all REQUIRED_COLUMNS."""
    from backtest.config import REQUIRED_COLUMNS
    df = load_announcements(ANNOUNCEMENTS_PATH)
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"


def test_load_announcements_raises_on_bad_path():
    """load_announcements should raise when the file path does not exist."""
    with pytest.raises(Exception):
        load_announcements("/nonexistent/path/to/file.parquet")


# ── filter_window ─────────────────────────────────────────────────────────────

def test_filter_window_returns_correct_rows():
    """filter_window should return only rows within [start, end)."""
    df = make_announcements_df(5)
    df["published_at"] = pd.to_datetime(df["published_at"])
    start = pd.Timestamp("2024-02-01")
    end   = pd.Timestamp("2024-05-01")
    result = filter_window(df, start, end)
    assert (result["published_at"] >= start).all()
    assert (result["published_at"] < end).all()


def test_filter_window_empty_result_for_out_of_range():
    """filter_window should return an empty DataFrame when no rows match the window."""
    df = make_announcements_df(3)
    df["published_at"] = pd.to_datetime(df["published_at"])
    result = filter_window(df, pd.Timestamp("2010-01-01"), pd.Timestamp("2010-06-01"))
    assert len(result) == 0


def test_filter_window_does_not_mutate_input():
    """filter_window should not modify the original DataFrame."""
    df = make_announcements_df(5)
    df["published_at"] = pd.to_datetime(df["published_at"])
    original_len = len(df)
    filter_window(df, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-01"))
    assert len(df) == original_len


def test_filter_window_excludes_end_boundary():
    """filter_window is half-open [start, end) — end itself should be excluded."""
    df = make_announcements_df(1)
    df["published_at"] = pd.to_datetime(["2024-03-01"])
    result = filter_window(df, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-01"))
    assert len(result) == 0


# ── load_benchmark ─────────────────────────────────────────────────────────────

def test_load_benchmark_raises_if_missing():
    """load_benchmark should raise FileNotFoundError when the benchmark file is absent."""
    with pytest.raises(FileNotFoundError):
        load_benchmark("/nonexistent/SPY.parquet")
