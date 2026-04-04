"""
data_loader.py — Load and filter announcements, OHLCV files, and benchmark data.
All functions return pandas DataFrames ready for downstream processing.
"""

import warnings
from pathlib import Path

import pandas as pd

from backtest.config import (
    ANNOUNCEMENTS_PATH,
    OHLCV_DIR,
    BENCHMARK_PATH,
    REQUIRED_COLUMNS,
)


def load_announcements(path: str | Path = ANNOUNCEMENTS_PATH) -> pd.DataFrame:
    """Load, filter, and type-cast the announcements parquet file.

    Drops EDGAR rows and rows where parse_status != 'OK'. Converts published_at
    to datetime. Returns a DataFrame with REQUIRED_COLUMNS plus any extras present.
    """
    df = pd.read_parquet(path)

    # Keep only successfully parsed rows
    if "parse_status" in df.columns:
        df = df[df["parse_status"] == "OK"].copy()

    # Drop EDGAR — case-insensitive
    df = df[~df["source"].str.contains("edgar", case=False, na=False)].copy()

    # Validate required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Announcements data missing required columns: {missing}")

    # Normalize published_at to tz-naive datetime
    df["published_at"] = pd.to_datetime(df["published_at"], utc=False, errors="coerce")
    if df["published_at"].dt.tz is not None:
        df["published_at"] = df["published_at"].dt.tz_localize(None)

    df = df.reset_index(drop=True)
    return df


def filter_window(
    df: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    """Return rows where published_at falls within [window_start, window_end).

    Takes a pre-loaded announcements DataFrame and two Timestamps; returns a
    filtered copy (does not mutate the input).
    """
    mask = (df["published_at"] >= window_start) & (df["published_at"] < window_end)
    return df[mask].copy()


def load_ohlcv(ticker: str, ohlcv_dir: str | Path = OHLCV_DIR) -> pd.DataFrame:
    """Load OHLCV data for a single ticker from the local parquet directory.

    Returns a DataFrame with a DatetimeIndex and columns Open/High/Low/Close/Volume
    (title-cased), as required by backtesting.py. Raises FileNotFoundError if the
    ticker file does not exist.
    """
    ohlcv_dir = Path(ohlcv_dir)
    candidates = [
        ohlcv_dir / f"{ticker}.parquet",
        ohlcv_dir / f"{ticker}.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"No OHLCV file found for ticker '{ticker}' in {ohlcv_dir}")

    if path.suffix == ".csv":
        price_df = pd.read_csv(path)
    else:
        price_df = pd.read_parquet(path)

    price_df = _normalize_ohlcv(price_df)
    return price_df


def load_benchmark(path: str | Path = BENCHMARK_PATH) -> pd.DataFrame:
    """Load SPY (or other benchmark) OHLCV data from a local parquet or CSV file.

    Returns a DataFrame with DatetimeIndex and Open/High/Low/Close/Volume columns,
    same format as load_ohlcv(). Raises FileNotFoundError if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    if path.suffix == ".csv":
        price_df = pd.read_csv(path)
    else:
        price_df = pd.read_parquet(path)

    return _normalize_ohlcv(price_df)


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw OHLCV DataFrame to backtesting.py format.

    Expects a date column (named 'date', 'Date', or index), and OHLCV columns
    (lowercase or title-case). If 'adjusted_close' is present, all OHLC prices
    are adjusted using the ratio adjusted_close/close so that splits and reverse
    splits are handled correctly. Returns a DataFrame with DatetimeIndex and
    columns exactly named: Open, High, Low, Close, Volume.
    """
    df = df.copy()

    # Identify and set the date index
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    if date_col is not None:
        df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.drop(columns=[date_col] if date_col != "Date" else [])
        df = df.set_index("Date")
    else:
        df.index = pd.to_datetime(df.index, errors="coerce")
        df.index.name = "Date"

    # Remove timezone info if present
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Detect adjusted_close column (case-insensitive)
    adj_col = next((c for c in df.columns if c.lower() == "adjusted_close"), None)

    # Title-case OHLCV columns
    rename_map = {}
    for col in df.columns:
        lower = col.lower()
        for target in ["open", "high", "low", "close", "volume"]:
            if lower == target:
                rename_map[col] = target.title()
    df = df.rename(columns=rename_map)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV data missing columns after normalization: {missing}")

    # Adjust OHLC for splits using adjusted_close / close ratio
    if adj_col is not None and adj_col not in rename_map:
        import numpy as np
        adj_values = df[adj_col].values.astype(float)
        raw_close = df["Close"].values.astype(float)
        # Compute adjustment factor; where close is 0 or NaN, factor = 1
        factor = np.where(
            (raw_close == 0) | np.isnan(raw_close),
            1.0,
            adj_values / raw_close,
        )
        factor = np.where(np.isfinite(factor), factor, 1.0)

        df["Open"] = df["Open"] * factor
        df["High"] = df["High"] * factor
        df["Low"] = df["Low"] * factor
        df["Close"] = adj_values  # use adjusted_close directly

    df = df[required].sort_index()
    df = df.dropna(subset=["Close"])
    # Drop zero-volume bars — these are bad data points (price with no trading activity)
    df = df[df["Volume"] > 0]
    return df
