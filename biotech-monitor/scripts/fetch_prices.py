"""Fetch 10 years of daily OHLCV data from EODHD for all tickers in announcements.parquet.

Saves one file per ticker to data/prices/TICKER.parquet, plus SPY.parquet.

Usage:
    python scripts/fetch_prices.py                  # All tickers + SPY
    python scripts/fetch_prices.py --ticker MRNA    # Single ticker
    python scripts/fetch_prices.py --skip-existing  # Skip already-downloaded tickers
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clients.eodhd import EODHDClient
from src.utils.logging import get_logger

load_dotenv()
logger = get_logger(__name__)

PRICES_DIR = Path("data/prices")
ANNOUNCEMENTS_PARQUET = Path("data/index/announcements.parquet")
DAYS_BACK = 3650  # 10 years


def get_tickers() -> list[str]:
    df = pd.read_parquet(ANNOUNCEMENTS_PARQUET)
    tickers = sorted(df["ticker"].dropna().unique().tolist())
    return tickers


async def fetch_and_save(client: EODHDClient, ticker: str, skip_existing: bool) -> bool:
    out_path = PRICES_DIR / f"{ticker}.parquet"

    if skip_existing and out_path.exists():
        print(f"  ⊘ {ticker}: already exists, skipping")
        return True

    from_date = (datetime.now() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")

    try:
        # SPY uses its own exchange suffix
        exchange = "US"
        rows = await client.get_historical_prices(
            ticker=ticker,
            exchange=exchange,
            from_date=from_date,
            to_date=to_date,
        )

        if not rows:
            print(f"  ✗ {ticker}: no data returned")
            return False

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Standardise columns to lowercase
        df.columns = [c.lower() for c in df.columns]

        df.to_parquet(out_path, index=False)
        print(f"  ✓ {ticker}: {len(df)} rows  ({df['date'].min().date()} → {df['date'].max().date()})")
        return True

    except Exception as e:
        print(f"  ✗ {ticker}: {e}")
        return False


async def main(ticker_arg: str | None, skip_existing: bool) -> None:
    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        print("ERROR: EODHD_API_KEY not set in .env")
        sys.exit(1)

    PRICES_DIR.mkdir(parents=True, exist_ok=True)

    if ticker_arg:
        tickers = [ticker_arg.upper()]
    else:
        tickers = get_tickers()
        # Always include SPY as benchmark
        if "SPY" not in tickers:
            tickers = ["SPY"] + tickers

    print(f"Fetching OHLCV for {len(tickers)} tickers  (10 years, from_date={( datetime.now() - timedelta(days=DAYS_BACK)).strftime('%Y-%m-%d')})")
    print(f"Output dir: {PRICES_DIR.resolve()}\n")

    ok = 0
    failed = 0

    async with EODHDClient(api_key=api_key, requests_per_second=5.0) as client:
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] {ticker}")
            success = await fetch_and_save(client, ticker, skip_existing)
            if success:
                ok += 1
            else:
                failed += 1

    print(f"\n{'='*50}")
    print(f"Done.  ✓ {ok} succeeded  ✗ {failed} failed")
    print(f"Files in {PRICES_DIR}: {len(list(PRICES_DIR.glob('*.parquet')))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", help="Fetch a single ticker only")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tickers that already have a parquet file",
    )
    args = parser.parse_args()

    asyncio.run(main(args.ticker, args.skip_existing))
