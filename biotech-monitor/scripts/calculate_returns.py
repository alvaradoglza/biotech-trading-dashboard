#!/usr/bin/env python3
"""Calculate post-announcement returns for all announcements.

This script calculates stock price returns at 30, 60, and 90 calendar days
after each announcement date.

Usage:
    python scripts/calculate_returns.py           # Calculate missing returns
    python scripts/calculate_returns.py --stats   # Show current statistics
    python scripts/calculate_returns.py --force   # Recalculate all returns
    python scripts/calculate_returns.py --ticker MRNA  # Single ticker only
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


async def run_calculation(args):
    """Run return calculation based on command line arguments."""
    from src.returns.pipeline import ReturnPipeline

    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        print("Error: EODHD_API_KEY not set in environment")
        return 1

    pipeline = ReturnPipeline(
        eodhd_api_key=api_key,
        announcements_csv=args.announcements,
    )

    if args.stats:
        stats = pipeline.get_stats()
        print("\n" + "=" * 60)
        print(" RETURN CALCULATION STATISTICS")
        print("=" * 60)
        print(f"  Total announcements: {stats['total']}")
        print(f"  With returns: {stats['with_returns']}")
        print(f"  Without returns: {stats['without_returns']}")
        print(f"\n  By parse status:")
        for status, count in sorted(stats["by_status"].items()):
            print(f"    {status}: {count}")
        return 0

    print("\n" + "=" * 60)
    print(" CALCULATING POST-ANNOUNCEMENT RETURNS")
    print("=" * 60)
    print(f"  Announcements: {args.announcements}")
    print(f"  Force recalculate: {args.force}")
    if args.ticker:
        print(f"  Ticker filter: {args.ticker}")
    print()

    async with pipeline:
        if args.ticker:
            # Process single ticker
            from src.returns.pipeline import RETURN_COLUMNS
            import csv

            # Load and filter announcements
            announcements = pipeline._load_announcements()
            ticker_anns = [a for a in announcements if a.get("ticker") == args.ticker]

            if not ticker_anns:
                print(f"No announcements found for ticker: {args.ticker}")
                return 1

            print(f"Processing {len(ticker_anns)} announcements for {args.ticker}...")

            stats = await pipeline.process_ticker(args.ticker, ticker_anns, args.force)

            # Save back
            fieldnames = list(announcements[0].keys())
            for col in RETURN_COLUMNS:
                if col not in fieldnames:
                    fieldnames.append(col)

            with open(args.announcements, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(announcements)

            print(f"\nResults for {args.ticker}:")
            print(f"  Calculated: {stats['calculated']}")
            print(f"  Skipped: {stats['skipped']}")
            print(f"  Failed: {stats['failed']}")
        else:
            # Process all tickers
            totals = await pipeline.run(force=args.force)

            print("\n" + "=" * 60)
            print(" CALCULATION COMPLETE")
            print("=" * 60)
            print(f"  Total announcements: {totals['total']}")
            print(f"  Calculated: {totals['calculated']}")
            print(f"  Skipped: {totals['skipped']}")
            print(f"  Failed: {totals['failed']}")

    # Sync Parquet dataset
    try:
        from src.storage.parquet_sync import ParquetSync
        print("\n📦 Syncing Parquet dataset...")
        parquet_sync = ParquetSync()
        parquet_stats = parquet_sync.update()
        print(f"   Parquet updated: {parquet_stats['text_loaded']} records with text")
    except FileNotFoundError:
        pass  # No CSV yet
    except Exception as e:
        print(f"   ⚠️  Parquet sync failed: {e}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Calculate post-announcement stock returns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/calculate_returns.py           # Calculate missing returns
    python scripts/calculate_returns.py --stats   # Show statistics
    python scripts/calculate_returns.py --force   # Recalculate all
    python scripts/calculate_returns.py --ticker MRNA  # Single ticker
        """
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics only, don't calculate",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recalculation of all returns",
    )
    parser.add_argument(
        "--ticker",
        help="Process specific ticker only",
    )
    parser.add_argument(
        "--announcements",
        default="data/index/announcements.csv",
        help="Path to announcements CSV file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    return asyncio.run(run_calculation(args))


if __name__ == "__main__":
    sys.exit(main())
