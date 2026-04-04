#!/usr/bin/env python3
"""Update Parquet dataset from CSV and text files.

Usage:
    python scripts/update_parquet.py
    python scripts/update_parquet.py --stats       # Show current stats only
    python scripts/update_parquet.py --ml-ready    # Show ML-ready dataset stats
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Update Parquet dataset from CSV and text files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/update_parquet.py                      # Regenerate Parquet
    python scripts/update_parquet.py --stats              # Show current stats
    python scripts/update_parquet.py --stats --ml-ready   # Include ML-ready count
        """,
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show current stats only (no update)",
    )
    parser.add_argument(
        "--ml-ready",
        action="store_true",
        help="Show ML-ready dataset stats (requires --stats or runs update first)",
    )
    parser.add_argument(
        "--csv",
        default="data/index/announcements.csv",
        help="Path to announcements CSV (default: data/index/announcements.csv)",
    )
    parser.add_argument(
        "--parquet",
        default="data/index/announcements.parquet",
        help="Path to output Parquet file (default: data/index/announcements.parquet)",
    )

    args = parser.parse_args()

    from src.storage.parquet_sync import ParquetSync
    import pandas as pd

    sync = ParquetSync(
        csv_path=Path(args.csv),
        parquet_path=Path(args.parquet),
    )

    if args.stats:
        parquet_path = Path(args.parquet)
        if not parquet_path.exists():
            print("❌ Parquet file not found. Run without --stats first to generate it.")
            return 1

        df = sync.get_dataframe()
        print("\n📊 Parquet Dataset Statistics")
        print("=" * 40)
        print(f"Total records:    {len(df)}")
        print(f"With text:        {(df['raw_text'].str.len() > 0).sum()}")
        if "return_30d" in df.columns:
            print(f"With 30d return:  {df['return_30d'].notna().sum()}")
        print(f"Unique tickers:   {df['ticker'].nunique()}")
        print(f"File size:        {parquet_path.stat().st_size / 1024 / 1024:.2f} MB")

        if args.ml_ready:
            ml_df = sync.get_ml_dataset()
            print(f"\n🤖 ML-Ready Dataset: {len(ml_df)} records")
            print(f"   (min 100 chars text + return_30d present)")

        return 0

    # Run update
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ CSV not found: {args.csv}")
        print("   Run 'python scripts/run_extraction.py' first.")
        return 1

    print("📦 Updating Parquet dataset...")
    stats = sync.update()

    print("\n✅ Parquet updated!")
    print(f"   Total records: {stats['total_records']}")
    print(f"   Text loaded:   {stats['text_loaded']}")
    print(f"   Text empty:    {stats['text_empty']}")
    print(f"   Text missing:  {stats['text_missing']}")
    print(f"   File size:     {stats['parquet_size_mb']} MB")
    print(f"\n📁 Output: {args.parquet}")

    if args.ml_ready:
        ml_df = sync.get_ml_dataset()
        print(f"\n🤖 ML-Ready Dataset: {len(ml_df)} records")
        print(f"   (min 100 chars text + return_30d present)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
