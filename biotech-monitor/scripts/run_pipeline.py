#!/usr/bin/env python3
"""Master pipeline orchestrator for the biotech monitoring system.

Runs the full data pipeline in order:
1. Fetch/update stock list (optional)
2. Extract announcements (ClinicalTrials + OpenFDA)
3. Calculate post-announcement returns (30/60/90 days)

Usage:
    python scripts/run_pipeline.py                     # Run everything
    python scripts/run_pipeline.py --no-stocks         # Skip stock fetch
    python scripts/run_pipeline.py --no-returns        # Skip return calculation
    python scripts/run_pipeline.py --limit 10          # Limit announcements per source
"""

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'=' * 60}")
    print(f" {description}")
    print(f"{'=' * 60}")
    print(f"  Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n  Error: Command failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n  Error: Python interpreter not found")
        return False


async def run_pipeline(args):
    """Run the full pipeline based on command line arguments."""
    from dotenv import load_dotenv
    import os

    load_dotenv()

    start_time = datetime.now()
    python_cmd = sys.executable

    if args.returns:
        if not os.getenv("EODHD_API_KEY"):
            print("Error: EODHD_API_KEY not set in environment (required for returns)")
            return 1

    print("\n" + "=" * 60)
    print(" BIOTECH MONITOR PIPELINE")
    print("=" * 60)
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Stocks: {'Yes' if args.stocks else 'Skip'}")
    print(f"  Announcements: {'Yes' if args.announcements else 'Skip'}")
    print(f"  Returns: {'Yes' if args.returns else 'Skip'}")
    if args.limit:
        print(f"  Limit per source: {args.limit}")
    if args.source:
        print(f"  Source filter: {args.source}")

    results = {}

    # Phase 1: Fetch stock list
    if args.stocks:
        cmd = [python_cmd, "scripts/fetch_stock_list.py"]
        results["stocks"] = run_command(cmd, "PHASE 1: Fetching Stock List")
        if not results["stocks"]:
            print("\nStock fetch failed. Continuing anyway...")

    # Phase 2: Extract announcements
    if args.announcements:
        cmd = [python_cmd, "scripts/run_extraction.py", "--all"]

        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        if args.source:
            cmd.extend(["--source", args.source])
        if args.stocks_csv:
            cmd.extend(["--stocks-csv", args.stocks_csv])
        if args.days_back != 1095:
            cmd.extend(["--days-back", str(args.days_back)])

        results["announcements"] = run_command(cmd, "PHASE 2: Extracting Announcements")
        if not results["announcements"]:
            print("\nAnnouncement extraction failed. Continuing anyway...")

    # Phase 3: Calculate returns
    if args.returns:
        announcements_path = Path("data/index/announcements.csv")
        if not announcements_path.exists():
            print("\nWarning: No announcements found. Skipping return calculation.")
            results["returns"] = False
        else:
            cmd = [python_cmd, "scripts/calculate_returns.py"]
            results["returns"] = run_command(cmd, "PHASE 3: Calculating Returns")
            if not results["returns"]:
                print("\nReturn calculation failed. Continuing anyway...")

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Duration: {duration}")
    print(f"\n  Results:")
    for phase, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"    {phase}: {status}")

    print(f"\n  Output files:")
    if args.stocks:
        print(f"    Stocks: data/stocks.csv")
    if args.announcements:
        print(f"    Announcements: data/index/announcements.csv")
        print(f"    Parquet: data/index/announcements.parquet")
        print(f"    Raw files: data/raw/")
        print(f"    Text files: data/text/")
    if args.returns:
        print(f"    Returns: data/index/announcements.csv (updated with return_5d/30d/60d/90d)")

    if False in results.values():
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run the biotech monitor data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_pipeline.py                          # Run full pipeline
    python scripts/run_pipeline.py --no-stocks              # Skip stock fetch
    python scripts/run_pipeline.py --no-returns             # Skip return calculation
    python scripts/run_pipeline.py --limit 10               # Limit to 10 per source
    python scripts/run_pipeline.py --source clinicaltrials  # Only fetch ClinicalTrials
        """
    )

    parser.add_argument(
        "--stocks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch/update stock list (default: True)",
    )
    parser.add_argument(
        "--announcements",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract announcements (default: True)",
    )
    parser.add_argument(
        "--returns",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Calculate post-announcement returns (default: True)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit announcements per source per stock",
    )
    parser.add_argument(
        "--source",
        choices=["clinicaltrials", "openfda"],
        help="Only fetch from specific source",
    )
    parser.add_argument(
        "--stocks-csv",
        default=None,
        help="Path to stocks CSV file (default: data/stocks.csv)",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=1095,
        help="Days of history to fetch (default: 1095 = 3 years)",
    )

    args = parser.parse_args()

    return asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    sys.exit(main())
