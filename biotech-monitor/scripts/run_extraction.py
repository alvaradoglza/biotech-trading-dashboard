#!/usr/bin/env python3
"""Run the full text extraction pipeline.

This script processes announcements from all sources (SEC EDGAR, ClinicalTrials.gov,
OpenFDA), extracts text content, and stores both raw files and extracted text.

Usage:
    python scripts/run_extraction.py
    python scripts/run_extraction.py --ticker MRNA
    python scripts/run_extraction.py --source edgar
    python scripts/run_extraction.py --retry-failed
    python scripts/run_extraction.py --stats
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import pandas as pd

load_dotenv()


async def run_for_all_stocks(
    pipeline,
    limit: int = 5,
    source: str = None,
    stocks_csv: str = "data/stocks.csv",
    days_back: int = 1095,
):
    """Run extraction for all stocks in stocks.csv.

    Args:
        pipeline: ExtractionPipeline instance
        limit: Maximum filings to fetch per source per stock
        source: Optional source filter
        stocks_csv: Path to stocks CSV file
    """
    csv_path = Path(stocks_csv)
    if not csv_path.exists():
        print(f"❌ Stocks file not found: {stocks_csv}")
        print("   Run 'python scripts/fetch_stock_list.py' first to create it.")
        return

    df = pd.read_csv(csv_path)
    total_stocks = len(df)

    print(f"\n📋 Processing {total_stocks} stocks from {stocks_csv}")
    print("=" * 60)

    errors = 0

    for idx, row in df.iterrows():
        ticker = row["ticker"]
        company_name = row.get("company_name", "")

        print(f"\n[{idx + 1}/{total_stocks}] {ticker}", end="")
        if company_name:
            print(f" - {company_name[:40]}", end="")
        print()

        try:
            await run_for_ticker(
                pipeline=pipeline,
                ticker=ticker,
                limit=limit,
                source=source,
                company_name=company_name if pd.notna(company_name) else None,
                days_back=days_back,
            )
        except Exception as e:
            print(f"  ❌ Error processing {ticker}: {e}")
            errors += 1

    stats = pipeline.get_stats()
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    print(f"Stocks processed: {total_stocks}")
    print(f"Total announcements: {stats['total']}")
    print(f"  OK: {stats['by_status'].get('OK', 0)}")
    print(f"  Failed: {stats['by_status'].get('FAILED', 0)}")
    print(f"  Errors: {errors}")


async def run_for_ticker(
    pipeline,
    ticker: str,
    limit: int = 10,
    source: str = None,
    company_name: str = None,
    days_back: int = 1095,
):
    """Run extraction for a specific ticker.

    Args:
        pipeline: ExtractionPipeline instance
        ticker: Stock ticker symbol (e.g., "MRNA")
        limit: Maximum filings to fetch per source
        source: Optional source filter ("clinicaltrials", "openfda")
        company_name: Company name for ClinicalTrials/OpenFDA searches
    """
    from src.clients.clinicaltrials import ClinicalTrialsClient
    from src.clients.openfda import OpenFDAClient

    openfda_key = os.getenv("OPENFDA_API_KEY")

    search_name = company_name or ticker

    print(f"\n🔍 Processing announcements for {ticker}" + (f" ({company_name})" if company_name else ""))
    print("=" * 50)

    processed = {"clinicaltrials": 0, "openfda": 0}

    # ClinicalTrials.gov
    if source is None or source == "clinicaltrials":
        print(f"\n🧪 Fetching ClinicalTrials.gov studies (last {days_back} days)...")
        try:
            ct_client = ClinicalTrialsClient()
            trials = await ct_client.search_by_sponsor(search_name, days_back=days_back)
            print(f"  Found {len(trials)} trials updated in last {days_back} days")

            for trial in trials:
                try:
                    study_json = await ct_client.get_study_json(trial.nct_id)
                    if study_json:
                        result = pipeline.process_clinical_trial(trial, study_json, ticker)
                        if result:
                            processed["clinicaltrials"] += 1
                            print(f"    ✓ {trial.nct_id}: {trial.status.value} - {trial.title[:50]}...")
                        else:
                            print(f"    ⊘ {trial.nct_id}: Already processed")
                    else:
                        print(f"    ⊘ {trial.nct_id}: Could not fetch study JSON")
                except Exception as e:
                    print(f"    ✗ {trial.nct_id}: Error - {e}")
        except Exception as e:
            print(f"  ✗ Error fetching trials: {e}")

    # OpenFDA approvals
    if source is None or source == "openfda":
        print(f"\n💊 Fetching OpenFDA approvals (last {days_back} days)...")
        try:
            from datetime import timedelta, date as date_type
            fda_client = OpenFDAClient(api_key=openfda_key)
            cutoff_date = date_type.today() - timedelta(days=days_back)

            approvals_with_raw, total = await fda_client.get_drug_approvals_raw(
                sponsor_name=search_name,
                limit=1000,
            )
            # Filter by date
            approvals_with_raw = [
                (a, r) for a, r in approvals_with_raw
                if not a.approval_date or a.approval_date >= cutoff_date
            ]
            print(f"  Found {len(approvals_with_raw)} approvals in last {days_back} days (of {total} total)")

            for approval, raw_result in approvals_with_raw:
                try:
                    result = pipeline.process_fda_approval(approval, raw_result, ticker)
                    if result:
                        processed["openfda"] += 1
                        print(f"    ✓ {approval.application_number}: {approval.brand_name}")
                    else:
                        print(f"    ⊘ {approval.application_number}: Already processed")
                except Exception as e:
                    print(f"    ✗ {approval.application_number}: Error - {e}")
        except Exception as e:
            print(f"  ✗ Error fetching FDA approvals: {e}")

        print(f"\n🚨 Fetching OpenFDA recalls (last {days_back} days)...")
        try:
            recalls_with_raw = await fda_client.get_recalls_raw_by_firm(
                firm_name=search_name,
                days_back=days_back,
            )
            print(f"  Found {len(recalls_with_raw)} recalls in last {days_back} days")

            for recall, raw_result in recalls_with_raw:
                try:
                    result = pipeline.process_fda_recall(recall, raw_result, ticker)
                    if result:
                        processed["openfda"] += 1
                        print(f"    ✓ {recall.recall_number}: {recall.classification} - {recall.recalling_firm[:40]}")
                    else:
                        print(f"    ⊘ {recall.recall_number}: Already processed")
                except Exception as e:
                    print(f"    ✗ {recall.recall_number}: Error - {e}")
        except Exception as e:
            print(f"  ✗ Error fetching FDA recalls: {e}")

    print(f"\n✅ Processed:")
    print(f"  ClinicalTrials: {processed['clinicaltrials']} studies")
    print(f"  OpenFDA: {processed['openfda']} approvals")


async def run_extraction(args):
    """Run the extraction pipeline based on command line arguments."""
    from src.extraction.pipeline import ExtractionPipeline

    openfda_key = os.getenv("OPENFDA_API_KEY")

    pipeline = ExtractionPipeline(
        openfda_key=openfda_key,
    )

    if args.stats:
        stats = pipeline.get_stats()
        print("\n📊 Extraction Statistics")
        print("=" * 50)
        print(f"Total announcements: {stats['total']}")
        print(f"\nBy source:")
        for source, count in sorted(stats["by_source"].items()):
            print(f"  {source}: {count}")
        print(f"\nBy status:")
        for status, count in stats["by_status"].items():
            print(f"  {status}: {count}")
        print(f"\nBy ticker (top 10):")
        sorted_tickers = sorted(stats["by_ticker"].items(), key=lambda x: x[1], reverse=True)
        for ticker, count in sorted_tickers[:10]:
            print(f"  {ticker}: {count}")
        return 0

    if args.retry_failed:
        print("🔄 Retrying failed extractions...")
        results = await pipeline.retry_failed(max_per_source=args.limit or 100)
        print(f"  Retried: {results['retried']}")
        print(f"  Succeeded: {results['succeeded']}")
        print(f"  Still failed: {results['failed']}")
        return 0

    # Run extraction for specific ticker, all stocks, or show usage
    if args.all:
        await run_for_all_stocks(
            pipeline=pipeline,
            limit=args.limit or 5,
            source=args.source,
            stocks_csv=args.stocks_csv or "data/stocks.csv",
            days_back=args.days_back,
        )
    elif args.ticker:
        await run_for_ticker(pipeline, args.ticker, args.limit or 10, args.source, days_back=args.days_back)
    elif args.source:
        print(f"⚠️  Processing by source requires --ticker or --all. Use --stats to see current data.")
    else:
        print("Usage:")
        print("  python scripts/run_extraction.py --ticker MRNA [--source edgar] [--limit 10]")
        print("  python scripts/run_extraction.py --all [--source edgar] [--limit 5]")
        print("  python scripts/run_extraction.py --stats")

    # Show current stats
    stats = pipeline.get_stats()
    print(f"\n📊 Current index stats:")
    print(f"  Total: {stats['total']}")
    print(f"  OK: {stats['by_status'].get('OK', 0)}")
    print(f"  Failed: {stats['by_status'].get('FAILED', 0)}")
    print(f"  Pending: {stats['by_status'].get('PENDING', 0)}")

    # Sync Parquet dataset
    try:
        from src.storage.parquet_sync import ParquetSync
        print("\n📦 Syncing Parquet dataset...")
        parquet_sync = ParquetSync()
        parquet_stats = parquet_sync.update()
        print(f"   Parquet updated: {parquet_stats['text_loaded']} records with text")
    except FileNotFoundError:
        pass  # No CSV yet (e.g. --stats only run)
    except Exception as e:
        print(f"   ⚠️  Parquet sync failed: {e}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run text extraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_extraction.py --stats                    # Show statistics
    python scripts/run_extraction.py --ticker MRNA              # Process specific ticker
    python scripts/run_extraction.py --ticker MRNA --limit 5    # Limit filings per source
    python scripts/run_extraction.py --all                      # Process ALL stocks from stocks.csv
    python scripts/run_extraction.py --all --source edgar       # All stocks, EDGAR only
    python scripts/run_extraction.py --all --limit 3            # All stocks, 3 filings each
    python scripts/run_extraction.py --retry-failed             # Retry failed extractions
        """
    )
    parser.add_argument(
        "--ticker",
        help="Process specific ticker only"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process ALL stocks from stocks.csv"
    )
    parser.add_argument(
        "--stocks-csv",
        default="data/stocks.csv",
        help="Path to stocks CSV file (default: data/stocks.csv)"
    )
    parser.add_argument(
        "--source",
        choices=["clinicaltrials", "openfda"],
        help="Process specific source only"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry previously failed extractions"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics only, don't process"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of records to process"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=1095,
        help="Days of history to fetch for ClinicalTrials and OpenFDA (default: 1095 = 3 years)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    return asyncio.run(run_extraction(args))


if __name__ == "__main__":
    sys.exit(main())
