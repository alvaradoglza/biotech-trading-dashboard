"""
run_daily.py — Daily pipeline orchestrator.

Execution order:
  1. Load stock universe (from Supabase or local stocks.csv)
  2. Fetch new FDA announcements → Supabase
  3. Fetch new ClinicalTrials announcements → Supabase
  4. Load labeled announcements from Supabase → train model → evaluate → log metrics
  5. Generate predictions for new announcements → Supabase
  6. Process signals: check exits, open new positions → Supabase
  7. Save portfolio snapshot → Supabase

Usage:
  python -m pipeline.run_daily          # run full pipeline
  python -m pipeline.run_daily --dry-run # run without writing to Supabase
  python -m pipeline.run_daily --skip-fetch  # skip API fetching (use existing announcements)
"""

import argparse
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Load .env for local development
load_dotenv(Path(__file__).parent.parent / ".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_daily")


def main(dry_run: bool = False, skip_fetch: bool = False) -> None:
    """Run the full daily pipeline."""

    logger.info("=" * 60)
    logger.info("Biotech Trading Dashboard — Daily Pipeline")
    logger.info("Date: %s | dry_run=%s | skip_fetch=%s", date.today(), dry_run, skip_fetch)
    logger.info("=" * 60)

    # Import pipeline modules
    from pipeline.fetch_fda import fetch_fda_announcements
    from pipeline.fetch_clinical_trials import fetch_clinical_trials_announcements
    from pipeline.predict import run_daily_prediction
    from pipeline.generate_trades import process_daily_signals, fetch_prices_for_pipeline
    from pipeline.supabase_writer import (
        get_client, upsert_announcements, upsert_model_run, upsert_predictions,
        insert_signals, insert_trades, upsert_positions, delete_position,
        upsert_portfolio_snapshot, update_cash, get_cash,
    )

    sb = None if dry_run else get_client()

    # ── Step 1: Load stock universe ────────────────────────────────────────────
    tickers = _load_tickers()
    logger.info("Step 1: Loaded %d tickers", len(tickers))

    # ── Step 2 & 3: Fetch announcements ────────────────────────────────────────
    all_new_announcements = []

    if not skip_fetch:
        logger.info("Step 2: Fetching FDA announcements...")
        try:
            fda_announcements = fetch_fda_announcements(tickers, days_back=7)
            logger.info("  Fetched %d FDA announcements", len(fda_announcements))
            all_new_announcements.extend(fda_announcements)
        except Exception as e:
            logger.error("FDA fetch failed: %s", e)

        logger.info("Step 3: Fetching ClinicalTrials announcements...")
        try:
            ct_announcements = fetch_clinical_trials_announcements(tickers, days_back=7)
            logger.info("  Fetched %d ClinicalTrials announcements", len(ct_announcements))
            all_new_announcements.extend(ct_announcements)
        except Exception as e:
            logger.error("ClinicalTrials fetch failed: %s", e)

        if all_new_announcements and not dry_run:
            n = upsert_announcements(all_new_announcements, sb)
            logger.info("  Wrote %d announcements to Supabase", n)
    else:
        logger.info("Steps 2-3: Skipped (--skip-fetch)")

    # ── Step 4: Train model & evaluate ────────────────────────────────────────
    logger.info("Step 4: Loading announcements for model training...")
    announcements_df = _load_announcements_from_supabase(sb)

    if announcements_df.empty:
        logger.warning("  No announcements in Supabase — falling back to local parquet if available")
        announcements_df = _load_local_parquet_fallback()

    if announcements_df.empty:
        logger.error("  No labeled data available — skipping model training")
    else:
        logger.info("  Loaded %d announcements for training", len(announcements_df))

        logger.info("Step 4b: Training model...")
        try:
            predictions_list, metrics = run_daily_prediction(announcements_df, horizon="30d")
            logger.info("  Model metrics: %s", {k: v for k, v in metrics.items() if isinstance(v, (int, float))})

            if not dry_run and metrics:
                model_run_id = upsert_model_run(metrics, sb)
                logger.info("  Saved model run: %s", model_run_id)
            else:
                model_run_id = None

        except Exception as e:
            logger.error("Model training failed: %s", e, exc_info=True)
            predictions_list, metrics, model_run_id = [], {}, None

    # ── Step 5: Write predictions ──────────────────────────────────────────────
    logger.info("Step 5: Writing predictions...")
    if predictions_list and not dry_run:
        # Attach model_run_id to each prediction
        for pred in predictions_list:
            pred["model_run_id"] = model_run_id

        n = upsert_predictions(predictions_list, sb)
        logger.info("  Wrote %d predictions", n)

    # ── Step 6: Process signals & trades ──────────────────────────────────────
    logger.info("Step 6: Processing signals and positions...")

    open_positions = _load_positions_from_supabase(sb)
    logger.info("  Open positions: %d", len(open_positions))

    all_tickers_needed = list({p["ticker"] for p in open_positions} | {p.get("ticker") for p in predictions_list if p.get("predicted_label") == 1})
    current_prices = fetch_prices_for_pipeline(all_tickers_needed)
    logger.info("  Fetched prices for %d tickers", len(current_prices))

    cash = get_cash(sb) if not dry_run else 1_000_000.0

    try:
        result = process_daily_signals(
            predictions=predictions_list,
            open_positions=open_positions,
            cash=cash,
            current_prices=current_prices,
        )
    except Exception as e:
        logger.error("Signal processing failed: %s", e, exc_info=True)
        result = None

    if result and not dry_run:
        # Write signals
        if result["new_signals"]:
            insert_signals(result["new_signals"], sb)

        # Write trades
        if result["new_trades"]:
            insert_trades(result["new_trades"], sb)

        # Update positions
        if result["updated_positions"]:
            upsert_positions(result["updated_positions"], sb)

        # Delete closed positions
        for ticker in result["closed_positions"]:
            delete_position(ticker, sb)

        # Update cash
        update_cash(result["updated_cash"], sb)

        logger.info(
            "  Signals: %d new | Trades: %d new | Positions: %d open | Cash: $%.2f",
            len(result["new_signals"]),
            len(result["new_trades"]),
            len(result["updated_positions"]),
            result["updated_cash"],
        )

        # ── Step 7: Portfolio snapshot ─────────────────────────────────────────
        logger.info("Step 7: Saving portfolio snapshot...")
        upsert_portfolio_snapshot(result["portfolio_snapshot"], sb)
        snap = result["portfolio_snapshot"]
        logger.info(
            "  Snapshot: total=$%.2f cash=$%.2f equity=$%.2f positions=%d",
            snap["total_value"], snap["cash"], snap["equity_value"], snap["n_positions"]
        )
    elif dry_run and result:
        logger.info("[DRY RUN] Would write: %d signals, %d trades, %d positions",
                    len(result["new_signals"]), len(result["new_trades"]), len(result["updated_positions"]))

    logger.info("=" * 60)
    logger.info("Daily pipeline completed successfully")
    logger.info("=" * 60)


# ── Helper functions ──────────────────────────────────────────────────────────

def _load_tickers() -> list[str]:
    """Load stock tickers from local stocks.csv or fall back to a small default list."""
    stocks_path = Path(__file__).parent.parent / "data" / "stocks.csv"

    if stocks_path.exists():
        try:
            df = pd.read_csv(stocks_path)
            ticker_col = next((c for c in df.columns if c.lower() in ["ticker", "symbol", "code"]), None)
            if ticker_col:
                tickers = df[ticker_col].dropna().str.upper().tolist()
                logger.info("Loaded %d tickers from %s", len(tickers), stocks_path)
                return tickers[:500]  # cap at 500 to avoid rate limit issues
        except Exception as e:
            logger.warning("Could not load stocks.csv: %s", e)

    # Minimal fallback list of well-known biotech stocks
    fallback = ["MRNA", "BNTX", "NVAX", "REGN", "BIIB", "GILD", "AMGN", "VRTX"]
    logger.warning("Using fallback ticker list (%d tickers)", len(fallback))
    return fallback


def _load_announcements_from_supabase(sb) -> pd.DataFrame:
    """Load all announcements from Supabase for model training using pagination."""
    if sb is None:
        return pd.DataFrame()
    try:
        all_rows = []
        page_size = 1000
        offset = 0

        while True:
            response = (
                sb.table("announcements")
                .select("id, ticker, source, event_type, published_at, raw_text, return_30d, return_5d")
                .order("published_at", desc=False)
                .range(offset, offset + page_size - 1)
                .execute()
            )
            batch = response.data or []
            all_rows.extend(batch)
            if len(batch) < page_size:
                break  # last page
            offset += page_size

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        logger.info("Loaded %d announcements from Supabase (paginated)", len(df))
        return df
    except Exception as e:
        logger.error("Failed to load announcements from Supabase: %s", e)
        return pd.DataFrame()


def _load_local_parquet_fallback() -> pd.DataFrame:
    """Load local announcements2.parquet as fallback for initial runs."""
    parquet_path = Path(__file__).parent.parent / "data" / "announcements2.parquet"

    if not parquet_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_parquet(parquet_path)
        # Filter for good data
        if "parse_status" in df.columns:
            df = df[df["parse_status"] == "OK"].copy()
        if "source" in df.columns:
            df = df[~df["source"].str.contains("edgar", case=False, na=False)].copy()
        logger.info("Loaded %d announcements from local parquet (fallback)", len(df))
        return df
    except Exception as e:
        logger.warning("Could not load local parquet: %s", e)
        return pd.DataFrame()


def _load_positions_from_supabase(sb) -> list[dict]:
    """Load current open positions from Supabase."""
    if sb is None:
        return []
    try:
        response = sb.table("positions").select("*").execute()
        return response.data or []
    except Exception as e:
        logger.error("Failed to load positions: %s", e)
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Biotech Trading Dashboard — Daily Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing to Supabase")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip API fetching")
    args = parser.parse_args()
    main(dry_run=args.dry_run, skip_fetch=args.skip_fetch)
