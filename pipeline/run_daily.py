"""
run_daily.py — Daily pipeline orchestrator.

Execution logic:

  ALWAYS:
    1. Load stock universe
    2. Fetch new announcements from FDA + ClinicalTrials APIs
    3. Upsert them to Supabase
    4. Load open positions + live prices
    5. Check exits (TP=30%, horizon=50d) — close if triggered

  ONLY IF new announcements were found today:
    6. Load all labeled announcements from Supabase
    7. Train model — Phase 1: eval on 80/20 temporal split → save metrics
                     Phase 2: retrain on 100% of labeled data
    8. Predict on today's new announcements only → BUY/HOLD signals
    9. Open new positions for BUY signals

  ALWAYS:
    10. Save portfolio snapshot

Usage:
  python -m pipeline.run_daily           # full pipeline
  python -m pipeline.run_daily --dry-run # no Supabase writes
  python -m pipeline.run_daily --skip-fetch  # skip API calls (useful for testing exit logic)
"""

import argparse
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_daily")


def main(dry_run: bool = False, skip_fetch: bool = False) -> None:
    logger.info("=" * 60)
    logger.info("Biotech Trading Dashboard — Daily Pipeline")
    logger.info("Date: %s | dry_run=%s | skip_fetch=%s", date.today(), dry_run, skip_fetch)
    logger.info("=" * 60)

    from pipeline.fetch_fda import fetch_fda_announcements
    from pipeline.fetch_clinical_trials import fetch_clinical_trials_announcements
    from pipeline.predict import run_daily_prediction
    from pipeline.generate_trades import process_daily_signals, fetch_prices_for_pipeline
    from pipeline.supabase_writer import (
        get_client, upsert_announcements, get_new_announcements_since,
        upsert_model_run, upsert_predictions,
        insert_signals, insert_trades, upsert_positions, delete_position,
        upsert_portfolio_snapshot, update_cash, get_cash,
    )

    sb = None if dry_run else get_client()

    # ── Step 1: Load stock universe ───────────────────────────────────────────
    tickers = _load_tickers()
    logger.info("Step 1: Loaded %d tickers", len(tickers))

    # ── Steps 2–3: Fetch + upsert announcements ───────────────────────────────
    new_announcement_rows: list[dict] = []

    if not skip_fetch:
        # Snapshot timestamp BEFORE the upsert so we can detect truly new rows
        fetch_ts = datetime.now(tz=timezone.utc).isoformat()

        fetched: list[dict] = []

        logger.info("Step 2: Fetching FDA announcements...")
        try:
            fda = fetch_fda_announcements(tickers, days_back=7)
            logger.info("  Fetched %d FDA announcements", len(fda))
            fetched.extend(fda)
        except Exception as e:
            logger.error("FDA fetch failed: %s", e)

        logger.info("Step 3: Fetching ClinicalTrials announcements...")
        try:
            ct = fetch_clinical_trials_announcements(tickers, days_back=7)
            logger.info("  Fetched %d ClinicalTrials announcements", len(ct))
            fetched.extend(ct)
        except Exception as e:
            logger.error("ClinicalTrials fetch failed: %s", e)

        if fetched and not dry_run:
            n_upserted = upsert_announcements(fetched, sb)
            logger.info("  Upserted %d announcement rows", n_upserted)

            # Detect which ones are genuinely new (created after our timestamp)
            new_announcement_rows = get_new_announcements_since(fetch_ts, sb)
            logger.info(
                "  New announcements (not previously in DB): %d",
                len(new_announcement_rows),
            )
        elif fetched and dry_run:
            logger.info("[DRY RUN] Would upsert %d announcements", len(fetched))
            # In dry-run, treat everything fetched as "new" for testing
            new_announcement_rows = fetched
    else:
        logger.info("Steps 2-3: Skipped (--skip-fetch)")

    # ── Step 4: Load open positions + live prices ─────────────────────────────
    logger.info("Step 4: Loading positions and live prices...")
    open_positions = _load_positions_from_supabase(sb)
    logger.info("  Open positions: %d", len(open_positions))

    # Always fetch prices for open positions (needed for exit checks)
    position_tickers = list({p["ticker"] for p in open_positions})
    current_prices = fetch_prices_for_pipeline(position_tickers) if position_tickers else {}
    logger.info("  Fetched prices for %d position tickers", len(current_prices))

    cash = get_cash(sb) if not dry_run else 1_000_000.0

    # ── Step 5: Check exits only (no new buys yet) ────────────────────────────
    logger.info("Step 5: Checking position exits...")
    from pipeline.generate_trades import process_exits_only
    exit_result = process_exits_only(open_positions, cash, current_prices)

    if exit_result["closed_positions"] and not dry_run:
        insert_trades(exit_result["new_trades"], sb)
        for ticker in exit_result["closed_positions"]:
            delete_position(ticker, sb)
        update_cash(exit_result["updated_cash"], sb)
        logger.info("  Closed %d positions", len(exit_result["closed_positions"]))
    elif exit_result["closed_positions"] and dry_run:
        logger.info("[DRY RUN] Would close positions: %s", exit_result["closed_positions"])

    # Update state after exits
    remaining_positions = exit_result["remaining_positions"]
    cash = exit_result["updated_cash"]

    # ── Branch: no new announcements → snapshot and done ─────────────────────
    if not new_announcement_rows:
        logger.info("No new announcements today — skipping model training and new entries.")

        # Update market values for remaining positions with fresh prices
        for pos in remaining_positions:
            price = current_prices.get(pos["ticker"])
            if price:
                pos["market_value"] = pos["quantity"] * price
                pos["unrealized_pnl"] = (price - pos["avg_cost"]) * pos["quantity"]

        if not dry_run:
            upsert_positions(remaining_positions, sb)

        _save_snapshot(remaining_positions, cash, sb, dry_run)
        _log_done()
        return

    # ── Step 6: Load labeled data for model training ──────────────────────────
    logger.info("Step 6: Loading labeled announcements for model training...")
    labeled_df = _load_labeled_announcements_from_supabase(sb)

    if labeled_df.empty:
        labeled_df = _load_local_parquet_fallback()

    if labeled_df.empty:
        logger.error("No labeled data — cannot train model, skipping new entries.")
        _save_snapshot(remaining_positions, cash, sb, dry_run)
        _log_done()
        return

    logger.info("  Loaded %d labeled announcements", len(labeled_df))

    # ── Step 7: Train model (two-phase) ──────────────────────────────────────
    logger.info("Step 7: Training model (80/20 eval + 100%% production refit)...")

    new_df = pd.DataFrame(new_announcement_rows)

    try:
        predictions_list, metrics = run_daily_prediction(
            labeled_df=labeled_df,
            predict_df=new_df,
            horizon="30d",
            eval_train_frac=0.80,
        )
        logger.info(
            "  Model metrics: %s",
            {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()
             if isinstance(v, (int, float))},
        )
    except Exception as e:
        logger.error("Model training failed: %s", e, exc_info=True)
        predictions_list, metrics = [], {}

    model_run_id = None
    if metrics and not dry_run:
        model_run_id = upsert_model_run(metrics, sb)
        logger.info("  Saved model run: %s", model_run_id)

    # ── Step 8: Write predictions ─────────────────────────────────────────────
    logger.info("Step 8: Writing predictions...")
    if predictions_list and not dry_run:
        for pred in predictions_list:
            pred["model_run_id"] = model_run_id
        n = upsert_predictions(predictions_list, sb)
        logger.info("  Wrote %d predictions", n)

    # ── Step 9: Open new positions from BUY signals ───────────────────────────
    logger.info("Step 9: Processing BUY signals...")

    # Fetch live prices for new BUY candidates
    buy_tickers = [
        p["ticker"] for p in predictions_list if p.get("predicted_label") == 1
    ]
    if buy_tickers:
        new_prices = fetch_prices_for_pipeline(buy_tickers)
        current_prices.update(new_prices)

    try:
        result = process_daily_signals(
            predictions=predictions_list,
            open_positions=remaining_positions,
            cash=cash,
            current_prices=current_prices,
        )
    except Exception as e:
        logger.error("Signal processing failed: %s", e, exc_info=True)
        result = None

    if result and not dry_run:
        if result["new_signals"]:
            insert_signals(result["new_signals"], sb)
        if result["new_trades"]:
            insert_trades(result["new_trades"], sb)
        if result["updated_positions"]:
            upsert_positions(result["updated_positions"], sb)
        update_cash(result["updated_cash"], sb)

        logger.info(
            "  Signals: %d | Trades: %d new | Positions now open: %d | Cash: $%.2f",
            len(result["new_signals"]),
            len(result["new_trades"]),
            len(result["updated_positions"]),
            result["updated_cash"],
        )
        _save_snapshot_from_result(result, sb)

    elif result and dry_run:
        logger.info(
            "[DRY RUN] Would write: %d signals, %d trades",
            len(result["new_signals"]), len(result["new_trades"]),
        )
        _save_snapshot_from_result(result, sb=None, dry_run=True)

    elif not result:
        # Signal processing failed — still save a snapshot with current state
        _save_snapshot(remaining_positions, cash, sb, dry_run)

    _log_done()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_tickers() -> list[str]:
    stocks_path = Path(__file__).parent.parent / "data" / "stocks.csv"
    if stocks_path.exists():
        try:
            df = pd.read_csv(stocks_path)
            col = next((c for c in df.columns if c.lower() in ["ticker", "symbol", "code"]), None)
            if col:
                return df[col].dropna().str.upper().tolist()[:500]
        except Exception as e:
            logger.warning("Could not load stocks.csv: %s", e)
    fallback = ["MRNA", "BNTX", "NVAX", "REGN", "BIIB", "GILD", "AMGN", "VRTX"]
    logger.warning("Using fallback ticker list (%d tickers)", len(fallback))
    return fallback


def _load_labeled_announcements_from_supabase(sb) -> pd.DataFrame:
    """Load ALL announcements that have a return label (return_30d is not null)."""
    if sb is None:
        return pd.DataFrame()
    try:
        all_rows = []
        page_size = 1000
        offset = 0
        while True:
            resp = (
                sb.table("announcements")
                .select("id, ticker, source, event_type, published_at, raw_text, return_30d, return_5d")
                .not_.is_("return_30d", "null")
                .order("published_at", desc=False)
                .range(offset, offset + page_size - 1)
                .execute()
            )
            batch = resp.data or []
            all_rows.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size
        if not all_rows:
            return pd.DataFrame()
        logger.info("Loaded %d labeled announcements from Supabase", len(all_rows))
        return pd.DataFrame(all_rows)
    except Exception as e:
        logger.error("Failed to load labeled announcements: %s", e)
        return pd.DataFrame()


def _load_local_parquet_fallback() -> pd.DataFrame:
    parquet_path = Path(__file__).parent.parent / "data" / "announcements2.parquet"
    if not parquet_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(parquet_path)
        if "parse_status" in df.columns:
            df = df[df["parse_status"] == "OK"].copy()
        if "source" in df.columns:
            df = df[~df["source"].str.contains("edgar", case=False, na=False)].copy()
        # Only rows with a return label
        if "return_30d" in df.columns:
            df = df[df["return_30d"].notna()].copy()
        logger.info("Loaded %d labeled announcements from local parquet (fallback)", len(df))
        return df
    except Exception as e:
        logger.warning("Could not load local parquet: %s", e)
        return pd.DataFrame()


def _load_positions_from_supabase(sb) -> list[dict]:
    if sb is None:
        return []
    try:
        resp = sb.table("positions").select("*").execute()
        return resp.data or []
    except Exception as e:
        logger.error("Failed to load positions: %s", e)
        return []


def _save_snapshot(positions: list[dict], cash: float, sb, dry_run: bool) -> None:
    from pipeline.supabase_writer import upsert_portfolio_snapshot
    equity = sum(p.get("market_value", p["quantity"] * p["avg_cost"]) for p in positions)
    snap = {
        "snapshot_date": date.today().isoformat(),
        "cash": cash,
        "equity_value": equity,
        "total_value": cash + equity,
        "n_positions": len(positions),
    }
    if not dry_run:
        upsert_portfolio_snapshot(snap, sb)
    logger.info(
        "%sSnapshot: total=$%.2f cash=$%.2f equity=$%.2f positions=%d",
        "[DRY RUN] " if dry_run else "",
        snap["total_value"], snap["cash"], snap["equity_value"], snap["n_positions"],
    )


def _save_snapshot_from_result(result: dict, sb, dry_run: bool = False) -> None:
    from pipeline.supabase_writer import upsert_portfolio_snapshot
    snap = result["portfolio_snapshot"]
    if not dry_run:
        upsert_portfolio_snapshot(snap, sb)
    logger.info(
        "%sSnapshot: total=$%.2f cash=$%.2f equity=$%.2f positions=%d",
        "[DRY RUN] " if dry_run else "",
        snap["total_value"], snap["cash"], snap["equity_value"], snap["n_positions"],
    )


def _log_done() -> None:
    logger.info("=" * 60)
    logger.info("Daily pipeline completed successfully")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Biotech Trading Dashboard — Daily Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing to Supabase")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip API fetching")
    args = parser.parse_args()
    main(dry_run=args.dry_run, skip_fetch=args.skip_fetch)
