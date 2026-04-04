"""
seed_supabase.py — Seed Supabase with historical announcements from local parquet.

Run this ONCE to populate the announcements table with historical data.
After this, the daily pipeline keeps Supabase up to date.

Usage:
    python scripts/seed_supabase.py              # dry run — count only
    python scripts/seed_supabase.py --write      # actually write to Supabase
    python scripts/seed_supabase.py --write --batch-size 200
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("seed_supabase")

PARQUET_PATH = Path(__file__).parent.parent / "data" / "announcements2.parquet"


def main(write: bool = False, batch_size: int = 100) -> None:
    import pandas as pd

    if not PARQUET_PATH.exists():
        logger.error("announcements2.parquet not found at %s", PARQUET_PATH)
        sys.exit(1)

    logger.info("Loading %s ...", PARQUET_PATH)
    df = pd.read_parquet(PARQUET_PATH)
    logger.info("Loaded %d rows total", len(df))

    # Apply same filters as the backtesting data_loader
    if "parse_status" in df.columns:
        df = df[df["parse_status"] == "OK"].copy()
        logger.info("After parse_status filter: %d rows", len(df))

    df = df[~df["source"].str.contains("edgar", case=False, na=False)].copy()
    logger.info("After EDGAR filter: %d rows", len(df))

    # Required columns — map from parquet schema to Supabase schema
    required = ["ticker", "source", "event_type", "published_at", "raw_text"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error("Parquet missing required columns: %s", missing)
        sys.exit(1)

    # Normalize
    df["published_at"] = pd.to_datetime(df["published_at"], utc=False, errors="coerce")
    if df["published_at"].dt.tz is not None:
        df["published_at"] = df["published_at"].dt.tz_localize(None)
    df = df.dropna(subset=["published_at", "ticker"])

    # Map parquet columns to Supabase schema
    url_col = next((c for c in ["url", "announcement_url", "source_url"] if c in df.columns), None)
    id_col = next((c for c in ["external_id", "id", "nct_id", "source_id"] if c in df.columns), None)
    title_col = next((c for c in ["title", "name", "brief_title"] if c in df.columns), None)
    company_col = next((c for c in ["company_name", "sponsor", "lead_sponsor"] if c in df.columns), None)

    import hashlib

    rows = []
    for _, row in df.iterrows():
        # Resolve external_id — if null in parquet, synthesize one from source+ticker+date
        # to keep the upsert ON CONFLICT (source, external_id) from violating uniqueness on NULLs
        raw_ext_id = (str(row[id_col]) if id_col and id_col in row and pd.notnull(row.get(id_col)) and str(row[id_col]).strip() else None)
        if not raw_ext_id:
            sig = f"{row['source']}|{row['ticker']}|{row['published_at'].isoformat()}"
            raw_ext_id = "synth_" + hashlib.sha1(sig.encode()).hexdigest()[:12]

        rows.append({
            "source": str(row["source"]),
            "ticker": str(row["ticker"]),
            "company_name": str(row[company_col]) if company_col and company_col in row and pd.notnull(row.get(company_col)) else None,
            "event_type": str(row["event_type"]),
            "title": str(row[title_col])[:500] if title_col and title_col in row and pd.notnull(row.get(title_col)) else None,
            "announcement_url": str(row[url_col]) if url_col and url_col in row and pd.notnull(row.get(url_col)) else None,
            "published_at": row["published_at"].isoformat(),
            "raw_text": str(row.get("raw_text", ""))[:50000],  # cap at 50KB per row
            "external_id": raw_ext_id,
            "return_30d": float(row["return_30d"]) if "return_30d" in row and pd.notnull(row.get("return_30d")) else None,
            "return_5d": float(row["return_5d"]) if "return_5d" in row and pd.notnull(row.get("return_5d")) else None,
        })

    # Deduplicate by (source, external_id) — keep latest published_at per pair
    # The parquet can have multiple rows for the same NCT ID (trial updated over time)
    seen = {}
    for row in rows:
        key = (row["source"], row["external_id"])
        if key not in seen or row["published_at"] > seen[key]["published_at"]:
            seen[key] = row
    rows = list(seen.values())

    logger.info("After dedup (source, external_id): %d unique rows", len(rows))

    if not write:
        logger.info("DRY RUN — pass --write to actually insert into Supabase")
        logger.info("Sample row: %s", {k: str(v)[:80] if isinstance(v, str) else v for k, v in rows[0].items()})
        return

    # Write to Supabase in batches
    from pipeline.supabase_writer import get_client
    sb = get_client()

    total_written = 0
    n_batches = (len(rows) + batch_size - 1) // batch_size

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        batch_num = i // batch_size + 1
        try:
            response = (
                sb.table("announcements")
                .upsert(batch, on_conflict="source,external_id")
                .execute()
            )
            written = len(response.data) if response.data else 0
            total_written += written
            logger.info("Batch %d/%d: wrote %d rows (total: %d)", batch_num, n_batches, written, total_written)
        except Exception as e:
            logger.error("Batch %d failed: %s", batch_num, e)
        time.sleep(0.1)  # gentle rate limiting

    logger.info("Seeding complete. Total rows written: %d", total_written)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Supabase with historical announcements")
    parser.add_argument("--write", action="store_true", help="Actually write to Supabase (default: dry run)")
    parser.add_argument("--batch-size", type=int, default=100, help="Rows per batch (default: 100)")
    args = parser.parse_args()
    main(write=args.write, batch_size=args.batch_size)
