"""
supabase_writer.py — Write pipeline results to Supabase.

All functions are idempotent: safe to re-run if the pipeline is interrupted.
Uses upserts with ON CONFLICT to prevent duplicates.
"""

import logging
import os
import re
from datetime import date, datetime
from typing import Any

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)


def _valid_uuid(value) -> str | None:
    """Return value if it is a valid UUID string, otherwise None."""
    if value and _UUID_RE.match(str(value)):
        return str(value)
    return None

from supabase import create_client, Client

logger = logging.getLogger(__name__)


def get_client() -> Client:
    """Create and return a Supabase client using environment variables."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SECRET_KEY")
    if not url or not key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_SECRET_KEY must be set in environment. "
            "Use the secret key for pipeline writes."
        )
    return create_client(url, key)


# ── announcements ─────────────────────────────────────────────────────────────

def upsert_announcements(announcements: list[dict], client: Client | None = None) -> int:
    """Upsert a list of announcement dicts into the announcements table.

    Deduplicates on (source, external_id). Returns number of rows upserted.
    Each announcement dict should have: source, ticker, company_name, event_type,
    title, announcement_url, published_at, raw_text, external_id.
    """
    if not announcements:
        return 0
    sb = client or get_client()

    rows = []
    for ann in announcements:
        rows.append({
            "source": ann.get("source"),
            "ticker": ann.get("ticker"),
            "company_name": ann.get("company_name"),
            "event_type": ann.get("event_type"),
            "title": ann.get("title"),
            "announcement_url": ann.get("announcement_url") or ann.get("url"),
            "published_at": _to_iso(ann.get("published_at")),
            "fetched_at": datetime.utcnow().isoformat(),
            "raw_text": ann.get("raw_text", ""),
            "external_id": ann.get("external_id"),
            "return_30d": ann.get("return_30d"),
            "return_5d": ann.get("return_5d"),
        })

    response = (
        sb.table("announcements")
        .upsert(rows, on_conflict="source,external_id")
        .execute()
    )
    count = len(response.data) if response.data else 0
    logger.info("Upserted %d announcements", count)
    return count


# ── new-announcement detection ────────────────────────────────────────────────

def get_new_announcements_since(timestamp_iso: str, client: Client | None = None) -> list[dict]:
    """Return announcements that were INSERTED (not updated) after timestamp_iso.

    Works because created_at is set once on INSERT and is NOT included in
    upsert rows, so on-conflict updates do not change it.

    Args:
        timestamp_iso: ISO 8601 string captured just before the upsert call.

    Returns:
        List of announcement dicts with Supabase UUIDs as 'id'.
    """
    sb = client or get_client()
    try:
        resp = (
            sb.table("announcements")
            .select("id, ticker, source, event_type, published_at, raw_text, return_30d, return_5d")
            .gte("created_at", timestamp_iso)
            .order("published_at", desc=False)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        logger.error("Failed to query new announcements since %s: %s", timestamp_iso, e)
        return []


# ── model_runs ────────────────────────────────────────────────────────────────

def upsert_model_run(metrics: dict, client: Client | None = None) -> str | None:
    """Upsert model run metrics. Returns the model_run id (uuid)."""
    sb = client or get_client()

    row = {
        "run_date": metrics.get("run_date", date.today().isoformat()),
        "horizon": metrics.get("horizon", "30d"),
        "mlflow_run_id": metrics.get("mlflow_run_id"),
        "mlflow_experiment_url": metrics.get("mlflow_experiment_url"),
        "accuracy": metrics.get("accuracy"),
        "precision_score": metrics.get("precision_score"),
        "recall": metrics.get("recall"),
        "specificity": metrics.get("specificity"),
        "f1_score": metrics.get("f1_score"),
        "roc_auc": metrics.get("roc_auc"),
        "n_train_samples": metrics.get("n_train_samples"),
        "n_test_samples": metrics.get("n_test_samples"),
        "n_positive_train": metrics.get("n_positive_train"),
        "n_positive_test": metrics.get("n_positive_test"),
        "model_version": metrics.get("model_version", "v1"),
    }

    response = (
        sb.table("model_runs")
        .upsert(row, on_conflict="run_date,horizon")
        .execute()
    )
    if response.data:
        run_id = response.data[0]["id"]
        logger.info("Upserted model run %s (date=%s, horizon=%s)", run_id, row["run_date"], row["horizon"])
        return run_id
    return None


# ── predictions ───────────────────────────────────────────────────────────────

def upsert_predictions(predictions: list[dict], client: Client | None = None) -> int:
    """Upsert ML predictions for announcements.

    Each prediction dict should have: announcement_id, model_run_id, model_version,
    predicted_label, predicted_probability, expected_return_30d.
    Deduplicates on (announcement_id, model_version).
    """
    if not predictions:
        return 0
    sb = client or get_client()

    rows = []
    for pred in predictions:
        rows.append({
            # Only set announcement_id if it's a real Supabase UUID.
            # Predictions from the local-parquet fallback carry a 16-char hex id
            # (not a UUID) so we leave it NULL until the data is seeded.
            "announcement_id": _valid_uuid(pred.get("announcement_id")),
            "model_run_id": _valid_uuid(pred.get("model_run_id")),
            "model_version": pred.get("model_version", "v1"),
            "predicted_label": int(pred["predicted_label"]),
            "predicted_probability": float(pred["predicted_probability"]),
            "expected_return_30d": pred.get("expected_return_30d"),
        })

    # If all announcement_ids are null (parquet fallback), plain insert is safe.
    # Once announcements are seeded in Supabase, upsert with proper conflict key.
    has_ann_ids = any(r["announcement_id"] for r in rows)
    if has_ann_ids:
        response = (
            sb.table("predictions")
            .upsert(rows, on_conflict="announcement_id,model_version")
            .execute()
        )
    else:
        response = sb.table("predictions").insert(rows).execute()
    count = len(response.data) if response.data else 0
    logger.info("Upserted %d predictions", count)
    # Return the inserted rows so callers can use the DB-assigned UUIDs
    return response.data or []


# ── signals ───────────────────────────────────────────────────────────────────

def insert_signals(signals: list[dict], client: Client | None = None) -> list[dict]:
    """Upsert trading signals. Returns the inserted/updated rows (with DB-assigned UUIDs).

    Each signal dict needs: prediction_id, signal_date, ticker, action, reason, score.
    Deduplicates on (signal_date, ticker, action).
    """
    if not signals:
        return []
    sb = client or get_client()

    rows = []
    for sig in signals:
        rows.append({
            "prediction_id": sig.get("prediction_id"),
            "signal_date": _to_iso(sig.get("signal_date")),
            "ticker": sig["ticker"],
            "action": sig.get("action", "BUY"),
            "reason": sig.get("reason"),
            "score": sig.get("score"),
        })

    response = (
        sb.table("signals")
        .upsert(rows, on_conflict="signal_date,ticker,action")
        .execute()
    )
    inserted = response.data or []
    logger.info("Upserted %d signals", len(inserted))
    return inserted


# ── trades ────────────────────────────────────────────────────────────────────

def insert_trades(trades: list[dict], client: Client | None = None) -> int:
    """Upsert executed trades. Deduplicates on (trade_date, ticker, side).

    Each trade dict needs: signal_id, trade_date, ticker, side, quantity, price,
    amount_usd, status. signal_id is optional but should be set for BUY trades.
    """
    if not trades:
        return 0
    sb = client or get_client()

    rows = []
    for trade in trades:
        rows.append({
            "signal_id": _valid_uuid(trade.get("signal_id")),
            "trade_date": _to_iso(trade.get("trade_date")),
            "ticker": trade["ticker"],
            "side": trade["side"],
            "quantity": float(trade["quantity"]),
            "price": float(trade["price"]),
            "amount_usd": float(trade["amount_usd"]),
            "status": trade.get("status", "filled"),
            "exit_reason": trade.get("exit_reason"),
        })

    response = (
        sb.table("trades")
        .upsert(rows, on_conflict="trade_date,ticker,side")
        .execute()
    )
    count = len(response.data) if response.data else 0
    logger.info("Upserted %d trades", count)
    return count


# ── positions ─────────────────────────────────────────────────────────────────

def upsert_positions(positions: list[dict], client: Client | None = None) -> int:
    """Upsert open positions. Each position dict needs: ticker, quantity, avg_cost,
    market_value, unrealized_pnl. Upserts on ticker (primary key).
    """
    if not positions:
        return 0
    sb = client or get_client()

    rows = []
    for pos in positions:
        rows.append({
            "ticker": pos["ticker"],
            "quantity": float(pos["quantity"]),
            "avg_cost": float(pos["avg_cost"]),
            "market_value": float(pos.get("market_value", 0)),
            "unrealized_pnl": float(pos.get("unrealized_pnl", 0)),
            "entry_date": _to_iso(pos.get("entry_date")),
            "updated_at": datetime.utcnow().isoformat(),
        })

    response = (
        sb.table("positions")
        .upsert(rows, on_conflict="ticker")
        .execute()
    )
    count = len(response.data) if response.data else 0
    logger.info("Upserted %d positions", count)
    return count


def delete_position(ticker: str, client: Client | None = None) -> None:
    """Remove a closed position from the positions table."""
    sb = client or get_client()
    sb.table("positions").delete().eq("ticker", ticker).execute()
    logger.info("Deleted position for %s", ticker)


# ── portfolio_snapshots ───────────────────────────────────────────────────────

def upsert_portfolio_snapshot(snapshot: dict, client: Client | None = None) -> None:
    """Upsert a daily portfolio snapshot. Deduplicates on snapshot_date."""
    sb = client or get_client()

    row = {
        "snapshot_date": _to_iso(snapshot.get("snapshot_date", date.today())),
        "cash": float(snapshot["cash"]),
        "equity_value": float(snapshot["equity_value"]),
        "total_value": float(snapshot["total_value"]),
        "n_positions": int(snapshot.get("n_positions", 0)),
    }

    sb.table("portfolio_snapshots").upsert(row, on_conflict="snapshot_date").execute()
    logger.info(
        "Portfolio snapshot: total=$%.2f cash=$%.2f equity=$%.2f",
        row["total_value"], row["cash"], row["equity_value"]
    )


# ── portfolio cash ─────────────────────────────────────────────────────────────

def update_cash(cash: float, client: Client | None = None) -> None:
    """Update the cash balance in portfolio_config."""
    sb = client or get_client()
    sb.table("portfolio_config").update({"cash": cash, "updated_at": datetime.utcnow().isoformat()}).eq("id", 1).execute()
    logger.info("Updated cash to $%.2f", cash)


def get_cash(client: Client | None = None) -> float:
    """Read current cash from portfolio_config."""
    sb = client or get_client()
    response = sb.table("portfolio_config").select("cash").eq("id", 1).execute()
    if response.data:
        return float(response.data[0]["cash"])
    return 1_000_000.0  # default initial capital


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_iso(value: Any) -> str | None:
    """Convert date/datetime/string to ISO 8601 string, or None if falsy."""
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)
