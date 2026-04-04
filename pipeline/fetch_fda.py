"""
fetch_fda.py — Fetch FDA announcements (approvals + recalls) via OpenFDA API.
Returns a list of announcement dicts ready for Supabase upsert.
"""

import logging
from datetime import datetime, timedelta

from pipeline.clients.openfda import OpenFDAClient

logger = logging.getLogger(__name__)


def fetch_fda_announcements(
    tickers: list[str],
    days_back: int = 7,
) -> list[dict]:
    """Fetch recent FDA announcements for the given tickers.

    Queries both drug approvals and drug recalls from OpenFDA.
    Returns a list of announcement dicts with keys matching the Supabase schema.

    Args:
        tickers: List of stock tickers to query.
        days_back: How many days back to look for announcements (default 7).

    Returns:
        List of announcement dicts with keys:
        source, ticker, company_name, event_type, title, announcement_url,
        published_at, raw_text, external_id.
    """
    since = datetime.utcnow() - timedelta(days=days_back)
    announcements = []

    client = OpenFDAClient()

    for ticker in tickers:
        try:
            approvals = _fetch_approvals(client, ticker, since)
            announcements.extend(approvals)
        except Exception as e:
            logger.warning("FDA approvals fetch failed for %s: %s", ticker, e)

        try:
            recalls = _fetch_recalls(client, ticker, since)
            announcements.extend(recalls)
        except Exception as e:
            logger.warning("FDA recalls fetch failed for %s: %s", ticker, e)

    logger.info("Fetched %d FDA announcements for %d tickers", len(announcements), len(tickers))
    return announcements


def _fetch_approvals(client, ticker: str, since: datetime) -> list[dict]:
    """Fetch FDA drug approvals for a single ticker and normalize to announcement dicts."""
    try:
        approvals = client.get_drug_approvals(company_name=ticker, limit=10)
    except Exception:
        return []

    results = []
    for approval in approvals:
        published_at = _parse_date(getattr(approval, "action_date", None) or getattr(approval, "date", None))
        if published_at and published_at < since:
            continue

        # Build a text summary for ML feature extraction
        raw_text = _build_approval_text(approval)
        external_id = getattr(approval, "application_number", None) or getattr(approval, "id", None)

        results.append({
            "source": "openfda",
            "ticker": ticker,
            "company_name": getattr(approval, "sponsor_name", None),
            "event_type": "FDA_APPROVAL",
            "title": getattr(approval, "brand_name", None) or getattr(approval, "generic_name", None) or "FDA Approval",
            "announcement_url": None,
            "published_at": published_at.isoformat() if published_at else None,
            "raw_text": raw_text,
            "external_id": str(external_id) if external_id else None,
        })
    return results


def _fetch_recalls(client, ticker: str, since: datetime) -> list[dict]:
    """Fetch FDA drug recalls for a single ticker and normalize to announcement dicts."""
    try:
        recalls = client.get_drug_recalls(company_name=ticker, limit=10)
    except Exception:
        return []

    results = []
    for recall in recalls:
        published_at = _parse_date(getattr(recall, "recall_initiation_date", None) or getattr(recall, "date", None))
        if published_at and published_at < since:
            continue

        raw_text = _build_recall_text(recall)
        external_id = getattr(recall, "recall_number", None) or getattr(recall, "id", None)

        results.append({
            "source": "openfda",
            "ticker": ticker,
            "company_name": getattr(recall, "recalling_firm", None),
            "event_type": "FDA_RECALL",
            "title": getattr(recall, "product_description", None) or "FDA Recall",
            "announcement_url": None,
            "published_at": published_at.isoformat() if published_at else None,
            "raw_text": raw_text,
            "external_id": str(external_id) if external_id else None,
        })
    return results


def _build_approval_text(approval) -> str:
    """Build a raw text summary from an FDA approval object for ML feature extraction."""
    parts = []
    for attr in ["brand_name", "generic_name", "sponsor_name", "product_type",
                 "action_type", "indication", "summary"]:
        val = getattr(approval, attr, None)
        if val:
            parts.append(f"{attr.replace('_', ' ').title()}: {val}")
    return " | ".join(parts)


def _build_recall_text(recall) -> str:
    """Build a raw text summary from an FDA recall object for ML feature extraction."""
    parts = []
    for attr in ["product_description", "reason_for_recall", "recalling_firm",
                 "classification", "voluntary_mandated", "status"]:
        val = getattr(recall, attr, None)
        if val:
            parts.append(f"{attr.replace('_', ' ').title()}: {val}")
    return " | ".join(parts)


def _parse_date(value) -> datetime | None:
    """Parse a date string or datetime into a datetime object."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        for fmt in ["%Y%m%d", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"]:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    return None
