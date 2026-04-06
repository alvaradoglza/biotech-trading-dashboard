"""
fetch_fda.py — Fetch FDA announcements (approvals + recalls) via OpenFDA API.

Strategy:
  - Fetch ALL recent approvals and recalls (date-based, not per-ticker)
  - Match results back to our ticker universe via sponsor/firm name
  - OpenFDA client methods are async; we run them with asyncio.run()
"""

import asyncio
import logging
from datetime import datetime, timedelta

from pipeline.clients.openfda import OpenFDAClient

logger = logging.getLogger(__name__)


def fetch_fda_announcements(
    tickers: list[str],
    days_back: int = 7,
    ticker_to_company: dict[str, str] | None = None,
) -> list[dict]:
    """Fetch recent FDA announcements and match to our ticker universe.

    Fetches ALL recent approvals + recalls from OpenFDA (date-range query),
    then matches each result to a ticker using sponsor/firm name lookup.

    Args:
        tickers: List of stock tickers in our universe.
        days_back: How many days back to search.
        ticker_to_company: Optional dict mapping ticker → company name.
                           If provided, used to build reverse lookup for matching.

    Returns:
        List of announcement dicts ready for Supabase upsert.
    """
    return asyncio.run(_fetch_fda_async(tickers, days_back, ticker_to_company or {}))


async def _fetch_fda_async(
    tickers: list[str],
    days_back: int,
    ticker_to_company: dict[str, str],
) -> list[dict]:
    client = OpenFDAClient()
    announcements = []

    # Build reverse lookup: normalized company name → ticker
    name_lookup = _build_name_lookup(tickers, ticker_to_company)

    # Fetch all recent approvals across all sponsors
    try:
        approvals = await client.get_recent_approvals(days_back=days_back)
        logger.info("OpenFDA: %d recent approvals fetched", len(approvals))
        for approval in approvals:
            ticker = _match_ticker(approval.sponsor_name, name_lookup)
            if ticker is None:
                continue
            published_at = approval.approval_date.isoformat() if approval.approval_date else None
            announcements.append({
                "source": "openfda",
                "ticker": ticker,
                "company_name": approval.sponsor_name,
                "event_type": "FDA_APPROVAL",
                "title": (approval.brand_name or approval.generic_name or "FDA Approval")[:500],
                "announcement_url": approval.url,
                "published_at": published_at,
                "raw_text": _approval_text(approval),
                "external_id": approval.application_number or None,
            })
    except Exception as e:
        logger.warning("FDA approvals fetch failed: %s", e)

    # Fetch all recent recalls across all firms
    try:
        recalls = await client.get_recent_recalls(days_back=days_back)
        logger.info("OpenFDA: %d recent recalls fetched", len(recalls))
        for recall in recalls:
            ticker = _match_ticker(recall.recalling_firm, name_lookup)
            if ticker is None:
                continue
            published_at = (recall.recall_initiation_date or recall.report_date)
            announcements.append({
                "source": "openfda",
                "ticker": ticker,
                "company_name": recall.recalling_firm,
                "event_type": "FDA_RECALL",
                "title": (recall.product_description or "FDA Recall")[:500],
                "announcement_url": recall.url,
                "published_at": published_at.isoformat() if published_at else None,
                "raw_text": _recall_text(recall),
                "external_id": recall.recall_number or None,
            })
    except Exception as e:
        logger.warning("FDA recalls fetch failed: %s", e)

    logger.info("Fetched %d FDA announcements matched to ticker universe", len(announcements))
    return announcements


def _build_name_lookup(
    tickers: list[str],
    ticker_to_company: dict[str, str],
) -> dict[str, tuple[str, str]]:
    """Build a {normalized_full_name: (ticker, first_word)} lookup.

    Returns dict mapping normalized full company name → ticker.
    Also stores the first significant word for prefix matching.
    """
    lookup: dict[str, str] = {}          # full normalized name → ticker
    prefix_lookup: dict[str, str] = {}   # first word (≥5 chars) → ticker
    for ticker in tickers:
        company = ticker_to_company.get(ticker, "")
        if not company:
            continue
        normalized = _normalize(company)
        if len(normalized) >= 4:
            lookup[normalized] = ticker
        parts = normalized.split()
        if parts and len(parts[0]) >= 5:
            # Only add prefix if not already claimed by another ticker
            if parts[0] not in prefix_lookup:
                prefix_lookup[parts[0]] = ticker
    return lookup, prefix_lookup


def _normalize(name: str) -> str:
    """Strip legal suffixes and lowercase a company name."""
    name = name.lower().strip()
    for suffix in [
        " incorporated", " inc.", " inc", " corporation", " corp.", " corp",
        " limited", " ltd.", " ltd", " llc", " plc", " sa", " ag", " nv",
        " therapeutics", " pharmaceutical", " pharmaceuticals",
        " biosciences", " bioscience", " biopharma", " biopharmaceuticals",
        " biopharmaceutical", " oncology", " genomics", " sciences", " science",
        " health", " healthcare", " medical", " medicine", " labs", " laboratory",
        " laboratories",
    ]:
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
    return name.strip()


def _match_ticker(
    company_name: str | None,
    lookup_pair: tuple[dict, dict],
) -> str | None:
    """Match a company name to a ticker with scored precision.

    Priority:
      1. Exact normalized match (highest confidence)
      2. Candidate's normalized name is a prefix of our name (≥8 chars)
      3. Long substring match only for fragments ≥ 10 chars (reduces false positives)
    """
    if not company_name:
        return None
    lookup, prefix_lookup = lookup_pair
    normalized = _normalize(company_name)
    if not normalized or len(normalized) < 4:
        return None

    # 1. Exact match
    if normalized in lookup:
        return lookup[normalized]

    # 2. Our known name is contained in candidate (prefix / full-name contained)
    for known, ticker in lookup.items():
        if len(known) >= 8 and known in normalized:
            return ticker
        if len(normalized) >= 8 and normalized in known:
            return ticker

    # 3. First-word prefix match (long words only, avoids "bio", "gen", etc.)
    parts = normalized.split()
    if parts and len(parts[0]) >= 7 and parts[0] in prefix_lookup:
        return prefix_lookup[parts[0]]

    return None


def _approval_text(approval) -> str:
    parts = []
    for attr, label in [
        ("brand_name", "Brand"), ("generic_name", "Generic"),
        ("sponsor_name", "Sponsor"), ("submission_type", "Type"),
        ("submission_status", "Status"), ("dosage_form", "Form"),
        ("route", "Route"),
    ]:
        val = getattr(approval, attr, None)
        if val:
            parts.append(f"{label}: {val}")
    if approval.active_ingredients:
        parts.append(f"Ingredients: {', '.join(approval.active_ingredients)}")
    return " | ".join(parts)


def _recall_text(recall) -> str:
    parts = []
    for attr, label in [
        ("product_description", "Product"), ("reason_for_recall", "Reason"),
        ("recalling_firm", "Firm"), ("classification", "Class"),
        ("voluntary_mandated", "Type"), ("status", "Status"),
    ]:
        val = getattr(recall, attr, None)
        if val:
            parts.append(f"{label}: {val}")
    return " | ".join(parts)
