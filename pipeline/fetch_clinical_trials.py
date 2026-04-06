"""
fetch_clinical_trials.py — Fetch ClinicalTrials.gov announcements.

Strategy:
  - Fetch ALL recently updated trials using a date-range query (one API call, not per-ticker)
  - Match results back to our ticker universe via lead sponsor name
  - ClinicalTrialsClient is async; we run it with asyncio.run()

NOTE: ClinicalTrials.gov client uses `requests` under the hood (not httpx)
due to TLS fingerprinting. The async interface is a thin wrapper.
"""

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Optional

from pipeline.clients.clinicaltrials import ClinicalTrialsClient

logger = logging.getLogger(__name__)


def fetch_clinical_trials_announcements(
    tickers: list[str],
    days_back: int = 7,
    ticker_to_company: dict[str, str] | None = None,
) -> list[dict]:
    """Fetch recently updated ClinicalTrials.gov studies matched to our ticker universe.

    Args:
        tickers: List of stock tickers in our universe.
        days_back: How many days back to search for updates.
        ticker_to_company: Optional dict mapping ticker → company name.

    Returns:
        List of announcement dicts ready for Supabase upsert.
    """
    return asyncio.run(_fetch_ct_async(tickers, days_back, ticker_to_company or {}))


async def _fetch_ct_async(
    tickers: list[str],
    days_back: int,
    ticker_to_company: dict[str, str],
) -> list[dict]:
    client = ClinicalTrialsClient()
    announcements = []

    # Build reverse lookup: normalized company name → ticker
    name_lookup = _build_name_lookup(tickers, ticker_to_company)

    # Fetch ALL studies updated in the last `days_back` days using a date-range term.
    # ClinicalTrials API query.term supports AREA[LastUpdatePostDate]RANGE[...] syntax.
    cutoff = (date.today() - timedelta(days=days_back)).isoformat()
    today = date.today().isoformat()
    date_term = f"AREA[LastUpdatePostDate]RANGE[{cutoff},{today}]"

    try:
        page_token: Optional[str] = None
        total_fetched = 0

        while True:
            trials, next_token, total = await client.search_studies(
                term=date_term,
                page_size=100,
                page_token=page_token,
            )

            for trial in trials:
                ticker = _match_ticker(trial.sponsor, name_lookup)
                if ticker is None:
                    continue

                update_date = trial.last_update_date or trial.study_first_posted_date
                published_at = update_date.isoformat() if update_date else None

                announcements.append({
                    "source": "clinicaltrials",
                    "ticker": ticker,
                    "company_name": trial.sponsor,
                    "event_type": _classify(trial),
                    "title": (trial.title or "Clinical Trial Update")[:500],
                    "announcement_url": f"https://clinicaltrials.gov/study/{trial.nct_id}" if trial.nct_id else None,
                    "published_at": published_at,
                    "raw_text": _trial_text(trial),
                    "external_id": trial.nct_id,
                })

            total_fetched += len(trials)
            logger.debug("ClinicalTrials: fetched %d/%d studies", total_fetched, total)

            if not next_token or not trials:
                break
            page_token = next_token

    except Exception as e:
        logger.warning("ClinicalTrials fetch failed: %s", e)

    logger.info(
        "Fetched %d ClinicalTrials announcements matched to ticker universe",
        len(announcements),
    )
    return announcements


def _build_name_lookup(
    tickers: list[str],
    ticker_to_company: dict[str, str],
) -> tuple[dict, dict]:
    """Build (full_name_lookup, prefix_lookup) for scored company matching."""
    lookup: dict[str, str] = {}
    prefix_lookup: dict[str, str] = {}
    for ticker in tickers:
        company = ticker_to_company.get(ticker, "")
        if not company:
            continue
        normalized = _normalize(company)
        if len(normalized) >= 4:
            lookup[normalized] = ticker
        parts = normalized.split()
        if parts and len(parts[0]) >= 5:
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
    sponsor: str | None,
    lookup_pair: tuple[dict, dict],
) -> str | None:
    """Match a sponsor name to a ticker with scored precision.

    Priority:
      1. Exact normalized match
      2. Long substring match (≥8 chars) in either direction
      3. First-word prefix match (≥7 chars)
    """
    if not sponsor:
        return None
    lookup, prefix_lookup = lookup_pair
    normalized = _normalize(sponsor)
    if not normalized or len(normalized) < 4:
        return None

    # 1. Exact match
    if normalized in lookup:
        return lookup[normalized]

    # 2. Long substring in either direction
    for known, ticker in lookup.items():
        if len(known) >= 8 and known in normalized:
            return ticker
        if len(normalized) >= 8 and normalized in known:
            return ticker

    # 3. First-word prefix match
    parts = normalized.split()
    if parts and len(parts[0]) >= 7 and parts[0] in prefix_lookup:
        return prefix_lookup[parts[0]]

    return None


def _classify(trial) -> str:
    status = (getattr(trial, "status", None) or "")
    status_str = status.value if hasattr(status, "value") else str(status).upper()

    if getattr(trial, "has_results", False):
        return "TRIAL_RESULTS"

    status_map = {
        "COMPLETED": "TRIAL_RESULTS",
        "TERMINATED": "TRIAL_UPDATE",
        "SUSPENDED": "TRIAL_UPDATE",
        "WITHDRAWN": "TRIAL_UPDATE",
        "ACTIVE_NOT_RECRUITING": "TRIAL_UPDATE",
        "RECRUITING": "TRIAL_START",
        "NOT_YET_RECRUITING": "TRIAL_START",
        "ENROLLING_BY_INVITATION": "TRIAL_START",
    }
    return status_map.get(status_str, "TRIAL_UPDATE")


def _trial_text(trial) -> str:
    parts = []
    attrs = [
        ("title", "Title"), ("brief_summary", "Summary"),
        ("sponsor", "Sponsor"), ("enrollment", "Enrollment"),
    ]
    for attr, label in attrs:
        val = getattr(trial, attr, None)
        if val:
            parts.append(f"{label}: {val}")
    if trial.phases:
        parts.append(f"Phase: {', '.join(trial.phases)}")
    if trial.conditions:
        parts.append(f"Conditions: {', '.join(trial.conditions[:3])}")
    if trial.interventions:
        parts.append(f"Interventions: {', '.join(trial.interventions[:3])}")
    status = getattr(trial, "status", None)
    if status:
        parts.append(f"Status: {status.value if hasattr(status, 'value') else status}")
    return " | ".join(parts)
