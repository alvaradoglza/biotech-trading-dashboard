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

    # Build reverse lookup: normalized_name_fragment → ticker
    name_to_ticker = _build_name_lookup(tickers, ticker_to_company)

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
                ticker = _match_ticker(trial.sponsor, name_to_ticker)
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
) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for ticker in tickers:
        company = ticker_to_company.get(ticker, "")
        if not company:
            continue
        for fragment in _name_fragments(company):
            lookup[fragment] = ticker
    return lookup


def _name_fragments(name: str) -> list[str]:
    name = name.lower().strip()
    for suffix in [" inc.", " inc", " corp.", " corp", " ltd.", " ltd",
                   " llc", " plc", " therapeutics", " pharmaceuticals",
                   " biosciences", " bioscience", " biopharma"]:
        name = name.replace(suffix, "")
    fragments = [name.strip()]
    parts = name.split()
    if len(parts) >= 2:
        fragments.append(parts[0].strip())
    return [f for f in fragments if len(f) >= 4]


def _match_ticker(sponsor: str | None, lookup: dict[str, str]) -> str | None:
    if not sponsor or not lookup:
        return None
    normalized = sponsor.lower().strip()
    for fragment, ticker in lookup.items():
        if fragment in normalized or normalized in fragment:
            return ticker
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
