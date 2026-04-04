"""
fetch_clinical_trials.py — Fetch ClinicalTrials.gov announcements.
Returns a list of announcement dicts ready for Supabase upsert.

NOTE: ClinicalTrials.gov client uses `requests` (not httpx) due to TLS fingerprinting.
"""

import logging
from datetime import datetime, timedelta

from pipeline.clients.clinicaltrials import ClinicalTrialsClient

logger = logging.getLogger(__name__)


def fetch_clinical_trials_announcements(
    tickers: list[str],
    days_back: int = 7,
) -> list[dict]:
    """Fetch recent ClinicalTrials.gov updates for the given tickers.

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

    client = ClinicalTrialsClient()

    for ticker in tickers:
        try:
            ticker_announcements = _fetch_ticker_trials(client, ticker, since)
            announcements.extend(ticker_announcements)
        except Exception as e:
            logger.warning("ClinicalTrials fetch failed for %s: %s", ticker, e)

    logger.info(
        "Fetched %d ClinicalTrials announcements for %d tickers",
        len(announcements), len(tickers)
    )
    return announcements


def _fetch_ticker_trials(client, ticker: str, since: datetime) -> list[dict]:
    """Fetch clinical trial updates for one ticker and normalize to announcement dicts."""
    results = []

    try:
        # Search by sponsor/company — the ClinicalTrials API v2 uses query.spons
        trials = client.search_trials(sponsor=ticker, max_results=20)
    except Exception as e:
        logger.debug("No trials found for %s: %s", ticker, e)
        return []

    for trial in trials:
        # Use lastUpdatePostDate as the signal date (most recent status change)
        published_at = _get_update_date(trial)
        if published_at and published_at < since:
            continue

        nct_id = getattr(trial, "nct_id", None) or getattr(trial, "id", None)
        event_type = _classify_event_type(trial)
        raw_text = _build_trial_text(trial)
        title = getattr(trial, "official_title", None) or getattr(trial, "brief_title", None) or "Clinical Trial Update"

        results.append({
            "source": "clinicaltrials",
            "ticker": ticker,
            "company_name": getattr(trial, "sponsor", None) or getattr(trial, "lead_sponsor", None),
            "event_type": event_type,
            "title": title,
            "announcement_url": f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else None,
            "published_at": published_at.isoformat() if published_at else None,
            "raw_text": raw_text,
            "external_id": nct_id,
        })

    return results


def _get_update_date(trial) -> datetime | None:
    """Extract the most relevant date from a trial object."""
    for attr in ["last_update_post_date", "lastUpdatePostDate", "results_first_post_date",
                 "primary_completion_date", "start_date"]:
        val = getattr(trial, attr, None)
        if val:
            return _parse_date(val)
    return None


def _classify_event_type(trial) -> str:
    """Classify a trial update into a broad event type category."""
    status = (getattr(trial, "overall_status", None) or "").upper()

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

    # Check if results are posted
    has_results = bool(getattr(trial, "has_results", False) or getattr(trial, "results_first_post_date", None))
    if has_results:
        return "TRIAL_RESULTS"

    return status_map.get(status, "TRIAL_UPDATE")


def _build_trial_text(trial) -> str:
    """Build a raw text summary from a ClinicalTrials trial object for ML features."""
    parts = []
    text_attrs = [
        "official_title", "brief_title", "brief_summary", "detailed_description",
        "phase", "overall_status", "condition", "intervention_name",
        "sponsor", "enrollment", "primary_outcome_measure",
    ]
    for attr in text_attrs:
        val = getattr(trial, attr, None)
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
        for fmt in ["%Y-%m-%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
                    "%B %d, %Y", "%b %d, %Y"]:
            try:
                return datetime.strptime(value.strip(), fmt)
            except ValueError:
                continue
    return None
