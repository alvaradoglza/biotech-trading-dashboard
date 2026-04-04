"""ClinicalTrials.gov API v2 client for trial monitoring."""

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from functools import partial
from typing import Any, Optional

import requests

from pipeline.clients.base import APIError
from pipeline.utils.logging import get_logger
from pipeline.utils.rate_limiter import RateLimiter

logger = get_logger(__name__)


class ClinicalTrialsAPIError(APIError):
    """ClinicalTrials.gov API-specific error."""

    pass


class TrialStatus(Enum):
    """Clinical trial status values from ClinicalTrials.gov API v2."""

    NOT_YET_RECRUITING = "NOT_YET_RECRUITING"
    RECRUITING = "RECRUITING"
    ENROLLING_BY_INVITATION = "ENROLLING_BY_INVITATION"
    ACTIVE_NOT_RECRUITING = "ACTIVE_NOT_RECRUITING"
    COMPLETED = "COMPLETED"
    SUSPENDED = "SUSPENDED"
    TERMINATED = "TERMINATED"
    WITHDRAWN = "WITHDRAWN"
    AVAILABLE = "AVAILABLE"
    NO_LONGER_AVAILABLE = "NO_LONGER_AVAILABLE"
    TEMPORARILY_NOT_AVAILABLE = "TEMPORARILY_NOT_AVAILABLE"
    APPROVED_FOR_MARKETING = "APPROVED_FOR_MARKETING"
    WITHHELD = "WITHHELD"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_string(cls, value: str) -> "TrialStatus":
        """Convert string to TrialStatus enum.

        Args:
            value: Status string from API

        Returns:
            TrialStatus enum value
        """
        try:
            # Handle API format with spaces/different casing
            normalized = value.upper().replace(" ", "_").replace("-", "_")
            return cls(normalized)
        except ValueError:
            logger.warning("Unknown trial status", status=value)
            return cls.UNKNOWN


class TrialPhase(Enum):
    """Clinical trial phases."""

    EARLY_PHASE_1 = "EARLY_PHASE1"
    PHASE_1 = "PHASE1"
    PHASE_2 = "PHASE2"
    PHASE_3 = "PHASE3"
    PHASE_4 = "PHASE4"
    NOT_APPLICABLE = "NA"

    @classmethod
    def from_string(cls, value: str) -> Optional["TrialPhase"]:
        """Convert string to TrialPhase enum.

        Args:
            value: Phase string from API

        Returns:
            TrialPhase enum value or None
        """
        phase_map = {
            "EARLY_PHASE1": cls.EARLY_PHASE_1,
            "PHASE1": cls.PHASE_1,
            "PHASE2": cls.PHASE_2,
            "PHASE3": cls.PHASE_3,
            "PHASE4": cls.PHASE_4,
            "NA": cls.NOT_APPLICABLE,
            "EARLY PHASE 1": cls.EARLY_PHASE_1,
            "PHASE 1": cls.PHASE_1,
            "PHASE 2": cls.PHASE_2,
            "PHASE 3": cls.PHASE_3,
            "PHASE 4": cls.PHASE_4,
            "NOT APPLICABLE": cls.NOT_APPLICABLE,
        }
        normalized = value.upper().replace("_", " ")
        return phase_map.get(normalized) or phase_map.get(value.upper())


@dataclass
class ClinicalTrial:
    """Represents a clinical trial from ClinicalTrials.gov."""

    nct_id: str
    title: str
    sponsor: str
    status: TrialStatus
    phases: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    interventions: list[str] = field(default_factory=list)
    start_date: Optional[date] = None
    completion_date: Optional[date] = None
    last_update_date: Optional[date] = None
    study_first_posted_date: Optional[date] = None
    enrollment: Optional[int] = None
    has_results: bool = False
    brief_summary: Optional[str] = None

    @property
    def url(self) -> str:
        """Get the ClinicalTrials.gov URL for this trial."""
        return f"https://clinicaltrials.gov/study/{self.nct_id}"

    @property
    def phase_display(self) -> str:
        """Get a human-readable phase string."""
        if not self.phases:
            return "N/A"
        return "/".join(p.replace("PHASE", "Phase ").replace("EARLY_", "Early ") for p in self.phases)


class ClinicalTrialsClient:
    """Client for ClinicalTrials.gov API v2.

    The API provides access to clinical trial data including:
    - Trial search by sponsor, condition, intervention
    - Trial status monitoring
    - Results availability

    API v2 documentation: https://clinicaltrials.gov/data-api/api

    Note: Uses requests library instead of httpx due to TLS fingerprinting
    by ClinicalTrials.gov that blocks httpx requests.
    """

    BASE_URL = "https://clinicaltrials.gov/api/v2"

    # Fields to request from the API
    DEFAULT_FIELDS = [
        "NCTId",
        "BriefTitle",
        "OfficialTitle",
        "OverallStatus",
        "Phase",
        "Condition",
        "InterventionName",
        "LeadSponsorName",
        "StartDate",
        "CompletionDate",
        "LastUpdatePostDate",
        "StudyFirstPostDate",
        "EnrollmentCount",
        "HasResults",
        "BriefSummary",
    ]

    def __init__(
        self,
        requests_per_second: float = 10.0,
    ):
        """Initialize the ClinicalTrials.gov client.

        Args:
            requests_per_second: Maximum request rate (be respectful, ~10/sec)
        """
        if requests_per_second > 10.0:
            logger.warning(
                "Capping ClinicalTrials.gov rate to 10 req/sec",
                requested=requests_per_second,
            )
            requests_per_second = 10.0

        self.rate_limiter = RateLimiter(requests_per_second=requests_per_second)

        # ClinicalTrials.gov requires a standard browser-like User-Agent
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        })

    async def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()

    async def __aenter__(self) -> "ClinicalTrialsClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def _request(
        self,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Make a GET request to the API.

        Uses requests library in a thread pool to maintain async interface.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response as dict
        """
        await self.rate_limiter.acquire()

        url = f"{self.BASE_URL}/{endpoint}"

        logger.debug("Making ClinicalTrials.gov request", url=url, params=params)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            partial(self._session.get, url, params=params, timeout=30),
        )

        if response.status_code == 404:
            raise ClinicalTrialsAPIError("Not found", status_code=404)

        if response.status_code != 200:
            raise ClinicalTrialsAPIError(
                f"API error: {response.status_code} - {response.text[:200]}",
                status_code=response.status_code,
            )

        return response.json()

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> Optional[date]:
        """Parse date string from API response.

        Args:
            date_str: Date string in various formats

        Returns:
            Parsed date or None
        """
        if not date_str:
            return None

        # Try different date formats
        formats = [
            "%Y-%m-%d",
            "%Y-%m",
            "%B %d, %Y",
            "%B %Y",
            "%Y",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue

        logger.debug("Could not parse date", date_str=date_str)
        return None

    def _parse_study(self, study: dict[str, Any]) -> ClinicalTrial:
        """Parse a study from API response into ClinicalTrial object.

        Args:
            study: Study data from API

        Returns:
            ClinicalTrial object
        """
        # API v2 uses nested structure
        protocol = study.get("protocolSection", {})
        identification = protocol.get("identificationModule", {})
        status_module = protocol.get("statusModule", {})
        design = protocol.get("designModule", {})
        sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
        conditions_module = protocol.get("conditionsModule", {})
        interventions_module = protocol.get("armsInterventionsModule", {})
        description_module = protocol.get("descriptionModule", {})
        results_section = study.get("resultsSection", {})

        # Extract fields
        nct_id = identification.get("nctId", "")
        title = identification.get("briefTitle", "") or identification.get("officialTitle", "")

        # Status
        status_str = status_module.get("overallStatus", "UNKNOWN")
        status = TrialStatus.from_string(status_str)

        # Phases
        phases = design.get("phases", [])
        if isinstance(phases, str):
            phases = [phases]

        # Sponsor
        lead_sponsor = sponsor_module.get("leadSponsor", {})
        sponsor = lead_sponsor.get("name", "")

        # Conditions
        conditions = conditions_module.get("conditions", [])

        # Interventions
        interventions = []
        for intervention in interventions_module.get("interventions", []):
            name = intervention.get("name", "")
            if name:
                interventions.append(name)

        # Dates
        start_date_struct = status_module.get("startDateStruct", {})
        completion_date_struct = status_module.get("completionDateStruct", {})
        last_update = status_module.get("lastUpdatePostDateStruct", {})
        study_first_posted = status_module.get("studyFirstPostDateStruct", {})

        start_date = self._parse_date(start_date_struct.get("date"))
        completion_date = self._parse_date(completion_date_struct.get("date"))
        last_update_date = self._parse_date(last_update.get("date"))
        study_first_posted_date = self._parse_date(study_first_posted.get("date"))

        # Enrollment
        enrollment_info = design.get("enrollmentInfo", {})
        enrollment = enrollment_info.get("count")
        if enrollment is not None:
            try:
                enrollment = int(enrollment)
            except (ValueError, TypeError):
                enrollment = None

        # Results
        has_results = bool(results_section)

        # Brief summary
        brief_summary = description_module.get("briefSummary")

        return ClinicalTrial(
            nct_id=nct_id,
            title=title,
            sponsor=sponsor,
            status=status,
            phases=phases,
            conditions=conditions,
            interventions=interventions,
            start_date=start_date,
            completion_date=completion_date,
            last_update_date=last_update_date,
            study_first_posted_date=study_first_posted_date,
            enrollment=enrollment,
            has_results=has_results,
            brief_summary=brief_summary,
        )

    async def search_studies(
        self,
        sponsor: Optional[str] = None,
        condition: Optional[str] = None,
        term: Optional[str] = None,
        status: Optional[list[TrialStatus]] = None,
        page_size: int = 100,
        page_token: Optional[str] = None,
    ) -> tuple[list[ClinicalTrial], Optional[str], int]:
        """Search for clinical trials.

        Args:
            sponsor: Sponsor name to search
            condition: Condition/disease to search
            term: General search term
            status: List of statuses to filter
            page_size: Number of results per page (max 1000)
            page_token: Token for pagination

        Returns:
            Tuple of (list of trials, next page token, total count)
        """
        params: dict[str, Any] = {
            "pageSize": min(page_size, 1000),
            "format": "json",
        }

        # Build query
        query_parts = []
        if sponsor:
            query_parts.append(f"AREA[LeadSponsorName]{sponsor}")
        if condition:
            query_parts.append(f"AREA[Condition]{condition}")
        if term:
            query_parts.append(term)

        if query_parts:
            params["query.term"] = " AND ".join(query_parts)

        # Filter by status
        if status:
            status_values = [s.value for s in status]
            params["filter.overallStatus"] = ",".join(status_values)

        if page_token:
            params["pageToken"] = page_token

        logger.debug("Searching clinical trials", params=params)

        try:
            response = await self._request("studies", params=params)
        except ClinicalTrialsAPIError:
            raise
        except Exception as e:
            raise ClinicalTrialsAPIError(str(e)) from e

        # Parse response
        studies = response.get("studies", [])
        next_page_token = response.get("nextPageToken")
        total_count = response.get("totalCount", 0)

        trials = [self._parse_study(study) for study in studies]

        logger.info(
            "Found clinical trials",
            count=len(trials),
            total=total_count,
        )

        return trials, next_page_token, total_count

    async def search_by_sponsor(
        self,
        sponsor_name: str,
        days_back: int = 1095,
    ) -> list[ClinicalTrial]:
        """Search for trials by sponsor name.

        Args:
            sponsor_name: Company/sponsor name
            days_back: Number of days to look back for updates

        Returns:
            List of trials for the sponsor
        """
        all_trials = []
        page_token = None

        while True:
            trials, next_token, _ = await self.search_studies(
                sponsor=sponsor_name,
                page_token=page_token,
            )

            # Filter by last update date
            cutoff_date = date.today() - timedelta(days=days_back)
            for trial in trials:
                if trial.last_update_date and trial.last_update_date >= cutoff_date:
                    all_trials.append(trial)
                elif not trial.last_update_date:
                    # Include trials without update date
                    all_trials.append(trial)

            if not next_token or len(trials) == 0:
                break

            page_token = next_token

        return all_trials

    async def get_study(self, nct_id: str) -> Optional[ClinicalTrial]:
        """Get a specific study by NCT ID.

        Args:
            nct_id: NCT identifier (e.g., "NCT04470427")

        Returns:
            ClinicalTrial or None if not found
        """
        endpoint = f"studies/{nct_id}"

        logger.debug("Fetching study", nct_id=nct_id)

        try:
            response = await self._request(endpoint)
            return self._parse_study(response)
        except ClinicalTrialsAPIError as e:
            if e.status_code == 404:
                logger.warning("Study not found", nct_id=nct_id)
                return None
            raise

    async def get_study_json(self, nct_id: str) -> Optional[dict]:
        """Get raw study JSON by NCT ID.

        Args:
            nct_id: NCT identifier (e.g., "NCT04470427")

        Returns:
            Raw study JSON dict or None if not found
        """
        endpoint = f"studies/{nct_id}"

        logger.debug("Fetching study JSON", nct_id=nct_id)

        try:
            return await self._request(endpoint)
        except ClinicalTrialsAPIError as e:
            if e.status_code == 404:
                logger.warning("Study not found", nct_id=nct_id)
                return None
            raise

    async def get_recently_updated(
        self,
        sponsor_names: list[str],
        days_back: int = 7,
    ) -> list[ClinicalTrial]:
        """Get recently updated trials for multiple sponsors.

        Args:
            sponsor_names: List of sponsor/company names
            days_back: Number of days to look back

        Returns:
            List of recently updated trials
        """
        all_trials = []

        for sponsor in sponsor_names:
            try:
                trials = await self.search_by_sponsor(sponsor, days_back=days_back)
                all_trials.extend(trials)
            except Exception as e:
                logger.error(
                    "Failed to fetch trials for sponsor",
                    sponsor=sponsor,
                    error=str(e),
                )

        # Deduplicate by NCT ID
        seen = set()
        unique_trials = []
        for trial in all_trials:
            if trial.nct_id not in seen:
                seen.add(trial.nct_id)
                unique_trials.append(trial)

        return unique_trials

    @staticmethod
    def match_sponsor_to_company(
        sponsor_name: str,
        company_names: list[str],
    ) -> Optional[str]:
        """Match a sponsor name to a company from our list.

        Uses case-insensitive and partial matching.

        Args:
            sponsor_name: Sponsor name from ClinicalTrials.gov
            company_names: List of company names to match against

        Returns:
            Matched company name or None
        """
        sponsor_lower = sponsor_name.lower()

        # Exact match (case-insensitive)
        for company in company_names:
            if company.lower() == sponsor_lower:
                return company

        # Partial match - sponsor contains company name
        for company in company_names:
            company_lower = company.lower()
            if company_lower in sponsor_lower:
                return company

        # Partial match - company contains sponsor name
        for company in company_names:
            company_lower = company.lower()
            if sponsor_lower in company_lower:
                return company

        # Common suffixes/prefixes removal
        suffixes = ["inc", "inc.", "corp", "corp.", "llc", "ltd", "ltd.", "plc", "pharmaceuticals", "therapeutics", "biosciences"]
        sponsor_clean = sponsor_lower
        for suffix in suffixes:
            sponsor_clean = sponsor_clean.replace(f" {suffix}", "").replace(f", {suffix}", "")

        for company in company_names:
            company_clean = company.lower()
            for suffix in suffixes:
                company_clean = company_clean.replace(f" {suffix}", "").replace(f", {suffix}", "")

            if company_clean == sponsor_clean:
                return company

        return None

    async def get_trials_by_nct_ids(
        self,
        nct_ids: list[str],
    ) -> list[ClinicalTrial]:
        """Get multiple trials by their NCT IDs.

        Args:
            nct_ids: List of NCT identifiers

        Returns:
            List of ClinicalTrial objects
        """
        trials = []

        for nct_id in nct_ids:
            trial = await self.get_study(nct_id)
            if trial:
                trials.append(trial)

        return trials
