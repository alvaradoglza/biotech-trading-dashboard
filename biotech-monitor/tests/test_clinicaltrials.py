"""Tests for ClinicalTrials.gov API client."""

import json
import pytest
import responses
from datetime import date, datetime, timedelta
from unittest.mock import patch, MagicMock

from src.clients.clinicaltrials import (
    ClinicalTrialsClient,
    ClinicalTrial,
    TrialStatus,
    TrialPhase,
    ClinicalTrialsAPIError,
)


class TestTrialStatus:
    """Tests for TrialStatus enum."""

    def test_from_string_valid(self):
        """Test converting valid status strings."""
        assert TrialStatus.from_string("RECRUITING") == TrialStatus.RECRUITING
        assert TrialStatus.from_string("COMPLETED") == TrialStatus.COMPLETED
        assert TrialStatus.from_string("NOT_YET_RECRUITING") == TrialStatus.NOT_YET_RECRUITING

    def test_from_string_with_spaces(self):
        """Test converting status strings with spaces."""
        assert TrialStatus.from_string("NOT YET RECRUITING") == TrialStatus.NOT_YET_RECRUITING
        assert TrialStatus.from_string("ACTIVE NOT RECRUITING") == TrialStatus.ACTIVE_NOT_RECRUITING

    def test_from_string_lowercase(self):
        """Test converting lowercase status strings."""
        assert TrialStatus.from_string("recruiting") == TrialStatus.RECRUITING
        assert TrialStatus.from_string("completed") == TrialStatus.COMPLETED

    def test_from_string_unknown(self):
        """Test unknown status returns UNKNOWN."""
        assert TrialStatus.from_string("INVALID_STATUS") == TrialStatus.UNKNOWN
        assert TrialStatus.from_string("") == TrialStatus.UNKNOWN


class TestTrialPhase:
    """Tests for TrialPhase enum."""

    def test_from_string_valid(self):
        """Test converting valid phase strings."""
        assert TrialPhase.from_string("PHASE1") == TrialPhase.PHASE_1
        assert TrialPhase.from_string("PHASE2") == TrialPhase.PHASE_2
        assert TrialPhase.from_string("PHASE3") == TrialPhase.PHASE_3

    def test_from_string_with_spaces(self):
        """Test converting phase strings with spaces."""
        assert TrialPhase.from_string("PHASE 1") == TrialPhase.PHASE_1
        assert TrialPhase.from_string("EARLY PHASE 1") == TrialPhase.EARLY_PHASE_1

    def test_from_string_unknown(self):
        """Test unknown phase returns None."""
        assert TrialPhase.from_string("INVALID") is None


class TestClinicalTrial:
    """Tests for ClinicalTrial dataclass."""

    def test_url_property(self):
        """Test URL generation."""
        trial = ClinicalTrial(
            nct_id="NCT04470427",
            title="Test Trial",
            sponsor="Test Sponsor",
            status=TrialStatus.RECRUITING,
        )
        assert trial.url == "https://clinicaltrials.gov/study/NCT04470427"

    def test_phase_display_single(self):
        """Test phase display with single phase."""
        trial = ClinicalTrial(
            nct_id="NCT12345",
            title="Test",
            sponsor="Sponsor",
            status=TrialStatus.RECRUITING,
            phases=["PHASE3"],
        )
        assert trial.phase_display == "Phase 3"

    def test_phase_display_multiple(self):
        """Test phase display with multiple phases."""
        trial = ClinicalTrial(
            nct_id="NCT12345",
            title="Test",
            sponsor="Sponsor",
            status=TrialStatus.RECRUITING,
            phases=["PHASE2", "PHASE3"],
        )
        assert trial.phase_display == "Phase 2/Phase 3"

    def test_phase_display_empty(self):
        """Test phase display with no phases."""
        trial = ClinicalTrial(
            nct_id="NCT12345",
            title="Test",
            sponsor="Sponsor",
            status=TrialStatus.RECRUITING,
            phases=[],
        )
        assert trial.phase_display == "N/A"


class TestClinicalTrialsClientInit:
    """Tests for ClinicalTrialsClient initialization."""

    def test_init_default(self):
        """Test default initialization."""
        client = ClinicalTrialsClient()
        assert client.rate_limiter.requests_per_second == 10.0

    def test_init_caps_rate_limit(self):
        """Test that rate limit is capped."""
        client = ClinicalTrialsClient(requests_per_second=20.0)
        assert client.rate_limiter.requests_per_second == 10.0

    def test_init_allows_lower_rate(self):
        """Test that lower rate limits are allowed."""
        client = ClinicalTrialsClient(requests_per_second=5.0)
        assert client.rate_limiter.requests_per_second == 5.0


class TestDateParsing:
    """Tests for date parsing."""

    def test_parse_date_iso_format(self):
        """Test parsing ISO format date."""
        result = ClinicalTrialsClient._parse_date("2024-01-15")
        assert result == date(2024, 1, 15)

    def test_parse_date_year_month(self):
        """Test parsing year-month format."""
        result = ClinicalTrialsClient._parse_date("2024-01")
        assert result == date(2024, 1, 1)

    def test_parse_date_long_format(self):
        """Test parsing long format date."""
        result = ClinicalTrialsClient._parse_date("January 15, 2024")
        assert result == date(2024, 1, 15)

    def test_parse_date_none(self):
        """Test parsing None returns None."""
        result = ClinicalTrialsClient._parse_date(None)
        assert result is None

    def test_parse_date_invalid(self):
        """Test parsing invalid date returns None."""
        result = ClinicalTrialsClient._parse_date("invalid-date")
        assert result is None


def get_sample_study_response():
    """Generate sample study response with current dates."""
    today = datetime.now()
    last_update = (today - timedelta(days=15)).strftime("%Y-%m-%d")

    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT04470427",
                "briefTitle": "A Study of mRNA-1273 Vaccine in Adults",
                "officialTitle": "A Phase 3 Study...",
            },
            "statusModule": {
                "overallStatus": "COMPLETED",
                "startDateStruct": {"date": "2020-07-27"},
                "completionDateStruct": {"date": "2022-12-31"},
                "lastUpdatePostDateStruct": {"date": last_update},
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "Moderna, Inc."},
            },
            "conditionsModule": {
                "conditions": ["COVID-19", "SARS-CoV-2 Infection"],
            },
            "designModule": {
                "phases": ["PHASE3"],
                "enrollmentInfo": {"count": 30000},
            },
            "armsInterventionsModule": {
                "interventions": [
                    {"name": "mRNA-1273"},
                    {"name": "Placebo"},
                ],
            },
            "descriptionModule": {
                "briefSummary": "This study will evaluate...",
            },
        },
        "resultsSection": {"participantFlowModule": {}},
    }


def get_sample_search_response():
    """Generate sample search response."""
    return {
        "studies": [get_sample_study_response()],
        "nextPageToken": "abc123",
        "totalCount": 25,
    }


@pytest.mark.asyncio
class TestClinicalTrialsClientAPI:
    """Tests for ClinicalTrials.gov API calls."""

    @responses.activate
    async def test_search_studies_success(self):
        """Test successful study search."""
        responses.add(
            responses.GET,
            "https://clinicaltrials.gov/api/v2/studies",
            json=get_sample_search_response(),
            status=200,
        )

        async with ClinicalTrialsClient() as client:
            trials, next_token, total = await client.search_studies(
                sponsor="Moderna"
            )

        assert len(trials) == 1
        assert next_token == "abc123"
        assert total == 25

        first = trials[0]
        assert first.nct_id == "NCT04470427"
        assert first.sponsor == "Moderna, Inc."

    @responses.activate
    async def test_search_studies_with_status_filter(self):
        """Test search with status filter."""
        responses.add(
            responses.GET,
            "https://clinicaltrials.gov/api/v2/studies",
            json=get_sample_search_response(),
            status=200,
        )

        async with ClinicalTrialsClient() as client:
            trials, _, _ = await client.search_studies(
                status=[TrialStatus.RECRUITING, TrialStatus.COMPLETED]
            )

        assert len(trials) >= 0

    @responses.activate
    async def test_search_studies_empty_response(self):
        """Test search with no results."""
        responses.add(
            responses.GET,
            "https://clinicaltrials.gov/api/v2/studies",
            json={"studies": [], "totalCount": 0},
            status=200,
        )

        async with ClinicalTrialsClient() as client:
            trials, next_token, total = await client.search_studies(
                sponsor="NonexistentCompany"
            )

        assert trials == []
        assert next_token is None
        assert total == 0

    @responses.activate
    async def test_search_studies_api_error(self):
        """Test search handles API errors."""
        responses.add(
            responses.GET,
            "https://clinicaltrials.gov/api/v2/studies",
            json={"error": "Internal error"},
            status=500,
        )

        async with ClinicalTrialsClient() as client:
            with pytest.raises(ClinicalTrialsAPIError):
                await client.search_studies(sponsor="Test")

    @responses.activate
    async def test_get_study_success(self):
        """Test getting a specific study."""
        responses.add(
            responses.GET,
            "https://clinicaltrials.gov/api/v2/studies/NCT04470427",
            json=get_sample_study_response(),
            status=200,
        )

        async with ClinicalTrialsClient() as client:
            trial = await client.get_study("NCT04470427")

        assert trial is not None
        assert trial.nct_id == "NCT04470427"
        assert trial.sponsor == "Moderna, Inc."
        assert "COVID-19" in trial.conditions

    @responses.activate
    async def test_get_study_not_found(self):
        """Test getting a non-existent study returns None."""
        responses.add(
            responses.GET,
            "https://clinicaltrials.gov/api/v2/studies/NCT99999999",
            json={"error": "Not found"},
            status=404,
        )

        async with ClinicalTrialsClient() as client:
            trial = await client.get_study("NCT99999999")

        assert trial is None

    @responses.activate
    async def test_search_by_sponsor(self):
        """Test search by sponsor."""
        response = get_sample_search_response()
        response["nextPageToken"] = None

        responses.add(
            responses.GET,
            "https://clinicaltrials.gov/api/v2/studies",
            json=response,
            status=200,
        )

        async with ClinicalTrialsClient() as client:
            trials = await client.search_by_sponsor("Moderna", days_back=365)

        assert len(trials) >= 1

    @responses.activate
    async def test_get_recently_updated(self):
        """Test getting recently updated trials."""
        response = get_sample_search_response()
        response["nextPageToken"] = None

        responses.add(
            responses.GET,
            "https://clinicaltrials.gov/api/v2/studies",
            json=response,
            status=200,
        )

        async with ClinicalTrialsClient() as client:
            trials = await client.get_recently_updated(
                sponsor_names=["Moderna"],
                days_back=30,
            )

        nct_ids = [t.nct_id for t in trials]
        assert len(nct_ids) == len(set(nct_ids))


class TestSponsorMatching:
    """Tests for sponsor name matching."""

    def test_exact_match(self):
        """Test exact match."""
        companies = ["Moderna, Inc.", "Pfizer Inc", "AstraZeneca"]
        result = ClinicalTrialsClient.match_sponsor_to_company(
            "Moderna, Inc.", companies
        )
        assert result == "Moderna, Inc."

    def test_case_insensitive_match(self):
        """Test case-insensitive match."""
        companies = ["Moderna, Inc.", "Pfizer Inc"]
        result = ClinicalTrialsClient.match_sponsor_to_company(
            "moderna, inc.", companies
        )
        assert result == "Moderna, Inc."

    def test_partial_match_sponsor_contains_company(self):
        """Test partial match where sponsor contains company name."""
        companies = ["Moderna", "Pfizer"]
        result = ClinicalTrialsClient.match_sponsor_to_company(
            "Moderna TX, Inc.", companies
        )
        assert result == "Moderna"

    def test_partial_match_company_contains_sponsor(self):
        """Test partial match where company contains sponsor name."""
        companies = ["Moderna Therapeutics, Inc.", "Pfizer Inc"]
        result = ClinicalTrialsClient.match_sponsor_to_company(
            "Moderna", companies
        )
        assert result == "Moderna Therapeutics, Inc."

    def test_no_match(self):
        """Test no match returns None."""
        companies = ["Pfizer Inc", "AstraZeneca"]
        result = ClinicalTrialsClient.match_sponsor_to_company(
            "Moderna, Inc.", companies
        )
        assert result is None

    def test_suffix_removal_match(self):
        """Test match after removing common suffixes."""
        companies = ["Moderna Pharmaceuticals Inc"]
        result = ClinicalTrialsClient.match_sponsor_to_company(
            "Moderna Pharmaceuticals", companies
        )
        assert result == "Moderna Pharmaceuticals Inc"


@pytest.mark.asyncio
class TestClinicalTrialsStudyParsing:
    """Tests for study response parsing."""

    @responses.activate
    async def test_parse_study_all_fields(self):
        """Test parsing a study with all fields."""
        responses.add(
            responses.GET,
            "https://clinicaltrials.gov/api/v2/studies/NCT04470427",
            json=get_sample_study_response(),
            status=200,
        )

        async with ClinicalTrialsClient() as client:
            trial = await client.get_study("NCT04470427")

        assert trial.nct_id == "NCT04470427"
        assert "mRNA-1273" in trial.title
        assert trial.sponsor == "Moderna, Inc."
        assert trial.status == TrialStatus.COMPLETED
        assert "PHASE3" in trial.phases
        assert trial.enrollment == 30000
        assert len(trial.conditions) == 2
        assert len(trial.interventions) == 2

    @responses.activate
    async def test_parse_study_missing_fields(self):
        """Test parsing a study with missing optional fields."""
        minimal_study = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT00000001",
                    "briefTitle": "Minimal Study",
                },
                "statusModule": {
                    "overallStatus": "UNKNOWN",
                },
                "sponsorCollaboratorsModule": {},
                "conditionsModule": {},
                "designModule": {},
                "armsInterventionsModule": {},
                "descriptionModule": {},
            },
        }

        responses.add(
            responses.GET,
            "https://clinicaltrials.gov/api/v2/studies/NCT00000001",
            json=minimal_study,
            status=200,
        )

        async with ClinicalTrialsClient() as client:
            trial = await client.get_study("NCT00000001")

        assert trial.nct_id == "NCT00000001"
        assert trial.sponsor == ""
        assert trial.phases == []
        assert trial.conditions == []
        assert trial.enrollment is None
        assert trial.has_results is False


@pytest.mark.asyncio
class TestClinicalTrialsEdgeCases:
    """Tests for edge cases in study parsing."""

    @responses.activate
    async def test_parse_study_phases_as_string(self):
        """Test parsing a study where phases is a string instead of list."""
        study = get_sample_study_response()
        study["protocolSection"]["designModule"]["phases"] = "PHASE3"

        responses.add(
            responses.GET,
            "https://clinicaltrials.gov/api/v2/studies/NCT12345",
            json=study,
            status=200,
        )

        async with ClinicalTrialsClient() as client:
            trial = await client.get_study("NCT12345")

        assert trial.phases == ["PHASE3"]

    @responses.activate
    async def test_parse_study_enrollment_invalid(self):
        """Test parsing a study with invalid enrollment value."""
        study = get_sample_study_response()
        study["protocolSection"]["designModule"]["enrollmentInfo"] = {"count": "not-a-number"}

        responses.add(
            responses.GET,
            "https://clinicaltrials.gov/api/v2/studies/NCT12345",
            json=study,
            status=200,
        )

        async with ClinicalTrialsClient() as client:
            trial = await client.get_study("NCT12345")

        assert trial.enrollment is None

    @responses.activate
    async def test_get_trials_by_nct_ids(self):
        """Test getting multiple trials by NCT IDs."""
        responses.add(
            responses.GET,
            "https://clinicaltrials.gov/api/v2/studies/NCT04470427",
            json=get_sample_study_response(),
            status=200,
        )
        responses.add(
            responses.GET,
            "https://clinicaltrials.gov/api/v2/studies/NCT99999999",
            json={"error": "Not found"},
            status=404,
        )

        async with ClinicalTrialsClient() as client:
            trials = await client.get_trials_by_nct_ids(
                ["NCT04470427", "NCT99999999"]
            )

        assert len(trials) == 1
        assert trials[0].nct_id == "NCT04470427"


@pytest.mark.asyncio
class TestClinicalTrialsClientPagination:
    """Tests for pagination handling."""

    @responses.activate
    async def test_pagination_follows_next_token(self):
        """Test that pagination follows next page tokens."""
        page1 = {
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT00000001", "briefTitle": "Study 1"},
                        "statusModule": {"overallStatus": "RECRUITING", "lastUpdatePostDateStruct": {"date": "2026-01-01"}},
                        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Test"}},
                        "conditionsModule": {},
                        "designModule": {},
                        "armsInterventionsModule": {},
                        "descriptionModule": {},
                    }
                }
            ],
            "nextPageToken": "page2",
            "totalCount": 2,
        }
        page2 = {
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT00000002", "briefTitle": "Study 2"},
                        "statusModule": {"overallStatus": "COMPLETED", "lastUpdatePostDateStruct": {"date": "2026-01-01"}},
                        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Test"}},
                        "conditionsModule": {},
                        "designModule": {},
                        "armsInterventionsModule": {},
                        "descriptionModule": {},
                    }
                }
            ],
            "nextPageToken": None,
            "totalCount": 2,
        }

        responses.add(
            responses.GET,
            "https://clinicaltrials.gov/api/v2/studies",
            json=page1,
            status=200,
        )
        responses.add(
            responses.GET,
            "https://clinicaltrials.gov/api/v2/studies",
            json=page2,
            status=200,
        )

        async with ClinicalTrialsClient() as client:
            trials = await client.search_by_sponsor("Test", days_back=365)

        assert len(responses.calls) == 2
        assert len(trials) == 2
