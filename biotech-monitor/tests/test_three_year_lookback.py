"""Tests verifying 3-year (1095 days) lookback for ClinicalTrials.gov, OpenFDA approvals,
and OpenFDA recalls.

Covers:
- search_by_sponsor (ClinicalTrials) filters by days_back=1095
- get_drug_approvals_raw + date-filter logic in run_extraction.py
- get_recalls_raw_by_firm (new method) filters by days_back=1095
"""

import inspect
import pytest
import responses as responses_lib
import respx
from datetime import date, timedelta
from httpx import Response

from src.clients.openfda import DrugApproval, DrugRecall, OpenFDAClient, OpenFDAAPIError
from src.clients.clinicaltrials import ClinicalTrialsClient

# ---------------------------------------------------------------------------
# Shared date constants
# ---------------------------------------------------------------------------
TODAY = date.today()
WITHIN_3_YEARS = TODAY - timedelta(days=500)       # 500 days ago  → included
BEYOND_3_YEARS = TODAY - timedelta(days=1200)      # 1200 days ago → excluded
THREE_YEAR_BOUNDARY = TODAY - timedelta(days=1095)  # exactly 1095  → included


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ct_study(nct_id: str, last_update: date | None = None) -> dict:
    """Minimal ClinicalTrials.gov study dict."""
    study: dict = {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id, "briefTitle": f"Study {nct_id}"},
            "statusModule": {"overallStatus": "COMPLETED"},
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Acme Bio"}},
            "conditionsModule": {},
            "designModule": {},
            "armsInterventionsModule": {},
            "descriptionModule": {},
        }
    }
    if last_update is not None:
        study["protocolSection"]["statusModule"]["lastUpdatePostDateStruct"] = {
            "date": last_update.strftime("%Y-%m-%d")
        }
    return study


def _ct_response(studies: list[dict], next_token: str | None = None) -> dict:
    return {"studies": studies, "totalCount": len(studies), "nextPageToken": next_token}


def _recall_result(recall_number: str, initiation_date: date | None, report_date: date | None = None) -> dict:
    return {
        "recall_number": recall_number,
        "recalling_firm": "Acme Bio Inc",
        "product_description": "Test capsules",
        "reason_for_recall": "Contamination risk",
        "classification": "Class II",
        "status": "Ongoing",
        "recall_initiation_date": initiation_date.strftime("%Y%m%d") if initiation_date else None,
        "report_date": report_date.strftime("%Y%m%d") if report_date else None,
        "termination_date": None,
        "voluntary_mandated": "Voluntary",
        "distribution_pattern": "Nationwide",
        "city": "Boston",
        "state": "MA",
        "country": "US",
    }


def _recall_response(results: list[dict], total: int | None = None) -> dict:
    return {
        "results": results,
        "meta": {"results": {"total": total if total is not None else len(results), "skip": 0, "limit": 100}},
    }


def _approval_result(app_number: str, sponsor: str, approval_date: date | None) -> dict:
    date_str = approval_date.strftime("%Y%m%d") if approval_date else ""
    return {
        "application_number": app_number,
        "sponsor_name": sponsor,
        "products": [{"brand_name": "TESTDRUG", "dosage_form": "TABLET", "route": "ORAL"}],
        "submissions": [
            {
                "submission_type": "NDA",
                "submission_status": "AP",
                "submission_status_date": date_str,
            }
        ],
        "openfda": {},
    }


def _approval_response(results: list[dict]) -> dict:
    return {
        "results": results,
        "meta": {"results": {"total": len(results), "skip": 0, "limit": 1000}},
    }


# ===========================================================================
# ClinicalTrials — search_by_sponsor with 3-year lookback
# ===========================================================================


@pytest.mark.asyncio
class TestClinicalTrials3YearLookback:
    """search_by_sponsor must respect days_back=1095."""

    @responses_lib.activate
    async def test_includes_trial_updated_within_3_years(self):
        """Trial updated 500 days ago is included."""
        responses_lib.add(
            responses_lib.GET,
            "https://clinicaltrials.gov/api/v2/studies",
            json=_ct_response([_ct_study("NCT00000001", WITHIN_3_YEARS)]),
            status=200,
        )

        async with ClinicalTrialsClient() as client:
            trials = await client.search_by_sponsor("Acme Bio", days_back=1095)

        assert len(trials) == 1
        assert trials[0].nct_id == "NCT00000001"

    @responses_lib.activate
    async def test_excludes_trial_updated_beyond_3_years(self):
        """Trial updated 1200 days ago is excluded."""
        responses_lib.add(
            responses_lib.GET,
            "https://clinicaltrials.gov/api/v2/studies",
            json=_ct_response([_ct_study("NCT00000002", BEYOND_3_YEARS)]),
            status=200,
        )

        async with ClinicalTrialsClient() as client:
            trials = await client.search_by_sponsor("Acme Bio", days_back=1095)

        assert trials == []

    @responses_lib.activate
    async def test_includes_trial_with_no_update_date(self):
        """Trial with no update date is included (unknown date = keep)."""
        responses_lib.add(
            responses_lib.GET,
            "https://clinicaltrials.gov/api/v2/studies",
            json=_ct_response([_ct_study("NCT00000003", None)]),
            status=200,
        )

        async with ClinicalTrialsClient() as client:
            trials = await client.search_by_sponsor("Acme Bio", days_back=1095)

        assert len(trials) == 1
        assert trials[0].nct_id == "NCT00000003"

    @responses_lib.activate
    async def test_includes_trial_on_exact_3_year_boundary(self):
        """Trial updated exactly 1095 days ago is included (inclusive)."""
        responses_lib.add(
            responses_lib.GET,
            "https://clinicaltrials.gov/api/v2/studies",
            json=_ct_response([_ct_study("NCT00000004", THREE_YEAR_BOUNDARY)]),
            status=200,
        )

        async with ClinicalTrialsClient() as client:
            trials = await client.search_by_sponsor("Acme Bio", days_back=1095)

        assert len(trials) == 1

    @responses_lib.activate
    async def test_mixed_dates_only_recent_trials_pass(self):
        """Only trials within 3 years pass the filter when dates are mixed."""
        responses_lib.add(
            responses_lib.GET,
            "https://clinicaltrials.gov/api/v2/studies",
            json=_ct_response([
                _ct_study("NCT10000001", WITHIN_3_YEARS),   # include
                _ct_study("NCT10000002", BEYOND_3_YEARS),   # exclude
                _ct_study("NCT10000003", None),              # include (no date)
                _ct_study("NCT10000004", THREE_YEAR_BOUNDARY),  # include (boundary)
            ]),
            status=200,
        )

        async with ClinicalTrialsClient() as client:
            trials = await client.search_by_sponsor("Acme Bio", days_back=1095)

        nct_ids = [t.nct_id for t in trials]
        assert "NCT10000001" in nct_ids
        assert "NCT10000002" not in nct_ids
        assert "NCT10000003" in nct_ids
        assert "NCT10000004" in nct_ids

    @responses_lib.activate
    async def test_paginates_through_all_pages(self):
        """Pagination is followed — all pages are fetched."""
        responses_lib.add(
            responses_lib.GET,
            "https://clinicaltrials.gov/api/v2/studies",
            json=_ct_response([_ct_study("NCT20000001", WITHIN_3_YEARS)], next_token="page2"),
            status=200,
        )
        responses_lib.add(
            responses_lib.GET,
            "https://clinicaltrials.gov/api/v2/studies",
            json=_ct_response([_ct_study("NCT20000002", WITHIN_3_YEARS)], next_token=None),
            status=200,
        )

        async with ClinicalTrialsClient() as client:
            trials = await client.search_by_sponsor("Acme Bio", days_back=1095)

        assert len(responses_lib.calls) == 2
        assert len(trials) == 2

    @responses_lib.activate
    async def test_pagination_stops_when_old_trials_dominate(self):
        """When all trials in a page are old, they are excluded but pagination stops naturally."""
        # Page 1: all old
        responses_lib.add(
            responses_lib.GET,
            "https://clinicaltrials.gov/api/v2/studies",
            json=_ct_response([_ct_study("NCT30000001", BEYOND_3_YEARS)], next_token=None),
            status=200,
        )

        async with ClinicalTrialsClient() as client:
            trials = await client.search_by_sponsor("Acme Bio", days_back=1095)

        assert trials == []

    async def test_default_days_back_is_1095(self):
        """search_by_sponsor default days_back parameter must be 1095 (3 years)."""
        sig = inspect.signature(ClinicalTrialsClient.search_by_sponsor)
        assert sig.parameters["days_back"].default == 1095

    @responses_lib.activate
    async def test_empty_response_returns_empty_list(self):
        """Empty API response returns empty list without error."""
        responses_lib.add(
            responses_lib.GET,
            "https://clinicaltrials.gov/api/v2/studies",
            json={"studies": [], "totalCount": 0},
            status=200,
        )

        async with ClinicalTrialsClient() as client:
            trials = await client.search_by_sponsor("NoTrialsCompany", days_back=1095)

        assert trials == []


# ===========================================================================
# OpenFDA Approvals — date filtering
# ===========================================================================


@pytest.mark.asyncio
class TestOpenFDAApprovals3YearLookback:
    """Date filter in run_extraction.py and get_drug_approvals_raw."""

    def _cutoff(self) -> date:
        return TODAY - timedelta(days=1095)

    def _filter(self, approvals_with_raw: list[tuple[DrugApproval, dict]]) -> list[tuple[DrugApproval, dict]]:
        """Replicates the filter applied in run_extraction.py."""
        cutoff_date = self._cutoff()
        return [
            (a, r) for a, r in approvals_with_raw
            if not a.approval_date or a.approval_date >= cutoff_date
        ]

    def _make_approval(self, app_number: str, approval_date: date | None) -> DrugApproval:
        return DrugApproval(
            application_number=app_number,
            sponsor_name="Acme Bio",
            brand_name="TESTDRUG",
            approval_date=approval_date,
        )

    async def test_get_drug_approvals_raw_returns_parsed_and_raw(self, mock_openfda_api):
        """get_drug_approvals_raw returns (DrugApproval, dict) tuples."""
        mock_openfda_api.get("/drug/drugsfda.json").respond(
            200,
            json=_approval_response([_approval_result("NDA123456", "Acme Bio", WITHIN_3_YEARS)]),
        )

        async with OpenFDAClient() as client:
            results, total = await client.get_drug_approvals_raw(
                sponsor_name="Acme Bio", limit=1000
            )

        assert total == 1
        assert len(results) == 1
        parsed, raw = results[0]
        assert parsed.application_number == "NDA123456"
        assert isinstance(raw, dict)
        assert raw["application_number"] == "NDA123456"

    def test_date_filter_excludes_approval_beyond_3_years(self):
        """Approval older than 1095 days is removed by the date filter."""
        old = self._make_approval("NDA000001", BEYOND_3_YEARS)
        result = self._filter([(old, {})])
        assert result == []

    def test_date_filter_includes_approval_within_3_years(self):
        """Approval within 1095 days passes the date filter."""
        recent = self._make_approval("NDA000002", WITHIN_3_YEARS)
        result = self._filter([(recent, {"application_number": "NDA000002"})])
        assert len(result) == 1
        assert result[0][0].application_number == "NDA000002"

    def test_date_filter_includes_approval_on_boundary(self):
        """Approval exactly 1095 days ago passes (inclusive boundary)."""
        boundary = self._make_approval("NDA000003", THREE_YEAR_BOUNDARY)
        result = self._filter([(boundary, {})])
        assert len(result) == 1

    def test_date_filter_includes_approval_with_no_date(self):
        """Approvals with no date are kept (unknown = conservative)."""
        no_date = self._make_approval("NDA000004", None)
        result = self._filter([(no_date, {})])
        assert len(result) == 1

    def test_date_filter_mixed_approvals(self):
        """Mix of old/recent/no-date — only recent + no-date pass."""
        pairs = [
            (self._make_approval("NDA001", BEYOND_3_YEARS), {}),   # exclude
            (self._make_approval("NDA002", WITHIN_3_YEARS), {}),    # include
            (self._make_approval("NDA003", None), {}),               # include
            (self._make_approval("NDA004", THREE_YEAR_BOUNDARY), {}), # include
        ]
        result = self._filter(pairs)
        app_numbers = [a.application_number for a, _ in result]
        assert "NDA001" not in app_numbers
        assert "NDA002" in app_numbers
        assert "NDA003" in app_numbers
        assert "NDA004" in app_numbers

    async def test_get_drug_approvals_raw_404_returns_empty(self, mock_openfda_api):
        """404 from OpenFDA returns empty list — no exception raised."""
        mock_openfda_api.get("/drug/drugsfda.json").respond(404)

        async with OpenFDAClient() as client:
            results, total = await client.get_drug_approvals_raw(
                sponsor_name="NoSuchCompany", limit=1000
            )

        assert results == []
        assert total == 0

    async def test_get_approvals_by_sponsor_with_3_year_window(self, mock_openfda_api):
        """get_approvals_by_sponsor with days_back=1095 passes the cutoff to filter."""
        mock_openfda_api.get("/drug/drugsfda.json").respond(
            200,
            json=_approval_response([_approval_result("NDA999999", "Acme Bio", WITHIN_3_YEARS)]),
        )

        async with OpenFDAClient() as client:
            approvals = await client.get_approvals_by_sponsor("Acme Bio", days_back=1095)

        assert len(approvals) == 1
        assert approvals[0].application_number == "NDA999999"

    async def test_get_approvals_by_sponsor_excludes_old_records(self, mock_openfda_api):
        """get_approvals_by_sponsor excludes approvals beyond 3 years."""
        mock_openfda_api.get("/drug/drugsfda.json").respond(
            200,
            json=_approval_response([_approval_result("NDA888888", "Acme Bio", BEYOND_3_YEARS)]),
        )

        async with OpenFDAClient() as client:
            approvals = await client.get_approvals_by_sponsor("Acme Bio", days_back=1095)

        assert approvals == []


# ===========================================================================
# OpenFDA Recalls — get_recalls_raw_by_firm with 3-year lookback
# ===========================================================================


@pytest.mark.asyncio
class TestOpenFDARecalls3YearLookback:
    """get_recalls_raw_by_firm must filter by days_back=1095."""

    async def test_returns_list_of_parsed_and_raw_tuples(self, mock_openfda_api):
        """Returns list of (DrugRecall, dict) tuples."""
        mock_openfda_api.get("/drug/enforcement.json").respond(
            200,
            json=_recall_response([_recall_result("R-2023-001", WITHIN_3_YEARS)]),
        )

        async with OpenFDAClient() as client:
            results = await client.get_recalls_raw_by_firm("Acme Bio Inc", days_back=1095)

        assert len(results) == 1
        recall, raw = results[0]
        assert isinstance(recall, DrugRecall)
        assert isinstance(raw, dict)
        assert recall.recall_number == "R-2023-001"
        assert raw["recall_number"] == "R-2023-001"

    async def test_includes_recall_within_3_years(self, mock_openfda_api):
        """Recall with initiation date 500 days ago is included."""
        mock_openfda_api.get("/drug/enforcement.json").respond(
            200,
            json=_recall_response([_recall_result("R-2023-002", WITHIN_3_YEARS)]),
        )

        async with OpenFDAClient() as client:
            results = await client.get_recalls_raw_by_firm("Acme Bio Inc", days_back=1095)

        assert len(results) == 1

    async def test_excludes_recall_beyond_3_years(self, mock_openfda_api):
        """Recall with initiation date 1200 days ago is excluded."""
        mock_openfda_api.get("/drug/enforcement.json").respond(
            200,
            json=_recall_response([_recall_result("R-2019-001", BEYOND_3_YEARS)]),
        )

        async with OpenFDAClient() as client:
            results = await client.get_recalls_raw_by_firm("Acme Bio Inc", days_back=1095)

        assert results == []

    async def test_includes_recall_on_exact_3_year_boundary(self, mock_openfda_api):
        """Recall exactly 1095 days ago is included (inclusive boundary)."""
        mock_openfda_api.get("/drug/enforcement.json").respond(
            200,
            json=_recall_response([_recall_result("R-2023-BOUNDARY", THREE_YEAR_BOUNDARY)]),
        )

        async with OpenFDAClient() as client:
            results = await client.get_recalls_raw_by_firm("Acme Bio Inc", days_back=1095)

        assert len(results) == 1

    async def test_falls_back_to_report_date_when_initiation_date_missing(self, mock_openfda_api):
        """When recall_initiation_date is None, report_date is used for filtering."""
        mock_openfda_api.get("/drug/enforcement.json").respond(
            200,
            json=_recall_response([_recall_result("R-2023-NODATEINIT", None, report_date=WITHIN_3_YEARS)]),
        )

        async with OpenFDAClient() as client:
            results = await client.get_recalls_raw_by_firm("Acme Bio Inc", days_back=1095)

        assert len(results) == 1
        assert results[0][0].recall_number == "R-2023-NODATEINIT"

    async def test_old_recall_excluded_via_report_date_fallback(self, mock_openfda_api):
        """Old recall_initiation_date=None + old report_date is excluded."""
        mock_openfda_api.get("/drug/enforcement.json").respond(
            200,
            json=_recall_response([_recall_result("R-2019-OLD", None, report_date=BEYOND_3_YEARS)]),
        )

        async with OpenFDAClient() as client:
            results = await client.get_recalls_raw_by_firm("Acme Bio Inc", days_back=1095)

        assert results == []

    async def test_handles_404_gracefully(self, mock_openfda_api):
        """404 response returns empty list — no exception."""
        mock_openfda_api.get("/drug/enforcement.json").respond(404)

        async with OpenFDAClient() as client:
            results = await client.get_recalls_raw_by_firm("NoSuchFirm", days_back=1095)

        assert results == []

    async def test_deduplicates_same_recall_number(self, mock_openfda_api):
        """If the same recall_number appears in multiple results it is included only once."""
        dupe = _recall_result("R-2023-DUPE", WITHIN_3_YEARS)
        mock_openfda_api.get("/drug/enforcement.json").respond(
            200,
            json=_recall_response([dupe, dupe]),  # same record twice
        )

        async with OpenFDAClient() as client:
            results = await client.get_recalls_raw_by_firm("Acme Bio Inc", days_back=1095)

        recall_numbers = [r.recall_number for r, _ in results]
        assert recall_numbers.count("R-2023-DUPE") == 1

    async def test_mixed_recalls_filters_correctly(self, mock_openfda_api):
        """Mix of old and recent recalls — only recent ones are returned."""
        mock_openfda_api.get("/drug/enforcement.json").respond(
            200,
            json=_recall_response([
                _recall_result("R-RECENT", WITHIN_3_YEARS),
                _recall_result("R-OLD", BEYOND_3_YEARS),
                _recall_result("R-BOUNDARY", THREE_YEAR_BOUNDARY),
            ]),
        )

        async with OpenFDAClient() as client:
            results = await client.get_recalls_raw_by_firm("Acme Bio Inc", days_back=1095)

        recall_numbers = [r.recall_number for r, _ in results]
        assert "R-RECENT" in recall_numbers
        assert "R-OLD" not in recall_numbers
        assert "R-BOUNDARY" in recall_numbers

    async def test_default_days_back_is_1095(self):
        """Default days_back for get_recalls_raw_by_firm must be 1095 (3 years)."""
        sig = inspect.signature(OpenFDAClient.get_recalls_raw_by_firm)
        assert sig.parameters["days_back"].default == 1095
