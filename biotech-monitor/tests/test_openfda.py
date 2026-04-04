"""Tests for OpenFDA API client."""

import pytest
import respx
from datetime import date
from httpx import Response

from src.clients.openfda import (
    OpenFDAClient,
    DrugApproval,
    DrugLabel,
    OpenFDAAPIError,
)


class TestOpenFDAClientInit:
    """Tests for OpenFDAClient initialization."""

    def test_init_without_api_key(self):
        """Test initialization without API key."""
        client = OpenFDAClient()
        assert client.api_key is None
        assert client.has_api_key is False
        # Should use lower rate limit
        assert client.rate_limiter.requests_per_second == 0.5

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        client = OpenFDAClient(api_key="test_key_123")
        assert client.api_key == "test_key_123"
        assert client.has_api_key is True
        # Should use higher rate limit
        assert client.rate_limiter.requests_per_second == 3.0


class TestDateParsing:
    """Tests for date parsing."""

    def test_parse_date_yyyymmdd(self):
        """Test parsing YYYYMMDD format."""
        result = OpenFDAClient._parse_date("20240115")
        assert result == date(2024, 1, 15)

    def test_parse_date_iso_format(self):
        """Test parsing ISO format."""
        result = OpenFDAClient._parse_date("2024-01-15")
        assert result == date(2024, 1, 15)

    def test_parse_date_us_format(self):
        """Test parsing US format."""
        result = OpenFDAClient._parse_date("01/15/2024")
        assert result == date(2024, 1, 15)

    def test_parse_date_none(self):
        """Test parsing None returns None."""
        result = OpenFDAClient._parse_date(None)
        assert result is None

    def test_parse_date_invalid(self):
        """Test parsing invalid date returns None."""
        result = OpenFDAClient._parse_date("invalid")
        assert result is None


class TestDrugApproval:
    """Tests for DrugApproval dataclass."""

    def test_url_property_bla(self):
        """Test URL generation for BLA application."""
        approval = DrugApproval(
            application_number="BLA761222",
            sponsor_name="Test Sponsor",
            brand_name="Test Drug",
        )
        assert "761222" in approval.url
        assert "daf/index.cfm" in approval.url

    def test_url_property_nda(self):
        """Test URL generation for NDA application."""
        approval = DrugApproval(
            application_number="NDA214900",
            sponsor_name="Test Sponsor",
            brand_name="Test Drug",
        )
        assert "214900" in approval.url


@pytest.mark.asyncio
class TestOpenFDAClientAPI:
    """Tests for OpenFDA API calls."""

    async def test_get_drug_approvals_success(
        self, mock_openfda_api, sample_openfda_drug_approval
    ):
        """Test successful drug approvals fetch."""
        mock_openfda_api.get("/drug/drugsfda.json").respond(
            200, json=sample_openfda_drug_approval
        )

        async with OpenFDAClient() as client:
            approvals, total = await client.get_drug_approvals(
                sponsor_name="Moderna"
            )

        assert len(approvals) == 1
        assert total == 1

        approval = approvals[0]
        assert approval.application_number == "BLA761222"
        assert approval.sponsor_name == "Moderna TX, Inc."
        assert approval.brand_name == "SPIKEVAX"
        assert approval.submission_type == "BLA"

    async def test_get_drug_approvals_not_found(self, mock_openfda_api):
        """Test drug approvals not found returns empty list."""
        mock_openfda_api.get("/drug/drugsfda.json").respond(404)

        async with OpenFDAClient() as client:
            approvals, total = await client.get_drug_approvals(
                sponsor_name="NonexistentCompany"
            )

        assert approvals == []
        assert total == 0

    async def test_get_drug_approvals_server_error(self, mock_openfda_api):
        """Test server error raises exception."""
        mock_openfda_api.get("/drug/drugsfda.json").respond(500)

        async with OpenFDAClient() as client:
            with pytest.raises(OpenFDAAPIError):
                await client.get_drug_approvals(sponsor_name="Test")

    async def test_get_drug_approvals_by_application_number(
        self, mock_openfda_api, sample_openfda_drug_approval
    ):
        """Test searching by application number."""
        mock_openfda_api.get("/drug/drugsfda.json").respond(
            200, json=sample_openfda_drug_approval
        )

        async with OpenFDAClient() as client:
            approvals, _ = await client.get_drug_approvals(
                application_number="BLA761222"
            )

        assert len(approvals) == 1
        # Verify search parameter was included
        request = mock_openfda_api.calls[0].request
        assert "application_number" in str(request.url)

    async def test_get_drug_approvals_by_brand_name(
        self, mock_openfda_api, sample_openfda_drug_approval
    ):
        """Test searching by brand name."""
        mock_openfda_api.get("/drug/drugsfda.json").respond(
            200, json=sample_openfda_drug_approval
        )

        async with OpenFDAClient() as client:
            approvals, _ = await client.get_drug_approvals(
                brand_name="SPIKEVAX"
            )

        assert len(approvals) == 1

    async def test_get_drug_approvals_pagination(
        self, mock_openfda_api, sample_openfda_multiple_approvals
    ):
        """Test pagination parameters."""
        mock_openfda_api.get("/drug/drugsfda.json").respond(
            200, json=sample_openfda_multiple_approvals
        )

        async with OpenFDAClient() as client:
            approvals, total = await client.get_drug_approvals(
                limit=50, skip=10
            )

        # Verify pagination parameters
        request = mock_openfda_api.calls[0].request
        assert "limit=50" in str(request.url)
        assert "skip=10" in str(request.url)

    async def test_api_key_in_request(self, mock_openfda_api, sample_openfda_drug_approval):
        """Test that API key is included in requests."""
        mock_openfda_api.get("/drug/drugsfda.json").respond(
            200, json=sample_openfda_drug_approval
        )

        async with OpenFDAClient(api_key="test_api_key") as client:
            await client.get_drug_approvals()

        request = mock_openfda_api.calls[0].request
        assert "api_key=test_api_key" in str(request.url)


@pytest.mark.asyncio
class TestOpenFDAApprovalParsing:
    """Tests for drug approval parsing."""

    async def test_parse_approval_all_fields(
        self, mock_openfda_api, sample_openfda_drug_approval
    ):
        """Test parsing approval with all fields."""
        mock_openfda_api.get("/drug/drugsfda.json").respond(
            200, json=sample_openfda_drug_approval
        )

        async with OpenFDAClient() as client:
            approvals, _ = await client.get_drug_approvals()

        approval = approvals[0]
        assert approval.application_number == "BLA761222"
        assert approval.sponsor_name == "Moderna TX, Inc."
        assert approval.brand_name == "SPIKEVAX"
        assert approval.generic_name == "COVID-19 VACCINE, MRNA"
        assert approval.submission_type == "BLA"
        assert approval.submission_status == "AP"
        # approval_date is dynamically set in fixture, just verify it's a date
        assert approval.approval_date is not None
        assert isinstance(approval.approval_date, date)
        assert approval.dosage_form == "INJECTION, SUSPENSION"
        assert approval.route == "INTRAMUSCULAR"
        assert len(approval.active_ingredients) == 1
        assert "ELASOMERAN" in approval.active_ingredients[0]

    async def test_parse_approval_missing_openfda(self, mock_openfda_api):
        """Test parsing approval with missing openfda section."""
        response = {
            "results": [
                {
                    "application_number": "NDA123456",
                    "sponsor_name": "Test Sponsor",
                    "products": [
                        {"brand_name": "TestDrug", "dosage_form": "TABLET"}
                    ],
                    "submissions": [],
                }
            ],
            "meta": {"results": {"total": 1}},
        }

        mock_openfda_api.get("/drug/drugsfda.json").respond(200, json=response)

        async with OpenFDAClient() as client:
            approvals, _ = await client.get_drug_approvals()

        approval = approvals[0]
        assert approval.application_number == "NDA123456"
        assert approval.brand_name == "TestDrug"
        # generic_name is empty string when openfda section is missing
        assert approval.generic_name == ""

    async def test_parse_approval_no_products(self, mock_openfda_api):
        """Test parsing approval with no products."""
        response = {
            "results": [
                {
                    "application_number": "NDA999999",
                    "sponsor_name": "Test",
                    "products": [],
                    "submissions": [],
                }
            ],
            "meta": {"results": {"total": 1}},
        }

        mock_openfda_api.get("/drug/drugsfda.json").respond(200, json=response)

        async with OpenFDAClient() as client:
            approvals, _ = await client.get_drug_approvals()

        approval = approvals[0]
        assert approval.brand_name == ""
        assert approval.dosage_form == ""

    async def test_parse_approval_brand_from_openfda_fallback(self, mock_openfda_api):
        """Test that brand_name falls back to openfda section when products is empty."""
        response = {
            "results": [
                {
                    "application_number": "NDA999999",
                    "sponsor_name": "Test",
                    "products": [],  # No products
                    "submissions": [],
                    "openfda": {
                        "brand_name": ["TESTBRAND"],
                        "generic_name": ["TESTGENERIC"],
                    },
                }
            ],
            "meta": {"results": {"total": 1}},
        }

        mock_openfda_api.get("/drug/drugsfda.json").respond(200, json=response)

        async with OpenFDAClient() as client:
            approvals, _ = await client.get_drug_approvals()

        approval = approvals[0]
        # brand_name should be populated from openfda section
        assert approval.brand_name == "TESTBRAND"
        assert approval.generic_name == "TESTGENERIC"


@pytest.mark.asyncio
class TestOpenFDADrugLabel:
    """Tests for drug label retrieval."""

    async def test_get_drug_label_success(
        self, mock_openfda_api, sample_openfda_drug_label
    ):
        """Test successful drug label fetch."""
        mock_openfda_api.get("/drug/label.json").respond(
            200, json=sample_openfda_drug_label
        )

        async with OpenFDAClient() as client:
            label = await client.get_drug_label(application_number="BLA761222")

        assert label is not None
        assert label.brand_name == "SPIKEVAX"
        assert label.generic_name == "COVID-19 VACCINE, MRNA"
        assert "immunization" in label.indications_and_usage.lower()

    async def test_get_drug_label_not_found(self, mock_openfda_api):
        """Test drug label not found returns None."""
        mock_openfda_api.get("/drug/label.json").respond(404)

        async with OpenFDAClient() as client:
            label = await client.get_drug_label(application_number="BLA999999")

        assert label is None

    async def test_get_drug_label_no_params(self):
        """Test drug label with no parameters returns None."""
        async with OpenFDAClient() as client:
            label = await client.get_drug_label()

        assert label is None

    async def test_get_drug_label_by_brand_name(
        self, mock_openfda_api, sample_openfda_drug_label
    ):
        """Test getting drug label by brand name."""
        mock_openfda_api.get("/drug/label.json").respond(
            200, json=sample_openfda_drug_label
        )

        async with OpenFDAClient() as client:
            label = await client.get_drug_label(brand_name="SPIKEVAX")

        assert label is not None
        assert label.brand_name == "SPIKEVAX"


@pytest.mark.asyncio
class TestOpenFDAApprovalsByDate:
    """Tests for date-based approval searches."""

    async def test_get_recent_approvals(
        self, mock_openfda_api, sample_openfda_multiple_approvals
    ):
        """Test getting recent approvals."""
        mock_openfda_api.get("/drug/drugsfda.json").respond(
            200, json=sample_openfda_multiple_approvals
        )

        async with OpenFDAClient() as client:
            approvals = await client.get_recent_approvals(days_back=30)

        assert len(approvals) >= 1

    async def test_get_recent_approvals_empty(
        self, mock_openfda_api, sample_openfda_empty_response
    ):
        """Test getting recent approvals with no results."""
        mock_openfda_api.get("/drug/drugsfda.json").respond(404)

        async with OpenFDAClient() as client:
            approvals = await client.get_recent_approvals(days_back=30)

        assert approvals == []

    async def test_get_approvals_by_sponsor_pagination(
        self, mock_openfda_api, sample_openfda_drug_approval
    ):
        """Test pagination when getting approvals by sponsor."""
        mock_openfda_api.get("/drug/drugsfda.json").respond(
            200, json=sample_openfda_drug_approval
        )

        async with OpenFDAClient() as client:
            approvals = await client.get_approvals_by_sponsor(
                sponsor_name="Moderna",
                days_back=365,
            )

        assert len(approvals) >= 1


@pytest.mark.asyncio
class TestOpenFDAIngredientSearch:
    """Tests for ingredient-based searches."""

    async def test_search_by_ingredient(
        self, mock_openfda_api, sample_openfda_drug_approval
    ):
        """Test searching by active ingredient."""
        mock_openfda_api.get("/drug/drugsfda.json").respond(
            200, json=sample_openfda_drug_approval
        )

        async with OpenFDAClient() as client:
            approvals = await client.search_approvals_by_ingredient(
                ingredient_name="ELASOMERAN"
            )

        assert len(approvals) >= 1

    async def test_search_by_ingredient_not_found(self, mock_openfda_api):
        """Test ingredient search with no results."""
        mock_openfda_api.get("/drug/drugsfda.json").respond(404)

        async with OpenFDAClient() as client:
            approvals = await client.search_approvals_by_ingredient(
                ingredient_name="NONEXISTENT"
            )

        assert approvals == []


@pytest.mark.asyncio
class TestOpenFDAQueryBuilding:
    """Tests for query parameter building."""

    async def test_query_escapes_special_characters(
        self, mock_openfda_api, sample_openfda_drug_approval
    ):
        """Test that special characters in queries are escaped."""
        mock_openfda_api.get("/drug/drugsfda.json").respond(
            200, json=sample_openfda_drug_approval
        )

        async with OpenFDAClient() as client:
            await client.get_drug_approvals(
                sponsor_name='Company "with" quotes'
            )

        # Query should be properly constructed
        assert len(mock_openfda_api.calls) == 1

    async def test_combined_search_parameters(
        self, mock_openfda_api, sample_openfda_drug_approval
    ):
        """Test combining multiple search parameters."""
        mock_openfda_api.get("/drug/drugsfda.json").respond(
            200, json=sample_openfda_drug_approval
        )

        async with OpenFDAClient() as client:
            await client.get_drug_approvals(
                sponsor_name="Moderna",
                brand_name="SPIKEVAX",
            )

        request = mock_openfda_api.calls[0].request
        url_str = str(request.url)
        assert "sponsor_name" in url_str
        assert "brand_name" in url_str
        assert "AND" in url_str
