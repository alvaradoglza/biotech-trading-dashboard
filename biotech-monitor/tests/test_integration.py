"""Integration tests for live API verification.

These tests make actual API calls and should only be run manually
or in CI with the RUN_INTEGRATION_TESTS=1 environment variable.

Usage:
    RUN_INTEGRATION_TESTS=1 pytest tests/test_integration.py -v
"""

import os
import pytest
from datetime import date

# Skip all tests in this module unless RUN_INTEGRATION_TESTS is set
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1",
    reason="Integration tests disabled. Set RUN_INTEGRATION_TESTS=1 to enable.",
)


@pytest.mark.asyncio
class TestClinicalTrialsIntegration:
    """Integration tests for ClinicalTrials.gov API."""

    async def test_search_moderna_trials(self):
        """Test searching for Moderna clinical trials."""
        from src.clients.clinicaltrials import ClinicalTrialsClient

        async with ClinicalTrialsClient() as client:
            trials, _, total = await client.search_studies(
                sponsor="Moderna",
                page_size=10,
            )

        assert isinstance(trials, list)
        assert total > 0, "Moderna should have registered trials"

    async def test_get_specific_trial(self):
        """Test fetching a specific trial by NCT ID."""
        from src.clients.clinicaltrials import ClinicalTrialsClient

        async with ClinicalTrialsClient() as client:
            trial = await client.get_study("NCT04470427")

        if trial is not None:
            assert trial.nct_id == "NCT04470427"
            assert trial.sponsor is not None

    async def test_trial_status_values(self):
        """Test that all returned statuses are valid."""
        from src.clients.clinicaltrials import ClinicalTrialsClient, TrialStatus

        async with ClinicalTrialsClient() as client:
            trials, _, _ = await client.search_studies(
                term="cancer",
                page_size=50,
            )

        for trial in trials:
            assert isinstance(trial.status, TrialStatus)


@pytest.mark.asyncio
class TestOpenFDAIntegration:
    """Integration tests for OpenFDA API."""

    async def test_search_drug_approvals(self):
        """Test searching for drug approvals."""
        from src.clients.openfda import OpenFDAClient

        async with OpenFDAClient() as client:
            approvals, total = await client.get_drug_approvals(
                sponsor_name="Moderna",
                limit=10,
            )

        assert isinstance(approvals, list)

    async def test_get_recent_approvals(self):
        """Test fetching recent approvals."""
        from src.clients.openfda import OpenFDAClient

        async with OpenFDAClient() as client:
            approvals = await client.get_recent_approvals(days_back=365)

        assert isinstance(approvals, list)

    async def test_drug_label_fetch(self):
        """Test fetching a drug label."""
        from src.clients.openfda import OpenFDAClient

        async with OpenFDAClient() as client:
            label = await client.get_drug_label(brand_name="ASPIRIN")

        if label is not None:
            assert label.brand_name is not None


@pytest.mark.asyncio
class TestErrorHandling:
    """Integration tests for error handling with live APIs."""

    async def test_clinicaltrials_nonexistent_nct(self):
        """Test handling of non-existent NCT ID."""
        from src.clients.clinicaltrials import ClinicalTrialsClient

        async with ClinicalTrialsClient() as client:
            trial = await client.get_study("NCT99999999")

        assert trial is None

    async def test_openfda_nonexistent_sponsor(self):
        """Test handling of non-existent sponsor."""
        from src.clients.openfda import OpenFDAClient

        async with OpenFDAClient() as client:
            approvals, total = await client.get_drug_approvals(
                sponsor_name="ThisCompanyDoesNotExist12345"
            )

        assert approvals == []
        assert total == 0
