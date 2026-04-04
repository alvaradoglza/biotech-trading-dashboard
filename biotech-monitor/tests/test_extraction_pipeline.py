"""Tests for the extraction pipeline."""

import pytest
from datetime import date
from unittest.mock import MagicMock, AsyncMock
from pathlib import Path
import json

from src.extraction.pipeline import ExtractionPipeline
from src.clients.clinicaltrials import ClinicalTrial, TrialStatus
from src.clients.openfda import DrugApproval, DrugRecall
from src.storage.csv_index import ParseStatus


@pytest.fixture
def temp_index(tmp_path):
    """Create temporary index path."""
    return tmp_path / "test_index.csv"


@pytest.fixture
def pipeline(temp_index):
    """Create extraction pipeline with temp index."""
    return ExtractionPipeline(
        openfda_key=None,
        index_path=str(temp_index),
    )


@pytest.fixture
def sample_clinical_trial():
    """Sample clinical trial."""
    return ClinicalTrial(
        nct_id="NCT04470427",
        title="A Phase 3 Study to Evaluate Efficacy of mRNA-1273",
        sponsor="ModernaTX, Inc.",
        status=TrialStatus.COMPLETED,
        phases=["PHASE3"],
        conditions=["COVID-19"],
        interventions=["mRNA-1273"],
        start_date=date(2020, 7, 27),
        completion_date=date(2023, 6, 30),
        last_update_date=date(2025, 2, 1),
        enrollment=30000,
        has_results=True,
    )


@pytest.fixture
def sample_study_json():
    """Sample study JSON for clinical trial."""
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT04470427",
                "officialTitle": "A Phase 3 Study to Evaluate Efficacy of mRNA-1273",
            },
            "statusModule": {
                "overallStatus": "COMPLETED",
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "ModernaTX, Inc."},
            },
        }
    }


@pytest.fixture
def sample_drug_approval():
    """Sample drug approval."""
    return DrugApproval(
        application_number="BLA761222",
        sponsor_name="ModernaTX, Inc.",
        brand_name="SPIKEVAX",
        generic_name="COVID-19 VACCINE, MRNA",
        approval_date=date(2022, 1, 31),
        submission_type="BLA",
        submission_status="AP",
        dosage_form="INJECTION",
        route="INTRAMUSCULAR",
        active_ingredients=["ELASOMERAN"],
    )


@pytest.fixture
def sample_fda_result():
    """Sample FDA result JSON."""
    return {
        "application_number": "BLA761222",
        "sponsor_name": "ModernaTX, Inc.",
        "products": [{"brand_name": "SPIKEVAX"}],
        "submissions": [{"submission_type": "BLA"}],
    }


@pytest.fixture
def sample_drug_recall():
    """Sample drug recall."""
    return DrugRecall(
        recall_number="D-2025-1234",
        recalling_firm="TestPharma Inc.",
        product_description="Test Drug 50mg",
        reason_for_recall="Contamination",
        classification="Class II",
        status="Ongoing",
        recall_initiation_date=date(2025, 2, 1),
        termination_date=None,
        voluntary_mandated="Voluntary",
        distribution_pattern="Nationwide",
        city="Princeton",
        state="NJ",
        country="US",
    )


@pytest.fixture
def sample_recall_result():
    """Sample recall result JSON."""
    return {
        "recall_number": "D-2025-1234",
        "recalling_firm": "TestPharma Inc.",
        "classification": "Class II",
    }


class TestExtractionPipelineInit:
    """Tests for pipeline initialization."""

    def test_init_creates_pipeline(self, temp_index):
        """Pipeline can be instantiated."""
        pipeline = ExtractionPipeline(
            openfda_key=None,
            index_path=str(temp_index),
        )
        assert pipeline is not None
        assert pipeline.openfda_key is None

    def test_init_with_openfda_key(self, temp_index):
        """Pipeline stores OpenFDA API key."""
        pipeline = ExtractionPipeline(
            openfda_key="test-key-123",
            index_path=str(temp_index),
        )
        assert pipeline.openfda_key == "test-key-123"

    def test_init_creates_extractors(self, temp_index):
        """Pipeline creates all extractors."""
        pipeline = ExtractionPipeline(
            index_path=str(temp_index),
        )
        assert pipeline.ct_extractor is not None
        assert pipeline.fda_extractor is not None

    def test_init_creates_index(self, temp_index):
        """Pipeline creates announcement index."""
        pipeline = ExtractionPipeline(
            index_path=str(temp_index),
        )
        assert pipeline.index is not None
        assert temp_index.exists()


class TestProcessClinicalTrial:
    """Tests for clinical trial processing."""

    def test_process_trial_success(self, pipeline, sample_clinical_trial, sample_study_json, tmp_path):
        """Successfully processes a clinical trial."""
        from src.extraction.clinicaltrials_extractor import CTExtractionResult

        raw_path = tmp_path / "raw.json"
        text_path = tmp_path / "text.txt"
        raw_path.write_text("{}")
        text_path.write_text("test")

        mock_result = CTExtractionResult(success=True, text="test")
        pipeline.ct_extractor.save_extraction = MagicMock(
            return_value=(raw_path, text_path, mock_result)
        )

        result = pipeline.process_clinical_trial(
            sample_clinical_trial, sample_study_json, "MRNA"
        )

        assert result is True
        records = pipeline.index.get_all()
        assert len(records) == 1

    def test_process_trial_skips_duplicate(self, pipeline, sample_clinical_trial, sample_study_json, tmp_path):
        """Skips duplicate trials."""
        from src.extraction.clinicaltrials_extractor import CTExtractionResult

        raw_path = tmp_path / "raw.json"
        text_path = tmp_path / "text.txt"
        raw_path.write_text("{}")
        text_path.write_text("test")

        mock_result = CTExtractionResult(success=True, text="test")
        pipeline.ct_extractor.save_extraction = MagicMock(
            return_value=(raw_path, text_path, mock_result)
        )

        result1 = pipeline.process_clinical_trial(
            sample_clinical_trial, sample_study_json, "MRNA"
        )
        result2 = pipeline.process_clinical_trial(
            sample_clinical_trial, sample_study_json, "MRNA"
        )

        assert result1 is True
        assert result2 is False

    def test_process_trial_creates_correct_record(self, pipeline, sample_clinical_trial, sample_study_json, tmp_path):
        """Creates record with correct fields."""
        from src.extraction.clinicaltrials_extractor import CTExtractionResult

        raw_path = tmp_path / "raw.json"
        text_path = tmp_path / "text.txt"
        raw_path.write_text("{}")
        text_path.write_text("test")

        mock_result = CTExtractionResult(success=True, text="test")
        pipeline.ct_extractor.save_extraction = MagicMock(
            return_value=(raw_path, text_path, mock_result)
        )

        pipeline.process_clinical_trial(
            sample_clinical_trial, sample_study_json, "MRNA"
        )

        records = pipeline.index.get_all()
        assert len(records) == 1
        record = records[0]

        assert record.ticker == "MRNA"
        assert record.source == "clinicaltrials"
        assert record.event_type == "CT_COMPLETED"
        assert record.external_id == "NCT04470427"
        assert "Phase 3" in record.title


class TestProcessFDAApproval:
    """Tests for FDA approval processing."""

    def test_process_approval_success(self, pipeline, sample_drug_approval, sample_fda_result, tmp_path):
        """Successfully processes an FDA approval."""
        from src.extraction.openfda_extractor import FDAExtractionResult

        raw_path = tmp_path / "raw.json"
        text_path = tmp_path / "text.json"
        raw_path.write_text("{}")
        text_path.write_text("{}")

        mock_result = FDAExtractionResult(success=True, json_data={})
        pipeline.fda_extractor.save_extraction = MagicMock(
            return_value=(raw_path, text_path, mock_result)
        )

        result = pipeline.process_fda_approval(
            sample_drug_approval, sample_fda_result, "MRNA"
        )

        assert result is True
        records = pipeline.index.get_all()
        assert len(records) == 1
        assert records[0].source == "openfda"

    def test_process_approval_skips_duplicate(self, pipeline, sample_drug_approval, sample_fda_result, tmp_path):
        """Skips duplicate approvals."""
        from src.extraction.openfda_extractor import FDAExtractionResult

        raw_path = tmp_path / "raw.json"
        text_path = tmp_path / "text.json"
        raw_path.write_text("{}")
        text_path.write_text("{}")

        mock_result = FDAExtractionResult(success=True, json_data={})
        pipeline.fda_extractor.save_extraction = MagicMock(
            return_value=(raw_path, text_path, mock_result)
        )

        result1 = pipeline.process_fda_approval(
            sample_drug_approval, sample_fda_result, "MRNA"
        )
        result2 = pipeline.process_fda_approval(
            sample_drug_approval, sample_fda_result, "MRNA"
        )

        assert result1 is True
        assert result2 is False

    def test_process_approval_creates_correct_record(self, pipeline, sample_drug_approval, sample_fda_result, tmp_path):
        """Creates record with correct fields."""
        from src.extraction.openfda_extractor import FDAExtractionResult

        raw_path = tmp_path / "raw.json"
        text_path = tmp_path / "text.json"
        raw_path.write_text("{}")
        text_path.write_text("{}")

        mock_result = FDAExtractionResult(success=True, json_data={})
        pipeline.fda_extractor.save_extraction = MagicMock(
            return_value=(raw_path, text_path, mock_result)
        )

        pipeline.process_fda_approval(
            sample_drug_approval, sample_fda_result, "MRNA"
        )

        records = pipeline.index.get_all()
        assert len(records) == 1
        record = records[0]

        assert record.ticker == "MRNA"
        assert record.source == "openfda"
        assert record.event_type == "FDA_BLA"
        assert record.external_id == "BLA761222"
        assert "SPIKEVAX" in record.title

    def test_process_approval_uses_today_if_no_date(self, pipeline, sample_fda_result, tmp_path):
        """Uses today's date if approval_date is None."""
        from src.extraction.openfda_extractor import FDAExtractionResult

        raw_path = tmp_path / "raw.json"
        text_path = tmp_path / "text.json"
        raw_path.write_text("{}")
        text_path.write_text("{}")

        mock_result = FDAExtractionResult(success=True, json_data={})
        pipeline.fda_extractor.save_extraction = MagicMock(
            return_value=(raw_path, text_path, mock_result)
        )

        approval_no_date = DrugApproval(
            application_number="NDA000001",
            sponsor_name="TestCo",
            brand_name="TestDrug",
            generic_name=None,
            approval_date=None,
            submission_type="NDA",
            submission_status="AP",
            dosage_form="TABLET",
            route="ORAL",
            active_ingredients=[],
        )

        result = pipeline.process_fda_approval(
            approval_no_date, sample_fda_result, "TEST"
        )
        assert result is True


class TestProcessFDARecall:
    """Tests for FDA recall processing."""

    def test_process_recall_success(self, pipeline, sample_drug_recall, sample_recall_result, tmp_path):
        """Successfully processes an FDA recall."""
        from src.extraction.openfda_extractor import FDAExtractionResult

        raw_path = tmp_path / "raw.json"
        text_path = tmp_path / "text.json"
        raw_path.write_text("{}")
        text_path.write_text("{}")

        mock_result = FDAExtractionResult(success=True, json_data={})
        pipeline.fda_extractor.save_recall_extraction = MagicMock(
            return_value=(raw_path, text_path, mock_result)
        )

        result = pipeline.process_fda_recall(
            sample_drug_recall, sample_recall_result, "TEST"
        )

        assert result is True
        records = pipeline.index.get_all()
        assert len(records) == 1

    def test_process_recall_skips_duplicate(self, pipeline, sample_drug_recall, sample_recall_result, tmp_path):
        """Skips duplicate recalls."""
        from src.extraction.openfda_extractor import FDAExtractionResult

        raw_path = tmp_path / "raw.json"
        text_path = tmp_path / "text.json"
        raw_path.write_text("{}")
        text_path.write_text("{}")

        mock_result = FDAExtractionResult(success=True, json_data={})
        pipeline.fda_extractor.save_recall_extraction = MagicMock(
            return_value=(raw_path, text_path, mock_result)
        )

        result1 = pipeline.process_fda_recall(
            sample_drug_recall, sample_recall_result, "TEST"
        )
        result2 = pipeline.process_fda_recall(
            sample_drug_recall, sample_recall_result, "TEST"
        )

        assert result1 is True
        assert result2 is False

    def test_process_recall_creates_correct_record(self, pipeline, sample_drug_recall, sample_recall_result, tmp_path):
        """Creates record with correct fields."""
        from src.extraction.openfda_extractor import FDAExtractionResult

        raw_path = tmp_path / "raw.json"
        text_path = tmp_path / "text.json"
        raw_path.write_text("{}")
        text_path.write_text("{}")

        mock_result = FDAExtractionResult(success=True, json_data={})
        pipeline.fda_extractor.save_recall_extraction = MagicMock(
            return_value=(raw_path, text_path, mock_result)
        )

        pipeline.process_fda_recall(
            sample_drug_recall, sample_recall_result, "TEST"
        )

        records = pipeline.index.get_all()
        assert len(records) == 1
        record = records[0]

        assert record.ticker == "TEST"
        assert record.source == "openfda"
        assert record.event_type == "FDA_RECALL"
        assert record.external_id == "D-2025-1234"
        assert "TestPharma" in record.title
        assert "Class II" in record.title


class TestGetStats:
    """Tests for statistics methods."""

    def test_get_stats_empty(self, pipeline):
        """Gets stats for empty index."""
        stats = pipeline.get_stats()

        assert stats["total"] == 0
        assert stats["by_source"] == {}
        assert stats["by_status"].get("OK", 0) == 0
        assert stats["by_status"].get("FAILED", 0) == 0
        assert stats["by_ticker"] == {}

    def test_get_stats_with_records(self, pipeline, sample_clinical_trial, sample_study_json, tmp_path):
        """Gets stats with records."""
        from src.extraction.clinicaltrials_extractor import CTExtractionResult

        raw_path = tmp_path / "raw.json"
        text_path = tmp_path / "text.txt"
        raw_path.write_text("{}")
        text_path.write_text("test")

        mock_result = CTExtractionResult(success=True, text="test")
        pipeline.ct_extractor.save_extraction = MagicMock(
            return_value=(raw_path, text_path, mock_result)
        )

        pipeline.process_clinical_trial(
            sample_clinical_trial, sample_study_json, "MRNA"
        )

        stats = pipeline.get_stats()

        assert stats["total"] == 1
        assert stats["by_source"]["clinicaltrials"] == 1
        assert stats["by_ticker"]["MRNA"] == 1

    def test_get_pending_count_empty(self, pipeline):
        """Gets pending count for empty index."""
        assert pipeline.get_pending_count() == 0

    def test_get_failed_count_empty(self, pipeline):
        """Gets failed count for empty index."""
        assert pipeline.get_failed_count() == 0


class TestRetryFailed:
    """Tests for retry functionality."""

    @pytest.mark.asyncio
    async def test_retry_failed_no_failures(self, pipeline):
        """Retry returns empty when no failures."""
        results = await pipeline.retry_failed()

        assert results["retried"] == 0
        assert results["succeeded"] == 0
        assert results["failed"] == 0

    @pytest.mark.asyncio
    async def test_retry_failed_placeholder(self, pipeline, sample_clinical_trial, sample_study_json, tmp_path):
        """Retry is a placeholder that tracks failed records."""
        from src.extraction.clinicaltrials_extractor import CTExtractionResult

        raw_path = tmp_path / "raw.json"
        text_path = tmp_path / "text.txt"
        raw_path.write_text("{}")
        text_path.write_text("test")

        mock_result = CTExtractionResult(success=False, text="", error="Extraction failed")
        pipeline.ct_extractor.save_extraction = MagicMock(
            return_value=(raw_path, text_path, mock_result)
        )

        pipeline.process_clinical_trial(sample_clinical_trial, sample_study_json, "MRNA")

        results = await pipeline.retry_failed()

        assert results["failed"] == 1
