"""Tests for OpenFDA data extraction/preservation."""

import pytest
from datetime import date
import json

from src.extraction.openfda_extractor import (
    OpenFDAExtractor,
    FDAExtractionResult,
)


@pytest.fixture
def extractor():
    """Create OpenFDA extractor instance."""
    return OpenFDAExtractor()


@pytest.fixture
def sample_drug_result():
    """Sample drug approval result from OpenFDA API."""
    return {
        "application_number": "BLA761222",
        "sponsor_name": "ModernaTX, Inc.",
        "products": [
            {
                "brand_name": "SPIKEVAX",
                "active_ingredients": [
                    {"name": "ELASOMERAN", "strength": "100 MCG/0.5ML"}
                ],
                "dosage_form": "INJECTION, SUSPENSION",
                "route": "INTRAMUSCULAR",
                "marketing_status": "Prescription",
            }
        ],
        "submissions": [
            {
                "submission_type": "BLA",
                "submission_number": "761222",
                "submission_status": "AP",
                "submission_status_date": "20220131",
                "submission_class_code": "TYPE 1",
                "submission_class_code_description": "New Molecular Entity",
            },
            {
                "submission_type": "SUPPL",
                "submission_number": "1",
                "submission_status": "AP",
                "submission_status_date": "20220401",
                "submission_class_code": "LABELING",
                "submission_class_code_description": "Labeling Revision",
            },
        ],
        "openfda": {
            "brand_name": ["SPIKEVAX"],
            "generic_name": ["COVID-19 VACCINE, MRNA"],
            "manufacturer_name": ["Moderna US, Inc."],
            "nui": ["N0000015436"],
            "package_ndc": ["80777-0273-10"],
        },
    }


@pytest.fixture
def sample_recall_result():
    """Sample recall result from OpenFDA API."""
    return {
        "recall_number": "D-2025-1234",
        "recalling_firm": "TestPharma Inc.",
        "product_description": "Test Drug 50mg Tablets",
        "reason_for_recall": "Potential contamination with particulate matter",
        "classification": "Class II",
        "status": "Ongoing",
        "recall_initiation_date": "20250201",
        "termination_date": None,
        "voluntary_mandated": "Voluntary: Firm Initiated",
        "distribution_pattern": "Nationwide",
        "product_quantity": "50,000 bottles",
        "openfda": {
            "brand_name": ["TEST DRUG"],
            "generic_name": ["TESTIUM"],
        },
    }


class TestOpenFDAExtractor:
    """Tests for OpenFDA extractor initialization."""

    def test_init_creates_extractor(self):
        """Extractor can be instantiated."""
        extractor = OpenFDAExtractor()
        assert extractor is not None


class TestSaveExtraction:
    """Tests for drug approval extraction."""

    def test_save_extraction_creates_files(self, extractor, sample_drug_result, tmp_path):
        """save_extraction creates raw and text JSON files."""
        import src.storage.paths as paths

        original_get_raw = paths.get_raw_path
        original_get_text = paths.get_text_path

        paths.get_raw_path = lambda *args, **kwargs: tmp_path / "raw.json"
        paths.get_text_path = lambda *args, **kwargs: tmp_path / "text.json"

        try:
            raw_path, text_path, result = extractor.save_extraction(
                fda_result=sample_drug_result,
                ticker="MRNA",
                published_date=date(2022, 1, 31),
                application_number="BLA761222",
            )

            assert raw_path.exists()
            assert text_path.exists()
            assert result.success is True

            # Verify raw JSON
            raw_content = json.loads(raw_path.read_text())
            assert raw_content["application_number"] == "BLA761222"
            assert raw_content["sponsor_name"] == "ModernaTX, Inc."

            # Verify extracted JSON
            text_content = json.loads(text_path.read_text())
            assert text_content["application_number"] == "BLA761222"
            assert text_content["sponsor_name"] == "ModernaTX, Inc."

        finally:
            paths.get_raw_path = original_get_raw
            paths.get_text_path = original_get_text

    def test_save_extraction_extracts_products(self, extractor, sample_drug_result, tmp_path):
        """save_extraction extracts product information."""
        import src.storage.paths as paths

        original_get_raw = paths.get_raw_path
        original_get_text = paths.get_text_path

        paths.get_raw_path = lambda *args, **kwargs: tmp_path / "raw.json"
        paths.get_text_path = lambda *args, **kwargs: tmp_path / "text.json"

        try:
            raw_path, text_path, result = extractor.save_extraction(
                fda_result=sample_drug_result,
                ticker="MRNA",
                published_date=date(2022, 1, 31),
                application_number="BLA761222",
            )

            text_content = json.loads(text_path.read_text())

            assert len(text_content["products"]) == 1
            product = text_content["products"][0]
            assert product["brand_name"] == "SPIKEVAX"
            assert product["dosage_form"] == "INJECTION, SUSPENSION"
            assert product["route"] == "INTRAMUSCULAR"

        finally:
            paths.get_raw_path = original_get_raw
            paths.get_text_path = original_get_text

    def test_save_extraction_extracts_submissions(self, extractor, sample_drug_result, tmp_path):
        """save_extraction extracts submission history."""
        import src.storage.paths as paths

        original_get_raw = paths.get_raw_path
        original_get_text = paths.get_text_path

        paths.get_raw_path = lambda *args, **kwargs: tmp_path / "raw.json"
        paths.get_text_path = lambda *args, **kwargs: tmp_path / "text.json"

        try:
            raw_path, text_path, result = extractor.save_extraction(
                fda_result=sample_drug_result,
                ticker="MRNA",
                published_date=date(2022, 1, 31),
                application_number="BLA761222",
            )

            text_content = json.loads(text_path.read_text())

            assert len(text_content["submissions"]) == 2
            assert text_content["submissions"][0]["submission_type"] == "BLA"
            assert text_content["submissions"][0]["submission_status"] == "AP"
            assert text_content["submissions"][1]["submission_type"] == "SUPPL"

        finally:
            paths.get_raw_path = original_get_raw
            paths.get_text_path = original_get_text

    def test_save_extraction_preserves_openfda(self, extractor, sample_drug_result, tmp_path):
        """save_extraction preserves openfda metadata."""
        import src.storage.paths as paths

        original_get_raw = paths.get_raw_path
        original_get_text = paths.get_text_path

        paths.get_raw_path = lambda *args, **kwargs: tmp_path / "raw.json"
        paths.get_text_path = lambda *args, **kwargs: tmp_path / "text.json"

        try:
            raw_path, text_path, result = extractor.save_extraction(
                fda_result=sample_drug_result,
                ticker="MRNA",
                published_date=date(2022, 1, 31),
                application_number="BLA761222",
            )

            text_content = json.loads(text_path.read_text())

            assert "openfda" in text_content
            assert text_content["openfda"]["brand_name"] == ["SPIKEVAX"]

        finally:
            paths.get_raw_path = original_get_raw
            paths.get_text_path = original_get_text

    def test_save_extraction_handles_missing_fields(self, extractor, tmp_path):
        """Handles results with missing optional fields."""
        import src.storage.paths as paths

        original_get_raw = paths.get_raw_path
        original_get_text = paths.get_text_path

        paths.get_raw_path = lambda *args, **kwargs: tmp_path / "raw.json"
        paths.get_text_path = lambda *args, **kwargs: tmp_path / "text.json"

        minimal_result = {
            "application_number": "NDA000001",
        }

        try:
            raw_path, text_path, result = extractor.save_extraction(
                fda_result=minimal_result,
                ticker="TEST",
                published_date=date(2025, 2, 7),
                application_number="NDA000001",
            )

            assert result.success is True
            text_content = json.loads(text_path.read_text())
            assert text_content["application_number"] == "NDA000001"
            assert text_content["products"] == []
            assert text_content["submissions"] == []

        finally:
            paths.get_raw_path = original_get_raw
            paths.get_text_path = original_get_text

    def test_save_extraction_handles_empty_arrays(self, extractor, tmp_path):
        """Handles results with empty arrays."""
        import src.storage.paths as paths

        original_get_raw = paths.get_raw_path
        original_get_text = paths.get_text_path

        paths.get_raw_path = lambda *args, **kwargs: tmp_path / "raw.json"
        paths.get_text_path = lambda *args, **kwargs: tmp_path / "text.json"

        result_with_empty = {
            "application_number": "NDA000001",
            "products": [],
            "submissions": [],
            "openfda": {},
        }

        try:
            raw_path, text_path, result = extractor.save_extraction(
                fda_result=result_with_empty,
                ticker="TEST",
                published_date=date(2025, 2, 7),
                application_number="NDA000001",
            )

            assert result.success is True

        finally:
            paths.get_raw_path = original_get_raw
            paths.get_text_path = original_get_text


class TestSaveRecallExtraction:
    """Tests for recall extraction."""

    def test_save_recall_creates_files(self, extractor, sample_recall_result, tmp_path):
        """save_recall_extraction creates raw and text files."""
        import src.storage.paths as paths

        original_get_raw = paths.get_raw_path
        original_get_text = paths.get_text_path

        paths.get_raw_path = lambda *args, **kwargs: tmp_path / "raw.json"
        paths.get_text_path = lambda *args, **kwargs: tmp_path / "text.json"

        try:
            raw_path, text_path, result = extractor.save_recall_extraction(
                recall_result=sample_recall_result,
                ticker="TEST",
                published_date=date(2025, 2, 1),
                recall_number="D-2025-1234",
            )

            assert raw_path.exists()
            assert text_path.exists()
            assert result.success is True

            # Verify raw JSON
            raw_content = json.loads(raw_path.read_text())
            assert raw_content["recall_number"] == "D-2025-1234"

            # Verify extracted JSON
            text_content = json.loads(text_path.read_text())
            assert text_content["recall_number"] == "D-2025-1234"
            assert text_content["recalling_firm"] == "TestPharma Inc."

        finally:
            paths.get_raw_path = original_get_raw
            paths.get_text_path = original_get_text

    def test_save_recall_extracts_all_fields(self, extractor, sample_recall_result, tmp_path):
        """save_recall_extraction extracts all recall fields."""
        import src.storage.paths as paths

        original_get_raw = paths.get_raw_path
        original_get_text = paths.get_text_path

        paths.get_raw_path = lambda *args, **kwargs: tmp_path / "raw.json"
        paths.get_text_path = lambda *args, **kwargs: tmp_path / "text.json"

        try:
            raw_path, text_path, result = extractor.save_recall_extraction(
                recall_result=sample_recall_result,
                ticker="TEST",
                published_date=date(2025, 2, 1),
                recall_number="D-2025-1234",
            )

            text_content = json.loads(text_path.read_text())

            assert text_content["classification"] == "Class II"
            assert text_content["status"] == "Ongoing"
            assert text_content["reason_for_recall"] == "Potential contamination with particulate matter"
            assert text_content["distribution_pattern"] == "Nationwide"
            assert text_content["product_quantity"] == "50,000 bottles"

        finally:
            paths.get_raw_path = original_get_raw
            paths.get_text_path = original_get_text

    def test_save_recall_handles_missing_fields(self, extractor, tmp_path):
        """Handles recalls with missing optional fields."""
        import src.storage.paths as paths

        original_get_raw = paths.get_raw_path
        original_get_text = paths.get_text_path

        paths.get_raw_path = lambda *args, **kwargs: tmp_path / "raw.json"
        paths.get_text_path = lambda *args, **kwargs: tmp_path / "text.json"

        minimal_recall = {
            "recall_number": "D-2025-0001",
            "recalling_firm": "TestCo",
        }

        try:
            raw_path, text_path, result = extractor.save_recall_extraction(
                recall_result=minimal_recall,
                ticker="TEST",
                published_date=date(2025, 2, 7),
                recall_number="D-2025-0001",
            )

            assert result.success is True
            text_content = json.loads(text_path.read_text())
            assert text_content["recall_number"] == "D-2025-0001"
            assert text_content["recalling_firm"] == "TestCo"

        finally:
            paths.get_raw_path = original_get_raw
            paths.get_text_path = original_get_text


class TestExtractKeyFields:
    """Tests for key field extraction."""

    def test_extracts_application_number(self, extractor, sample_drug_result):
        """Extracts application number."""
        extracted = extractor._extract_key_fields(sample_drug_result)
        assert extracted["application_number"] == "BLA761222"

    def test_extracts_sponsor_name(self, extractor, sample_drug_result):
        """Extracts sponsor name."""
        extracted = extractor._extract_key_fields(sample_drug_result)
        assert extracted["sponsor_name"] == "ModernaTX, Inc."

    def test_extracts_products_list(self, extractor, sample_drug_result):
        """Extracts products as list."""
        extracted = extractor._extract_key_fields(sample_drug_result)
        assert isinstance(extracted["products"], list)
        assert len(extracted["products"]) == 1

    def test_extracts_product_details(self, extractor, sample_drug_result):
        """Extracts product details."""
        extracted = extractor._extract_key_fields(sample_drug_result)
        product = extracted["products"][0]

        assert product["brand_name"] == "SPIKEVAX"
        assert product["dosage_form"] == "INJECTION, SUSPENSION"
        assert product["route"] == "INTRAMUSCULAR"
        assert product["marketing_status"] == "Prescription"
        assert product["active_ingredients"] == [
            {"name": "ELASOMERAN", "strength": "100 MCG/0.5ML"}
        ]

    def test_extracts_submissions_list(self, extractor, sample_drug_result):
        """Extracts submissions as list."""
        extracted = extractor._extract_key_fields(sample_drug_result)
        assert isinstance(extracted["submissions"], list)
        assert len(extracted["submissions"]) == 2

    def test_extracts_submission_details(self, extractor, sample_drug_result):
        """Extracts submission details."""
        extracted = extractor._extract_key_fields(sample_drug_result)
        submission = extracted["submissions"][0]

        assert submission["submission_type"] == "BLA"
        assert submission["submission_number"] == "761222"
        assert submission["submission_status"] == "AP"
        assert submission["submission_status_date"] == "20220131"
        assert submission["submission_class_code"] == "TYPE 1"

    def test_handles_empty_result(self, extractor):
        """Handles empty result gracefully."""
        extracted = extractor._extract_key_fields({})

        assert extracted["application_number"] is None
        assert extracted["sponsor_name"] is None
        assert extracted["products"] == []
        assert extracted["submissions"] == []
        assert extracted["openfda"] == {}


class TestExtractRecallFields:
    """Tests for recall field extraction."""

    def test_extracts_recall_number(self, extractor, sample_recall_result):
        """Extracts recall number."""
        extracted = extractor._extract_recall_fields(sample_recall_result)
        assert extracted["recall_number"] == "D-2025-1234"

    def test_extracts_recalling_firm(self, extractor, sample_recall_result):
        """Extracts recalling firm."""
        extracted = extractor._extract_recall_fields(sample_recall_result)
        assert extracted["recalling_firm"] == "TestPharma Inc."

    def test_extracts_classification(self, extractor, sample_recall_result):
        """Extracts recall classification."""
        extracted = extractor._extract_recall_fields(sample_recall_result)
        assert extracted["classification"] == "Class II"

    def test_extracts_reason(self, extractor, sample_recall_result):
        """Extracts reason for recall."""
        extracted = extractor._extract_recall_fields(sample_recall_result)
        assert "contamination" in extracted["reason_for_recall"]

    def test_extracts_dates(self, extractor, sample_recall_result):
        """Extracts recall dates."""
        extracted = extractor._extract_recall_fields(sample_recall_result)
        assert extracted["recall_initiation_date"] == "20250201"
        assert extracted["termination_date"] is None

    def test_extracts_distribution(self, extractor, sample_recall_result):
        """Extracts distribution pattern."""
        extracted = extractor._extract_recall_fields(sample_recall_result)
        assert extracted["distribution_pattern"] == "Nationwide"
        assert extracted["product_quantity"] == "50,000 bottles"

    def test_handles_empty_recall(self, extractor):
        """Handles empty recall gracefully."""
        extracted = extractor._extract_recall_fields({})

        assert extracted["recall_number"] is None
        assert extracted["recalling_firm"] is None
        assert extracted["classification"] is None
        assert extracted["openfda"] == {}


class TestFDAExtractionResult:
    """Tests for FDAExtractionResult dataclass."""

    def test_success_result(self):
        """Creates successful result."""
        result = FDAExtractionResult(
            success=True,
            json_data={"key": "value"},
        )
        assert result.success is True
        assert result.json_data == {"key": "value"}
        assert result.error is None

    def test_failed_result(self):
        """Creates failed result with error."""
        result = FDAExtractionResult(
            success=False,
            json_data={},
            error="Test error message",
        )
        assert result.success is False
        assert result.json_data == {}
        assert result.error == "Test error message"
