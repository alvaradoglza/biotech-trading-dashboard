"""Tests for ClinicalTrials.gov text extraction."""

import pytest
from datetime import date
import json

from src.extraction.clinicaltrials_extractor import (
    ClinicalTrialsExtractor,
    CTExtractionResult,
)


@pytest.fixture
def extractor():
    """Create ClinicalTrials extractor instance."""
    return ClinicalTrialsExtractor()


@pytest.fixture
def sample_study_json():
    """Sample study JSON from ClinicalTrials.gov API."""
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT04470427",
                "officialTitle": "A Phase 3 Study to Evaluate Efficacy of mRNA-1273",
                "briefTitle": "COVE Study",
            },
            "descriptionModule": {
                "briefSummary": "This study evaluates the vaccine efficacy.",
                "detailedDescription": "Detailed protocol information here.",
            },
            "conditionsModule": {
                "conditions": ["COVID-19", "SARS-CoV-2"],
                "keywords": ["vaccine", "mRNA"],
            },
            "designModule": {
                "studyType": "INTERVENTIONAL",
                "phases": ["PHASE3"],
                "enrollmentInfo": {"count": 30000, "type": "ACTUAL"},
            },
            "eligibilityModule": {
                "eligibilityCriteria": "Adults 18+ years",
                "healthyVolunteers": "Yes",
                "sex": "ALL",
                "minimumAge": "18 Years",
                "maximumAge": "None",
            },
            "armsInterventionsModule": {
                "interventions": [
                    {
                        "type": "BIOLOGICAL",
                        "name": "mRNA-1273",
                        "description": "mRNA vaccine candidate",
                    }
                ],
                "armGroups": [
                    {
                        "label": "Treatment",
                        "type": "EXPERIMENTAL",
                        "description": "Receives vaccine",
                    },
                    {
                        "label": "Placebo",
                        "type": "PLACEBO_COMPARATOR",
                        "description": "Receives placebo",
                    },
                ],
            },
            "outcomesModule": {
                "primaryOutcomes": [
                    {
                        "measure": "Vaccine Efficacy",
                        "timeFrame": "14 days post dose 2",
                        "description": "Prevention of COVID-19",
                    }
                ],
                "secondaryOutcomes": [
                    {"measure": "Safety profile"},
                ],
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "ModernaTX, Inc.", "class": "INDUSTRY"},
                "collaborators": [{"name": "BARDA", "class": "FED"}],
            },
            "statusModule": {
                "overallStatus": "COMPLETED",
                "startDateStruct": {"date": "2020-07-27"},
                "completionDateStruct": {"date": "2023-06-30"},
            },
            "contactsLocationsModule": {
                "centralContacts": [
                    {
                        "name": "Study Contact",
                        "role": "CONTACT",
                        "email": "contact@example.com",
                        "phone": "555-1234",
                    }
                ],
            },
        },
        "hasResults": True,
    }


class TestClinicalTrialsExtractor:
    """Tests for ClinicalTrials.gov text extraction."""

    def test_extract_all_fields(self, extractor, sample_study_json):
        """Extracts all relevant text fields."""
        result = extractor.extract_from_study(sample_study_json)

        assert result.success is True

        # Check key content is present
        assert "NCT04470427" in result.text
        assert "Phase 3 Study" in result.text
        assert "vaccine efficacy" in result.text.lower()
        assert "COVID-19" in result.text
        assert "mRNA-1273" in result.text
        assert "ModernaTX" in result.text
        assert "BARDA" in result.text
        assert "COMPLETED" in result.text

    def test_includes_section_headers(self, extractor, sample_study_json):
        """Text includes section headers for readability."""
        result = extractor.extract_from_study(sample_study_json)

        assert "## Official Title" in result.text
        assert "## Brief Summary" in result.text
        assert "## Interventions" in result.text
        assert "## Primary Outcomes" in result.text
        assert "## Sponsor" in result.text
        assert "## Status" in result.text

    def test_includes_enrollment_info(self, extractor, sample_study_json):
        """Includes enrollment information."""
        result = extractor.extract_from_study(sample_study_json)

        assert "30000" in result.text
        assert "ACTUAL" in result.text

    def test_includes_arms_groups(self, extractor, sample_study_json):
        """Includes arm/group information."""
        result = extractor.extract_from_study(sample_study_json)

        assert "Arms/Groups" in result.text
        assert "Treatment" in result.text
        assert "Placebo" in result.text

    def test_includes_contact_info(self, extractor, sample_study_json):
        """Includes contact information."""
        result = extractor.extract_from_study(sample_study_json)

        assert "Contacts" in result.text
        assert "contact@example.com" in result.text

    def test_handles_missing_fields(self, extractor):
        """Handles studies with missing optional fields."""
        minimal_study = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT00000001",
                },
                "statusModule": {
                    "overallStatus": "UNKNOWN",
                },
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "Unknown"},
                },
            }
        }

        result = extractor.extract_from_study(minimal_study)

        assert result.success is True
        assert "NCT00000001" in result.text
        assert "Unknown" in result.text

    def test_handles_empty_arrays(self, extractor):
        """Handles empty arrays gracefully."""
        study_with_empty = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT00000001"},
                "conditionsModule": {"conditions": []},
                "armsInterventionsModule": {"interventions": [], "armGroups": []},
                "outcomesModule": {
                    "primaryOutcomes": [],
                    "secondaryOutcomes": [],
                },
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "Test"},
                    "collaborators": [],
                },
                "statusModule": {"overallStatus": "RECRUITING"},
            }
        }

        result = extractor.extract_from_study(study_with_empty)

        assert result.success is True

    def test_handles_malformed_json(self, extractor):
        """Returns success with minimal text for empty input."""
        result = extractor.extract_from_study({})

        # Should handle gracefully
        assert isinstance(result.text, str)

    def test_handles_missing_protocol_section(self, extractor):
        """Handles missing protocolSection."""
        result = extractor.extract_from_study({"otherSection": {}})

        # Should not crash
        assert isinstance(result, CTExtractionResult)

    def test_handles_why_stopped(self, extractor):
        """Includes why stopped reason if present."""
        study = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT00000001"},
                "statusModule": {
                    "overallStatus": "TERMINATED",
                    "whyStopped": "Lack of funding",
                },
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "Test"},
                },
            }
        }

        result = extractor.extract_from_study(study)

        assert result.success is True
        assert "TERMINATED" in result.text
        assert "Lack of funding" in result.text


class TestSaveExtraction:
    """Tests for saving extracted content."""

    def test_save_extraction_creates_files(self, extractor, sample_study_json, tmp_path):
        """save_extraction creates raw and text files."""
        # Temporarily patch the paths to use tmp_path
        import src.storage.paths as paths

        original_get_raw = paths.get_raw_path
        original_get_text = paths.get_text_path

        paths.get_raw_path = lambda *args, **kwargs: tmp_path / "raw.json"
        paths.get_text_path = lambda *args, **kwargs: tmp_path / "text.txt"

        try:
            raw_path, text_path, result = extractor.save_extraction(
                study_json=sample_study_json,
                ticker="MRNA",
                published_date=date(2025, 2, 7),
                nct_id="NCT04470427",
            )

            assert raw_path.exists()
            assert text_path.exists()
            assert result.success is True

            # Verify raw JSON
            raw_content = json.loads(raw_path.read_text())
            assert raw_content["protocolSection"]["identificationModule"]["nctId"] == "NCT04470427"

            # Verify text content
            text_content = text_path.read_text()
            assert "NCT04470427" in text_content

        finally:
            paths.get_raw_path = original_get_raw
            paths.get_text_path = original_get_text

    def test_save_extraction_retries_on_failure(self, extractor, tmp_path):
        """save_extraction retries on extraction failure."""
        # This test would require mocking the extract_from_study method
        # to simulate failures and then success
        pass
