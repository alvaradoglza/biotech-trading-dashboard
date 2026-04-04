"""ClinicalTrials.gov text extraction.

This module handles extracting text from ClinicalTrials.gov study data,
concatenating all relevant fields into a single document.

Example:
    >>> extractor = ClinicalTrialsExtractor()
    >>> raw_path, text_path, result = extractor.save_extraction(
    ...     study_json=study_data,
    ...     ticker="MRNA",
    ...     published_date=date(2025, 2, 7),
    ...     nct_id="NCT04470427",
    ... )
"""

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

from src.storage.paths import generate_id, get_raw_path, get_text_path, ensure_parent_dirs
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CTExtractionResult:
    """Result of ClinicalTrials.gov text extraction.

    Attributes:
        success: Whether extraction succeeded
        text: Extracted text content
        error: Error message if extraction failed
    """

    success: bool
    text: str
    error: Optional[str] = None


class ClinicalTrialsExtractor:
    """Extract text from ClinicalTrials.gov study data.

    Concatenates all relevant text fields into a single document with
    section headers for readability.

    Fields extracted:
    - Official title
    - Brief summary
    - Detailed description
    - Conditions
    - Interventions (name + description)
    - Eligibility criteria
    - Primary/secondary outcomes
    - Study design
    - Sponsor/collaborators
    - Status information

    Example:
        >>> extractor = ClinicalTrialsExtractor()
        >>> result = extractor.extract_from_study(study_json)
        >>> print(result.text)
    """

    # Fields to extract (module, field, label)
    TEXT_FIELDS = [
        ("identificationModule", "officialTitle", "Official Title"),
        ("identificationModule", "briefTitle", "Brief Title"),
        ("descriptionModule", "briefSummary", "Brief Summary"),
        ("descriptionModule", "detailedDescription", "Detailed Description"),
        ("conditionsModule", "conditions", "Conditions"),
        ("conditionsModule", "keywords", "Keywords"),
        ("designModule", "studyType", "Study Type"),
        ("designModule", "phases", "Phases"),
        ("eligibilityModule", "eligibilityCriteria", "Eligibility Criteria"),
        ("eligibilityModule", "healthyVolunteers", "Healthy Volunteers"),
        ("eligibilityModule", "sex", "Sex"),
        ("eligibilityModule", "minimumAge", "Minimum Age"),
        ("eligibilityModule", "maximumAge", "Maximum Age"),
    ]

    def extract_from_study(self, study_json: dict) -> CTExtractionResult:
        """Extract all text from a study JSON response.

        Args:
            study_json: Full study object from ClinicalTrials.gov API

        Returns:
            CTExtractionResult with concatenated text
        """
        try:
            protocol = study_json.get("protocolSection", {})
            text_parts = []

            # Extract NCT ID first
            nct_id = protocol.get("identificationModule", {}).get("nctId", "Unknown")
            text_parts.append(f"=== Clinical Trial: {nct_id} ===\n")

            # Extract standard fields
            for module_name, field_name, label in self.TEXT_FIELDS:
                module = protocol.get(module_name, {})
                value = module.get(field_name)
                if value:
                    text_parts.append(f"\n## {label}\n")
                    if isinstance(value, list):
                        text_parts.append(", ".join(str(v) for v in value))
                    else:
                        text_parts.append(str(value))

            # Extract interventions (nested)
            interventions = protocol.get("armsInterventionsModule", {}).get(
                "interventions", []
            )
            if interventions:
                text_parts.append("\n\n## Interventions\n")
                for interv in interventions:
                    name = interv.get("name", "Unknown")
                    itype = interv.get("type", "")
                    desc = interv.get("description", "")
                    text_parts.append(f"- {name} ({itype}): {desc}\n")

            # Extract arms/groups
            arms = protocol.get("armsInterventionsModule", {}).get("armGroups", [])
            if arms:
                text_parts.append("\n\n## Arms/Groups\n")
                for arm in arms:
                    label = arm.get("label", "Unknown")
                    arm_type = arm.get("type", "")
                    desc = arm.get("description", "")
                    text_parts.append(f"- {label} ({arm_type}): {desc}\n")

            # Extract outcomes
            outcomes_module = protocol.get("outcomesModule", {})
            primary = outcomes_module.get("primaryOutcomes", [])
            if primary:
                text_parts.append("\n\n## Primary Outcomes\n")
                for outcome in primary:
                    measure = outcome.get("measure", "")
                    timeframe = outcome.get("timeFrame", "")
                    desc = outcome.get("description", "")
                    text_parts.append(f"- {measure}")
                    if timeframe:
                        text_parts.append(f" (Timeframe: {timeframe})")
                    if desc:
                        text_parts.append(f"\n  {desc}")
                    text_parts.append("\n")

            secondary = outcomes_module.get("secondaryOutcomes", [])
            if secondary:
                text_parts.append("\n\n## Secondary Outcomes\n")
                for outcome in secondary:
                    measure = outcome.get("measure", "")
                    text_parts.append(f"- {measure}\n")

            # Extract sponsor info
            sponsors = protocol.get("sponsorCollaboratorsModule", {})
            lead = sponsors.get("leadSponsor", {})
            if lead:
                text_parts.append(f"\n\n## Sponsor\n{lead.get('name', 'Unknown')}")
                sponsor_class = lead.get("class", "")
                if sponsor_class:
                    text_parts.append(f" ({sponsor_class})")

            collaborators = sponsors.get("collaborators", [])
            if collaborators:
                text_parts.append("\n\n## Collaborators\n")
                for collab in collaborators:
                    name = collab.get("name", "Unknown")
                    collab_class = collab.get("class", "")
                    text_parts.append(f"- {name}")
                    if collab_class:
                        text_parts.append(f" ({collab_class})")
                    text_parts.append("\n")

            # Extract status info
            status = protocol.get("statusModule", {})
            if status:
                text_parts.append("\n\n## Status\n")
                text_parts.append(
                    f"Overall Status: {status.get('overallStatus', 'Unknown')}\n"
                )
                start_date = status.get("startDateStruct", {}).get("date", "Unknown")
                text_parts.append(f"Start Date: {start_date}\n")
                completion_date = status.get("completionDateStruct", {}).get(
                    "date", "Unknown"
                )
                text_parts.append(f"Completion Date: {completion_date}\n")

                # Why stopped (if applicable)
                why_stopped = status.get("whyStopped")
                if why_stopped:
                    text_parts.append(f"Why Stopped: {why_stopped}\n")

            # Extract enrollment info
            design = protocol.get("designModule", {})
            enrollment_info = design.get("enrollmentInfo", {})
            if enrollment_info:
                count = enrollment_info.get("count")
                enroll_type = enrollment_info.get("type", "")
                if count:
                    text_parts.append(f"\n\n## Enrollment\n{count} ({enroll_type})\n")

            # Extract contacts
            contacts = protocol.get("contactsLocationsModule", {})
            central_contacts = contacts.get("centralContacts", [])
            if central_contacts:
                text_parts.append("\n\n## Contacts\n")
                for contact in central_contacts:
                    name = contact.get("name", "")
                    role = contact.get("role", "")
                    email = contact.get("email", "")
                    phone = contact.get("phone", "")
                    text_parts.append(f"- {name}")
                    if role:
                        text_parts.append(f" ({role})")
                    if email:
                        text_parts.append(f" - {email}")
                    if phone:
                        text_parts.append(f" - {phone}")
                    text_parts.append("\n")

            final_text = "".join(text_parts)
            return CTExtractionResult(success=True, text=final_text.strip())

        except Exception as e:
            logger.error(f"Failed to extract clinical trial text: {e}")
            return CTExtractionResult(success=False, text="", error=str(e))

    def save_extraction(
        self,
        study_json: dict,
        ticker: str,
        published_date: date,
        nct_id: str,
        max_retries: int = 3,
    ) -> tuple[Path, Path, CTExtractionResult]:
        """Save raw JSON and extracted text for a study.

        Args:
            study_json: Full study JSON from ClinicalTrials.gov API
            ticker: Stock ticker symbol
            published_date: Publication/update date
            nct_id: NCT identifier
            max_retries: Max extraction attempts

        Returns:
            Tuple of (raw_path, text_path, extraction_result)
        """
        url = f"https://clinicaltrials.gov/study/{nct_id}"
        announcement_id = generate_id(url=url, published_date=published_date)

        # Save raw JSON
        raw_path = get_raw_path(
            "clinicaltrials", published_date, ticker, announcement_id, "json"
        )
        ensure_parent_dirs(raw_path)
        raw_path.write_text(json.dumps(study_json, indent=2), encoding="utf-8")
        logger.debug(f"Saved raw JSON: {raw_path}")

        # Extract text with retry
        result = None
        for attempt in range(max_retries):
            result = self.extract_from_study(study_json)
            if result.success:
                break
            logger.warning(f"Extraction attempt {attempt + 1} failed: {result.error}")

        # Save text file (empty if all retries failed)
        text_path = get_text_path(
            "clinicaltrials", published_date, ticker, announcement_id
        )
        ensure_parent_dirs(text_path)
        text_path.write_text(result.text if result.success else "", encoding="utf-8")
        logger.debug(f"Saved text file: {text_path}")

        return raw_path, text_path, result
