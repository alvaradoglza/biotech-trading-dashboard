"""OpenFDA data preservation.

This module handles preserving OpenFDA data as structured JSON.
Unlike other sources, OpenFDA data is kept as JSON rather than
converted to plain text, as the structure is valuable for analysis.

Example:
    >>> extractor = OpenFDAExtractor()
    >>> raw_path, text_path, result = extractor.save_extraction(
    ...     fda_result=fda_data,
    ...     ticker="MRNA",
    ...     published_date=date(2022, 1, 31),
    ...     application_number="BLA761222",
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
class FDAExtractionResult:
    """Result of OpenFDA extraction.

    Attributes:
        success: Whether extraction succeeded
        json_data: Preserved JSON data
        error: Error message if extraction failed
    """

    success: bool
    json_data: dict
    error: Optional[str] = None


class OpenFDAExtractor:
    """Extract/preserve OpenFDA data.

    Unlike other sources, OpenFDA data is preserved as structured JSON
    rather than converted to plain text. This is because:
    1. Drug labels are already structured (indications, warnings, dosage, etc.)
    2. The structure is valuable for downstream analysis
    3. Converting to plain text loses important categorization

    The "text" file is actually a JSON file with extension .json

    Example:
        >>> extractor = OpenFDAExtractor()
        >>> raw_path, text_path, result = extractor.save_extraction(
        ...     fda_result=result_dict,
        ...     ticker="MRNA",
        ...     published_date=date(2022, 1, 31),
        ...     application_number="BLA761222",
        ... )
    """

    def save_extraction(
        self,
        fda_result: dict,
        ticker: str,
        published_date: date,
        application_number: str,
    ) -> tuple[Path, Path, FDAExtractionResult]:
        """Save OpenFDA result as JSON.

        Both raw and "text" paths point to JSON files.

        Args:
            fda_result: Single result from OpenFDA API
            ticker: Stock ticker symbol
            published_date: Approval/submission date
            application_number: NDA/BLA number

        Returns:
            Tuple of (raw_path, text_path, extraction_result)
        """
        url = f"https://api.fda.gov/drug/drugsfda.json?search=application_number:{application_number}"
        announcement_id = generate_id(url=url, published_date=published_date)

        # Initialize paths
        raw_path = get_raw_path(
            "openfda", published_date, ticker, announcement_id, "json"
        )
        text_path = get_text_path(
            "openfda", published_date, ticker, announcement_id, "json"
        )

        try:
            # Save raw JSON
            ensure_parent_dirs(raw_path)
            raw_path.write_text(json.dumps(fda_result, indent=2), encoding="utf-8")
            logger.debug(f"Saved raw JSON: {raw_path}")

            # For OpenFDA, "text" is also JSON (preserved structure)
            # We extract just the key fields we care about
            ensure_parent_dirs(text_path)
            extracted = self._extract_key_fields(fda_result)
            text_path.write_text(json.dumps(extracted, indent=2), encoding="utf-8")
            logger.debug(f"Saved extracted JSON: {text_path}")

            return (
                raw_path,
                text_path,
                FDAExtractionResult(
                    success=True,
                    json_data=extracted,
                ),
            )

        except Exception as e:
            logger.error(f"Failed to save OpenFDA extraction: {e}")
            # Still try to save raw if possible
            return (
                raw_path,
                text_path,
                FDAExtractionResult(
                    success=False,
                    json_data={},
                    error=str(e),
                ),
            )

    def _extract_key_fields(self, result: dict) -> dict:
        """Extract key fields from OpenFDA result.

        Preserves structure but filters to relevant fields.

        Args:
            result: Full OpenFDA result

        Returns:
            Filtered dictionary with key fields
        """
        extracted = {
            "application_number": result.get("application_number"),
            "sponsor_name": result.get("sponsor_name"),
            "products": [],
            "submissions": [],
            "openfda": result.get("openfda", {}),
        }

        # Extract product info
        for product in result.get("products", []):
            extracted["products"].append(
                {
                    "brand_name": product.get("brand_name"),
                    "active_ingredients": product.get("active_ingredients", []),
                    "dosage_form": product.get("dosage_form"),
                    "route": product.get("route"),
                    "marketing_status": product.get("marketing_status"),
                }
            )

        # Extract submission info
        for submission in result.get("submissions", []):
            extracted["submissions"].append(
                {
                    "submission_type": submission.get("submission_type"),
                    "submission_number": submission.get("submission_number"),
                    "submission_status": submission.get("submission_status"),
                    "submission_status_date": submission.get("submission_status_date"),
                    "submission_class_code": submission.get("submission_class_code"),
                    "submission_class_code_description": submission.get(
                        "submission_class_code_description"
                    ),
                }
            )

        return extracted

    def save_recall_extraction(
        self,
        recall_result: dict,
        ticker: str,
        published_date: date,
        recall_number: str,
    ) -> tuple[Path, Path, FDAExtractionResult]:
        """Save OpenFDA recall result as JSON.

        Args:
            recall_result: Single recall result from OpenFDA API
            ticker: Stock ticker symbol
            published_date: Recall initiation date
            recall_number: Recall number

        Returns:
            Tuple of (raw_path, text_path, extraction_result)
        """
        url = f"https://api.fda.gov/drug/enforcement.json?search=recall_number:{recall_number}"
        announcement_id = generate_id(url=url, published_date=published_date)

        # Initialize paths
        raw_path = get_raw_path(
            "openfda", published_date, ticker, announcement_id, "json"
        )
        text_path = get_text_path(
            "openfda", published_date, ticker, announcement_id, "json"
        )

        try:
            # Save raw JSON
            ensure_parent_dirs(raw_path)
            raw_path.write_text(json.dumps(recall_result, indent=2), encoding="utf-8")
            logger.debug(f"Saved raw recall JSON: {raw_path}")

            # Extract key recall fields
            ensure_parent_dirs(text_path)
            extracted = self._extract_recall_fields(recall_result)
            text_path.write_text(json.dumps(extracted, indent=2), encoding="utf-8")
            logger.debug(f"Saved extracted recall JSON: {text_path}")

            return (
                raw_path,
                text_path,
                FDAExtractionResult(
                    success=True,
                    json_data=extracted,
                ),
            )

        except Exception as e:
            logger.error(f"Failed to save OpenFDA recall extraction: {e}")
            return (
                raw_path,
                text_path,
                FDAExtractionResult(
                    success=False,
                    json_data={},
                    error=str(e),
                ),
            )

    def _extract_recall_fields(self, result: dict) -> dict:
        """Extract key fields from OpenFDA recall result.

        Args:
            result: Full OpenFDA recall result

        Returns:
            Filtered dictionary with key fields
        """
        return {
            "recall_number": result.get("recall_number"),
            "recalling_firm": result.get("recalling_firm"),
            "product_description": result.get("product_description"),
            "reason_for_recall": result.get("reason_for_recall"),
            "classification": result.get("classification"),
            "status": result.get("status"),
            "recall_initiation_date": result.get("recall_initiation_date"),
            "termination_date": result.get("termination_date"),
            "voluntary_mandated": result.get("voluntary_mandated"),
            "distribution_pattern": result.get("distribution_pattern"),
            "product_quantity": result.get("product_quantity"),
            "openfda": result.get("openfda", {}),
        }
