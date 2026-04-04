"""Main extraction pipeline for announcement content.

This module provides the ExtractionPipeline class that orchestrates
fetching and extracting content from ClinicalTrials.gov and OpenFDA.
"""

import json
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from src.storage.csv_index import AnnouncementIndex, AnnouncementRecord, ParseStatus
from src.storage.paths import generate_id
from src.extraction.clinicaltrials_extractor import ClinicalTrialsExtractor
from src.extraction.openfda_extractor import OpenFDAExtractor
from src.clients.clinicaltrials import ClinicalTrial
from src.clients.openfda import DrugApproval, DrugRecall
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ExtractionPipeline:
    """Main pipeline for fetching and extracting announcement content.

    Workflow for each announcement:
    1. Fetch raw content + metadata
    2. Write raw file(s) -> get raw_path
    3. Extract plaintext -> write text file -> get text_path
    4. Compute ID (hash) and dedupe
    5. Append row to announcements.csv
    6. Set parse_status=OK or FAILED with error
    """

    def __init__(
        self,
        openfda_key: Optional[str] = None,
        index_path: str = "data/index/announcements.csv",
    ):
        """Initialize the extraction pipeline.

        Args:
            openfda_key: Optional OpenFDA API key
            index_path: Path to the CSV index file
        """
        self.ct_extractor = ClinicalTrialsExtractor()
        self.fda_extractor = OpenFDAExtractor()
        self.index = AnnouncementIndex(index_path)
        self.openfda_key = openfda_key

    def process_clinical_trial(
        self,
        trial: ClinicalTrial,
        study_json: dict,
        ticker: str,
    ) -> bool:
        """Process a single ClinicalTrials.gov study.

        Args:
            trial: ClinicalTrial object with trial metadata
            study_json: Full study JSON from API
            ticker: Stock ticker symbol

        Returns:
            True if processed (new), False if skipped (duplicate)
        """
        url = f"https://clinicaltrials.gov/study/{trial.nct_id}"
        announcement_id = generate_id(url, trial.last_update_date)

        if self.index.exists(announcement_id):
            logger.debug(f"Skipping duplicate trial: {trial.nct_id}")
            return False

        logger.info(f"Processing clinical trial: {trial.nct_id} for {ticker}")

        # Extract and save
        raw_path, text_path, result = self.ct_extractor.save_extraction(
            study_json=study_json,
            ticker=ticker,
            published_date=trial.last_update_date,
            nct_id=trial.nct_id,
            max_retries=3,
        )

        # Create record
        record = AnnouncementRecord(
            id=announcement_id,
            ticker=ticker,
            source="clinicaltrials",
            event_type=f"CT_{trial.status.value}",
            published_at=trial.last_update_date.isoformat(),
            fetched_at=datetime.now().isoformat(),
            title=trial.title[:200] if trial.title else trial.nct_id,
            url=url,
            external_id=trial.nct_id,
            raw_path=str(raw_path),
            raw_paths_extra=None,
            text_path=str(text_path),
            raw_mime="application/json",
            raw_size_bytes=raw_path.stat().st_size if raw_path.exists() else 0,
            text_size_bytes=text_path.stat().st_size if text_path.exists() else 0,
            parse_status=ParseStatus.OK.value if result.success else ParseStatus.FAILED.value,
            parse_attempts=1 if result.success else 3,
            error=result.error,
            extra_json=json.dumps(
                {
                    "status": trial.status.value,
                    "phases": trial.phases,
                    "conditions": trial.conditions,
                    "study_first_posted": trial.study_first_posted_date.isoformat() if trial.study_first_posted_date else None,
                    "start_date": trial.start_date.isoformat() if trial.start_date else None,
                    "last_update_date": trial.last_update_date.isoformat() if trial.last_update_date else None,
                }
            ),
        )

        logger.info(
            f"Extracted clinical trial: {trial.nct_id} - "
            f"status={record.parse_status}, text_size={record.text_size_bytes}"
        )

        self.index.append(record)
        return True

    def process_fda_approval(
        self,
        approval: DrugApproval,
        fda_result: dict,
        ticker: str,
    ) -> bool:
        """Process a single OpenFDA approval.

        Args:
            approval: DrugApproval object with approval metadata
            fda_result: Full result from OpenFDA API
            ticker: Stock ticker symbol

        Returns:
            True if processed (new), False if skipped (duplicate)
        """
        url = f"https://api.fda.gov/drug/drugsfda.json?search=application_number:{approval.application_number}"
        published_date = approval.approval_date or date.today()
        announcement_id = generate_id(url, published_date)

        if self.index.exists(announcement_id):
            logger.debug(f"Skipping duplicate approval: {approval.application_number}")
            return False

        logger.info(
            f"Processing FDA approval: {approval.application_number} for {ticker}"
        )

        # Save extraction
        raw_path, text_path, result = self.fda_extractor.save_extraction(
            fda_result=fda_result,
            ticker=ticker,
            published_date=published_date,
            application_number=approval.application_number,
        )

        # Create record
        record = AnnouncementRecord(
            id=announcement_id,
            ticker=ticker,
            source="openfda",
            event_type=f"FDA_{approval.submission_type}",
            published_at=published_date.isoformat(),
            fetched_at=datetime.now().isoformat(),
            title=f"{approval.brand_name} ({approval.application_number})",
            url=url,
            external_id=approval.application_number,
            raw_path=str(raw_path),
            raw_paths_extra=None,
            text_path=str(text_path),
            raw_mime="application/json",
            raw_size_bytes=raw_path.stat().st_size if raw_path.exists() else 0,
            text_size_bytes=text_path.stat().st_size if text_path.exists() else 0,
            parse_status=ParseStatus.OK.value if result.success else ParseStatus.FAILED.value,
            parse_attempts=1 if result.success else 3,
            error=result.error,
            extra_json=json.dumps(
                {
                    "submission_type": approval.submission_type,
                    "brand_name": approval.brand_name,
                    "generic_name": approval.generic_name,
                }
            ),
        )

        logger.info(
            f"Extracted FDA approval: {approval.application_number} - "
            f"status={record.parse_status}"
        )

        self.index.append(record)
        return True

    def process_fda_recall(
        self,
        recall: DrugRecall,
        recall_result: dict,
        ticker: str,
    ) -> bool:
        """Process a single OpenFDA recall.

        Args:
            recall: DrugRecall object with recall metadata
            recall_result: Full result from OpenFDA API
            ticker: Stock ticker symbol

        Returns:
            True if processed (new), False if skipped (duplicate)
        """
        url = f"https://api.fda.gov/drug/enforcement.json?search=recall_number:{recall.recall_number}"
        published_date = recall.recall_initiation_date or date.today()
        announcement_id = generate_id(url, published_date)

        if self.index.exists(announcement_id):
            logger.debug(f"Skipping duplicate recall: {recall.recall_number}")
            return False

        logger.info(f"Processing FDA recall: {recall.recall_number} for {ticker}")

        # Save extraction
        raw_path, text_path, result = self.fda_extractor.save_recall_extraction(
            recall_result=recall_result,
            ticker=ticker,
            published_date=published_date,
            recall_number=recall.recall_number,
        )

        # Create record
        record = AnnouncementRecord(
            id=announcement_id,
            ticker=ticker,
            source="openfda",
            event_type="FDA_RECALL",
            published_at=published_date.isoformat(),
            fetched_at=datetime.now().isoformat(),
            title=f"Recall: {recall.recalling_firm} - {recall.classification}",
            url=url,
            external_id=recall.recall_number,
            raw_path=str(raw_path),
            raw_paths_extra=None,
            text_path=str(text_path),
            raw_mime="application/json",
            raw_size_bytes=raw_path.stat().st_size if raw_path.exists() else 0,
            text_size_bytes=text_path.stat().st_size if text_path.exists() else 0,
            parse_status=ParseStatus.OK.value if result.success else ParseStatus.FAILED.value,
            parse_attempts=1 if result.success else 3,
            error=result.error,
            extra_json=json.dumps(
                {
                    "classification": recall.classification,
                    "reason": recall.reason_for_recall[:200] if recall.reason_for_recall else None,
                    "status": recall.status,
                }
            ),
        )

        logger.info(f"Extracted FDA recall: {recall.recall_number} - status={record.parse_status}")

        self.index.append(record)
        return True

    def get_stats(self) -> dict:
        """Get extraction statistics.

        Returns:
            Dictionary with extraction statistics
        """
        return self.index.get_stats()

    def get_pending_count(self) -> int:
        """Get count of pending extractions.

        Returns:
            Number of records with PENDING status
        """
        return len(self.index.get_pending())

    def get_failed_count(self) -> int:
        """Get count of failed extractions.

        Returns:
            Number of records with FAILED status
        """
        return len(self.index.get_failed())

    async def retry_failed(self, max_per_source: int = 100) -> dict:
        """Retry failed extractions.

        Args:
            max_per_source: Maximum records to retry per source

        Returns:
            Dictionary with retry results
        """
        failed = self.index.get_failed()
        results = {"retried": 0, "succeeded": 0, "failed": 0}

        for record in failed[:max_per_source]:
            # Implementation depends on having access to original data
            # This is a placeholder for future implementation
            logger.warning(f"Retry not yet implemented for: {record.id}")
            results["failed"] += 1

        return results
