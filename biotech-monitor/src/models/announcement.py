"""Announcement data model for unified announcement storage."""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional


class AnnouncementSource(Enum):
    """Source of the announcement."""

    CLINICALTRIALS = "clinicaltrials"
    OPENFDA = "openfda"
    IR_SCRAPE = "ir_scrape"
    FDA_SCRAPE = "fda_scrape"


class AnnouncementCategory(Enum):
    """Category of the announcement."""

    EARNINGS = "earnings"
    TRIAL_START = "trial_start"
    TRIAL_UPDATE = "trial_update"
    TRIAL_RESULTS = "trial_results"
    TRIAL_TERMINATED = "trial_terminated"
    TRIAL_SUSPENDED = "trial_suspended"
    FDA_APPROVAL = "fda_approval"
    FDA_REJECTION = "fda_rejection"
    FDA_SUBMISSION = "fda_submission"
    FDA_RECALL = "fda_recall"
    FDA_WARNING = "fda_warning"
    FDA_ADCOM = "fda_adcom"
    FDA_CRL = "fda_crl"  # Complete Response Letter
    FDA_PDUFA = "fda_pdufa"  # PDUFA action date
    PARTNERSHIP = "partnership"
    FINANCING = "financing"
    EXECUTIVE_CHANGE = "executive_change"
    OTHER = "other"


class Sentiment(Enum):
    """Sentiment of the announcement."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class Announcement:
    """Unified announcement model for all sources."""

    ticker: str
    source: AnnouncementSource
    source_id: str
    announcement_date: date
    title: str
    url: Optional[str] = None
    content: Optional[str] = None
    category: AnnouncementCategory = AnnouncementCategory.OTHER

    # AI enrichment (Phase 5)
    ai_summary: Optional[str] = None
    sentiment: Optional[Sentiment] = None
    is_processed: bool = False

    # Metadata
    raw_data: Optional[dict[str, Any]] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            "ticker": self.ticker,
            "source": self.source.value,
            "source_id": self.source_id,
            "announcement_date": self.announcement_date.isoformat(),
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "category": self.category.value,
            "ai_summary": self.ai_summary,
            "sentiment": self.sentiment.value if self.sentiment else None,
            "is_processed": self.is_processed,
            "raw_data": self.raw_data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Announcement":
        """Create from dictionary (e.g., database row)."""
        return cls(
            id=data.get("id"),
            ticker=data["ticker"],
            source=AnnouncementSource(data["source"]),
            source_id=data["source_id"],
            announcement_date=(
                data["announcement_date"]
                if isinstance(data["announcement_date"], date)
                else datetime.fromisoformat(data["announcement_date"]).date()
            ),
            title=data["title"],
            url=data.get("url"),
            content=data.get("content"),
            category=AnnouncementCategory(data.get("category", "other")),
            ai_summary=data.get("ai_summary"),
            sentiment=Sentiment(data["sentiment"]) if data.get("sentiment") else None,
            is_processed=data.get("is_processed", False),
            raw_data=data.get("raw_data"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    @classmethod
    def from_clinical_trial(
        cls,
        ticker: str,
        trial: "ClinicalTrial",
    ) -> "Announcement":
        """Create announcement from ClinicalTrials.gov trial.

        Args:
            ticker: Stock ticker
            trial: ClinicalTrial object from clinicaltrials client

        Returns:
            Announcement object
        """
        # Determine category based on trial status
        category = cls._categorize_trial_status(trial.status.value)

        # Build title
        phase_str = trial.phase_display
        title = f"[{phase_str}] {trial.title[:200]}"

        return cls(
            ticker=ticker,
            source=AnnouncementSource.CLINICALTRIALS,
            source_id=trial.nct_id,
            announcement_date=trial.last_update_date or date.today(),
            title=title,
            url=trial.url,
            content=trial.brief_summary,
            category=category,
            raw_data={
                "nct_id": trial.nct_id,
                "sponsor": trial.sponsor,
                "status": trial.status.value,
                "phases": trial.phases,
                "conditions": trial.conditions,
                "interventions": trial.interventions,
                "enrollment": trial.enrollment,
                "has_results": trial.has_results,
            },
        )

    @classmethod
    def from_fda_approval(
        cls,
        ticker: str,
        approval: "DrugApproval",
    ) -> "Announcement":
        """Create announcement from OpenFDA drug approval.

        Args:
            ticker: Stock ticker
            approval: DrugApproval object from openfda client

        Returns:
            Announcement object
        """
        # Determine category based on submission status
        if approval.submission_status == "AP":
            category = AnnouncementCategory.FDA_APPROVAL
        else:
            category = AnnouncementCategory.FDA_SUBMISSION

        # Build title
        title = f"FDA {approval.submission_type}: {approval.brand_name or approval.generic_name or approval.application_number}"

        return cls(
            ticker=ticker,
            source=AnnouncementSource.OPENFDA,
            source_id=approval.application_number,
            announcement_date=approval.approval_date or date.today(),
            title=title,
            url=approval.url,
            content=None,
            category=category,
            raw_data={
                "application_number": approval.application_number,
                "sponsor_name": approval.sponsor_name,
                "brand_name": approval.brand_name,
                "generic_name": approval.generic_name,
                "submission_type": approval.submission_type,
                "submission_status": approval.submission_status,
                "dosage_form": approval.dosage_form,
                "route": approval.route,
                "active_ingredients": approval.active_ingredients,
            },
        )

    @classmethod
    def from_fda_recall(
        cls,
        ticker: str,
        recall: "DrugRecall",
    ) -> "Announcement":
        """Create announcement from OpenFDA drug recall/enforcement.

        Args:
            ticker: Stock ticker
            recall: DrugRecall object from openfda client

        Returns:
            Announcement object
        """
        # Determine category based on classification
        # Class I = most severe (FDA_WARNING), Class II/III = recall
        if recall.classification == "Class I":
            category = AnnouncementCategory.FDA_WARNING
        else:
            category = AnnouncementCategory.FDA_RECALL

        # Build title
        title = f"FDA {recall.classification} Recall: {recall.product_description[:100]}"
        if len(recall.product_description) > 100:
            title += "..."

        return cls(
            ticker=ticker,
            source=AnnouncementSource.OPENFDA,
            source_id=recall.recall_number,
            announcement_date=recall.recall_initiation_date or date.today(),
            title=title,
            url=recall.url,
            content=recall.reason_for_recall,
            category=category,
            raw_data={
                "recall_number": recall.recall_number,
                "recalling_firm": recall.recalling_firm,
                "classification": recall.classification,
                "status": recall.status,
                "product_description": recall.product_description,
                "reason_for_recall": recall.reason_for_recall,
                "distribution_pattern": recall.distribution_pattern,
                "voluntary_mandated": recall.voluntary_mandated,
            },
        )

    @staticmethod
    def _categorize_trial_status(status: str) -> AnnouncementCategory:
        """Categorize clinical trial based on status.

        Args:
            status: Trial status string

        Returns:
            AnnouncementCategory
        """
        status_upper = status.upper()

        if status_upper == "NOT_YET_RECRUITING":
            return AnnouncementCategory.TRIAL_START
        if status_upper in ("RECRUITING", "ENROLLING_BY_INVITATION", "ACTIVE_NOT_RECRUITING"):
            return AnnouncementCategory.TRIAL_UPDATE
        if status_upper == "COMPLETED":
            return AnnouncementCategory.TRIAL_RESULTS
        if status_upper == "TERMINATED":
            return AnnouncementCategory.TRIAL_TERMINATED
        if status_upper == "SUSPENDED":
            return AnnouncementCategory.TRIAL_SUSPENDED
        if status_upper == "WITHDRAWN":
            return AnnouncementCategory.TRIAL_TERMINATED

        return AnnouncementCategory.TRIAL_UPDATE
