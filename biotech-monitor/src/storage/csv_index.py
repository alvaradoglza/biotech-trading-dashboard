"""CSV-based index for announcement tracking and deduplication.

This module provides a thread-safe CSV index for tracking all processed
announcements. It supports deduplication, status updates, and querying.

Example:
    >>> index = AnnouncementIndex("data/index/announcements.csv")
    >>> if not index.exists("a1b2c3d4"):
    ...     index.append(record)
    >>> index.update_status("a1b2c3d4", ParseStatus.OK)
"""

import csv
import fcntl
import json
import platform
from dataclasses import asdict, dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class ParseStatus(Enum):
    """Status of text extraction for an announcement."""

    PENDING = "PENDING"
    OK = "OK"
    FAILED = "FAILED"


class Source(Enum):
    """Data source for announcements."""

    EDGAR = "edgar"
    CLINICALTRIALS = "clinicaltrials"
    OPENFDA = "openfda"
    IR = "ir"


@dataclass
class AnnouncementRecord:
    """Single row in the announcements CSV.

    Attributes:
        id: SHA256 hash (16 chars), PRIMARY KEY
        ticker: Stock ticker symbol
        source: Data source (edgar, clinicaltrials, openfda)
        event_type: Specific event type (EDGAR_8K, CT_STATUS_CHANGE, etc.)
        published_at: Publication date (ISO format)
        fetched_at: When we fetched it (ISO datetime)
        title: Short title/description
        url: Original URL
        external_id: Source-specific ID (accession number, NCT ID)
        raw_path: Relative path to raw file
        raw_paths_extra: JSON array of exhibit paths (nullable)
        text_path: Relative path to extracted text
        raw_mime: MIME type of raw content
        raw_size_bytes: Size of raw file(s)
        text_size_bytes: Size of extracted text
        parse_status: Extraction status (OK, FAILED, PENDING)
        parse_attempts: Number of extraction attempts
        error: Error message if failed (nullable)
        extra_json: Source-specific metadata as JSON (nullable)
        return_30d: % return 30 days post-announcement (nullable, Phase 3.5)
        return_60d: % return 60 days post-announcement (nullable, Phase 3.5)
        return_90d: % return 90 days post-announcement (nullable, Phase 3.5)
    """

    id: str
    ticker: str
    source: str
    event_type: str
    published_at: str
    fetched_at: str
    title: str
    url: str
    external_id: Optional[str]
    raw_path: str
    raw_paths_extra: Optional[str]
    text_path: str
    raw_mime: str
    raw_size_bytes: int
    text_size_bytes: int
    parse_status: str
    parse_attempts: int
    error: Optional[str]
    extra_json: Optional[str]
    # Return columns (Phase 3.5) - calculated post-announcement returns
    return_30d: Optional[str] = None
    return_60d: Optional[str] = None
    return_90d: Optional[str] = None


class AnnouncementIndex:
    """CSV-based index for announcement tracking.

    Thread-safe with file locking for concurrent access.
    Uses an in-memory cache for fast deduplication checks.

    Example:
        >>> index = AnnouncementIndex("data/index/announcements.csv")
        >>>
        >>> # Check if exists
        >>> if not index.exists("a1b2c3d4"):
        ...     index.append(record)
        >>>
        >>> # Update status
        >>> index.update_status("a1b2c3d4", ParseStatus.OK)
    """

    COLUMNS = [
        "id",
        "ticker",
        "source",
        "event_type",
        "published_at",
        "fetched_at",
        "title",
        "url",
        "external_id",
        "raw_path",
        "raw_paths_extra",
        "text_path",
        "raw_mime",
        "raw_size_bytes",
        "text_size_bytes",
        "parse_status",
        "parse_attempts",
        "error",
        "extra_json",
        # Return columns (Phase 3.5) - calculated post-announcement returns
        "return_30d",
        "return_60d",
        "return_90d",
    ]

    def __init__(self, path: str | Path = "data/index/announcements.csv"):
        """Initialize the announcement index.

        Args:
            path: Path to the CSV file
        """
        self.path = Path(path)
        self._ensure_file_exists()
        self._id_cache: Optional[set[str]] = None

    def _ensure_file_exists(self) -> None:
        """Create CSV with headers if it doesn't exist."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                writer.writeheader()

    def _load_id_cache(self) -> set[str]:
        """Load all IDs into memory for fast dedup checking."""
        if self._id_cache is None:
            self._id_cache = set()
            with open(self.path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self._id_cache.add(row["id"])
        return self._id_cache

    def _lock_file(self, f) -> None:
        """Acquire exclusive lock on file (platform-specific)."""
        if platform.system() != "Windows":
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)

    def _unlock_file(self, f) -> None:
        """Release lock on file (platform-specific)."""
        if platform.system() != "Windows":
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def exists(self, announcement_id: str) -> bool:
        """Check if announcement ID already exists (O(1) with cache).

        Args:
            announcement_id: The ID to check

        Returns:
            True if the ID exists in the index
        """
        return announcement_id in self._load_id_cache()

    def append(self, record: AnnouncementRecord) -> bool:
        """Append a new record to the CSV.

        Thread-safe with file locking.

        Args:
            record: The announcement record to append

        Returns:
            True if appended, False if ID already exists (dedup)
        """
        if self.exists(record.id):
            return False

        with open(self.path, "a", newline="", encoding="utf-8") as f:
            self._lock_file(f)
            try:
                writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                writer.writerow(asdict(record))
                if self._id_cache is not None:
                    self._id_cache.add(record.id)
            finally:
                self._unlock_file(f)

        return True

    def update_status(
        self,
        announcement_id: str,
        status: ParseStatus,
        text_size_bytes: int = 0,
        error: Optional[str] = None,
        parse_attempts: Optional[int] = None,
    ) -> bool:
        """Update parse status for an existing record.

        Reads all rows, modifies the matching one, and writes back.
        Thread-safe with file locking.

        Args:
            announcement_id: ID of the record to update
            status: New parse status
            text_size_bytes: Size of extracted text (if applicable)
            error: Error message (if failed)
            parse_attempts: Number of attempts made

        Returns:
            True if record was found and updated, False otherwise
        """
        rows = []
        found = False

        with open(self.path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["id"] == announcement_id:
                    row["parse_status"] = status.value
                    row["text_size_bytes"] = str(text_size_bytes)
                    if error is not None:
                        row["error"] = error
                    if parse_attempts is not None:
                        row["parse_attempts"] = str(parse_attempts)
                    found = True
                rows.append(row)

        if not found:
            return False

        with open(self.path, "w", newline="", encoding="utf-8") as f:
            self._lock_file(f)
            try:
                writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            finally:
                self._unlock_file(f)

        return True

    def get_pending(self) -> list[AnnouncementRecord]:
        """Get all records with PENDING status.

        Returns:
            List of AnnouncementRecord objects with PENDING status
        """
        pending = []
        with open(self.path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["parse_status"] == ParseStatus.PENDING.value:
                    pending.append(self._row_to_record(row))
        return pending

    def get_failed(self) -> list[AnnouncementRecord]:
        """Get all records with FAILED status.

        Returns:
            List of AnnouncementRecord objects with FAILED status
        """
        failed = []
        with open(self.path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["parse_status"] == ParseStatus.FAILED.value:
                    failed.append(self._row_to_record(row))
        return failed

    def get_by_ticker(self, ticker: str) -> list[AnnouncementRecord]:
        """Get all records for a specific ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of AnnouncementRecord objects for the ticker
        """
        results = []
        with open(self.path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["ticker"] == ticker:
                    results.append(self._row_to_record(row))
        return results

    def get_by_source(self, source: str) -> list[AnnouncementRecord]:
        """Get all records for a specific source.

        Args:
            source: Data source (edgar, clinicaltrials, openfda)

        Returns:
            List of AnnouncementRecord objects for the source
        """
        results = []
        with open(self.path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["source"] == source:
                    results.append(self._row_to_record(row))
        return results

    def get_all(self) -> list[AnnouncementRecord]:
        """Get all records.

        Returns:
            List of all AnnouncementRecord objects
        """
        results = []
        with open(self.path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(self._row_to_record(row))
        return results

    def get_stats(self) -> dict:
        """Get summary statistics.

        Returns:
            Dictionary with total count, breakdowns by source, status, and ticker
        """
        stats = {
            "total": 0,
            "by_source": {},
            "by_status": {"OK": 0, "FAILED": 0, "PENDING": 0},
            "by_ticker": {},
        }
        with open(self.path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stats["total"] += 1
                stats["by_source"][row["source"]] = (
                    stats["by_source"].get(row["source"], 0) + 1
                )
                stats["by_status"][row["parse_status"]] = (
                    stats["by_status"].get(row["parse_status"], 0) + 1
                )
                stats["by_ticker"][row["ticker"]] = (
                    stats["by_ticker"].get(row["ticker"], 0) + 1
                )
        return stats

    def clear_cache(self) -> None:
        """Clear the in-memory ID cache.

        Call this if the CSV file was modified externally.
        """
        self._id_cache = None

    @staticmethod
    def _row_to_record(row: dict) -> AnnouncementRecord:
        """Convert CSV row dict to AnnouncementRecord.

        Args:
            row: Dictionary from csv.DictReader

        Returns:
            AnnouncementRecord object
        """
        return AnnouncementRecord(
            id=row["id"],
            ticker=row["ticker"],
            source=row["source"],
            event_type=row["event_type"],
            published_at=row["published_at"],
            fetched_at=row["fetched_at"],
            title=row["title"],
            url=row["url"],
            external_id=row["external_id"] or None,
            raw_path=row["raw_path"],
            raw_paths_extra=row["raw_paths_extra"] or None,
            text_path=row["text_path"],
            raw_mime=row["raw_mime"],
            raw_size_bytes=int(row["raw_size_bytes"]) if row["raw_size_bytes"] else 0,
            text_size_bytes=int(row["text_size_bytes"]) if row["text_size_bytes"] else 0,
            parse_status=row["parse_status"],
            parse_attempts=int(row["parse_attempts"]) if row["parse_attempts"] else 0,
            error=row["error"] or None,
            extra_json=row["extra_json"] or None,
            # Return columns (Phase 3.5)
            return_30d=row.get("return_30d") or None,
            return_60d=row.get("return_60d") or None,
            return_90d=row.get("return_90d") or None,
        )
