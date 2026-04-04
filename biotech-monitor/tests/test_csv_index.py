"""Tests for CSV index management."""

import pytest
from pathlib import Path
import csv

from src.storage.csv_index import (
    AnnouncementIndex,
    AnnouncementRecord,
    ParseStatus,
    Source,
)


@pytest.fixture
def temp_csv(tmp_path):
    """Create temporary CSV file path."""
    return tmp_path / "test_index.csv"


@pytest.fixture
def sample_record():
    """Sample announcement record."""
    return AnnouncementRecord(
        id="a1b2c3d4e5f6g7h8",
        ticker="MRNA",
        source="edgar",
        event_type="EDGAR_8K",
        published_at="2025-02-07",
        fetched_at="2025-02-07T14:30:00",
        title="Form 8-K: Results",
        url="https://sec.gov/...",
        external_id="0001682852-25-000008",
        raw_path="raw/edgar/2025-02-07/MRNA/a1b2c3d4.html",
        raw_paths_extra=None,
        text_path="text/edgar/2025-02-07/MRNA/a1b2c3d4.txt",
        raw_mime="text/html",
        raw_size_bytes=125000,
        text_size_bytes=45000,
        parse_status="OK",
        parse_attempts=1,
        error=None,
        extra_json='{"items": ["2.02"]}',
    )


class TestAnnouncementIndex:
    """Tests for CSV index management."""

    def test_creates_file_with_headers(self, temp_csv):
        """Index creates CSV with correct headers."""
        # File shouldn't exist yet
        assert not temp_csv.exists()

        index = AnnouncementIndex(temp_csv)

        assert temp_csv.exists()
        with open(temp_csv) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == AnnouncementIndex.COLUMNS

    def test_append_record(self, temp_csv, sample_record):
        """Appends record to CSV."""
        index = AnnouncementIndex(temp_csv)
        result = index.append(sample_record)

        assert result is True
        assert index.exists(sample_record.id)

    def test_dedup_prevents_duplicate(self, temp_csv, sample_record):
        """Duplicate ID is rejected."""
        index = AnnouncementIndex(temp_csv)

        result1 = index.append(sample_record)
        result2 = index.append(sample_record)  # Same ID

        assert result1 is True
        assert result2 is False

    def test_exists_uses_cache(self, temp_csv, sample_record):
        """Exists check uses in-memory cache."""
        index = AnnouncementIndex(temp_csv)
        index.append(sample_record)

        # Should use cache, not read file
        assert index.exists(sample_record.id) is True
        assert index.exists("nonexistent") is False

    def test_update_status(self, temp_csv, sample_record):
        """Updates parse status."""
        index = AnnouncementIndex(temp_csv)
        index.append(sample_record)

        result = index.update_status(
            sample_record.id,
            ParseStatus.FAILED,
            text_size_bytes=0,
            error="Test error",
            parse_attempts=3,
        )

        assert result is True

        # Re-read to verify
        records = index.get_failed()
        assert len(records) == 1
        assert records[0].error == "Test error"
        assert records[0].parse_attempts == 3

    def test_update_status_not_found(self, temp_csv, sample_record):
        """Update returns False for non-existent ID."""
        index = AnnouncementIndex(temp_csv)
        index.append(sample_record)

        result = index.update_status(
            "nonexistent",
            ParseStatus.FAILED,
        )

        assert result is False

    def test_get_pending(self, temp_csv):
        """Gets records with PENDING status."""
        index = AnnouncementIndex(temp_csv)

        pending_record = AnnouncementRecord(
            id="pending123",
            ticker="TEST",
            source="edgar",
            event_type="EDGAR_8K",
            published_at="2025-02-07",
            fetched_at="2025-02-07T14:30:00",
            title="Pending",
            url="https://...",
            external_id=None,
            raw_path="",
            raw_paths_extra=None,
            text_path="",
            raw_mime="text/html",
            raw_size_bytes=0,
            text_size_bytes=0,
            parse_status="PENDING",
            parse_attempts=0,
            error=None,
            extra_json=None,
        )

        index.append(pending_record)

        pending = index.get_pending()
        assert len(pending) == 1
        assert pending[0].id == "pending123"

    def test_get_failed(self, temp_csv):
        """Gets records with FAILED status."""
        index = AnnouncementIndex(temp_csv)

        failed_record = AnnouncementRecord(
            id="failed123",
            ticker="TEST",
            source="edgar",
            event_type="EDGAR_8K",
            published_at="2025-02-07",
            fetched_at="2025-02-07T14:30:00",
            title="Failed",
            url="https://...",
            external_id=None,
            raw_path="raw/test.html",
            raw_paths_extra=None,
            text_path="text/test.txt",
            raw_mime="text/html",
            raw_size_bytes=100,
            text_size_bytes=0,
            parse_status="FAILED",
            parse_attempts=3,
            error="Extraction failed",
            extra_json=None,
        )

        index.append(failed_record)

        failed = index.get_failed()
        assert len(failed) == 1
        assert failed[0].id == "failed123"
        assert failed[0].error == "Extraction failed"

    def test_get_by_ticker(self, temp_csv, sample_record):
        """Gets records for specific ticker."""
        index = AnnouncementIndex(temp_csv)
        index.append(sample_record)

        records = index.get_by_ticker("MRNA")
        assert len(records) == 1

        records = index.get_by_ticker("OTHER")
        assert len(records) == 0

    def test_get_by_source(self, temp_csv, sample_record):
        """Gets records for specific source."""
        index = AnnouncementIndex(temp_csv)
        index.append(sample_record)

        records = index.get_by_source("edgar")
        assert len(records) == 1

        records = index.get_by_source("clinicaltrials")
        assert len(records) == 0

    def test_get_all(self, temp_csv, sample_record):
        """Gets all records."""
        index = AnnouncementIndex(temp_csv)
        index.append(sample_record)

        # Add another record
        record2 = AnnouncementRecord(
            id="record2",
            ticker="PFE",
            source="clinicaltrials",
            event_type="CT_COMPLETED",
            published_at="2025-02-08",
            fetched_at="2025-02-08T10:00:00",
            title="Trial completed",
            url="https://clinicaltrials.gov/...",
            external_id="NCT12345",
            raw_path="raw/ct.json",
            raw_paths_extra=None,
            text_path="text/ct.txt",
            raw_mime="application/json",
            raw_size_bytes=5000,
            text_size_bytes=2000,
            parse_status="OK",
            parse_attempts=1,
            error=None,
            extra_json=None,
        )
        index.append(record2)

        records = index.get_all()
        assert len(records) == 2

    def test_get_stats(self, temp_csv, sample_record):
        """Gets summary statistics."""
        index = AnnouncementIndex(temp_csv)
        index.append(sample_record)

        stats = index.get_stats()

        assert stats["total"] == 1
        assert stats["by_source"]["edgar"] == 1
        assert stats["by_status"]["OK"] == 1
        assert stats["by_ticker"]["MRNA"] == 1

    def test_get_stats_multiple_records(self, temp_csv):
        """Gets statistics with multiple records."""
        index = AnnouncementIndex(temp_csv)

        # Add multiple records
        for i, (source, status, ticker) in enumerate([
            ("edgar", "OK", "MRNA"),
            ("edgar", "FAILED", "MRNA"),
            ("clinicaltrials", "OK", "PFE"),
            ("openfda", "PENDING", "MRNA"),
        ]):
            record = AnnouncementRecord(
                id=f"record{i}",
                ticker=ticker,
                source=source,
                event_type="TEST",
                published_at="2025-02-07",
                fetched_at="2025-02-07T14:30:00",
                title=f"Record {i}",
                url=f"https://example.com/{i}",
                external_id=None,
                raw_path="",
                raw_paths_extra=None,
                text_path="",
                raw_mime="text/html",
                raw_size_bytes=0,
                text_size_bytes=0,
                parse_status=status,
                parse_attempts=1,
                error=None,
                extra_json=None,
            )
            index.append(record)

        stats = index.get_stats()

        assert stats["total"] == 4
        assert stats["by_source"]["edgar"] == 2
        assert stats["by_source"]["clinicaltrials"] == 1
        assert stats["by_source"]["openfda"] == 1
        assert stats["by_status"]["OK"] == 2
        assert stats["by_status"]["FAILED"] == 1
        assert stats["by_status"]["PENDING"] == 1
        assert stats["by_ticker"]["MRNA"] == 3
        assert stats["by_ticker"]["PFE"] == 1

    def test_clear_cache(self, temp_csv, sample_record):
        """Clears the in-memory ID cache."""
        index = AnnouncementIndex(temp_csv)
        index.append(sample_record)

        # Verify cache is populated
        assert index.exists(sample_record.id)

        # Clear cache
        index.clear_cache()

        # Should still work (reloads from file)
        assert index.exists(sample_record.id)

    def test_handles_nullable_fields(self, temp_csv):
        """Handles nullable fields correctly."""
        record = AnnouncementRecord(
            id="nullable_test",
            ticker="TEST",
            source="edgar",
            event_type="EDGAR_8K",
            published_at="2025-02-07",
            fetched_at="2025-02-07T14:30:00",
            title="Test",
            url="https://...",
            external_id=None,
            raw_path="",
            raw_paths_extra=None,
            text_path="",
            raw_mime="text/html",
            raw_size_bytes=0,
            text_size_bytes=0,
            parse_status="PENDING",
            parse_attempts=0,
            error=None,
            extra_json=None,
        )

        index = AnnouncementIndex(temp_csv)
        index.append(record)

        # Read back
        records = index.get_all()
        assert len(records) == 1
        assert records[0].external_id is None
        assert records[0].raw_paths_extra is None
        assert records[0].error is None
        assert records[0].extra_json is None


class TestParseStatus:
    """Tests for ParseStatus enum."""

    def test_parse_status_values(self):
        """ParseStatus has correct values."""
        assert ParseStatus.PENDING.value == "PENDING"
        assert ParseStatus.OK.value == "OK"
        assert ParseStatus.FAILED.value == "FAILED"


class TestSource:
    """Tests for Source enum."""

    def test_source_values(self):
        """Source has correct values."""
        assert Source.EDGAR.value == "edgar"
        assert Source.CLINICALTRIALS.value == "clinicaltrials"
        assert Source.OPENFDA.value == "openfda"
        assert Source.IR.value == "ir"
