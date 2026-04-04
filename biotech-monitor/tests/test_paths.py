"""Tests for storage path generation."""

import pytest
from datetime import date
from pathlib import Path

from src.storage.paths import (
    generate_id,
    get_raw_path,
    get_text_path,
    get_index_path,
    ensure_parent_dirs,
)


class TestGenerateId:
    """Tests for deterministic ID generation."""

    def test_generate_id_deterministic(self):
        """Same URL + date always produces same ID."""
        url = "https://example.com/filing"
        d = date(2025, 2, 7)

        id1 = generate_id(url, d)
        id2 = generate_id(url, d)

        assert id1 == id2
        assert len(id1) == 16  # Truncated SHA256

    def test_generate_id_different_urls(self):
        """Different URLs produce different IDs."""
        d = date(2025, 2, 7)

        id1 = generate_id("https://example.com/filing1", d)
        id2 = generate_id("https://example.com/filing2", d)

        assert id1 != id2

    def test_generate_id_different_dates(self):
        """Different dates produce different IDs."""
        url = "https://example.com/filing"

        id1 = generate_id(url, date(2025, 2, 7))
        id2 = generate_id(url, date(2025, 2, 8))

        assert id1 != id2

    def test_generate_id_all_different(self):
        """All different inputs produce unique IDs."""
        url = "https://example.com/filing"

        id1 = generate_id(url, date(2025, 2, 7))
        id2 = generate_id(url, date(2025, 2, 8))
        id3 = generate_id("https://other.com", date(2025, 2, 7))

        assert id1 != id2
        assert id1 != id3
        assert id2 != id3

    def test_generate_id_length(self):
        """Generated ID is exactly 16 characters."""
        id1 = generate_id("https://example.com", date(2025, 1, 1))
        assert len(id1) == 16

    def test_generate_id_hexadecimal(self):
        """Generated ID contains only hexadecimal characters."""
        id1 = generate_id("https://example.com", date(2025, 1, 1))
        assert all(c in "0123456789abcdef" for c in id1)


class TestGetRawPath:
    """Tests for raw file path generation."""

    def test_get_raw_path_basic(self):
        """Basic raw path generation."""
        path = get_raw_path(
            source="edgar",
            published_date=date(2025, 2, 7),
            ticker="MRNA",
            announcement_id="a1b2c3d4",
            extension="html",
        )

        assert path == Path("data/raw/edgar/2025-02-07/MRNA/a1b2c3d4.html")

    def test_get_raw_path_with_exhibit(self):
        """Raw path with exhibit name."""
        path = get_raw_path(
            source="edgar",
            published_date=date(2025, 2, 7),
            ticker="MRNA",
            announcement_id="a1b2c3d4",
            extension="htm",
            exhibit_name="ex99-1",
        )

        assert path == Path("data/raw/edgar/2025-02-07/MRNA/a1b2c3d4_ex99-1.htm")

    def test_get_raw_path_clinicaltrials(self):
        """Raw path for clinicaltrials source."""
        path = get_raw_path(
            source="clinicaltrials",
            published_date=date(2025, 2, 7),
            ticker="MRNA",
            announcement_id="e5f6g7h8",
            extension="json",
        )

        assert path == Path("data/raw/clinicaltrials/2025-02-07/MRNA/e5f6g7h8.json")

    def test_get_raw_path_openfda(self):
        """Raw path for openfda source."""
        path = get_raw_path(
            source="openfda",
            published_date=date(2025, 2, 7),
            ticker="MRNA",
            announcement_id="i9j0k1l2",
            extension="json",
        )

        assert path == Path("data/raw/openfda/2025-02-07/MRNA/i9j0k1l2.json")

    def test_get_raw_path_pdf_extension(self):
        """Raw path with PDF extension."""
        path = get_raw_path(
            source="edgar",
            published_date=date(2025, 2, 7),
            ticker="PFE",
            announcement_id="abc123",
            extension="pdf",
            exhibit_name="ex99-2",
        )

        assert path == Path("data/raw/edgar/2025-02-07/PFE/abc123_ex99-2.pdf")


class TestGetTextPath:
    """Tests for text file path generation."""

    def test_get_text_path_default_extension(self):
        """Text path defaults to .txt extension."""
        path = get_text_path(
            source="edgar",
            published_date=date(2025, 2, 7),
            ticker="MRNA",
            announcement_id="a1b2c3d4",
        )

        assert path == Path("data/text/edgar/2025-02-07/MRNA/a1b2c3d4.txt")

    def test_get_text_path_custom_extension(self):
        """Text path with custom extension (for JSON)."""
        path = get_text_path(
            source="openfda",
            published_date=date(2025, 2, 7),
            ticker="MRNA",
            announcement_id="a1b2c3d4",
            extension="json",
        )

        assert path == Path("data/text/openfda/2025-02-07/MRNA/a1b2c3d4.json")

    def test_get_text_path_clinicaltrials(self):
        """Text path for clinicaltrials source."""
        path = get_text_path(
            source="clinicaltrials",
            published_date=date(2025, 2, 7),
            ticker="MRNA",
            announcement_id="nct12345",
        )

        assert path == Path("data/text/clinicaltrials/2025-02-07/MRNA/nct12345.txt")


class TestGetIndexPath:
    """Tests for index path generation."""

    def test_get_index_path(self):
        """Index path is correctly generated."""
        path = get_index_path()
        assert path == Path("data/index/announcements.csv")


class TestEnsureParentDirs:
    """Tests for parent directory creation."""

    def test_ensure_parent_dirs_creates_dirs(self, tmp_path):
        """Creates parent directories if they don't exist."""
        test_path = tmp_path / "level1" / "level2" / "file.txt"

        assert not test_path.parent.exists()

        ensure_parent_dirs(test_path)

        assert test_path.parent.exists()
        assert test_path.parent.parent.exists()

    def test_ensure_parent_dirs_already_exists(self, tmp_path):
        """Does nothing if parent directories already exist."""
        test_path = tmp_path / "existing" / "file.txt"
        test_path.parent.mkdir(parents=True)

        # Should not raise
        ensure_parent_dirs(test_path)

        assert test_path.parent.exists()
