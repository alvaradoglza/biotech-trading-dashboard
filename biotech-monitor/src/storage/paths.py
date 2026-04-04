"""Deterministic path generation for announcement storage.

This module provides functions to generate consistent file paths for storing
raw announcement content and extracted text. The paths are deterministic based
on the announcement URL and publication date, enabling deduplication.

Example:
    >>> from datetime import date
    >>> generate_id("https://sec.gov/filing", date(2025, 2, 7))
    'a1b2c3d4e5f6g7h8'
    >>> get_raw_path("edgar", date(2025, 2, 7), "MRNA", "a1b2c3d4", "html")
    Path('data/raw/edgar/2025-02-07/MRNA/a1b2c3d4.html')
"""

import hashlib
from datetime import date
from pathlib import Path
from typing import Optional


def generate_id(url: str, published_date: date) -> str:
    """Generate deterministic ID from URL and date.

    Creates a unique identifier by hashing the URL and publication date.
    The same inputs will always produce the same output, enabling deduplication.

    Args:
        url: Full URL of the announcement
        published_date: Publication/filing date

    Returns:
        First 16 characters of SHA256 hash (sufficient for uniqueness)

    Example:
        >>> generate_id("https://sec.gov/filing", date(2025, 2, 7))
        'a1b2c3d4e5f6g7h8'
    """
    content = f"{url}|{published_date.isoformat()}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def get_raw_path(
    source: str,
    published_date: date,
    ticker: str,
    announcement_id: str,
    extension: str,
    exhibit_name: Optional[str] = None,
) -> Path:
    """Generate path for raw file storage.

    Creates a hierarchical path structure:
    data/raw/{source}/{date}/{ticker}/{id}[_{exhibit}].{ext}

    Args:
        source: Data source (edgar, clinicaltrials, openfda, ir)
        published_date: Publication date
        ticker: Stock ticker symbol
        announcement_id: Unique announcement ID (from generate_id)
        extension: File extension (html, pdf, json, etc.)
        exhibit_name: Optional exhibit identifier for SEC filings

    Returns:
        Path object for the raw file

    Examples:
        >>> get_raw_path("edgar", date(2025, 2, 7), "MRNA", "a1b2c3d4", "html")
        Path('data/raw/edgar/2025-02-07/MRNA/a1b2c3d4.html')

        >>> get_raw_path("edgar", date(2025, 2, 7), "MRNA", "a1b2c3d4", "htm", "ex99-1")
        Path('data/raw/edgar/2025-02-07/MRNA/a1b2c3d4_ex99-1.htm')
    """
    base = Path("data/raw") / source / published_date.isoformat() / ticker

    if exhibit_name:
        return base / f"{announcement_id}_{exhibit_name}.{extension}"
    return base / f"{announcement_id}.{extension}"


def get_text_path(
    source: str,
    published_date: date,
    ticker: str,
    announcement_id: str,
    extension: str = "txt",
) -> Path:
    """Generate path for extracted text storage.

    Creates a hierarchical path structure:
    data/text/{source}/{date}/{ticker}/{id}.{ext}

    Args:
        source: Data source (edgar, clinicaltrials, openfda, ir)
        published_date: Publication date
        ticker: Stock ticker symbol
        announcement_id: Unique announcement ID (from generate_id)
        extension: File extension (default "txt", use "json" for OpenFDA)

    Returns:
        Path object for the text file

    Examples:
        >>> get_text_path("edgar", date(2025, 2, 7), "MRNA", "a1b2c3d4")
        Path('data/text/edgar/2025-02-07/MRNA/a1b2c3d4.txt')

        >>> get_text_path("openfda", date(2025, 2, 7), "MRNA", "a1b2c3d4", "json")
        Path('data/text/openfda/2025-02-07/MRNA/a1b2c3d4.json')
    """
    base = Path("data/text") / source / published_date.isoformat() / ticker
    return base / f"{announcement_id}.{extension}"


def get_index_path() -> Path:
    """Get path to the master announcements CSV index.

    Returns:
        Path object for the index file
    """
    return Path("data/index/announcements.csv")


def ensure_parent_dirs(path: Path) -> None:
    """Create parent directories for a path if they don't exist.

    Args:
        path: File path whose parent directories should be created
    """
    path.parent.mkdir(parents=True, exist_ok=True)
