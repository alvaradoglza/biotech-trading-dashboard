"""Tests for ParquetSync — CSV → Parquet mirror with raw_text column."""

import pytest
import pandas as pd
from pathlib import Path

from src.storage.parquet_sync import ParquetSync


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temp data structure with a CSV and matching text files."""
    # Create CSV
    csv_path = tmp_path / "index" / "announcements.csv"
    csv_path.parent.mkdir(parents=True)

    csv_content = (
        "id,ticker,source,published_at,title,text_path,parse_status,return_30d\n"
        "abc123,MRNA,edgar,2025-01-15,8-K Filing,"
        "data/text/edgar/2025-01-15/MRNA/abc123.txt,OK,15.43\n"
        "def456,MRNA,edgar,2025-02-01,Another Filing,"
        "data/text/edgar/2025-02-01/MRNA/def456.txt,OK,\n"
        "ghi789,PFE,edgar,2025-01-20,PFE Filing,,FAILED,\n"
    )
    csv_path.write_text(csv_content)

    # Create text files
    text_dir1 = tmp_path / "data" / "text" / "edgar" / "2025-01-15" / "MRNA"
    text_dir1.mkdir(parents=True)
    (text_dir1 / "abc123.txt").write_text("Full text content of the filing...")

    text_dir2 = tmp_path / "data" / "text" / "edgar" / "2025-02-01" / "MRNA"
    text_dir2.mkdir(parents=True)
    (text_dir2 / "def456.txt").write_text("Another filing text content")

    return tmp_path


@pytest.fixture
def sync(temp_data_dir):
    """ParquetSync configured against the temp data dir."""
    return ParquetSync(
        csv_path=temp_data_dir / "index" / "announcements.csv",
        parquet_path=temp_data_dir / "index" / "announcements.parquet",
        text_base_path=temp_data_dir / "data" / "text",
    )


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_creates_parquet_file(self, sync, temp_data_dir):
        """update() creates a Parquet file on disk."""
        stats = sync.update()
        assert (temp_data_dir / "index" / "announcements.parquet").exists()

    def test_returns_correct_total_records(self, sync):
        """Stats dict contains accurate total_records count."""
        stats = sync.update()
        assert stats["total_records"] == 3

    def test_returns_stats_keys(self, sync):
        """Stats dict has all expected keys."""
        stats = sync.update()
        assert "total_records" in stats
        assert "text_loaded" in stats
        assert "text_missing" in stats
        assert "text_empty" in stats
        assert "parquet_size_mb" in stats

    def test_raises_when_csv_missing(self, tmp_path):
        """Raises FileNotFoundError if CSV doesn't exist."""
        sync = ParquetSync(
            csv_path=tmp_path / "nonexistent.csv",
            parquet_path=tmp_path / "out.parquet",
        )
        with pytest.raises(FileNotFoundError, match="CSV not found"):
            sync.update()

    def test_parquet_size_mb_is_positive(self, sync):
        stats = sync.update()
        assert stats["parquet_size_mb"] >= 0


# ---------------------------------------------------------------------------
# raw_text column
# ---------------------------------------------------------------------------

class TestRawTextColumn:
    def test_parquet_has_raw_text_column(self, sync):
        """Parquet file includes a raw_text column."""
        sync.update()
        df = sync.get_dataframe()
        assert "raw_text" in df.columns

    def test_raw_text_populated_from_text_file(self, sync, temp_data_dir):
        """raw_text matches file content for records with a text_path."""
        # The text files reference paths like "data/text/..." but they're relative
        # to the project root, not temp_data_dir. We write absolute-style paths
        # that match files in temp_data_dir by using the actual resolved paths.
        abc_text_file = temp_data_dir / "data" / "text" / "edgar" / "2025-01-15" / "MRNA" / "abc123.txt"

        # Override CSV with absolute paths so the sync can find the files
        csv_path = temp_data_dir / "index" / "announcements.csv"
        def_text_file = temp_data_dir / "data" / "text" / "edgar" / "2025-02-01" / "MRNA" / "def456.txt"

        csv_content = (
            "id,ticker,source,published_at,title,text_path,parse_status,return_30d\n"
            f"abc123,MRNA,edgar,2025-01-15,8-K Filing,{abc_text_file},OK,15.43\n"
            f"def456,MRNA,edgar,2025-02-01,Another Filing,{def_text_file},OK,\n"
            "ghi789,PFE,edgar,2025-01-20,PFE Filing,,FAILED,\n"
        )
        csv_path.write_text(csv_content)

        sync.update()
        df = sync.get_dataframe()
        assert df.loc[df["id"] == "abc123", "raw_text"].iloc[0] == "Full text content of the filing..."

    def test_missing_text_path_gives_empty_string(self, sync):
        """Records with no text_path have empty string in raw_text."""
        sync.update()
        df = sync.get_dataframe()
        assert df.loc[df["id"] == "ghi789", "raw_text"].iloc[0] == ""

    def test_text_missing_count_reflects_no_path_records(self, sync):
        """text_missing stat counts records without a text_path."""
        stats = sync.update()
        assert stats["text_missing"] >= 1  # ghi789 has no text_path

    def test_missing_text_file_gives_empty_string(self, temp_data_dir):
        """If text_path points to a nonexistent file, raw_text is empty string."""
        csv_path = temp_data_dir / "index" / "announcements.csv"
        csv_content = (
            "id,ticker,source,published_at,title,text_path,parse_status,return_30d\n"
            "zzz999,TEST,edgar,2025-01-01,Test,/nonexistent/path/file.txt,OK,\n"
        )
        csv_path.write_text(csv_content)

        sync = ParquetSync(
            csv_path=csv_path,
            parquet_path=temp_data_dir / "index" / "announcements.parquet",
        )
        stats = sync.update()
        df = sync.get_dataframe()

        assert df.loc[df["id"] == "zzz999", "raw_text"].iloc[0] == ""
        assert stats["text_missing"] >= 1


# ---------------------------------------------------------------------------
# Return columns
# ---------------------------------------------------------------------------

class TestReturnColumns:
    def test_return_columns_are_numeric_dtype(self, sync):
        """return_30d/60d/90d columns are float, not object/string."""
        sync.update()
        df = sync.get_dataframe()
        assert df["return_30d"].dtype == float

    def test_return_value_preserved(self, sync, temp_data_dir):
        """Existing return values are preserved accurately."""
        abc_path = temp_data_dir / "data" / "text" / "edgar" / "2025-01-15" / "MRNA" / "abc123.txt"
        def_path = temp_data_dir / "data" / "text" / "edgar" / "2025-02-01" / "MRNA" / "def456.txt"

        csv_path = temp_data_dir / "index" / "announcements.csv"
        csv_content = (
            "id,ticker,source,published_at,title,text_path,parse_status,return_30d\n"
            f"abc123,MRNA,edgar,2025-01-15,8-K Filing,{abc_path},OK,15.43\n"
            f"def456,MRNA,edgar,2025-02-01,Another Filing,{def_path},OK,\n"
            "ghi789,PFE,edgar,2025-01-20,PFE Filing,,FAILED,\n"
        )
        csv_path.write_text(csv_content)

        sync = ParquetSync(
            csv_path=csv_path,
            parquet_path=temp_data_dir / "index" / "announcements.parquet",
        )
        sync.update()
        df = sync.get_dataframe()

        assert df.loc[df["id"] == "abc123", "return_30d"].iloc[0] == 15.43

    def test_empty_return_becomes_nan(self, sync, temp_data_dir):
        """Empty return strings are converted to NaN."""
        def_path = temp_data_dir / "data" / "text" / "edgar" / "2025-02-01" / "MRNA" / "def456.txt"
        csv_path = temp_data_dir / "index" / "announcements.csv"
        csv_content = (
            "id,ticker,source,published_at,title,text_path,parse_status,return_30d\n"
            f"def456,MRNA,edgar,2025-02-01,Another Filing,{def_path},OK,\n"
        )
        csv_path.write_text(csv_content)

        sync = ParquetSync(
            csv_path=csv_path,
            parquet_path=temp_data_dir / "index" / "announcements.parquet",
        )
        sync.update()
        df = sync.get_dataframe()

        assert pd.isna(df.loc[df["id"] == "def456", "return_30d"].iloc[0])


# ---------------------------------------------------------------------------
# get_dataframe / get_ml_dataset
# ---------------------------------------------------------------------------

class TestGetDataframe:
    def test_raises_if_parquet_missing(self, tmp_path):
        sync = ParquetSync(
            csv_path=tmp_path / "a.csv",
            parquet_path=tmp_path / "b.parquet",
        )
        with pytest.raises(FileNotFoundError, match="Parquet not found"):
            sync.get_dataframe()

    def test_returns_dataframe(self, sync):
        sync.update()
        df = sync.get_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3


class TestGetMlDataset:
    def test_filters_by_text_length(self, sync, temp_data_dir):
        """Only records with >= min_text_length chars in raw_text are included."""
        abc_path = temp_data_dir / "data" / "text" / "edgar" / "2025-01-15" / "MRNA" / "abc123.txt"
        def_path = temp_data_dir / "data" / "text" / "edgar" / "2025-02-01" / "MRNA" / "def456.txt"

        csv_path = temp_data_dir / "index" / "announcements.csv"
        csv_content = (
            "id,ticker,source,published_at,title,text_path,parse_status,return_30d\n"
            f"abc123,MRNA,edgar,2025-01-15,8-K Filing,{abc_path},OK,15.43\n"
            f"def456,MRNA,edgar,2025-02-01,Another Filing,{def_path},OK,\n"
            "ghi789,PFE,edgar,2025-01-20,PFE Filing,,FAILED,\n"
        )
        csv_path.write_text(csv_content)

        sync = ParquetSync(
            csv_path=csv_path,
            parquet_path=temp_data_dir / "index" / "announcements.parquet",
        )
        sync.update()

        # Both abc123 and def456 have text; only abc123 has return_30d
        ml_df = sync.get_ml_dataset(min_text_length=10, require_returns=True)
        assert len(ml_df) == 1
        assert ml_df.iloc[0]["id"] == "abc123"

    def test_no_require_returns_includes_more(self, sync, temp_data_dir):
        """With require_returns=False, records without returns are included."""
        abc_path = temp_data_dir / "data" / "text" / "edgar" / "2025-01-15" / "MRNA" / "abc123.txt"
        def_path = temp_data_dir / "data" / "text" / "edgar" / "2025-02-01" / "MRNA" / "def456.txt"

        csv_path = temp_data_dir / "index" / "announcements.csv"
        csv_content = (
            "id,ticker,source,published_at,title,text_path,parse_status,return_30d\n"
            f"abc123,MRNA,edgar,2025-01-15,8-K Filing,{abc_path},OK,15.43\n"
            f"def456,MRNA,edgar,2025-02-01,Another Filing,{def_path},OK,\n"
            "ghi789,PFE,edgar,2025-01-20,PFE Filing,,FAILED,\n"
        )
        csv_path.write_text(csv_content)

        sync = ParquetSync(
            csv_path=csv_path,
            parquet_path=temp_data_dir / "index" / "announcements.parquet",
        )
        sync.update()

        ml_df = sync.get_ml_dataset(min_text_length=5, require_returns=False)
        # abc123 and def456 both have text; ghi789 has no text → excluded by length filter
        assert len(ml_df) == 2

    def test_raises_if_parquet_missing(self, tmp_path):
        sync = ParquetSync(
            csv_path=tmp_path / "a.csv",
            parquet_path=tmp_path / "b.parquet",
        )
        with pytest.raises(FileNotFoundError):
            sync.get_ml_dataset()
