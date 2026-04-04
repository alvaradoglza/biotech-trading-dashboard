"""Parquet mirror of the announcements CSV with full text content for ML/NLP analysis."""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

CSV_PATH = Path("data/index/announcements.csv")
PARQUET_PATH = Path("data/index/announcements.parquet")
TEXT_BASE_PATH = Path("data/text")


class ParquetSync:
    """
    Maintains a Parquet mirror of the announcements CSV with full text content.

    The Parquet file is always kept in sync with the CSV and includes
    the full extracted text for each announcement, making it suitable
    for ML/NLP analysis.

    Example:
        sync = ParquetSync()
        sync.update()  # Regenerate Parquet from CSV + text files

        # Load for analysis
        df = pd.read_parquet("data/index/announcements.parquet")
        print(df[["ticker", "title", "raw_text", "return_30d"]].head())
    """

    def __init__(
        self,
        csv_path: Path = CSV_PATH,
        parquet_path: Path = PARQUET_PATH,
        text_base_path: Path = TEXT_BASE_PATH,
    ):
        self.csv_path = Path(csv_path)
        self.parquet_path = Path(parquet_path)
        self.text_base_path = Path(text_base_path)

    def update(self) -> dict:
        """
        Regenerate Parquet file from CSV and text files.

        Reads all records from the CSV (including FAILED ones), loads the
        extracted text from each record's text_path, and writes a Parquet
        file that includes all CSV columns plus a `raw_text` column.

        Returns:
            Stats dict with counts: total_records, text_loaded, text_missing,
            text_empty, parquet_size_mb
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        stats: dict = {
            "total_records": len(df),
            "text_loaded": 0,
            "text_missing": 0,
            "text_empty": 0,
        }

        raw_texts = []

        for _, row in df.iterrows():
            text_path = row.get("text_path", "")

            if not text_path or (isinstance(text_path, float) and pd.isna(text_path)):
                raw_texts.append("")
                stats["text_missing"] += 1
                continue

            full_path = Path(str(text_path))
            if not full_path.is_absolute():
                # If path is relative, resolve against the project root
                full_path = Path(str(text_path))

            try:
                if full_path.exists():
                    content = full_path.read_text(encoding="utf-8", errors="replace")
                    raw_texts.append(content)
                    if content.strip():
                        stats["text_loaded"] += 1
                    else:
                        stats["text_empty"] += 1
                else:
                    raw_texts.append("")
                    stats["text_missing"] += 1
                    logger.debug("Text file not found", path=str(full_path))
            except Exception as e:
                raw_texts.append("")
                stats["text_missing"] += 1
                logger.warning("Failed to read text file", path=str(full_path), error=str(e))

        df["raw_text"] = raw_texts

        # Convert return columns to float (handle empty strings / missing values)
        for col in ["return_5d", "return_30d", "return_60d", "return_90d"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.parquet_path, index=False, engine="pyarrow")

        stats["parquet_size_mb"] = round(self.parquet_path.stat().st_size / 1024 / 1024, 2)

        logger.info(
            "Parquet updated",
            total=stats["total_records"],
            text_loaded=stats["text_loaded"],
            text_missing=stats["text_missing"],
            size_mb=stats["parquet_size_mb"],
        )

        return stats

    def get_dataframe(self) -> pd.DataFrame:
        """Load Parquet as DataFrame.

        Raises:
            FileNotFoundError: If Parquet file does not exist.
        """
        if not self.parquet_path.exists():
            raise FileNotFoundError(
                f"Parquet not found: {self.parquet_path}. Run update() first."
            )
        return pd.read_parquet(self.parquet_path)

    def get_ml_dataset(
        self,
        min_text_length: int = 100,
        require_returns: bool = True,
    ) -> pd.DataFrame:
        """
        Get filtered dataset suitable for ML training.

        Args:
            min_text_length: Minimum characters in raw_text
            require_returns: Only include records with return_30d calculated

        Returns:
            Filtered DataFrame
        """
        df = self.get_dataframe()

        df = df[df["raw_text"].str.len() >= min_text_length]

        if require_returns:
            df = df[df["return_30d"].notna()]

        return df
