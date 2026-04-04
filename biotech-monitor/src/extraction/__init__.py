"""Extraction module for extracting text from announcement sources."""

from src.extraction.clinicaltrials_extractor import (
    ClinicalTrialsExtractor,
    CTExtractionResult,
)
from src.extraction.openfda_extractor import (
    OpenFDAExtractor,
    FDAExtractionResult,
)
from src.extraction.pipeline import ExtractionPipeline

__all__ = [
    "ClinicalTrialsExtractor",
    "CTExtractionResult",
    "OpenFDAExtractor",
    "FDAExtractionResult",
    "ExtractionPipeline",
]
