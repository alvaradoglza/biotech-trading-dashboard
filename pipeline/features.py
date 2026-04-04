"""
features.py — Feature engineering for the clinical trial announcement classifier.
Verbatim replication of extract_structured() and keyword lists from ann_v10_new_data.ipynb Cell 8.
"""

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from pipeline.ml_config import SF_COLS


# ── Keyword lists (copied verbatim from ann_v10_new_data.ipynb) ───────────────

POSITIVE_ENDPOINT = [
    "met primary", "met the primary", "achieved primary", "statistically significant",
    "p < 0.05", "p<0.05", "significant improvement", "significant reduction",
    "significant benefit", "demonstrated efficacy", "showed significant",
    "positive results", "met its primary",
]

NEGATIVE_ENDPOINT = [
    "failed to meet", "did not meet", "no significant difference", "not significant",
    "p > 0.05", "p>0.05", "missed primary", "did not achieve",
    "no statistically significant", "failed to demonstrate", "not statistically significant",
]

MECHANISM_KEYWORDS = [
    "antibody", "monoclonal", "inhibitor", "car-t", "car t", "gene therapy",
    "cell therapy", "checkpoint", "kinase", "receptor", "agonist", "antagonist",
    "bispecific", "adc", "antibody-drug conjugate", "sirna", "mrna", "protein degrader",
]

DISEASE_KEYWORDS = [
    "cancer", "tumor", "tumour", "carcinoma", "lymphoma", "leukemia", "melanoma",
    "rare disease", "orphan", "autoimmune", "cardiovascular", "neurological",
    "diabetes", "infectious", "respiratory",
]


# ── Core feature extraction (copied verbatim from ann_v10_new_data.ipynb) ─────

def extract_structured(text: Any) -> dict[str, int | float]:
    """Extract 14 structured clinical features from raw announcement text.

    Takes a raw text string (or any value) and returns a dict keyed by SF_COLS.
    Returns all-zero dict for non-string or empty input. Copied verbatim from notebook.
    """
    if not isinstance(text, str) or not text:
        return {k: 0 for k in SF_COLS}
    tl = text.lower()
    phase_map = {"1": 1, "i": 1, "2": 2, "ii": 2, "3": 3, "iii": 3, "4": 4, "iv": 4}
    pm = re.search(r"phase\s+(iv|iii|ii|i|4|3|2|1)\b", tl)
    sf_phase = phase_map.get(pm.group(1), 0) if pm else 0
    nm = (
        re.search(r"n\s*=\s*(\d+)", tl)
        or re.search(r"(\d+)\s+patients", tl)
        or re.search(r"enrolled\s+(\d+)", tl)
        or re.search(r"(\d+)\s+participants", tl)
    )
    sf_n_patients = min(int(nm.group(1)), 5000) if nm else 0
    sf_pos = sum(1 for kw in POSITIVE_ENDPOINT if kw in tl)
    sf_neg = sum(1 for kw in NEGATIVE_ENDPOINT if kw in tl)
    return dict(
        sf_phase=sf_phase,
        sf_n_patients=sf_n_patients,
        sf_pos_endpoint=sf_pos,
        sf_neg_endpoint=sf_neg,
        sf_net_endpoint=sf_pos - sf_neg,
        sf_mech_hits=sum(1 for kw in MECHANISM_KEYWORDS if kw in tl),
        sf_disease_hits=sum(1 for kw in DISEASE_KEYWORDS if kw in tl),
        sf_is_oncology=int(
            any(kw in tl for kw in ["cancer", "tumor", "tumour", "oncology",
                                     "leukemia", "lymphoma", "melanoma", "carcinoma"])
        ),
        sf_is_rare=int(any(kw in tl for kw in ["rare disease", "orphan", "rare"])),
        sf_is_phase3=int(sf_phase == 3),
        sf_is_phase2=int(sf_phase == 2),
        sf_has_approval=int(
            any(kw in tl for kw in ["approval", "approved", "breakthrough therapy",
                                     "accelerated approval"])
        ),
        sf_has_nct=int(bool(re.search(r"nct\d{8}", tl))),
        sf_word_count=len(tl.split()),
    )


def preprocess_for_embedding(text: Any) -> str:
    """Clean raw announcement text for embedding: strip section dividers and normalize whitespace.

    Takes any value and returns a cleaned string (empty string for non-string input).
    Copied verbatim from ann_v10_new_data.ipynb.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"={3,}.*?={3,}", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ── Structured feature matrix ─────────────────────────────────────────────────

def build_structured_features(df: pd.DataFrame) -> np.ndarray:
    """Apply extract_structured() to every row in df and return a float32 array.

    Takes a DataFrame with a 'raw_text' column. Returns a (len(df), 14) float32 ndarray
    with columns in SF_COLS order.
    """
    records = df["raw_text"].apply(extract_structured)
    sf_df = pd.DataFrame(list(records), columns=SF_COLS)
    return sf_df.values.astype(np.float32)


# ── Global OHE (fitted once on full dataset) ──────────────────────────────────

def fit_ohe(df: pd.DataFrame) -> OneHotEncoder:
    """Fit a OneHotEncoder on the entire announcements dataset's source and event_type columns.

    Takes the full (filtered) announcements DataFrame. Returns a fitted OneHotEncoder.
    Must be called once and reused for all rolling windows to prevent category mismatch.
    """
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    ohe.fit(df[["source", "event_type"]])
    return ohe


def transform_ohe(ohe: OneHotEncoder, df: pd.DataFrame) -> np.ndarray:
    """Apply a pre-fitted OHE to a DataFrame's source and event_type columns.

    Takes a fitted OneHotEncoder and a DataFrame with 'source' and 'event_type' columns.
    Returns a float32 array of shape (len(df), n_categories).
    """
    return ohe.transform(df[["source", "event_type"]]).astype(np.float32)


# ── Full feature matrix ───────────────────────────────────────────────────────

def build_feature_matrix(
    ohe: OneHotEncoder,
    df: pd.DataFrame,
) -> np.ndarray:
    """Build the D_cat_sf feature matrix: OHE(source, event_type) + 14 structured features.

    Takes a fitted OneHotEncoder and a DataFrame with 'source', 'event_type', and 'raw_text'.
    Returns a float32 array of shape (len(df), n_ohe_dims + 14).
    """
    cat_features = transform_ohe(ohe, df)
    sf_features = build_structured_features(df)
    return np.hstack([cat_features, sf_features]).astype(np.float32)


# ── Label construction ────────────────────────────────────────────────────────

def build_labels(df: pd.DataFrame, horizon: str, threshold: float) -> np.ndarray:
    """Build binary classification labels using a fixed return threshold.

    Takes a DataFrame with return_5d or return_30d column, a horizon ('5d' or '30d'),
    and the fixed P85 threshold. Fills nulls with 0. Returns int array of 0s and 1s.
    """
    if horizon not in ("5d", "30d"):
        raise ValueError(f"horizon must be '5d' or '30d', got '{horizon}'")
    col = f"return_{horizon}"
    if col not in df.columns:
        raise ValueError(f"DataFrame missing column '{col}'")
    returns = df[col].fillna(0.0)
    return (returns >= threshold).astype(int).values
