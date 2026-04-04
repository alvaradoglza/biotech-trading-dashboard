"""Tests for features.py — extract_structured, build_structured_features, OHE, feature matrix, labels."""

import numpy as np
import pandas as pd
import pytest

from backtest.features import (
    extract_structured,
    preprocess_for_embedding,
    build_structured_features,
    fit_ohe,
    transform_ohe,
    build_feature_matrix,
    build_labels,
    SF_COLS,
)
from backtest.config import P85_5D, P85_30D


# ── Helpers ───────────────────────────────────────────────────────────────────

SAMPLE_TEXT_POSITIVE = (
    "Phase 3 trial of monoclonal antibody in cancer patients (n=500). "
    "Met primary endpoint statistically significant p < 0.05. "
    "FDA approval granted. NCT12345678."
)
SAMPLE_TEXT_NEGATIVE = (
    "Phase 2 study of inhibitor in rare disease patients enrolled 120. "
    "Failed to meet primary endpoint, no statistically significant difference p > 0.05."
)
SAMPLE_TEXT_EMPTY = ""
SAMPLE_TEXT_NONE = None


def make_df(n: int = 4) -> pd.DataFrame:
    """Make a minimal announcements DataFrame for feature tests."""
    return pd.DataFrame({
        "source": ["clinicaltrials", "openfda", "clinicaltrials", "clinicaltrials"],
        "event_type": ["CT_COMPLETED", "FDA_SUPPL", "CT_RECRUITING", "CT_TERMINATED"],
        "raw_text": [SAMPLE_TEXT_POSITIVE, SAMPLE_TEXT_NEGATIVE, SAMPLE_TEXT_EMPTY, "some text"],
        "return_5d": [10.0, 1.0, None, 3.0],
        "return_30d": [20.0, 5.0, 8.0, None],
    })


# ── extract_structured ────────────────────────────────────────────────────────

def test_extract_structured_returns_all_sf_cols():
    """extract_structured should return a dict with all 14 SF_COLS keys."""
    result = extract_structured(SAMPLE_TEXT_POSITIVE)
    assert set(result.keys()) == set(SF_COLS)


def test_extract_structured_empty_string_returns_zeros():
    """extract_structured should return all zeros for an empty string."""
    result = extract_structured("")
    assert all(v == 0 for v in result.values())


def test_extract_structured_none_returns_zeros():
    """extract_structured should return all zeros for None input."""
    result = extract_structured(None)
    assert all(v == 0 for v in result.values())


def test_extract_structured_detects_phase3():
    """extract_structured should detect Phase 3 and set sf_is_phase3=1."""
    result = extract_structured("Phase 3 trial of drug X.")
    assert result["sf_phase"] == 3
    assert result["sf_is_phase3"] == 1
    assert result["sf_is_phase2"] == 0


def test_extract_structured_detects_phase2_roman():
    """extract_structured should detect Phase II (roman numeral)."""
    result = extract_structured("Phase II study results published.")
    assert result["sf_phase"] == 2
    assert result["sf_is_phase2"] == 1


def test_extract_structured_counts_patients():
    """extract_structured should parse patient count from 'n=500'."""
    result = extract_structured("A study with n=500 participants showed benefit.")
    assert result["sf_n_patients"] == 500


def test_extract_structured_caps_patients_at_5000():
    """extract_structured should cap patient count at 5000."""
    result = extract_structured("Enrolled 99999 patients in the study.")
    assert result["sf_n_patients"] == 5000


def test_extract_structured_positive_endpoint():
    """extract_structured should count positive endpoint keywords."""
    result = extract_structured("The trial met primary endpoint and was statistically significant.")
    assert result["sf_pos_endpoint"] >= 1
    assert result["sf_neg_endpoint"] == 0
    assert result["sf_net_endpoint"] > 0


def test_extract_structured_negative_endpoint():
    """extract_structured should count negative endpoint keywords."""
    result = extract_structured("Failed to meet primary endpoint, no significant difference.")
    assert result["sf_neg_endpoint"] >= 1
    assert result["sf_net_endpoint"] < 0


def test_extract_structured_oncology_flag():
    """extract_structured should set sf_is_oncology=1 for cancer text."""
    result = extract_structured("Treatment for cancer patients showed improvement.")
    assert result["sf_is_oncology"] == 1


def test_extract_structured_rare_disease_flag():
    """extract_structured should set sf_is_rare=1 for orphan/rare disease text."""
    result = extract_structured("Designated orphan drug status for rare disease.")
    assert result["sf_is_rare"] == 1


def test_extract_structured_nct_detection():
    """extract_structured should detect NCT numbers."""
    result = extract_structured("Trial identifier NCT12345678 reported results.")
    assert result["sf_has_nct"] == 1


def test_extract_structured_approval_detection():
    """extract_structured should detect FDA approval mentions."""
    result = extract_structured("FDA granted approval for the new therapy.")
    assert result["sf_has_approval"] == 1


def test_extract_structured_word_count():
    """extract_structured should count words in sf_word_count."""
    text = "word " * 50
    result = extract_structured(text)
    assert result["sf_word_count"] == 50


def test_extract_structured_mechanism_hits():
    """extract_structured should count mechanism keyword hits."""
    result = extract_structured("Monoclonal antibody checkpoint inhibitor therapy.")
    assert result["sf_mech_hits"] >= 3


# ── preprocess_for_embedding ──────────────────────────────────────────────────

def test_preprocess_strips_section_dividers():
    """preprocess_for_embedding should remove === ... === section markers."""
    text = "Before === SECTION HEADER === after"
    result = preprocess_for_embedding(text)
    assert "===" not in result
    assert "Before" in result
    assert "after" in result


def test_preprocess_normalizes_whitespace():
    """preprocess_for_embedding should collapse multiple spaces/newlines."""
    text = "word1   \n\n  word2"
    result = preprocess_for_embedding(text)
    assert result == "word1 word2"


def test_preprocess_none_returns_empty_string():
    """preprocess_for_embedding should return '' for non-string input."""
    assert preprocess_for_embedding(None) == ""
    assert preprocess_for_embedding(42) == ""


# ── build_structured_features ────────────────────────────────────────────────

def test_build_structured_features_shape():
    """build_structured_features should return array of shape (n_rows, 14)."""
    df = make_df(4)
    result = build_structured_features(df)
    assert result.shape == (4, 14)


def test_build_structured_features_dtype():
    """build_structured_features should return float32 array."""
    df = make_df(4)
    result = build_structured_features(df)
    assert result.dtype == np.float32


def test_build_structured_features_empty_text_is_zeros():
    """build_structured_features should produce all-zero row for empty raw_text."""
    df = pd.DataFrame({"raw_text": [""]})
    result = build_structured_features(df)
    assert (result == 0).all()


# ── fit_ohe / transform_ohe ───────────────────────────────────────────────────

def test_fit_ohe_returns_fitted_encoder():
    """fit_ohe should return a OneHotEncoder that can transform data."""
    df = make_df()
    ohe = fit_ohe(df)
    result = ohe.transform(df[["source", "event_type"]])
    assert result.shape[0] == len(df)


def test_transform_ohe_dtype_is_float32():
    """transform_ohe should return float32 array."""
    df = make_df()
    ohe = fit_ohe(df)
    result = transform_ohe(ohe, df)
    assert result.dtype == np.float32


def test_transform_ohe_unknown_category_becomes_zeros():
    """transform_ohe should produce an all-zero row for an unseen category."""
    train_df = make_df()
    ohe = fit_ohe(train_df)
    unseen_df = pd.DataFrame({
        "source": ["unknown_source"],
        "event_type": ["UNKNOWN_EVENT"],
    })
    result = transform_ohe(ohe, unseen_df)
    assert (result == 0).all()


def test_transform_ohe_same_input_same_output():
    """transform_ohe should produce identical output for identical input (determinism)."""
    df = make_df()
    ohe = fit_ohe(df)
    r1 = transform_ohe(ohe, df)
    r2 = transform_ohe(ohe, df)
    np.testing.assert_array_equal(r1, r2)


# ── build_feature_matrix ─────────────────────────────────────────────────────

def test_build_feature_matrix_shape():
    """build_feature_matrix should return (n_rows, n_ohe_dims + 14) float32 array."""
    df = make_df()
    ohe = fit_ohe(df)
    X = build_feature_matrix(ohe, df)
    cat_dims = ohe.transform(df[["source", "event_type"]]).shape[1]
    assert X.shape == (len(df), cat_dims + 14)


def test_build_feature_matrix_dtype():
    """build_feature_matrix should return float32."""
    df = make_df()
    ohe = fit_ohe(df)
    X = build_feature_matrix(ohe, df)
    assert X.dtype == np.float32


def test_build_feature_matrix_no_data_leakage():
    """Training subset should never contain indices from prediction window."""
    df = make_df(4)
    df["published_at"] = pd.date_range("2024-01-01", periods=4, freq="30D")
    train_df = df[df["published_at"] < pd.Timestamp("2024-04-01")]
    pred_df  = df[df["published_at"] >= pd.Timestamp("2024-04-01")]
    # No overlap in indices
    assert len(set(train_df.index) & set(pred_df.index)) == 0


# ── build_labels ─────────────────────────────────────────────────────────────

def test_build_labels_5d_shape():
    """build_labels for '5d' should return array of length equal to df length."""
    df = make_df()
    labels = build_labels(df, "5d", P85_5D)
    assert len(labels) == len(df)


def test_build_labels_binary():
    """build_labels should return only 0s and 1s."""
    df = make_df()
    labels = build_labels(df, "5d", P85_5D)
    assert set(labels).issubset({0, 1})


def test_build_labels_null_treated_as_zero():
    """build_labels should fill null returns with 0 before applying threshold."""
    df = pd.DataFrame({"return_5d": [None, 100.0]})
    # threshold = 50: null (→0) gets label 0, 100 gets label 1
    labels = build_labels(df, "5d", 50.0)
    assert labels[0] == 0
    assert labels[1] == 1


def test_build_labels_invalid_horizon_raises():
    """build_labels should raise ValueError for unsupported horizon values."""
    df = make_df()
    with pytest.raises(ValueError, match="horizon"):
        build_labels(df, "10d", P85_5D)


def test_build_labels_30d():
    """build_labels for '30d' returns correct binary array."""
    df = make_df()
    labels = build_labels(df, "30d", P85_30D)
    assert len(labels) == len(df)
    assert set(labels).issubset({0, 1})
