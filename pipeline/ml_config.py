"""
ml_config.py — ML model constants for the daily prediction pipeline.

Extracted from backtesting-biotech/backtest/config.py.
Only contains the constants actually used by pipeline/predict.py and pipeline/features.py.
"""

# ── Label thresholds (85th percentile of full dataset) ────────────────────────
P85_5D  = 2.224    # 85th pct of return_5d
P85_30D = 7.2225   # 85th pct of return_30d

# ── Model hyperparameters ─────────────────────────────────────────────────────
RANDOM_STATE      = 42
GBM_N_ESTIMATORS  = 200
GBM_MAX_DEPTH     = 4
GBM_LEARNING_RATE = 0.05
DROP_FEATURES     = ["sf_word_count"]

# ── Structured feature columns ────────────────────────────────────────────────
SF_COLS = [
    "sf_phase", "sf_n_patients", "sf_pos_endpoint", "sf_neg_endpoint", "sf_net_endpoint",
    "sf_mech_hits", "sf_disease_hits", "sf_is_oncology", "sf_is_rare",
    "sf_is_phase3", "sf_is_phase2", "sf_has_approval", "sf_has_nct", "sf_word_count",
]
