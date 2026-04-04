"""
config.py — All constants, file paths, and hyperparameters for the backtester.
Everything configurable lives here. Import this module; never hardcode values elsewhere.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent

ANNOUNCEMENTS_PATH = ROOT_DIR / "announcements2.parquet"
OHLCV_DIR          = ROOT_DIR / "data" / "ohlcv"
BENCHMARK_PATH     = ROOT_DIR / "data" / "ohlcv" / "SPY.parquet"
OUTPUT_DIR         = ROOT_DIR / "output"

# ── Label thresholds (computed once from full filtered dataset) ────────────────
# These are fixed constants — NOT recomputed per rolling window.
# Source: announcements2.parquet, after dropping EDGAR + parse_status != 'OK'
P85_5D  = 2.224   # 85th percentile of return_5d  across full dataset
P85_30D = 7.2225  # 85th percentile of return_30d across full dataset

# ── Rolling window hyperparameters ────────────────────────────────────────────
TRAIN_MONTHS     = 9    # 9-month training window (validated: Sharpe 2.28 vs 1.57 at 12m)
PREDICT_WEEKS    = 4    # length of prediction/test window in weeks
STEP_WEEKS       = 4    # how far the window slides each iteration
START_DATE       = "2024-01-01"   # first day of first prediction window
END_DATE         = "2025-12-31"   # last day of backtesting period
MIN_TRAIN_SAMPLES = 30  # minimum announcements to train; skip window if below

# ── Model hyperparameters ─────────────────────────────────────────────────────
RANDOM_STATE     = 42
GBM_N_ESTIMATORS = 200
GBM_MAX_DEPTH    = 4
GBM_LEARNING_RATE = 0.05
DROP_FEATURES    = ["sf_word_count"]  # features excluded from model training

# ── Position sizing & capital ─────────────────────────────────────────────────
INITIAL_CAPITAL      = 1_000_000.0
MAX_OPEN_POSITIONS   = 20   # accounts for ~1.5x window overlap (30d hold, 4wk step)
MAX_WEIGHT           = 0.07  # max 7% of capital per position (inverse-vol sizing)
VOL_WINDOW           = 60   # lookback for inverse-volatility weighting
COMMISSION_PCT       = 0.001   # 0.1% per trade
SLIPPAGE_PCT         = 0.001   # 0.1% slippage

# ── Default TP/SL per horizon ─────────────────────────────────────────────────
TP_SL_CONFIG = {
    "5d":  {"take_profit": 0.08, "stop_loss": 0.04, "horizon_bars": 30},
    "30d": {"take_profit": 0.30, "stop_loss": 1.00, "horizon_bars": 50},  # TP 30%, no SL, 50-bar horizon
}

# ── Realistic execution friction ─────────────────────────────────────────────
TIERED_SLIPPAGE    = True    # price-dependent slippage: <$2 → 5%, $2-$5 → 2%, ≥$5 → 0.1%
ADV_CAP_PCT        = 0.05    # cap each position at 5% of 20-day avg daily dollar volume

# ── Sweep grid ────────────────────────────────────────────────────────────────
TP_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
SL_GRID = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25]

# ── Structured feature columns (order must match extract_structured output) ───
SF_COLS = [
    "sf_phase", "sf_n_patients", "sf_pos_endpoint", "sf_neg_endpoint", "sf_net_endpoint",
    "sf_mech_hits", "sf_disease_hits", "sf_is_oncology", "sf_is_rare",
    "sf_is_phase3", "sf_is_phase2", "sf_has_approval", "sf_has_nct", "sf_word_count",
]

# ── Data filtering ────────────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "ticker", "source", "event_type", "published_at", "raw_text",
    "return_5d", "return_30d",
]
