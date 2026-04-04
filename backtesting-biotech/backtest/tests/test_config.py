"""Tests for config.py — verify constants, paths, and structure."""

from backtest.config import (
    P85_5D, P85_30D, SF_COLS, TP_SL_CONFIG, TP_GRID, SL_GRID,
    TRAIN_MONTHS, PREDICT_WEEKS, STEP_WEEKS, MIN_TRAIN_SAMPLES,
    INITIAL_CAPITAL, MAX_OPEN_POSITIONS,
)


def test_p85_thresholds_are_positive():
    """P85 thresholds should be positive return values."""
    assert P85_5D > 0
    assert P85_30D > 0


def test_p85_30d_greater_than_5d():
    """30d threshold should be higher than 5d threshold."""
    assert P85_30D > P85_5D


def test_sf_cols_has_14_entries():
    """SF_COLS must contain exactly 14 structured feature names."""
    assert len(SF_COLS) == 14


def test_sf_cols_starts_with_sf_prefix():
    """All SF_COLS should start with 'sf_'."""
    assert all(col.startswith("sf_") for col in SF_COLS)


def test_tp_sl_config_has_both_horizons():
    """TP/SL config must define entries for '5d' and '30d'."""
    assert "5d" in TP_SL_CONFIG
    assert "30d" in TP_SL_CONFIG
    for h in ("5d", "30d"):
        assert "take_profit" in TP_SL_CONFIG[h]
        assert "stop_loss" in TP_SL_CONFIG[h]


def test_sweep_grids_are_nonempty():
    """TP and SL grids must be non-empty lists."""
    assert len(TP_GRID) > 0
    assert len(SL_GRID) > 0


def test_rolling_window_params_are_positive():
    """All rolling window parameters must be positive integers."""
    assert TRAIN_MONTHS > 0
    assert PREDICT_WEEKS > 0
    assert STEP_WEEKS > 0
    assert MIN_TRAIN_SAMPLES > 0


def test_initial_capital_positive():
    """Initial capital and max positions must be positive."""
    assert INITIAL_CAPITAL > 0
    assert MAX_OPEN_POSITIONS > 0
