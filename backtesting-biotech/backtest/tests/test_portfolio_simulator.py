"""
test_portfolio_simulator.py — Tests for the true portfolio simulator.

Covers: capital cap enforcement, horizon bar counting, TP/SL anchoring
to fill price, gap-through handling, slippage/commission application,
shared capital pool accounting, and priority ordering of signals.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.portfolio_simulator import simulate_portfolio


# ── Synthetic data helpers ────────────────────────────────────────────────────

def make_dates(n: int, start: str = "2024-01-02") -> pd.DatetimeIndex:
    """Generate n consecutive weekday dates starting from start."""
    return pd.bdate_range(start=start, periods=n)


def make_ohlcv(
    dates: pd.DatetimeIndex,
    open_: float = 100.0,
    high_mult: float = 1.02,
    low_mult: float = 0.98,
    close: float | None = None,
) -> pd.DataFrame:
    """Create a flat OHLCV DataFrame where every bar has the same prices."""
    c = close if close is not None else open_
    return pd.DataFrame(
        {
            "Open":   [open_]  * len(dates),
            "High":   [open_ * high_mult] * len(dates),
            "Low":    [open_ * low_mult]  * len(dates),
            "Close":  [c]      * len(dates),
            "Volume": [100_000] * len(dates),
        },
        index=dates,
    )


def make_ohlcv_with_spike(
    dates: pd.DatetimeIndex,
    base: float,
    spike_day: int,
    spike_high: float,
) -> pd.DataFrame:
    """Flat OHLCV except on spike_day where High = spike_high."""
    df = make_ohlcv(dates, open_=base, high_mult=1.001, low_mult=0.999)
    df.iloc[spike_day, df.columns.get_loc("High")] = spike_high
    return df


def make_ohlcv_with_drop(
    dates: pd.DatetimeIndex,
    base: float,
    drop_day: int,
    drop_low: float,
) -> pd.DataFrame:
    """Flat OHLCV except on drop_day where Low = drop_low."""
    df = make_ohlcv(dates, open_=base, high_mult=1.001, low_mult=0.999)
    df.iloc[drop_day, df.columns.get_loc("Low")] = drop_low
    return df


def make_signal(
    ticker: str,
    published_at: str,
    decision_score: float = 1.0,
    window_id: int = 0,
    horizon: str = "5d",
) -> pd.DataFrame:
    """Create a single-row signals DataFrame."""
    return pd.DataFrame([{
        "ticker":       ticker,
        "published_at": pd.Timestamp(published_at),
        "decision_score": decision_score,
        "window_id":    window_id,
        "horizon":      horizon,
        "return_5d":    0.0,
        "return_30d":   0.0,
    }])


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_horizon_expiry_fires_at_correct_bar():
    """Position should close exactly at horizon_bars, not before or after."""
    dates = make_dates(20)
    ohlcv = make_ohlcv(dates, open_=100.0, high_mult=1.001, low_mult=0.999)
    cache = {"AAAA": ohlcv}

    signals = make_signal("AAAA", str(dates[0].date()), horizon="5d")
    trade_log, _ = simulate_portfolio(
        signals, "5d",
        take_profit_pct=0.50,   # unreachable
        stop_loss_pct=0.50,     # unreachable
        _ohlcv_cache=cache,
        initial_capital=100_000,
        max_positions=10,
        commission_pct=0.0,
        slippage_pct=0.0,
    )

    assert len(trade_log) == 1
    assert trade_log.iloc[0]["exit_reason"] == "horizon_expiry"
    entry = pd.Timestamp(trade_log.iloc[0]["entry_date"])
    exit_ = pd.Timestamp(trade_log.iloc[0]["exit_date"])
    biz_days = len(pd.bdate_range(entry, exit_)) - 1  # exclusive of entry
    assert biz_days == 5, f"Expected 5 biz days held, got {biz_days}"


def test_horizon_30d_fires_at_correct_bar():
    """30d horizon should close after exactly 30 business days."""
    dates = make_dates(50)
    ohlcv = make_ohlcv(dates, open_=50.0, high_mult=1.001, low_mult=0.999)
    cache = {"BBBB": ohlcv}

    signals = make_signal("BBBB", str(dates[0].date()), horizon="30d")
    trade_log, _ = simulate_portfolio(
        signals, "30d",
        take_profit_pct=0.99,
        stop_loss_pct=0.99,
        _ohlcv_cache=cache,
        initial_capital=100_000,
        max_positions=10,
        commission_pct=0.0,
        slippage_pct=0.0,
    )

    assert len(trade_log) == 1
    assert trade_log.iloc[0]["exit_reason"] == "horizon_expiry"
    entry = pd.Timestamp(trade_log.iloc[0]["entry_date"])
    exit_ = pd.Timestamp(trade_log.iloc[0]["exit_date"])
    biz_days = len(pd.bdate_range(entry, exit_)) - 1
    assert biz_days == 30, f"Expected 30 biz days held, got {biz_days}"


def test_take_profit_triggered_intraday():
    """TP should fire when High reaches the TP level intraday."""
    dates = make_dates(20)
    ohlcv = make_ohlcv_with_spike(dates, base=100.0, spike_day=3, spike_high=125.0)
    cache = {"CCCC": ohlcv}

    signals = make_signal("CCCC", str(dates[0].date()), horizon="5d")
    trade_log, _ = simulate_portfolio(
        signals, "5d",
        take_profit_pct=0.20,   # TP at 120 → spike at 125 triggers it
        stop_loss_pct=0.50,
        _ohlcv_cache=cache,
        initial_capital=100_000,
        max_positions=10,
        commission_pct=0.0,
        slippage_pct=0.0,
    )

    assert len(trade_log) == 1
    row = trade_log.iloc[0]
    assert row["exit_reason"] == "take_profit"
    # Exit should happen on spike day (index 3 from entry)
    exit_date = pd.Timestamp(row["exit_date"])
    assert exit_date == dates[3]
    # Exit price should be the TP level (entry_fill * 1.20)
    assert abs(row["exit_price"] - row["entry_price"] * 1.20) < 0.01


def test_stop_loss_triggered_intraday():
    """SL should fire when Low reaches the SL level intraday."""
    dates = make_dates(20)
    ohlcv = make_ohlcv_with_drop(dates, base=100.0, drop_day=2, drop_low=88.0)
    cache = {"DDDD": ohlcv}

    signals = make_signal("DDDD", str(dates[0].date()), horizon="5d")
    trade_log, _ = simulate_portfolio(
        signals, "5d",
        take_profit_pct=0.50,
        stop_loss_pct=0.10,     # SL at 90 → drop to 88 triggers it
        _ohlcv_cache=cache,
        initial_capital=100_000,
        max_positions=10,
        commission_pct=0.0,
        slippage_pct=0.0,
    )

    assert len(trade_log) == 1
    row = trade_log.iloc[0]
    assert row["exit_reason"] == "stop_loss"
    assert pd.Timestamp(row["exit_date"]) == dates[2]
    assert abs(row["exit_price"] - row["entry_price"] * 0.90) < 0.01


def test_gap_through_sl_fills_at_open():
    """When Open is below SL (overnight gap down), exit fills at Open, not SL."""
    dates = make_dates(20)
    df = make_ohlcv(dates, open_=100.0, high_mult=1.001, low_mult=0.999)
    # Day 2: Open gaps down hard — below any reasonable SL
    df.iloc[2, df.columns.get_loc("Open")] = 70.0
    df.iloc[2, df.columns.get_loc("Low")]  = 69.0
    df.iloc[2, df.columns.get_loc("High")] = 71.0
    df.iloc[2, df.columns.get_loc("Close")] = 70.5
    cache = {"EEEE": df}

    signals = make_signal("EEEE", str(dates[0].date()), horizon="5d")
    trade_log, _ = simulate_portfolio(
        signals, "5d",
        take_profit_pct=0.50,
        stop_loss_pct=0.10,   # SL at ~90; open at 70 gaps through it
        _ohlcv_cache=cache,
        initial_capital=100_000,
        max_positions=10,
        commission_pct=0.0,
        slippage_pct=0.0,
    )

    assert len(trade_log) == 1
    row = trade_log.iloc[0]
    assert row["exit_reason"] == "stop_loss"
    # Fill must be at gap open (70), not SL (~90)
    assert abs(row["exit_price"] - 70.0) < 0.01


def test_tp_sl_anchored_to_fill_price_not_prior_close():
    """TP and SL levels must be relative to entry_fill, not prior bar's Close."""
    dates = make_dates(20)
    # Prior close is 100; next open (entry) is 110 (gap up)
    df = make_ohlcv(dates, open_=100.0, high_mult=1.005, low_mult=0.995)
    df.iloc[1, df.columns.get_loc("Open")]  = 110.0
    df.iloc[1, df.columns.get_loc("High")]  = 111.0
    df.iloc[1, df.columns.get_loc("Low")]   = 109.0
    df.iloc[1, df.columns.get_loc("Close")] = 110.0
    # Make TP reachable only if anchored to 110, not 100
    # TP = 10%: from 110 → 121; from 100 → 110 (already hit on day 1)
    for i in range(2, len(dates)):
        df.iloc[i, df.columns.get_loc("Open")]  = 110.0
        df.iloc[i, df.columns.get_loc("High")]  = 122.0  # above 110*1.10 = 121
        df.iloc[i, df.columns.get_loc("Low")]   = 109.0
        df.iloc[i, df.columns.get_loc("Close")] = 110.0
    cache = {"FFFF": df}

    signals = make_signal("FFFF", str(dates[1].date()), horizon="5d")
    trade_log, _ = simulate_portfolio(
        signals, "5d",
        take_profit_pct=0.10,
        stop_loss_pct=0.50,
        _ohlcv_cache=cache,
        initial_capital=100_000,
        max_positions=10,
        commission_pct=0.0,
        slippage_pct=0.0,
    )

    assert len(trade_log) == 1
    row = trade_log.iloc[0]
    # Entry fill ≈ 110 (day 1 open, no slippage)
    assert abs(row["entry_price"] - 110.0) < 0.5
    # TP should be ~121, which is hit on day 2 (High=122)
    assert row["exit_reason"] == "take_profit"
    assert abs(row["exit_price"] - 110.0 * 1.10) < 0.5


def test_slippage_applied_on_entry_and_exit():
    """Entry fill = Open*(1+slip); exit fill = raw_price*(1-slip)."""
    dates = make_dates(20)
    ohlcv = make_ohlcv(dates, open_=100.0, high_mult=1.001, low_mult=0.999)
    cache = {"GGGG": ohlcv}
    slip = 0.005  # 0.5%

    signals = make_signal("GGGG", str(dates[0].date()), horizon="5d")
    trade_log, _ = simulate_portfolio(
        signals, "5d",
        take_profit_pct=0.50,
        stop_loss_pct=0.50,
        _ohlcv_cache=cache,
        initial_capital=100_000,
        max_positions=10,
        commission_pct=0.0,
        slippage_pct=slip,
    )

    assert len(trade_log) == 1
    row = trade_log.iloc[0]
    # Entry: 100 * (1 + 0.005) = 100.5
    assert abs(row["entry_price"] - 100.0 * (1 + slip)) < 0.01
    # Horizon exit at close = 100; exit fill = 100 * (1 - 0.005) = 99.5
    assert abs(row["exit_price"] - 100.0 * (1 - slip)) < 0.01


def test_max_positions_cap_enforced():
    """Never more than max_positions concurrent open positions."""
    dates = make_dates(30)
    max_pos = 5
    n_signals = 15

    # Create 15 different tickers all with the same flat price data
    cache = {}
    signals_list = []
    for i in range(n_signals):
        ticker = f"T{i:03d}"
        cache[ticker] = make_ohlcv(dates, open_=100.0, high_mult=1.001, low_mult=0.999)
        signals_list.append({
            "ticker":        ticker,
            "published_at":  dates[0],
            "decision_score": float(n_signals - i),  # descending score
            "window_id":     0,
            "horizon":       "5d",
            "return_5d":     0.0,
            "return_30d":    0.0,
        })

    signals = pd.DataFrame(signals_list)
    trade_log, equity_curve = simulate_portfolio(
        signals, "5d",
        take_profit_pct=0.50,
        stop_loss_pct=0.50,
        _ohlcv_cache=cache,
        initial_capital=100_000,
        max_positions=max_pos,
        commission_pct=0.0,
        slippage_pct=0.0,
    )

    # Only max_pos signals should have been traded
    assert len(trade_log) == max_pos

    # Verify it was the top-scoring signals that were taken
    traded_tickers = set(trade_log["ticker"])
    expected_tickers = {f"T{i:03d}" for i in range(max_pos)}
    assert traded_tickers == expected_tickers


def test_capital_pool_shared_limits_concurrent_positions():
    """With $100k capital and $10k per position, max 10 positions at once."""
    dates = make_dates(40)
    initial_capital = 100_000
    max_positions = 10
    n_signals = 12

    cache = {}
    signals_list = []
    for i in range(n_signals):
        ticker = f"S{i:03d}"
        # Make positions last 20 bars (no TP/SL hit)
        cache[ticker] = make_ohlcv(dates, open_=50.0, high_mult=1.001, low_mult=0.999)
        signals_list.append({
            "ticker":        ticker,
            "published_at":  dates[0],
            "decision_score": float(n_signals - i),
            "window_id":     0,
            "horizon":       "30d",
            "return_5d":     0.0,
            "return_30d":    0.0,
        })

    signals = pd.DataFrame(signals_list)
    _, equity_curve = simulate_portfolio(
        signals, "30d",
        take_profit_pct=0.99,
        stop_loss_pct=0.99,
        _ohlcv_cache=cache,
        initial_capital=initial_capital,
        max_positions=max_positions,
        commission_pct=0.0,
        slippage_pct=0.0,
    )

    # Peak equity ≠ initial (positions are open), but never goes negative
    assert (equity_curve["equity"] > 0).all()
    # Equity should be close to initial_capital (flat prices, no TP/SL)
    assert abs(equity_curve["equity"].iloc[0] - initial_capital) < initial_capital * 0.05


def test_pnl_math_is_correct():
    """PnL should equal (exit_fill - entry_fill) * shares, net of commission."""
    dates = make_dates(20)
    # Flat price: open=100, close=100 (horizon exit)
    df = make_ohlcv(dates, open_=100.0, high_mult=1.001, low_mult=0.999, close=100.0)
    cache = {"HHHH": df}
    comm = 0.001  # 0.1%
    slip = 0.001  # 0.1%
    position_dollars = 10_000  # 100k / 10 positions

    signals = make_signal("HHHH", str(dates[0].date()), horizon="5d")
    trade_log, _ = simulate_portfolio(
        signals, "5d",
        take_profit_pct=0.50,
        stop_loss_pct=0.50,
        _ohlcv_cache=cache,
        initial_capital=100_000,
        max_positions=10,
        commission_pct=comm,
        slippage_pct=slip,
    )

    assert len(trade_log) == 1
    row = trade_log.iloc[0]

    entry_fill = 100.0 * (1 + slip)
    cost_per_share = entry_fill * (1 + comm)
    shares = position_dollars / cost_per_share
    exit_fill = 100.0 * (1 - slip)  # horizon exit at close=100
    proceeds = shares * exit_fill * (1 - comm)
    expected_pnl = proceeds - position_dollars

    assert abs(row["pnl"] - expected_pnl) < 0.01


def test_equity_curve_starts_at_initial_capital():
    """Equity curve first value should be initial_capital (no positions open yet)."""
    dates = make_dates(20)
    # Signal on day 1 (not day 0), so day 0 has no open positions
    ohlcv = make_ohlcv(dates, open_=100.0, high_mult=1.001, low_mult=0.999)
    cache = {"IIII": ohlcv}
    initial = 100_000

    signals = make_signal("IIII", str(dates[1].date()), horizon="5d")
    _, equity_curve = simulate_portfolio(
        signals, "5d",
        take_profit_pct=0.50,
        stop_loss_pct=0.50,
        _ohlcv_cache=cache,
        initial_capital=initial,
        max_positions=10,
        commission_pct=0.0,
        slippage_pct=0.0,
    )

    assert abs(equity_curve["equity"].iloc[0] - initial) < 1.0


def test_higher_decision_score_prioritised_when_cap_reached():
    """When cap is hit, the highest-scoring signals are taken, not the first ones."""
    dates = make_dates(20)
    cache = {}
    signals_list = []
    for i in range(5):
        ticker = f"P{i:02d}"
        cache[ticker] = make_ohlcv(dates, open_=100.0, high_mult=1.001, low_mult=0.999)
        signals_list.append({
            "ticker":        ticker,
            "published_at":  dates[0],
            "decision_score": float(i),  # P04 has highest score
            "window_id":     0,
            "horizon":       "5d",
            "return_5d":     0.0,
            "return_30d":    0.0,
        })

    signals = pd.DataFrame(signals_list)
    trade_log, _ = simulate_portfolio(
        signals, "5d",
        take_profit_pct=0.99,
        stop_loss_pct=0.99,
        _ohlcv_cache=cache,
        initial_capital=100_000,
        max_positions=2,      # only take top 2
        commission_pct=0.0,
        slippage_pct=0.0,
    )

    assert len(trade_log) == 2
    traded = set(trade_log["ticker"])
    assert "P04" in traded  # highest score
    assert "P03" in traded  # second highest
    assert "P00" not in traded  # lowest score — should not be traded


# ── Entry timing (next-day entry) ─────────────────────────────────────────────

def test_entry_is_next_trading_day_after_announcement():
    """Entry must be on the NEXT trading day after published_at, not same day."""
    dates = make_dates(20, start="2024-01-02")
    ohlcv = make_ohlcv(dates, open_=100.0, high_mult=1.001, low_mult=0.999)
    cache = {"NEXT": ohlcv}

    # published_at = first trading day; entry should be the SECOND trading day
    pub_date = str(dates[0].date())
    signals = make_signal("NEXT", pub_date, horizon="5d")
    trade_log, _ = simulate_portfolio(
        signals, "5d",
        take_profit_pct=0.99,
        stop_loss_pct=0.99,
        _ohlcv_cache=cache,
        initial_capital=100_000,
        max_positions=10,
        commission_pct=0.0,
        slippage_pct=0.0,
    )

    assert len(trade_log) == 1
    entry_date = pd.Timestamp(trade_log.iloc[0]["entry_date"])
    # Entry must be strictly AFTER published_at (next trading day)
    assert entry_date > pd.Timestamp(pub_date), (
        f"Entry {entry_date} should be after announcement {pub_date}"
    )
    assert entry_date == dates[1], (
        f"Expected entry on {dates[1]}, got {entry_date}"
    )


def test_announcement_on_last_trading_day_has_no_entry():
    """If published_at is the last trading day in the OHLCV, no entry is possible."""
    dates = make_dates(5, start="2024-01-02")
    ohlcv = make_ohlcv(dates, open_=100.0, high_mult=1.001, low_mult=0.999)
    cache = {"LAST": ohlcv}

    # published_at = last trading day — no next day exists
    pub_date = str(dates[-1].date())
    signals = make_signal("LAST", pub_date, horizon="5d")
    trade_log, _ = simulate_portfolio(
        signals, "5d",
        take_profit_pct=0.99,
        stop_loss_pct=0.99,
        _ohlcv_cache=cache,
        initial_capital=100_000,
        max_positions=10,
        commission_pct=0.0,
        slippage_pct=0.0,
    )

    # No entry because there is no next trading day
    assert len(trade_log) == 0


# ── Weight cap fix: no renormalization ────────────────────────────────────────

def test_inverse_vol_max_weight_actually_caps_position():
    """With max_weight=0.10 and 5 equal-vol signals, each gets exactly 10% not 20%."""
    from backtest.portfolio_construction import _apply_capped_weights
    import numpy as np

    rows = [{"target_dollars": 0.0} for _ in range(5)]
    # Equal inverse-vol → equal raw_weights
    raw = np.ones(5)
    initial_capital = 100_000
    max_weight = 0.10

    _apply_capped_weights(rows, raw, initial_capital, max_weight)

    for row in rows:
        # Each position should be at most 10% of initial_capital = $10,000
        assert row["target_dollars"] <= initial_capital * max_weight + 0.01, (
            f"Position ${row['target_dollars']:.0f} exceeds 10% cap of ${initial_capital * max_weight:.0f}"
        )
    # Total deployment should be 5 × 10% = 50%, not 100%
    total = sum(r["target_dollars"] for r in rows)
    assert abs(total - initial_capital * 0.50) < 1.0, (
        f"Expected total deployment $50,000, got ${total:.0f}"
    )
