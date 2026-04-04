# Exit Strategy Analysis Report — Post-Audit Revision

**Date**: 2026-03-26 (revised from 2026-03-25)
**Horizon**: 30d | **Capital**: $1,000,000 | **Sizing**: Inverse-volatility (60d, 10% max weight)
**Backtest period**: 2024-01-01 to 2025-12-31 | **Signals**: 433 across 24 rolling windows
**MAX_OPEN_POSITIONS**: 20

---

## Audit Fixes Applied

An external audit identified 8 bugs (Critical/High/Medium/Low). All were fixed before this report.

### Critical fixes
| # | Bug | Fix |
|---|-----|-----|
| C1 | Training used rows whose return window extended past `pred_start` (14.91% label leakage) | Excluded training rows with `published_at >= pred_start - 30 days` |
| C2a | `_apply_capped_weights` renormalized after capping, negating the 10% max_weight | Removed renormalization — uninvested weight stays as cash |
| C2b | `max_positions` cap was only enforced in `equal` sizing mode; bypassed by `weight_fn` | Cap now enforced regardless of sizing mode |
| C3 | 94.5% of trades entered same calendar day as announcement (`>= pub` instead of `> pub`) | Entry now strictly next trading day after `published_at` |

### High fixes
| # | Bug | Fix |
|---|-----|-----|
| H1 | Ticker dedup kept highest-score announcement per ticker (used future scores) | Now keeps earliest (first) announcement per ticker |
| H2 | Stale output files documented old strategies | Removed `comparison.csv`, `results.md`, `project_report.md` |

### Medium / Low fixes
| # | Fix |
|---|-----|
| M2 | `run_strategy_comparison.py` now passes `tiered_slippage` / `adv_cap_pct` |
| L1 | `_compute_exposure` now uses true interval-merge instead of heuristic avg_hold × count |

---

## Pre-Fix vs Post-Fix Results

| Metric | Pre-fix (buggy) | Post-fix (correct) | SPY |
|--------|----------------|-------------------|-----|
| Total return | +54.5% | **+70.2%** | +44.3% |
| Annualized return | +24.6% | +32.4% | +20.1% |
| Sharpe ratio | 0.76 | 0.64 | 1.21 |
| Max drawdown | 41.5% | 43.3% | 19.0% |
| Win rate | 50.4% | 44.5% | — |
| Profit factor | 1.39 | 1.73 | — |
| Total trades | 365 | 247 | 1 |
| Avg hold (cal days) | 38.4 | 38.8 | 729 |

**Headline is misleading.** The +70.2% is driven entirely by 2 GNPX trades ($767K PnL = 109% of total).

---

## Critical Concentration Finding

The strategy generates **-6.5%** ex-GNPX (two trades on one ticker). All other 245 trades lose money in aggregate.

### PnL by price tier

| Tier | Trades | PnL | Win rate | PnL share |
|------|--------|-----|----------|-----------|
| <$2 (penny) | 56 | +$656,847 | 35.7% | 93.5% |
| $2–$5 | 57 | +$23,142 | 50.9% | 3.3% |
| >$5 | 134 | +$22,208 | 45.5% | 3.2% |

### Exit type breakdown

| Exit type | Count | Avg sim return | Avg actual 30d return |
|-----------|-------|---------------|----------------------|
| Take profit | 58 | +113.6% | +21.9% |
| Horizon expiry | 189 | −11.0% | −5.7% |

The 58 TP exits average +113.6% because they include extreme penny-stock gap-throughs (GNPX: $0.35→$12.97 overnight). The 189 horizon expiry trades — **76% of all trades** — average −11%.

### Top trades by PnL

| Ticker | Entry price | Return | PnL | Exit |
|--------|------------|--------|-----|------|
| GNPX | $0.35 | +3,552% | +$767,634 | take_profit |
| KPTI | $0.69 | +1,163% | +$55,747 | take_profit |
| VIR | $7.31 | +34.6% | +$34,596 | take_profit |
| ALDX | $5.09 | +34.6% | +$34,596 | take_profit |
| SPRY | $11.15 | +34.6% | +$34,596 | take_profit |

---

## What the Bugs Were Hiding

### Same-day entry (Critical 3) was the biggest issue

- Pre-fix: 94.5% of trades entered same calendar day as announcement
- These captured the intraday move when a catalyst news broke — not reproducible without knowing the announcement before market open
- After the fix, entry is the next day, and most stocks have already repriced
- Only extreme movers (penny stocks with overnight gaps) generate large returns post-entry

### max_positions bypass (Critical 2b)

- Pre-fix: inverse-vol sizing bypassed the 10-position cap entirely (59 concurrent positions observed)
- After fix with MAX_OPEN_POSITIONS=10: 281 of 432 signals skipped — too restrictive due to window overlap
- Solution: updated MAX_OPEN_POSITIONS to 20 to account for ~1.5x window overlap (30-day holds, 4-week steps)

---

## Production Configuration (config.py)

```python
TP_SL_CONFIG = {"30d": {"take_profit": 0.35, "stop_loss": 1.00}}
TIERED_SLIPPAGE = True
ADV_CAP_PCT = 0.10
MAX_OPEN_POSITIONS = 20
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_PCT = 0.001
SLIPPAGE_PCT = 0.001
```

---

## Honest Assessment

**The model has weak signal quality.** On non-penny stocks (>$5), 134 trades produced only $22K PnL over 2 years — essentially zero alpha. The only genuine source of return is:

1. A small number of extreme penny stock catalyst events (GNPX, KPTI)
2. Where the model's signal happens to coincide with a 10–3500% overnight gap
3. And the next-day entry still captures most of the move (because the stock continues rallying beyond the entry day)

This is a **lottery-ticket strategy**, not a consistent alpha source. It survives backtesting because of 1–2 extreme outliers per 2-year period.

### For the strategy to be investable:
- The outlier alpha needs to hold out-of-sample (not overfit to GNPX specifically)
- Maximum loss per 2-year period (ex-outliers) is approximately −6.5% on $1M
- Expected return is highly right-skewed: most periods will underperform SPY until an outlier hits

---

## Files Generated

- `output/trade_log_30d.csv` — 247 trades, production strategy with realistic friction
- `output/metrics_30d.csv` — summary metrics
- `output/trade_log_B3_realistic.csv` — B3 with friction, pre-audit (reference)
- `output/equity_curve_B3_realistic.csv` — B3 pre-audit equity curve (reference)
- `output/strategy_analysis_report.md` — this file
