# Post Split-Adjustment Analysis

**Date**: 2026-03-27
**Bug**: OHLCV data loader used unadjusted `close` instead of `adjusted_close`, causing reverse stock splits to appear as 1,000-3,500% returns.

---

## The Bug

`_normalize_ohlcv()` in `data_loader.py` mapped the raw `close` column to `Close` and discarded `adjusted_close`. When a company does a reverse stock split (e.g., 1:50), the unadjusted close jumps overnight by the split ratio, creating phantom returns.

**192 tickers** in the OHLCV universe had splits during the 2024-2025 backtest period.

### The Two "Star" Trades Were Fake

| Ticker | Old Entry | Old Return | Old PnL | What Actually Happened |
|--------|-----------|-----------|---------|----------------------|
| GNPX | $0.35 | +3,552% | +$767,634 | 1:50 reverse split on Oct 21, 2025. Adjusted price fell from $18 to $11. |
| KPTI | $0.69 | +1,163% | +$55,747 | 1:15 reverse split on Feb 26, 2025. Adjusted price went from ~$10 to ~$10. |

### After Adjustment

| Ticker | Adjusted Entry | Adjusted Return | Adjusted PnL |
|--------|---------------|----------------|-------------|
| GNPX (Oct 2025) | $16.88 | **-90.1%** | **-$6,758** |
| KPTI (Jan 2025) | $9.91 | **-59.9%** | **-$4,495** |

The trades that appeared to be +3,552% and +1,163% winners were actually -90% and -60% losers.

---

## Fix Applied

In `_normalize_ohlcv()`: when `adjusted_close` column exists, compute `factor = adjusted_close / close` and multiply Open, High, Low by this factor. Close is replaced with `adjusted_close` directly. This makes all OHLC prices continuous across splits.

---

## Corrected Results

### Model-Filtered Strategy (30d)

| Metric | Before (unadjusted) | After (adjusted) | SPY |
|--------|--------------------|-----------------|----|
| Total return | +70.2% | **-25.3%** | +47.8% |
| Annualized return | +32.4% | -14.3% | +21.6% |
| Sharpe ratio | 0.64 | **-0.36** | 1.28 |
| Max drawdown | 43.3% | 52.5% | 18.8% |
| Win rate | 44.5% | 44.5% | -- |
| Profit factor | 1.73 | **0.76** | -- |

**The model loses money. It has negative alpha.**

### Path C — No Model, Best Configuration

Optimized config: iv 0.75%, mp=100, TP=35%, 50-bar horizon, ADV=none, grace=0.

| Metric | Before (unadjusted) | After (adjusted) | SPY |
|--------|--------------------|-----------------|----|
| Total return | +140.4% | **+14.0%** | +47.8% |
| Annualized return | +48.8% | +6.3% | +21.6% |
| Sharpe ratio | 1.54 | **0.49** | 1.28 |
| Max drawdown | 16.5% | 25.8% | 18.8% |
| Win rate | 50.4% | 51.0% | -- |
| Profit factor | 2.65 | **1.19** | -- |

**Path C is slightly positive but dramatically underperforms SPY on every metric.**

### PnL by Price Tier (Path C Final)

| Tier | Before (unadjusted) | After (adjusted) |
|------|--------------------| -----------------|
| < $2 (penny) | +$1,300,244 (93%) | +$29,274 (21%) |
| $2–$5 | -- | +$64,134 (46%) |
| >= $5 | +$94,128 (7%) | +$46,095 (33%) |

**The penny stock "alpha" was entirely reverse-split artifacts.** After adjustment, PnL is distributed across tiers with $2-$5 stocks actually contributing the most.

### Phase 6: Price Tier Filter Results (Corrected)

| Filter | Return | Sharpe |
|--------|--------|--------|
| Exclude penny (>=$2) | **+16.2%** | **0.57** |
| Exclude sub-$5 (>=$5) | +13.8% | 0.54 |
| ALL (baseline) | +14.0% | 0.49 |
| Mid-cap ($2-$10) | +7.9% | 0.59 |
| Only penny (<$2) | **-1.7%** | **-0.64** |

**Penny stocks are now net losers.** Excluding them actually improves both return and Sharpe. The best filter is "exclude penny" at +16.2%, Sharpe 0.57 — but still far below SPY.

---

## Top Trades (Corrected)

| Ticker | Entry Price | Return | PnL | Exit |
|--------|------------|--------|-----|------|
| AVTX | $4.59 | +306.2% | +$22,964 | take_profit |
| OLMA | $10.59 | +149.0% | +$11,179 | take_profit |
| CNSP | $13.69 | +113.2% | +$8,489 | take_profit |
| ALZN | $3.91 | +98.0% | +$7,353 | take_profit |
| NKTR | $10.35 | +91.9% | +$6,891 | take_profit |

These are real trades on real (adjusted) prices. The best trade is now +$23K (AVTX), not +$768K (GNPX). Returns are in the +60% to +306% range, not +3,552%.

### Top Tickers by Total PnL

| Ticker | Trades | PnL | Share |
|--------|--------|-----|-------|
| AVTX | 3 | +$28,153 | 20% |
| NKTR | 6 | +$16,319 | 12% |
| OLMA | 2 | +$13,773 | 10% |
| ARVN | 6 | +$11,935 | 9% |
| TNDM | 8 | +$11,403 | 8% |

Concentration is still high (top-5 = ~59% of PnL) but not as extreme as before (was 82-109%).

---

## Conclusion

**There is no alpha in this system.** Neither the model-filtered approach nor the buy-everything approach produces risk-adjusted returns competitive with SPY buy-and-hold.

| Approach | Return | Sharpe | vs SPY |
|----------|--------|--------|--------|
| Model-filtered (30d) | -25.3% | -0.36 | Loses money |
| Path C (optimized) | +14.0% | 0.49 | Underperforms by 34pp, 2.6x worse Sharpe |
| Path C (ex-penny) | +16.2% | 0.57 | Underperforms by 32pp, 2.2x worse Sharpe |
| SPY buy & hold | +47.8% | 1.28 | -- |

The previous "outperformance" findings were entirely driven by reverse stock split artifacts in 192 tickers worth of unadjusted price data. Once prices are properly adjusted for corporate actions, the biotech catalyst universe shows no exploitable edge.

### What the Data Shows

1. **The model actively destroys value**: -25.3% return, negative Sharpe. It's worse than random.
2. **Buying everything is slightly positive** (+14%) but with 25.8% drawdown and Sharpe of 0.49 — you'd never allocate to this over a passive index.
3. **Penny stocks are net losers** when properly adjusted. The "fat tail" story was entirely fabricated by split artifacts.
4. **The real PnL distribution** is thin — best trade is +$23K, worst is -$7K. No 3,500% lottery tickets. The average trade makes $174 on $7,500 position size.
5. **Year split is extreme**: Y1=+0.4%, Y2=+14.0%. Almost all return comes from 2025.

### Remaining Concern: Announcement Return Labels

The `return_5d` and `return_30d` columns in the announcements dataset may also use unadjusted prices. If so, the training labels themselves are corrupted — the model was trained to predict split artifacts, not real returns. **This should be verified in the upstream data pipeline.**

---

*Report generated 2026-03-27 after fixing the split-adjustment bug in `data_loader.py`.*
