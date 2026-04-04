# Clinical Trial Announcement Backtester — Project Specification

## 1. Project Overview

Build a backtesting system that evaluates a **rolling-window retrained clinical trial signal model** against real historical price data. The model predicts whether a clinical trial or FDA announcement will produce an outsized stock return (>85th percentile) over 5-day and 30-day horizons.

The system simulates trading on these predictions using the `backtesting.py` library, with configurable profit-taker/stop-loss levels, position sizing, and capital constraints.

### Key Properties

- **Horizons**: 5-day and 30-day (treated as separate backtests, with a combined comparison report)
- **Model**: `LinearSVC(C=0.1, class_weight='balanced')` retrained on each rolling window
- **Features**: `D_cat_sf` = OneHotEncoded(source, event_type) + 14 structured regex features
- **Label**: Binary, 1 if return >= 85th percentile (fixed threshold from full dataset)
- **Price data**: Pre-downloaded EODHD daily OHLCV in local parquet/CSV files
- **Library**: `backtesting.py` for trade simulation

---

## 2. Data Sources

### 2.1 Announcements Dataset

**File**: `announcements2.parquet` (current: ~18,884 rows, 2006–2026)

**Full dataset distribution** (before filtering):

| Source | Count | Earliest | Latest |
|--------|-------|----------|--------|
| SEC EDGAR | 12,842 | 2023-02-21 | 2026-03-19 |
| ClinicalTrials.gov | 6,020 | 2006-09-19 | 2026-03-19 |
| OpenFDA | 22 | 2017-08-15 | 2025-12-23 |

**After filtering** (drop EDGAR, keep parse_status=='OK'): ~6,042 usable rows (ClinicalTrials + OpenFDA).

**Important: `published_at` = lastUpdatePostDate** (the date of the status change event, NOT the original trial registration date). This is correct for trading — the signal is the event (phase completion, results posted, approval), not when the trial was first registered. The heavy concentration of records in 2024-2025 is expected because active studies get updated frequently.

**Preprocessing** (must match model training):
- Filter: `parse_status == 'OK'`
- Drop: rows where `source` contains 'edgar' (case-insensitive)
- Keep: both `clinicaltrials` and `openfda` sources
- Required columns: `ticker`, `source`, `event_type`, `published_at`, `raw_text`, `return_5d`, `return_30d`

**Schema** (relevant columns):

| Column | Type | Notes |
|--------|------|-------|
| ticker | str | Stock ticker (e.g., MRNA) |
| source | str | `clinicaltrials` or `openfda` |
| event_type | str | e.g., CT_COMPLETED, FDA_SUPPL |
| published_at | datetime | Last update post date — the tradeable event date |
| raw_text | str | Full announcement text |
| return_5d | float | 5-day return in % (e.g., 8.55 = +8.55%) |
| return_30d | float | 30-day return in % |

### 2.2 OHLCV Price Data

**Source**: EODHD (eodhistoricaldata.com), pre-downloaded to local files.
**Format**: One parquet file per ticker, or one consolidated file — to be determined during implementation. Minimum columns: `date`, `open`, `high`, `low`, `close`, `volume`.
**Usage**: Fed into `backtesting.py` for trade simulation per ticker.

### 2.3 S&P 500 Benchmark

Daily OHLCV for `SPY` (or `^GSPC`), same source and format as individual ticker data, used for benchmark comparison.

---

## 3. Rolling Window Architecture

### 3.1 Hyperparameters (all configurable)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `train_months` | Length of training window in months | 12 |
| `predict_weeks` | Length of prediction/test window in weeks | 4 |
| `step_weeks` | How far the window slides forward each iteration | 4 |
| `start_date` | First day of the first prediction window | 2025-01-01 |
| `end_date` | Last day of backtesting period | 2025-12-31 |
| `min_train_samples` | Minimum announcements required in a training window to run it | 30 |

### 3.2 Data Distribution Warning

The filtered dataset (~6,042 rows) is **heavily skewed toward recent years** (2024-2025). Early-year windows (2006-2020) will be very sparse. The `min_train_samples` threshold ensures the model is never trained on too little data — windows below this threshold are skipped entirely, and a warning is logged.

When choosing `start_date` and `train_months`, consider:
- ClinicalTrials.gov data goes back to 2006, but volume is thin before ~2022
- Most data is concentrated in 2024-2025
- A reasonable starting point for backtesting might be 2023-2024 with 12-month lookback

### 3.3 Window Logic

For each iteration `i`:
1. **Prediction window**: `start_date + i * step_weeks` → `start_date + i * step_weeks + predict_weeks`
2. **Training window**: `prediction_start - train_months` → `prediction_start`  (sliding, NOT expanding)
3. Train a fresh `LinearSVC` on the training window's announcements
4. Predict on all announcements within the prediction window
5. Simulate trades on predictions using `backtesting.py` against real OHLCV data
6. Record results, advance to next window

### 3.4 Edge Cases

- **Sparse training window**: If a training window has fewer than `min_train_samples` announcements, **skip this window entirely**. Log a warning: `"Window N skipped: only X announcements in training period (min: {min_train_samples})"`.
- **Sparse prediction window**: If a prediction window has very few announcements (even 0-5), **still run it** — the result is simply no/few trades for that window.
- If the training window has insufficient data for a given ticker's event type, the OHE handles unknowns via `handle_unknown='ignore'`.
- Windows continue until `prediction_end > end_date`.

---

## 4. Model Retraining Per Window

### 4.1 Label Construction

- **Fixed threshold**: Use the 85th percentile return value computed from the **full dataset** (not per-window). Store these as constants: `P85_5D` and `P85_30D`.
- Label: `y = 1 if return >= P85 else 0`
- Null returns: fill with 0 (same as training notebooks).

### 4.2 Feature Engineering

Exactly replicate the `D_cat_sf` feature set from the notebooks:

**OneHotEncoder** (global):
- Fit **once** on the entire dataset's `source` and `event_type` columns (all time periods).
- Reuse this single fitted encoder for every rolling window. This avoids category mismatch across windows.
- Settings: `handle_unknown='ignore'`, `sparse_output=False`, `dtype=np.float32`

**14 Structured Features** (`SF_COLS`):
Extracted via `extract_structured()` function from `raw_text`:
```
sf_phase, sf_n_patients, sf_pos_endpoint, sf_neg_endpoint, sf_net_endpoint,
sf_mech_hits, sf_disease_hits, sf_is_oncology, sf_is_rare,
sf_is_phase3, sf_is_phase2, sf_has_approval, sf_has_nct, sf_word_count
```

**The `extract_structured()` function and all keyword lists (POSITIVE_ENDPOINT, NEGATIVE_ENDPOINT, MECHANISM_KEYWORDS, DISEASE_KEYWORDS) must be copied verbatim from the notebooks.** These are the ground-truth implementations.

### 4.3 Training

- Algorithm: `LinearSVC(C=0.1, class_weight='balanced', max_iter=2000, random_state=42)`
- Input: `np.hstack([ohe_features, structured_features])` as float32
- No SMOTE, no embeddings, no PCA.

### 4.4 Prediction

- `clf.predict(X)` → binary prediction (0 or 1)
- `clf.decision_function(X)` → confidence score (useful for logging, not used for trade entry)
- A prediction of 1 = "take a long position on this ticker"

---

## 5. Trading Simulation

### 5.1 Trade Entry

- **Trigger**: Model predicts 1 for an announcement
- **Entry date**: Next trading day's market open after `published_at`
- **One position per ticker per horizon**: If multiple announcements for the same ticker fire within the same prediction window for the same horizon, take only one position.

### 5.2 Trade Exit (whichever comes first)

1. **Profit-taker hit**: Price reaches `entry_price * (1 + take_profit_pct)` during the horizon
2. **Stop-loss hit**: Price reaches `entry_price * (1 - stop_loss_pct)` during the horizon
3. **Horizon expiry**: If neither TP nor SL is hit within 5 or 30 trading days, exit at close on the last day

### 5.3 TP/SL Configuration

Two modes:

**Manual mode**: User specifies fixed TP and SL values per horizon:
```python
config = {
    "5d":  {"take_profit": 0.08, "stop_loss": 0.04},
    "30d": {"take_profit": 0.15, "stop_loss": 0.08},
}
```

**Sweep mode**: System tests a grid of TP/SL combinations and reports results for each:
```python
tp_grid = [0.05, 0.08, 0.10, 0.15, 0.20]
sl_grid = [0.03, 0.05, 0.08, 0.10]
```

### 5.4 Position Sizing & Capital

Configurable hyperparameters:
- `initial_capital`: Starting portfolio value (e.g., $100,000)
- `max_open_positions`: Maximum simultaneous positions (e.g., 10, 20, or unlimited)
- `position_sizing`: Strategy for allocating capital per trade — at minimum support equal-weight (divide available capital equally among max positions). More advanced methods (e.g., fractional Kelly) are optional stretch goals.

### 5.5 Transaction Costs

- `commission_pct`: Percentage per trade (e.g., 0.001 = 0.1%)
- `slippage_pct`: Simulated slippage (e.g., 0.001 = 0.1%)
- Both configurable, default to 0 if not specified.

---

## 6. Output & Metrics

### 6.1 Per-Window Results

For each rolling window iteration, record:
- Window dates (train start/end, predict start/end)
- Number of announcements in training set, number of positives
- Number of announcements in prediction window
- Number of signals generated (predictions = 1)
- Number of trades executed (after dedup per ticker)
- Per-trade results (see trade log)

### 6.2 Trade Log

A DataFrame with one row per trade:

| Column | Description |
|--------|-------------|
| window_id | Which rolling window generated this trade |
| horizon | '5d' or '30d' |
| ticker | Stock ticker |
| published_at | Announcement date |
| entry_date | Actual trade entry date (next trading day) |
| entry_price | Open price on entry date |
| exit_date | Date position was closed |
| exit_price | Price at exit |
| exit_reason | 'take_profit', 'stop_loss', or 'horizon_expiry' |
| return_pct | Realized return percentage |
| pnl | Dollar P&L for this trade |
| decision_score | LinearSVC decision_function score |
| actual_return_5d | Actual 5d return from announcements dataset (for comparison) |
| actual_return_30d | Actual 30d return from announcements dataset (for comparison) |

### 6.3 Aggregate Metrics

Compute across all windows combined:
- **Total return** (%)
- **Annualized return** (%)
- **Sharpe ratio** (annualized, assuming 252 trading days)
- **Max drawdown** (%)
- **Win rate** (% of trades with positive return)
- **Profit factor** (gross profit / gross loss)
- **Total trades**
- **Average return per trade**
- **Average holding period** (days)
- **Exposure time** (% of time with open positions)

### 6.4 Benchmark Comparison

- Compute the same metrics for a buy-and-hold SPY strategy over the same backtesting period.
- Present side-by-side: strategy vs benchmark.

### 6.5 Comparison: 5d vs 30d

- Run both horizons as separate backtests.
- Present a comparison table of aggregate metrics.
- Optionally: equity curve chart for both on the same plot.

---

## 7. Module Structure

```
backtest/
├── config.py              # All hyperparameters, file paths, constants
├── data_loader.py         # Load announcements, load OHLCV, load benchmark
├── features.py            # extract_structured(), preprocess_for_embedding(), build feature matrix
├── model.py               # train_model(), predict(), rolling window orchestration
├── strategy.py            # backtesting.py Strategy class (minimal OOP, just the wrapper)
├── simulator.py           # Run backtesting.py per window, collect results
├── metrics.py             # Compute aggregate metrics, benchmark comparison
├── sweep.py               # TP/SL grid sweep logic
├── main.py                # CLI entry point: run full backtest
├── tests/
│   ├── test_data_loader.py
│   ├── test_features.py
│   ├── test_model.py
│   ├── test_strategy.py
│   ├── test_simulator.py
│   ├── test_metrics.py
│   └── test_sweep.py
└── README.md
```

### Design Principles

- **No classes** except the one `Strategy` subclass required by `backtesting.py`.
- All other code is **pure functions** organized by module.
- Keep functions small, testable, and composable.
- Every function that touches data should have a corresponding test.

---

## 8. Implementation Order

Build and test in this sequence:

### Phase 1: Data & Features (no trading yet)
1. `config.py` — all constants, paths, hyperparameters
2. `data_loader.py` — load and filter announcements, load OHLCV files
3. `features.py` — extract_structured(), OHE fitting, feature matrix building
4. Tests for Phase 1

### Phase 2: Rolling Window Model
5. `model.py` — train_model(), predict(), generate_windows(), run rolling loop
6. Tests for Phase 2 (verify window generation, model retraining, prediction output shapes)

### Phase 3: Trade Simulation
7. `strategy.py` — backtesting.py Strategy class
8. `simulator.py` — run one window's trades, collect per-window results
9. Tests for Phase 3

### Phase 4: Metrics & Output
10. `metrics.py` — aggregate metrics, benchmark comparison, 5d vs 30d comparison
11. `sweep.py` — TP/SL grid sweep
12. `main.py` — CLI orchestration
13. Tests for Phase 4

### Phase 5: Integration Testing
14. End-to-end test with a small synthetic dataset
15. End-to-end test with real data (small time range)

---

## 9. Key Constraints & Gotchas

1. **No look-ahead bias**: The model must only see announcements with `published_at` within the training window. Price data for TP/SL evaluation uses only data from entry_date onward.
2. **Fixed P85 threshold**: `P85_5D` and `P85_30D` are computed once from the full announcements dataset and stored in config. They are NOT recomputed per window.
3. **Global OHE**: Fitted once on all announcements data, reused for every window.
4. **Drop EDGAR**: Filter out `source` containing 'edgar' (case-insensitive), matching the model training notebooks.
5. **Null returns**: Rows with null `return_5d` or `return_30d` are NOT dropped. Nulls are filled with 0 for label construction.
6. **One position per ticker per horizon per window**: Dedup multiple signals for the same ticker.
7. **backtesting.py expects OHLCV DataFrame with DatetimeIndex**: Format price data accordingly.
8. **Entry on next trading day**: If `published_at` falls on a weekend or holiday, entry is the next available trading day's open.
9. **The dataset is skewed and will grow**: After EDGAR filtering, ~6,042 rows remain, heavily concentrated in 2024-2025. Code must handle sparse early-year windows gracefully (via `min_train_samples` skip logic). Do not hardcode date ranges.
10. **Structured features function `extract_structured()` must be copied exactly from the notebooks** — do not simplify, rename, or modify the regex patterns or keyword lists.
11. **`published_at` is lastUpdatePostDate**: This is the date of the status change event (phase completion, results posted, etc.), NOT the original trial registration. This is the correct tradeable event date.
