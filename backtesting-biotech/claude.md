# CLAUDE.md — Working Instructions for Clinical Trial Backtester

## Project Context

You are building a rolling-window backtesting system for a clinical trial announcement classifier. The full specification is in `prompt.md` — read it before every coding session.

The model predicts whether clinical trial / FDA announcements will produce outsized stock returns (>85th percentile) over 5d and 30d horizons. The backtester retrains this model on sliding time windows and simulates trades using `backtesting.py`.

---

## Code Style Rules

### Strictly Functional
- **NO classes, NO OOP** — the only exception is the `Strategy` subclass required by `backtesting.py`.
- All logic lives in **pure functions** that take inputs and return outputs.
- No global mutable state. Configuration is passed explicitly via dicts or function arguments.
- Prefer simple, flat code. Avoid abstractions unless they eliminate real duplication.

### Simplicity Over Cleverness
- Write code that reads top-to-bottom like a recipe.
- Avoid decorators, metaclasses, ABCs, mixins, or any indirection that obscures what the code does.
- If a function is getting long, split it into smaller named functions — not into a class hierarchy.
- Use descriptive variable names. `train_df` not `d`. `window_start` not `ws`.

### Type Hints & Docstrings
- Add type hints to all function signatures.
- Add a one-line docstring to every function explaining what it does, what it takes, and what it returns.
- No multi-page docstrings. Keep them concise.

### Dependencies
- Core: `pandas`, `numpy`, `scikit-learn`, `backtesting` (backtesting.py)
- Standard lib only for everything else (no unnecessary packages).
- Pin versions in `requirements.txt`.

---

## Testing Protocol — MANDATORY

### Test Every Function
- **Every function you write must have at least one test** before you move to the next function.
- Tests go in `backtest/tests/test_<module>.py`.
- Use `pytest`. No unittest classes — just plain test functions.
- Run tests after writing each function: `pytest backtest/tests/test_<module>.py -v`

### Test Structure
```python
def test_function_name_does_what():
    """One sentence describing what's being tested."""
    # Arrange
    input_data = ...
    # Act
    result = function_under_test(input_data)
    # Assert
    assert result == expected
```

### What to Test
- **Happy path**: Does it work with normal input?
- **Edge cases**: Empty DataFrames, windows with 0 announcements, unknown tickers, single-row inputs.
- **Shape/type checks**: Output DataFrame has expected columns? Output array has right shape?
- **No data leakage**: Training window does not include data from prediction window dates.
- **Determinism**: Same inputs → same outputs (set random_state=42 everywhere).

### Integration Tests
- After completing each phase (see prompt.md §8), run an end-to-end test for that phase.
- Use small synthetic data for fast iteration — a handful of tickers, a few dozen announcements, a short date range.

### Running Tests
```bash
# Single module
pytest backtest/tests/test_features.py -v

# All tests
pytest backtest/tests/ -v

# With output on failure
pytest backtest/tests/ -v --tb=short
```

---

## Implementation Sequence

Follow the phases in `prompt.md §8` strictly. **Do not skip ahead.**

1. **Phase 1**: `config.py` → `data_loader.py` → `features.py` → tests
2. **Phase 2**: `model.py` → tests
3. **Phase 3**: `strategy.py` → `simulator.py` → tests
4. **Phase 4**: `metrics.py` → `sweep.py` → `main.py` → tests
5. **Phase 5**: Integration tests

Within each phase:
- Write one function at a time.
- Test it immediately.
- Only move to the next function after the test passes.

---

## Critical Implementation Details

### Feature Engineering — Copy Exactly
The `extract_structured()` function, `preprocess_for_embedding()`, and all keyword lists (`POSITIVE_ENDPOINT`, `NEGATIVE_ENDPOINT`, `MECHANISM_KEYWORDS`, `DISEASE_KEYWORDS`, `SF_COLS`) must be **copied verbatim** from the notebooks. Do NOT:
- Rename features
- Simplify regex patterns
- Merge or split keyword lists
- Change the order of SF_COLS

Reference notebook: `ann_v10_new_data.ipynb`, Cell 8.

### OHE — Global Fit
The `OneHotEncoder` is fitted **once** on the entire announcements dataset (all time periods) and reused for every rolling window. This is intentional — it prevents category mismatch across windows at the cost of minor (acceptable) data leakage.

### P85 Threshold — Fixed
`P85_5D` and `P85_30D` are computed once from the full dataset and stored as constants. They are NOT recomputed per rolling window.

### backtesting.py Integration
- The library expects a `Strategy` subclass — this is the ONE allowed class.
- OHLCV data must be a pandas DataFrame with a DatetimeIndex and columns named exactly: `Open`, `High`, `Low`, `Close`, `Volume`.
- The Strategy's `init()` and `next()` methods drive trade logic.
- Keep the Strategy class as thin as possible — compute signals outside, pass them in via class attributes or parameters.

### No Look-Ahead Bias
- Training window: only announcements with `published_at < prediction_window_start`
- Trade entry: next trading day's open AFTER `published_at`
- TP/SL evaluation: only uses price data from entry_date onward
- Never use `return_5d` or `return_30d` columns during backtesting simulation — those are only for label construction during training and for post-hoc comparison in the trade log.

### Data Distribution Awareness
- After dropping EDGAR, the dataset is ~6,042 rows (ClinicalTrials.gov + OpenFDA).
- Data is **heavily skewed toward 2024-2025**. Early years (2006-2020) are very sparse.
- `published_at` = `lastUpdatePostDate` — the date of the status change event, not original registration. This is the correct tradeable signal date.
- `min_train_samples` (default: 30) prevents training on insufficient data. If a rolling window's training period has fewer announcements than this threshold, skip the window and log a warning.
- When writing tests, use date ranges in 2024-2025 for realistic scenarios and 2008-2010 for sparse/edge-case scenarios.

---

## Error Handling & Logging

- Use `print()` for progress logging (e.g., "Window 3/12: training on 234 announcements, 5 signals generated").
- Use `warnings.warn()` for non-fatal issues (e.g., "Ticker XYZ has no OHLCV data, skipping").
- Raise `ValueError` with descriptive messages for invalid configuration.
- Never silently swallow errors. If something unexpected happens, fail loudly.

---

## Error Log — UPDATE THIS SECTION

**Every time an error occurs during development, add an entry below.** This prevents repeating the same mistakes. Format:

```
### Error: [Short description]
- **Date**: YYYY-MM-DD
- **File**: which file
- **Root cause**: what went wrong
- **Fix**: what was done
- **Prevention**: what to check next time
```

### Errors encountered:
(none yet — add entries here as they occur)

---

## File Paths & Configuration

- Announcements data: path configured in `config.py` (default: `data/announcements2.parquet`)
- OHLCV price data: directory configured in `config.py` (default: `data/ohlcv/`)
- Benchmark data: path configured in `config.py` (default: `data/ohlcv/SPY.parquet`)
- Output directory: `output/` (trade logs, metrics reports, equity curves)
- All paths should be configurable, not hardcoded.

---

## Git Practices

- Commit after each function + test pair passes.
- Commit messages: `feat(module): add function_name — short description`
- Branch: `feat/backtesting`

---

## Checklist Before Marking a Phase Complete

- [ ] All functions in the phase have tests
- [ ] All tests pass (`pytest backtest/tests/ -v`)
- [ ] No hardcoded paths or magic numbers (everything in config.py)
- [ ] No look-ahead bias in any function
- [ ] Functions have type hints and docstrings
- [ ] Error log section above is updated if any errors were encountered
