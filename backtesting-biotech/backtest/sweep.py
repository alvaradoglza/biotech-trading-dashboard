"""
sweep.py — TP/SL grid sweep: run the full backtest for each (take_profit, stop_loss) pair
and return a comparison table of aggregate metrics.
"""

from typing import Callable

import pandas as pd

from backtest.config import TP_GRID, SL_GRID


def run_sweep(
    run_backtest_fn: Callable[[float, float], pd.DataFrame],
    tp_grid: list[float] = TP_GRID,
    sl_grid: list[float] = SL_GRID,
) -> pd.DataFrame:
    """Run a TP/SL grid sweep by calling run_backtest_fn for each (tp, sl) combination.

    Takes a callable run_backtest_fn(take_profit_pct, stop_loss_pct) -> metrics dict,
    and the grid lists. Returns a DataFrame with one row per (tp, sl) pair and metric
    columns, sorted by total_return_pct descending.
    """
    records = []
    total = len(tp_grid) * len(sl_grid)
    i = 0
    for tp in tp_grid:
        for sl in sl_grid:
            i += 1
            print(f"Sweep {i}/{total}: TP={tp:.0%} SL={sl:.0%}")
            metrics = run_backtest_fn(tp, sl)
            row = {"take_profit_pct": tp, "stop_loss_pct": sl}
            row.update(metrics)
            records.append(row)

    df = pd.DataFrame(records)
    if "total_return_pct" in df.columns:
        df = df.sort_values("total_return_pct", ascending=False).reset_index(drop=True)
    return df


def best_params(sweep_df: pd.DataFrame, metric: str = "total_return_pct") -> dict[str, float]:
    """Extract the best TP/SL parameters from a sweep result DataFrame.

    Takes the sweep DataFrame and the metric to optimize. Returns a dict with
    take_profit_pct and stop_loss_pct for the row with the highest metric value.
    Raises ValueError if sweep_df is empty.
    """
    if len(sweep_df) == 0:
        raise ValueError("sweep_df is empty — no results to extract best params from.")
    if metric not in sweep_df.columns:
        raise ValueError(f"Metric '{metric}' not found in sweep_df columns.")

    best_row = sweep_df.loc[sweep_df[metric].idxmax()]
    return {
        "take_profit_pct": float(best_row["take_profit_pct"]),
        "stop_loss_pct":   float(best_row["stop_loss_pct"]),
    }
