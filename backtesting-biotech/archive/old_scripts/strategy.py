"""
strategy.py — backtesting.py Strategy class (minimal OOP wrapper).
This is the ONE allowed class in the project. All trade logic is driven by
pre-computed signals passed in as class attributes; next() just acts on them.
"""

import pandas as pd
from backtesting import Strategy


class ClinicalTrialStrategy(Strategy):
    """A thin backtesting.py wrapper that executes pre-computed clinical trial signals.

    Class attributes (set before Backtest.run()):
      - signals_df:      pd.DataFrame with columns [entry_date, take_profit, stop_loss]
                         and a DatetimeIndex or 'entry_date' column (as Timestamps).
      - take_profit_pct: float, e.g. 0.08 = 8%
      - stop_loss_pct:   float, e.g. 0.04 = 4%
      - horizon_bars:    int, number of bars (trading days) before forced exit
      - max_positions:   int, maximum simultaneous open positions
      - position_size:   float fraction of equity per trade, e.g. 0.1 = 10%

    Entry happens on the open of the bar matching entry_date. Exit on TP/SL hit or
    after horizon_bars bars from entry.
    """

    # Class-level defaults — override before instantiating Backtest
    signals_df:      pd.DataFrame = None
    take_profit_pct: float = 0.08
    stop_loss_pct:   float = 0.04
    horizon_bars:    int   = 5
    max_positions:   int   = 10
    position_size:   float = 0.1

    def init(self):
        """Preprocess signals into per-date TP/SL lookup tables for O(1) access in next()."""
        if self.signals_df is not None and len(self.signals_df) > 0:
            entry_col = pd.to_datetime(self.signals_df["entry_date"]).dt.normalize()
            self._entry_dates = set(entry_col)
            # Precomputed absolute TP/SL prices anchored to entry Open + slippage
            self._tp_by_date = dict(zip(entry_col, self.signals_df["tp_price"]))
            self._sl_by_date = dict(zip(entry_col, self.signals_df["sl_price"]))
        else:
            self._entry_dates = set()
            self._tp_by_date = {}
            self._sl_by_date = {}

        # Track when each position was opened (bar index) for horizon expiry
        self._open_bar: dict[int, int] = {}   # position_id → bar_index_at_entry

    def next(self):
        """On each bar: open new positions on signal dates; close expired positions."""
        current_date = pd.Timestamp(self.data.index[-1]).normalize()

        # Close positions that have exceeded the horizon
        for trade in list(self.trades):
            entry_bar = self._open_bar.get(id(trade))
            if entry_bar is not None:
                bars_held = len(self.data) - 1 - entry_bar
                if bars_held >= self.horizon_bars:
                    trade.close()
                    del self._open_bar[id(trade)]

        # Open new positions on signal dates
        if current_date in self._entry_dates:
            n_open = len(self.trades)
            if n_open < self.max_positions:
                tp = self._tp_by_date.get(current_date)
                sl = self._sl_by_date.get(current_date)
                self.buy(size=self.position_size, tp=tp, sl=sl)
                # Record entry bar index for horizon tracking
                if self.trades:
                    self._open_bar[id(self.trades[-1])] = len(self.data) - 1
