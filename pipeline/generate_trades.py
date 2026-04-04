"""
generate_trades.py — Convert ML predictions into trading signals and manage positions.

Daily workflow:
  1. Check existing open positions for exits (TP hit, SL hit, or horizon expired)
  2. Convert new BUY predictions into entry signals and trades
  3. Update position table
  4. Return updated trades, positions, and portfolio snapshot

Uses EODHD for current prices (via dashboard/prices.py logic for consistency).
"""

import logging
import os
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Portfolio config — these mirror backtesting-biotech/backtest/config.py
INITIAL_CAPITAL = float(os.environ.get("INITIAL_CAPITAL", "1000000.0"))
MAX_OPEN_POSITIONS = int(os.environ.get("MAX_OPEN_POSITIONS", "20"))
MAX_WEIGHT = float(os.environ.get("MAX_WEIGHT", "0.07"))  # 7% max per position
COMMISSION_PCT = 0.001   # 0.1%
TAKE_PROFIT_PCT = 0.30   # 30% TP for 30d horizon
STOP_LOSS_PCT = 1.00     # No stop loss for 30d (SL = 100%)
HORIZON_DAYS = 50        # Max holding period (calendar days)


def process_daily_signals(
    predictions: list[dict],
    open_positions: list[dict],
    cash: float,
    current_prices: dict[str, float],
    today: date | None = None,
) -> dict:
    """Main entry point: process predictions and positions for the day.

    Args:
        predictions: List of prediction dicts from predict.py (all predictions today).
        open_positions: Current positions from Supabase positions table.
        cash: Current cash balance.
        current_prices: Dict of {ticker: price} from EODHD (today's prices).
        today: Date to use (defaults to today).

    Returns:
        Dict with keys:
        - new_signals: list of signal dicts to insert
        - new_trades: list of trade dicts to insert
        - updated_positions: list of position dicts to upsert
        - closed_positions: list of ticker strings to delete
        - updated_cash: updated cash balance
        - portfolio_snapshot: snapshot dict
    """
    today = today or date.today()

    new_signals = []
    new_trades = []
    updated_positions = list(open_positions)  # start from current state
    closed_tickers = []

    # Step 1: Check existing positions for exits
    positions_to_keep = []
    for pos in open_positions:
        ticker = pos["ticker"]
        current_price = current_prices.get(ticker)
        if current_price is None:
            logger.warning("No price for open position %s — holding", ticker)
            positions_to_keep.append(pos)
            continue

        exit_result = _check_exit(pos, current_price, today)

        if exit_result["should_exit"]:
            # Record closing trade
            qty = pos["quantity"]
            exit_price = exit_result["exit_price"]
            exit_trade = {
                "ticker": ticker,
                "side": "SELL",
                "quantity": qty,
                "price": exit_price,
                "amount_usd": qty * exit_price * (1 - COMMISSION_PCT),
                "trade_date": today.isoformat(),
                "status": "filled",
                "exit_reason": exit_result["reason"],
            }
            new_trades.append(exit_trade)
            cash += exit_trade["amount_usd"]
            closed_tickers.append(ticker)
            logger.info(
                "Exit %s: %s at $%.2f (reason: %s)",
                ticker, exit_result["reason"], exit_price, exit_result["reason"]
            )
        else:
            # Update market value
            pos = pos.copy()
            pos["market_value"] = pos["quantity"] * current_price
            pos["unrealized_pnl"] = (current_price - pos["avg_cost"]) * pos["quantity"]
            positions_to_keep.append(pos)

    updated_positions = positions_to_keep

    # Step 2: Open new positions from BUY signals
    open_tickers = {pos["ticker"] for pos in updated_positions}
    buy_predictions = [
        p for p in predictions
        if p.get("predicted_label") == 1
        and p.get("ticker") not in open_tickers
    ]

    # Sort by predicted probability (highest confidence first)
    buy_predictions = sorted(buy_predictions, key=lambda p: p.get("predicted_probability", 0), reverse=True)

    available_slots = MAX_OPEN_POSITIONS - len(updated_positions)

    for pred in buy_predictions[:available_slots]:
        ticker = pred["ticker"]
        # Re-check inside loop: a previous iteration may have opened this ticker
        if ticker in open_tickers:
            continue
        current_price = current_prices.get(ticker)
        if current_price is None or current_price <= 0:
            logger.warning("No price for BUY signal %s — skipping", ticker)
            continue

        # Position sizing: MAX_WEIGHT of portfolio, limited by available cash
        portfolio_value = cash + sum(p.get("market_value", p["quantity"] * p["avg_cost"]) for p in updated_positions)
        target_notional = portfolio_value * MAX_WEIGHT
        target_notional = min(target_notional, cash * 0.95)  # don't use >95% of remaining cash

        if target_notional < 100:  # minimum $100 position
            logger.warning("Insufficient cash for %s position — skipping", ticker)
            continue

        qty = target_notional / current_price
        entry_cost = qty * current_price * (1 + COMMISSION_PCT)
        cash -= entry_cost

        # Signal
        signal = {
            "prediction_id": pred.get("id"),
            "signal_date": today.isoformat(),
            "ticker": ticker,
            "action": "BUY",
            "reason": f"ML BUY signal (prob={pred.get('predicted_probability', 0):.3f})",
            "score": pred.get("predicted_probability"),
        }
        new_signals.append(signal)

        # Trade
        trade = {
            "ticker": ticker,
            "side": "BUY",
            "quantity": qty,
            "price": current_price,
            "amount_usd": entry_cost,
            "trade_date": today.isoformat(),
            "status": "filled",
        }
        new_trades.append(trade)

        # Position
        updated_positions.append({
            "ticker": ticker,
            "quantity": qty,
            "avg_cost": current_price,
            "market_value": qty * current_price,
            "unrealized_pnl": 0.0,
            "entry_date": today.isoformat(),
        })
        open_tickers.add(ticker)

        logger.info(
            "Entry %s: %.2f shares at $%.2f = $%.2f (prob=%.3f)",
            ticker, qty, current_price, entry_cost, pred.get("predicted_probability", 0)
        )

    # Step 3: Portfolio snapshot
    equity_value = sum(p.get("market_value", p["quantity"] * p["avg_cost"]) for p in updated_positions)
    total_value = cash + equity_value

    snapshot = {
        "snapshot_date": today.isoformat(),
        "cash": cash,
        "equity_value": equity_value,
        "total_value": total_value,
        "n_positions": len(updated_positions),
    }

    return {
        "new_signals": new_signals,
        "new_trades": new_trades,
        "updated_positions": updated_positions,
        "closed_positions": closed_tickers,
        "updated_cash": cash,
        "portfolio_snapshot": snapshot,
    }


def _check_exit(position: dict, current_price: float, today: date) -> dict:
    """Determine if a position should be exited today.

    Checks: take profit, stop loss, and max holding period.
    Returns: {should_exit: bool, reason: str, exit_price: float}
    """
    avg_cost = position.get("avg_cost", current_price)
    entry_date_str = position.get("entry_date")

    if avg_cost <= 0:
        return {"should_exit": False, "reason": None, "exit_price": current_price}

    pnl_pct = (current_price - avg_cost) / avg_cost

    # Take profit
    if pnl_pct >= TAKE_PROFIT_PCT:
        return {"should_exit": True, "reason": "take_profit", "exit_price": current_price}

    # Stop loss (only if SL < 100%)
    if STOP_LOSS_PCT < 1.0 and pnl_pct <= -STOP_LOSS_PCT:
        return {"should_exit": True, "reason": "stop_loss", "exit_price": current_price}

    # Horizon expired
    if entry_date_str:
        try:
            entry_date = date.fromisoformat(str(entry_date_str)[:10])
            days_held = (today - entry_date).days
            if days_held >= HORIZON_DAYS:
                return {"should_exit": True, "reason": "horizon", "exit_price": current_price}
        except (ValueError, TypeError):
            pass

    return {"should_exit": False, "reason": None, "exit_price": current_price}


def fetch_prices_for_pipeline(tickers: list[str]) -> dict[str, float]:
    """Fetch current prices from EODHD for the daily pipeline.

    This is a lightweight version (no Streamlit caching) for use in the pipeline.
    Returns {ticker: price} dict, skipping any tickers with missing/bad data.
    """
    api_key = os.environ.get("EODHD_API_KEY")
    if not api_key:
        logger.error("EODHD_API_KEY not set — cannot fetch live prices")
        return {}

    prices = {}
    for ticker in tickers:
        try:
            url = f"https://eodhd.com/api/real-time/{ticker}.US"
            response = requests.get(url, params={"api_token": api_key, "fmt": "json"}, timeout=10)
            if response.status_code == 200:
                data = response.json()
                price = data.get("close") or data.get("previousClose")
                if price is not None and str(price).upper() not in ("NA", "N/A", "NULL", "NONE", ""):
                    val = float(price)
                    if val > 0:
                        prices[ticker] = val
        except (ValueError, TypeError) as e:
            logger.warning("Price parse failed for %s: %s", ticker, e)
        except Exception as e:
            logger.warning("Price fetch failed for %s: %s", ticker, e)

    logger.info("Fetched prices for %d/%d tickers", len(prices), len(tickers))
    return prices
