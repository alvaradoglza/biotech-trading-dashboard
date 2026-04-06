"""
ui_helpers.py — Streamlit UI helper functions: formatting, badges, metric display.
"""

from datetime import date, timedelta

import pandas as pd
import streamlit as st

# Must match generate_trades.py constants
_HORIZON_DAYS = 50
_TAKE_PROFIT_PCT = 0.30


# ── Metric display ────────────────────────────────────────────────────────────

def metric_card(label: str, value: str, delta: str | None = None, help_text: str | None = None) -> None:
    """Display a single metric card."""
    st.metric(label=label, value=value, delta=delta, help=help_text)


def portfolio_metrics_row(summary: dict) -> None:
    """Display the top-level portfolio summary metrics in a row of 5 columns."""
    cols = st.columns(5)

    cols[0].metric(
        "Total Value",
        fmt_currency(summary.get("total_value", 0)),
        help="Total portfolio value (cash + equity)",
    )
    cols[1].metric(
        "Cash",
        fmt_currency(summary.get("cash", 0)),
    )
    cols[2].metric(
        "Equity Value",
        fmt_currency(summary.get("equity_value", 0)),
    )
    cols[3].metric(
        "Open Positions",
        str(summary.get("n_positions", 0)),
    )
    cols[4].metric(
        "Trades (30d)",
        str(summary.get("n_recent_trades", 0)),
    )


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_currency(value: float | None, prefix: str = "$") -> str:
    """Format a number as currency string."""
    if value is None:
        return "—"
    try:
        return f"{prefix}{float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


def fmt_pct(value: float | None, decimal_places: int = 2) -> str:
    """Format a number as percentage string."""
    if value is None:
        return "—"
    try:
        return f"{float(value):+.{decimal_places}f}%"
    except (TypeError, ValueError):
        return "—"


def fmt_shares(value: float | None) -> str:
    """Format a share count."""
    if value is None:
        return "—"
    try:
        v = float(value)
        if v >= 1000:
            return f"{v:,.1f}"
        return f"{v:.4f}"
    except (TypeError, ValueError):
        return "—"


def fmt_date(value) -> str:
    """Format a date string to readable format."""
    if value is None:
        return "—"
    try:
        dt = pd.to_datetime(value)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return str(value)[:10]


# ── Badge helpers ─────────────────────────────────────────────────────────────

def side_badge(side: str) -> str:
    """Return colored markdown badge for trade side."""
    side = str(side).upper()
    if side == "BUY":
        return "🟢 BUY"
    elif side == "SELL":
        return "🔴 SELL"
    return side


def label_badge(label: int | None) -> str:
    """Return readable badge for ML prediction label."""
    if label == 1:
        return "✅ BUY"
    elif label == 0:
        return "⬜ HOLD"
    return "—"


def source_badge(source: str) -> str:
    """Return readable badge for announcement source."""
    source = str(source).lower()
    if "openfda" in source or "fda" in source:
        return "🏥 FDA"
    elif "clinicaltrials" in source or "clinical" in source:
        return "🧬 ClinicalTrials"
    elif "edgar" in source:
        return "📄 EDGAR"
    return source


# ── Table formatting ──────────────────────────────────────────────────────────

def compute_position_extras(df: pd.DataFrame, total_value: float) -> pd.DataFrame:
    """Add derived columns to positions DataFrame: days_held, days_remaining,
    expiry_date, tp_price, portfolio_weight_pct.

    Call this AFTER enrich_positions_with_prices so live_market_value is present.
    """
    if df.empty:
        return df

    today = date.today()
    df = df.copy()

    def _days_held(entry_str):
        if not entry_str:
            return None
        try:
            return (today - date.fromisoformat(str(entry_str)[:10])).days
        except (ValueError, TypeError):
            return None

    df["days_held"] = df["entry_date"].apply(_days_held)
    df["days_remaining"] = df["days_held"].apply(
        lambda d: max(0, _HORIZON_DAYS - d) if d is not None else None
    )
    df["expiry_date"] = df["entry_date"].apply(
        lambda s: (date.fromisoformat(str(s)[:10]) + timedelta(days=_HORIZON_DAYS)).isoformat()
        if s else None
    )
    df["tp_price"] = df["avg_cost"] * (1 + _TAKE_PROFIT_PCT)

    if total_value and total_value > 0:
        mv_col = "live_market_value" if "live_market_value" in df.columns else "market_value"
        df["portfolio_weight_pct"] = df[mv_col].fillna(0) / total_value * 100

    return df


def format_holdings_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format the enriched holdings DataFrame for display.

    Expects compute_position_extras() to have been called first.
    """
    if df.empty:
        return df

    display = pd.DataFrame()
    display["Ticker"] = df["ticker"]
    display["Entry Date"] = df.get("entry_date", pd.Series([None] * len(df))).apply(fmt_date)
    display["Days Held"] = df.get("days_held", pd.Series([None] * len(df))).apply(
        lambda v: str(int(v)) if pd.notnull(v) else "—"
    )
    display["Days Left"] = df.get("days_remaining", pd.Series([None] * len(df))).apply(
        lambda v: f"{int(v)}d" if pd.notnull(v) else "—"
    )
    display["Expiry"] = df.get("expiry_date", pd.Series([None] * len(df))).apply(fmt_date)
    display["Entry Price"] = df["avg_cost"].apply(fmt_currency)
    display["TP Price"] = df.get("tp_price", pd.Series([None] * len(df))).apply(fmt_currency)
    display["Live Price"] = df.get("live_price", pd.Series([None] * len(df))).apply(fmt_currency)
    display["Return"] = df.get("unrealized_pnl_pct", pd.Series([None] * len(df))).apply(fmt_pct)
    display["PnL $"] = df.get("unrealized_pnl_live", df.get("unrealized_pnl", pd.Series([None] * len(df)))).apply(fmt_currency)
    display["Weight"] = df.get("portfolio_weight_pct", pd.Series([None] * len(df))).apply(
        lambda v: f"{v:.1f}%" if pd.notnull(v) else "—"
    )
    display["Mkt Value"] = df.get("live_market_value", df.get("market_value", pd.Series([None] * len(df)))).apply(fmt_currency)
    display["Shares"] = df["quantity"].apply(fmt_shares)
    return display


def format_positions_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format the positions DataFrame for display (compact version)."""
    if df.empty:
        return df
    display = pd.DataFrame()
    display["Ticker"] = df["ticker"]
    display["Shares"] = df["quantity"].apply(fmt_shares)
    display["Avg Cost"] = df["avg_cost"].apply(fmt_currency)
    display["Live Price"] = df.get("live_price", pd.Series([None] * len(df))).apply(fmt_currency)
    display["Market Value"] = df.get("live_market_value", df.get("market_value")).apply(fmt_currency)
    display["Unrealized PnL"] = df.get("unrealized_pnl_live", df.get("unrealized_pnl")).apply(fmt_currency)
    display["PnL %"] = df.get("unrealized_pnl_pct", pd.Series([None] * len(df))).apply(fmt_pct)
    display["Entry Date"] = df.get("entry_date", pd.Series([None] * len(df))).apply(fmt_date)
    return display


def format_trades_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format the trades DataFrame for display."""
    if df.empty:
        return df
    display = pd.DataFrame()
    display["Date"] = df["trade_date"].apply(fmt_date)
    display["Ticker"] = df["ticker"]
    display["Side"] = df["side"].apply(side_badge)
    display["Shares"] = df["quantity"].apply(fmt_shares)
    display["Price"] = df["price"].apply(fmt_currency)
    display["Amount"] = df["amount_usd"].apply(fmt_currency)
    display["Status"] = df.get("status", pd.Series(["filled"] * len(df)))
    return display


def format_announcements_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format the announcements DataFrame for display."""
    if df.empty:
        return df
    display = pd.DataFrame()
    display["Source"] = df["source"].apply(source_badge)
    display["Ticker"] = df["ticker"]
    display["Event"] = df["event_type"]
    display["Title"] = df["title"].str[:80] + "..." if "title" in df.columns else "—"
    display["Date"] = df["published_at"].apply(fmt_date)
    if "announcement_url" in df.columns:
        display["URL"] = df["announcement_url"].apply(lambda u: f"[Link]({u})" if u else "—")
    return display


def format_predictions_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format the predictions DataFrame for display."""
    if df.empty:
        return df
    display = pd.DataFrame()
    display["Ticker"] = df["ticker"]
    display["Signal"] = df["predicted_label"].apply(label_badge)
    display["Confidence"] = df["predicted_probability"].apply(lambda v: f"{v:.1%}" if v is not None else "—")
    display["Event"] = df.get("event_type", pd.Series([None] * len(df)))
    display["Model"] = df.get("model_version", pd.Series([None] * len(df)))
    display["Date"] = df.get("created_at", pd.Series([None] * len(df))).apply(fmt_date)
    return display


def format_journal_df(trades_df: pd.DataFrame, signals_df: pd.DataFrame, total_value: float) -> pd.DataFrame:
    """Format the daily journal view: BUY trades merged with signal confidence scores.

    Args:
        trades_df: Trades for the selected date (any side).
        signals_df: BUY signals for the selected date (with score column).
        total_value: Estimated total portfolio value for weight calculation.
    """
    if trades_df.empty:
        return pd.DataFrame()

    buys = trades_df[trades_df["side"] == "BUY"].copy()
    sells = trades_df[trades_df["side"] == "SELL"].copy()

    result_rows = []

    # BUY entries
    if not buys.empty:
        # Merge confidence score from signals
        score_map = {}
        if not signals_df.empty and "ticker" in signals_df.columns:
            score_map = dict(zip(signals_df["ticker"], signals_df["score"]))

        for _, row in buys.iterrows():
            ticker = row["ticker"]
            amount = row.get("amount_usd", 0) or 0
            weight = (amount / total_value * 100) if total_value > 0 else None
            result_rows.append({
                "Side": "🟢 ENTRY",
                "Ticker": ticker,
                "Entry Price": fmt_currency(row.get("price")),
                "Shares": fmt_shares(row.get("quantity")),
                "Position Size": fmt_currency(amount),
                "Portfolio Weight": f"{weight:.1f}%" if weight is not None else "—",
                "Confidence": f"{score_map.get(ticker, 0):.1%}" if score_map.get(ticker) else "—",
            })

    # SELL exits
    if not sells.empty:
        for _, row in sells.iterrows():
            result_rows.append({
                "Side": "🔴 EXIT",
                "Ticker": row["ticker"],
                "Entry Price": fmt_currency(row.get("price")),
                "Shares": fmt_shares(row.get("quantity")),
                "Position Size": fmt_currency(row.get("amount_usd")),
                "Portfolio Weight": "—",
                "Confidence": f"({row.get('exit_reason', '—')})",
            })

    return pd.DataFrame(result_rows)


def format_model_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format the model metrics DataFrame for display."""
    if df.empty:
        return df
    display = pd.DataFrame()
    display["Date"] = df["run_date"].apply(fmt_date)
    display["Accuracy"] = df.get("accuracy", pd.Series()).apply(lambda v: f"{v:.1%}" if pd.notnull(v) else "—")
    display["Precision"] = df.get("precision_score", pd.Series()).apply(lambda v: f"{v:.1%}" if pd.notnull(v) else "—")
    display["Recall"] = df.get("recall", pd.Series()).apply(lambda v: f"{v:.1%}" if pd.notnull(v) else "—")
    display["Specificity"] = df.get("specificity", pd.Series()).apply(lambda v: f"{v:.1%}" if pd.notnull(v) else "—")
    display["F1"] = df.get("f1_score", pd.Series()).apply(lambda v: f"{v:.1%}" if pd.notnull(v) else "—")
    display["ROC AUC"] = df.get("roc_auc", pd.Series()).apply(lambda v: f"{v:.3f}" if pd.notnull(v) else "—")
    display["Train N"] = df.get("n_train_samples", pd.Series()).fillna(0).astype(int)
    display["Test N"] = df.get("n_test_samples", pd.Series()).fillna(0).astype(int)
    return display
