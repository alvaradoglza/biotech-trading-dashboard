"""
ui_helpers.py — Streamlit UI helper functions: formatting, badges, metric display.
"""

import pandas as pd
import streamlit as st


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

def format_positions_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format the positions DataFrame for display."""
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
