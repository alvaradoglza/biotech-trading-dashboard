"""
db.py — Supabase read functions for the Streamlit dashboard.

All functions use @st.cache_data with appropriate TTLs to prevent excessive
database queries. The dashboard is read-only; all writes happen in the pipeline.
"""

import os
from datetime import datetime

import pandas as pd
import streamlit as st
from supabase import create_client, Client


@st.cache_resource
def get_supabase_client() -> Client:
    """Create and cache a Supabase client (created once per session)."""
    # Try Streamlit secrets first (for Streamlit Cloud deployment)
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets.get("SUPABASE_ANON_KEY") or st.secrets.get("SUPABASE_KEY")
    except (AttributeError, KeyError):
        # Fall back to environment variables (local dev / GitHub Actions)
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY", "")

    if not url or not key:
        st.error(
            "Supabase credentials not configured. "
            "Set SUPABASE_URL and SUPABASE_ANON_KEY in .streamlit/secrets.toml or environment variables."
        )
        st.stop()

    return create_client(url, key)


# ── Portfolio summary ─────────────────────────────────────────────────────────

@st.cache_data(ttl=300)  # refresh every 5 minutes
def load_portfolio_summary() -> dict:
    """Load latest portfolio snapshot + position count + recent trades count."""
    sb = get_supabase_client()

    # Latest snapshot
    snap_resp = (
        sb.table("portfolio_snapshots")
        .select("*")
        .order("snapshot_date", desc=True)
        .limit(1)
        .execute()
    )
    snapshot = snap_resp.data[0] if snap_resp.data else {}

    # Open position count
    pos_resp = sb.table("positions").select("ticker", count="exact").execute()
    n_positions = pos_resp.count or 0

    # Recent trades (last 30 days)
    trades_resp = (
        sb.table("trades")
        .select("id", count="exact")
        .gte("trade_date", _days_ago(30))
        .execute()
    )
    n_recent_trades = trades_resp.count or 0

    return {
        "cash": float(snapshot.get("cash", 0)),
        "equity_value": float(snapshot.get("equity_value", 0)),
        "total_value": float(snapshot.get("total_value", 0)),
        "n_positions": n_positions,
        "n_recent_trades": n_recent_trades,
        "snapshot_date": snapshot.get("snapshot_date"),
    }


# ── Positions ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_positions() -> pd.DataFrame:
    """Load all open positions."""
    sb = get_supabase_client()
    resp = sb.table("positions").select("*").order("market_value", desc=True).execute()
    if not resp.data:
        return pd.DataFrame(columns=["ticker", "quantity", "avg_cost", "market_value", "unrealized_pnl", "entry_date"])
    return pd.DataFrame(resp.data)


# ── Trades ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_recent_trades(limit: int = 50) -> pd.DataFrame:
    """Load the most recent trades."""
    sb = get_supabase_client()
    resp = (
        sb.table("trades")
        .select("*")
        .order("trade_date", desc=True)
        .limit(limit)
        .execute()
    )
    if not resp.data:
        return pd.DataFrame(columns=["trade_date", "ticker", "side", "quantity", "price", "amount_usd", "status"])
    return pd.DataFrame(resp.data)


# ── Announcements ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=600)  # 10 minutes
def load_recent_announcements(limit: int = 50, source: str | None = None) -> pd.DataFrame:
    """Load the most recent announcements, optionally filtered by source."""
    sb = get_supabase_client()
    query = (
        sb.table("announcements")
        .select("id, source, ticker, company_name, event_type, title, announcement_url, published_at")
        .order("published_at", desc=True)
        .limit(limit)
    )
    if source:
        query = query.eq("source", source)

    resp = query.execute()
    if not resp.data:
        return pd.DataFrame()
    return pd.DataFrame(resp.data)


# ── Predictions ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def load_recent_predictions(limit: int = 50) -> pd.DataFrame:
    """Load recent ML predictions with announcement details."""
    sb = get_supabase_client()
    resp = (
        sb.table("predictions")
        .select(
            "id, predicted_label, predicted_probability, model_version, created_at, "
            "announcements(ticker, event_type, title, published_at)"
        )
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    if not resp.data:
        return pd.DataFrame()

    rows = []
    for row in resp.data:
        ann = row.get("announcements") or {}
        rows.append({
            "prediction_id": row["id"],
            "ticker": ann.get("ticker"),
            "event_type": ann.get("event_type"),
            "title": ann.get("title"),
            "published_at": ann.get("published_at"),
            "predicted_label": row["predicted_label"],
            "predicted_probability": row["predicted_probability"],
            "model_version": row["model_version"],
            "created_at": row["created_at"],
        })
    return pd.DataFrame(rows)


# ── Portfolio history ─────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)  # 1 hour — history doesn't change intraday
def load_portfolio_history(days: int = 180) -> pd.DataFrame:
    """Load portfolio snapshot history for the equity curve chart."""
    sb = get_supabase_client()
    resp = (
        sb.table("portfolio_snapshots")
        .select("snapshot_date, cash, equity_value, total_value, n_positions")
        .gte("snapshot_date", _days_ago(days))
        .order("snapshot_date", desc=False)
        .execute()
    )
    if not resp.data:
        return pd.DataFrame(columns=["snapshot_date", "total_value", "cash", "equity_value"])
    return pd.DataFrame(resp.data)


# ── Model performance ─────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_model_runs(limit: int = 90) -> pd.DataFrame:
    """Load recent model run metrics for performance tracking chart."""
    sb = get_supabase_client()
    resp = (
        sb.table("model_runs")
        .select("*")
        .order("run_date", desc=True)
        .limit(limit)
        .execute()
    )
    if not resp.data:
        return pd.DataFrame()
    df = pd.DataFrame(resp.data)
    df = df.sort_values("run_date")  # chronological for charting
    return df


# ── Cache invalidation ────────────────────────────────────────────────────────

def invalidate_cache() -> None:
    """Clear all cached data (call after pipeline runs or manual refresh)."""
    load_portfolio_summary.clear()
    load_positions.clear()
    load_recent_trades.clear()
    load_recent_announcements.clear()
    load_recent_predictions.clear()
    load_portfolio_history.clear()
    load_model_runs.clear()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _days_ago(n: int) -> str:
    """Return ISO date string for n days ago."""
    from datetime import timedelta
    return (datetime.utcnow() - timedelta(days=n)).strftime("%Y-%m-%d")
