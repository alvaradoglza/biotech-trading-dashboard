"""
app.py — Biotech Trading Dashboard (Streamlit)

Main entry point for the Streamlit app. Reads all data from Supabase.
Does NOT run any pipeline logic — all computation happens in pipeline/run_daily.py.

Deploy: streamlit run app.py
"""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load .env for local development
load_dotenv()

# Page config — must be first Streamlit call
st.set_page_config(
    page_title="Biotech Trading Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import dashboard modules
from dashboard.db import (
    load_portfolio_summary,
    load_positions,
    load_recent_trades,
    load_recent_announcements,
    load_recent_predictions,
    load_portfolio_history,
    load_model_runs,
    invalidate_cache,
)
from dashboard.prices import get_live_prices, enrich_positions_with_prices
from dashboard.charts import (
    portfolio_history_chart,
    model_metrics_chart,
    model_roc_auc_chart,
    positions_chart,
    prediction_distribution_chart,
)
from dashboard.ui_helpers import (
    portfolio_metrics_row,
    format_positions_df,
    format_trades_df,
    format_announcements_df,
    format_predictions_df,
    format_model_metrics_df,
    source_badge,
    label_badge,
    fmt_currency,
    fmt_pct,
)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧬 Biotech Trading")
    st.caption("Regulatory announcement alpha strategy")
    st.divider()

    if st.button("🔄 Refresh Data", use_container_width=True):
        invalidate_cache()
        st.rerun()

    st.divider()
    st.subheader("Navigation")
    section = st.radio(
        "Go to",
        ["Portfolio Summary", "Positions", "Trades", "Announcements", "ML Predictions", "Model Performance"],
        label_visibility="collapsed",
    )

    # MLflow link
    st.divider()
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        st.markdown(f"[📊 MLflow Experiments]({mlflow_uri})")

    st.divider()
    st.caption("Data from Supabase · Prices from EODHD")
    st.caption("Pipeline runs daily at 7:07 AM MX")


# ── Load data ─────────────────────────────────────────────────────────────────

summary = load_portfolio_summary()

# ── Section: Portfolio Summary ────────────────────────────────────────────────

if section == "Portfolio Summary":
    st.title("Portfolio Summary")

    # Top metrics
    portfolio_metrics_row(summary)

    st.divider()

    # Portfolio history chart
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Portfolio Value Over Time")
        history_df = load_portfolio_history(days=180)
        st.plotly_chart(portfolio_history_chart(history_df), use_container_width=True)

    with col2:
        st.subheader("Quick Stats")
        if not history_df.empty:
            first_val = history_df["total_value"].iloc[0]
            last_val = history_df["total_value"].iloc[-1]
            total_return_pct = (last_val - first_val) / first_val * 100 if first_val > 0 else 0
            st.metric("Total Return", fmt_pct(total_return_pct), delta=fmt_currency(last_val - first_val))
            st.metric("Days Tracked", str(len(history_df)))
            st.metric("Peak Value", fmt_currency(history_df["total_value"].max()))
            st.metric("Min Value", fmt_currency(history_df["total_value"].min()))

    # Recent model performance mini-view
    st.divider()
    st.subheader("Latest Model Run")
    model_df = load_model_runs(limit=5)
    if not model_df.empty:
        latest = model_df.iloc[-1]
        cols = st.columns(6)
        cols[0].metric("Accuracy", f"{latest.get('accuracy', 0):.1%}" if pd.notnull(latest.get('accuracy')) else "—")
        cols[1].metric("Precision", f"{latest.get('precision_score', 0):.1%}" if pd.notnull(latest.get('precision_score')) else "—")
        cols[2].metric("Recall", f"{latest.get('recall', 0):.1%}" if pd.notnull(latest.get('recall')) else "—")
        cols[3].metric("Specificity", f"{latest.get('specificity', 0):.1%}" if pd.notnull(latest.get('specificity')) else "—")
        cols[4].metric("F1 Score", f"{latest.get('f1_score', 0):.1%}" if pd.notnull(latest.get('f1_score')) else "—")
        cols[5].metric("ROC AUC", f"{latest.get('roc_auc', 0):.3f}" if pd.notnull(latest.get('roc_auc')) else "—")

        if latest.get("mlflow_experiment_url"):
            st.markdown(f"[View in MLflow ↗]({latest['mlflow_experiment_url']})")
    else:
        st.info("No model runs recorded yet. Run the daily pipeline to train the model.")


# ── Section: Positions ────────────────────────────────────────────────────────

elif section == "Positions":
    st.title("Open Positions")
    st.caption("Live prices fetched from EODHD · Cached 15 minutes")

    positions_df = load_positions()

    if positions_df.empty:
        st.info("No open positions. Signals from the next pipeline run will open positions.")
    else:
        # Fetch live prices
        tickers_tuple = tuple(positions_df["ticker"].tolist())
        with st.spinner("Fetching live prices..."):
            prices = get_live_prices(tickers_tuple)

        positions_enriched = enrich_positions_with_prices(positions_df, prices)

        # Summary metrics
        total_equity = positions_enriched["live_market_value"].sum() if "live_market_value" in positions_enriched.columns else 0
        total_pnl = positions_enriched.get("unrealized_pnl_live", pd.Series([0])).sum()

        cols = st.columns(3)
        cols[0].metric("Open Positions", str(len(positions_enriched)))
        cols[1].metric("Total Equity Value", fmt_currency(total_equity))
        cols[2].metric("Total Unrealized PnL", fmt_currency(total_pnl))

        st.divider()

        # Positions bar chart
        col1, col2 = st.columns([3, 2])
        with col1:
            st.plotly_chart(positions_chart(positions_enriched), use_container_width=True)

        with col2:
            st.subheader("Positions Table")
            st.dataframe(
                format_positions_df(positions_enriched),
                use_container_width=True,
                hide_index=True,
            )


# ── Section: Trades ───────────────────────────────────────────────────────────

elif section == "Trades":
    st.title("Recent Trades")

    n_trades = st.selectbox("Show last N trades", [25, 50, 100, 200], index=1)
    trades_df = load_recent_trades(limit=n_trades)

    if trades_df.empty:
        st.info("No trades recorded yet.")
    else:
        # Summary
        buys = trades_df[trades_df["side"] == "BUY"]
        sells = trades_df[trades_df["side"] == "SELL"]
        cols = st.columns(3)
        cols[0].metric("Total Trades", str(len(trades_df)))
        cols[1].metric("Buys", str(len(buys)))
        cols[2].metric("Sells", str(len(sells)))

        st.divider()
        st.dataframe(
            format_trades_df(trades_df),
            use_container_width=True,
            hide_index=True,
        )


# ── Section: Announcements ────────────────────────────────────────────────────

elif section == "Announcements":
    st.title("Recent Announcements")

    col1, col2 = st.columns([2, 1])
    with col1:
        source_filter = st.selectbox(
            "Filter by source",
            ["All", "openfda", "clinicaltrials"],
            format_func=lambda s: source_badge(s) if s != "All" else "All Sources",
        )
    with col2:
        n_ann = st.selectbox("Show last N", [25, 50, 100], index=1)

    source = None if source_filter == "All" else source_filter
    ann_df = load_recent_announcements(limit=n_ann, source=source)

    if ann_df.empty:
        st.info("No announcements in the database yet. Run the pipeline to fetch them.")
    else:
        st.caption(f"Showing {len(ann_df)} announcements")
        formatted = format_announcements_df(ann_df)
        st.dataframe(formatted, use_container_width=True, hide_index=True)


# ── Section: ML Predictions ───────────────────────────────────────────────────

elif section == "ML Predictions":
    st.title("ML Predictions")
    st.caption("GradientBoosting classifier trained daily on all labeled announcements")

    preds_df = load_recent_predictions(limit=100)

    if preds_df.empty:
        st.info("No predictions yet. The model generates predictions after each daily run.")
    else:
        # Summary metrics
        buy_signals = preds_df[preds_df["predicted_label"] == 1]
        cols = st.columns(3)
        cols[0].metric("Total Predictions", str(len(preds_df)))
        cols[1].metric("BUY Signals", str(len(buy_signals)))
        cols[2].metric("Signal Rate", f"{len(buy_signals)/len(preds_df):.1%}")

        st.divider()

        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(prediction_distribution_chart(preds_df), use_container_width=True)
        with col2:
            st.subheader("Top BUY Signals")
            if not buy_signals.empty:
                top_signals = buy_signals.sort_values("predicted_probability", ascending=False).head(10)
                st.dataframe(
                    format_predictions_df(top_signals),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No BUY signals generated.")

        st.divider()
        st.subheader("All Recent Predictions")
        st.dataframe(format_predictions_df(preds_df), use_container_width=True, hide_index=True)


# ── Section: Model Performance ────────────────────────────────────────────────

elif section == "Model Performance":
    st.title("Model Performance")
    st.caption(
        "GBM trained daily · Train: all data except last 4 weeks · Test: last 4 weeks\n\n"
        "Metrics track whether the model is improving or degrading over time."
    )

    model_df = load_model_runs(limit=90)

    if model_df.empty:
        st.info("No model runs recorded. The pipeline logs metrics after each daily training run.")
    else:
        # Latest run summary
        latest = model_df.iloc[-1]
        st.subheader(f"Latest Run: {latest.get('run_date', 'Unknown')}")

        cols = st.columns(6)
        metrics_to_show = [
            ("accuracy", "Accuracy"),
            ("precision_score", "Precision"),
            ("recall", "Recall"),
            ("specificity", "Specificity"),
            ("f1_score", "F1 Score"),
            ("roc_auc", "ROC AUC"),
        ]
        for i, (col_key, label) in enumerate(metrics_to_show):
            val = latest.get(col_key)
            if pd.notnull(val):
                if col_key == "roc_auc":
                    cols[i].metric(label, f"{val:.3f}")
                else:
                    cols[i].metric(label, f"{val:.1%}")
            else:
                cols[i].metric(label, "—")

        # MLflow link
        if latest.get("mlflow_experiment_url"):
            st.markdown(f"[📊 View MLflow experiment ↗]({latest['mlflow_experiment_url']})")

        st.divider()

        # Charts
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Metrics Over Time")
            st.plotly_chart(model_metrics_chart(model_df), use_container_width=True)
        with col2:
            st.subheader("ROC AUC Over Time")
            st.plotly_chart(model_roc_auc_chart(model_df), use_container_width=True)

        # Full table
        st.divider()
        st.subheader("All Model Runs")
        st.dataframe(
            format_model_metrics_df(model_df.iloc[::-1]),  # newest first
            use_container_width=True,
            hide_index=True,
        )
