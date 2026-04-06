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
    load_trade_dates,
    load_trades_for_date,
    load_signals_for_date,
    invalidate_cache,
)
from dashboard.prices import get_live_prices, enrich_positions_with_prices
from dashboard.charts import (
    portfolio_history_chart,
    model_metrics_chart,
    model_roc_auc_chart,
    positions_chart,
    prediction_distribution_chart,
    portfolio_composition_chart,
    tp_progress_chart,
)
from dashboard.ui_helpers import (
    portfolio_metrics_row,
    compute_position_extras,
    format_holdings_df,
    format_positions_df,
    format_trades_df,
    format_announcements_df,
    format_predictions_df,
    format_journal_df,
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
        ["Portfolio Summary", "Holdings", "Trade Journal", "Trades", "Announcements", "ML Predictions", "Model Performance"],
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
        initial_capital = float(summary.get("initial_capital", 1_000_000.0))
        current_val = float(summary.get("total_value", 0))
        if current_val > 0:
            total_return_pct = (current_val - initial_capital) / initial_capital * 100
            st.metric(
                "Total Return",
                fmt_pct(total_return_pct),
                delta=fmt_currency(current_val - initial_capital),
            )
        if not history_df.empty:
            st.metric("Days Tracked", str(len(history_df)))
            st.metric("Peak Value", fmt_currency(history_df["total_value"].max()))
            st.metric("Min Value", fmt_currency(history_df["total_value"].min()))
        st.metric("Initial Capital", fmt_currency(initial_capital))

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


# ── Section: Holdings ────────────────────────────────────────────────────────

elif section == "Holdings":
    st.title("Holdings")
    st.caption("Live prices from EODHD · Cached 15 min · TP = +30% · Horizon = 50 days")

    positions_df = load_positions()

    if positions_df.empty:
        st.info("No open positions. Signals from the next pipeline run will open positions.")
    else:
        tickers_tuple = tuple(positions_df["ticker"].tolist())
        with st.spinner("Fetching live prices..."):
            prices = get_live_prices(tickers_tuple)

        positions_enriched = enrich_positions_with_prices(positions_df, prices)

        # Cash from latest snapshot
        cash = float(summary.get("cash", 0))
        total_equity = positions_enriched["live_market_value"].dropna().sum() if "live_market_value" in positions_enriched.columns else float(summary.get("equity_value", 0))
        total_value = cash + total_equity

        positions_enriched = compute_position_extras(positions_enriched, total_value)

        total_pnl = positions_enriched.get("unrealized_pnl_live", pd.Series(dtype=float)).dropna().sum()
        total_pnl_pct = total_pnl / total_equity * 100 if total_equity > 0 else 0
        n_winning = int((positions_enriched.get("unrealized_pnl_pct", pd.Series(dtype=float)).fillna(0) > 0).sum())

        # Top metrics
        cols = st.columns(5)
        cols[0].metric("Open Positions", str(len(positions_enriched)))
        cols[1].metric("Total Equity", fmt_currency(total_equity))
        cols[2].metric("Unrealized PnL", fmt_currency(total_pnl), delta=fmt_pct(total_pnl_pct))
        cols[3].metric("Cash", fmt_currency(cash))
        cols[4].metric("Winning Positions", f"{n_winning}/{len(positions_enriched)}")

        st.divider()

        # Composition donut + TP progress side by side
        col1, col2 = st.columns([1, 2])
        with col1:
            st.plotly_chart(
                portfolio_composition_chart(positions_enriched, cash, total_value),
                use_container_width=True,
            )
        with col2:
            st.plotly_chart(
                tp_progress_chart(positions_enriched),
                use_container_width=True,
            )

        st.divider()

        # Full holdings table
        st.subheader("All Holdings")
        st.caption("Entry Price → TP Price marks the 30% take-profit target. Days Left = calendar days before 50-day horizon exit.")
        st.dataframe(
            format_holdings_df(positions_enriched.sort_values("unrealized_pnl_pct", ascending=False, na_position="last")),
            use_container_width=True,
            hide_index=True,
        )


# ── Section: Trade Journal ────────────────────────────────────────────────────

elif section == "Trade Journal":
    st.title("Trade Journal")
    st.caption("Browse positions opened or closed on any given trading day.")

    trade_dates = load_trade_dates()

    if not trade_dates:
        st.info("No trades recorded yet. Run the daily pipeline to generate trades.")
    else:
        col1, col2 = st.columns([2, 3])
        with col1:
            selected_date = st.selectbox(
                "Select trading day",
                options=trade_dates,
                format_func=lambda d: d,
            )

        trades_on_date = load_trades_for_date(selected_date)
        signals_on_date = load_signals_for_date(selected_date)

        buys_on_date = trades_on_date[trades_on_date["side"] == "BUY"] if not trades_on_date.empty else pd.DataFrame()
        sells_on_date = trades_on_date[trades_on_date["side"] == "SELL"] if not trades_on_date.empty else pd.DataFrame()

        with col2:
            entry_count = len(buys_on_date)
            exit_count = len(sells_on_date)
            total_deployed = buys_on_date["amount_usd"].sum() if not buys_on_date.empty else 0
            st.markdown(f"**{selected_date}** — {entry_count} entr{'y' if entry_count == 1 else 'ies'}, {exit_count} exit{'s' if exit_count != 1 else ''}")

        if trades_on_date.empty:
            st.info("No trades on this date.")
        else:
            # Use today's total value as proxy for weight calculation
            total_value_est = float(summary.get("total_value", 1_000_000))

            journal_df = format_journal_df(trades_on_date, signals_on_date, total_value_est)

            if not journal_df.empty:
                st.dataframe(journal_df, use_container_width=True, hide_index=True)

            # Summary metrics for the day
            if not buys_on_date.empty:
                st.divider()
                cols = st.columns(3)
                cols[0].metric("Positions Opened", str(entry_count))
                cols[1].metric("Capital Deployed", fmt_currency(total_deployed))
                cols[2].metric(
                    "Avg Confidence",
                    f"{signals_on_date['score'].mean():.1%}" if not signals_on_date.empty and "score" in signals_on_date.columns else "—"
                )

                st.caption(
                    "**Portfolio Weight** = position size ÷ total portfolio value at time of entry. "
                    "Positions are sized at ~7% of portfolio (equal-weight). "
                    "Confidence = GBM model predicted probability (higher = stronger signal)."
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
    st.caption("GBM trained on 100% of labeled data · Confidence = P(return > 7.22% in 30d) · BUY signal when confidence ≥ 50%")

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
        "GBM trained daily · Two-phase: eval on 80/20 temporal split, then retrain on 100% for live predictions\n\n"
        "Train = first 80% of labeled announcements (chronological) · Test = most recent 20% · "
        "Confidence = P(return > 7.22% in 30d)"
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
