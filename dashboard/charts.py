"""
charts.py — Plotly chart builders for the Streamlit dashboard.

All functions accept DataFrames and return Plotly figures ready for st.plotly_chart().
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Portfolio history ─────────────────────────────────────────────────────────

def portfolio_history_chart(df: pd.DataFrame) -> go.Figure:
    """Line chart of total portfolio value over time.

    Args:
        df: DataFrame with columns: snapshot_date, total_value, cash, equity_value.

    Returns:
        Plotly figure.
    """
    if df.empty:
        return _empty_chart("No portfolio history yet")

    df = df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["snapshot_date"],
        y=df["total_value"],
        mode="lines",
        name="Total Value",
        line=dict(color="#0A84FF", width=2),
        fill="tozeroy",
        fillcolor="rgba(10, 132, 255, 0.1)",
    ))

    fig.add_trace(go.Scatter(
        x=df["snapshot_date"],
        y=df["equity_value"],
        mode="lines",
        name="Equity",
        line=dict(color="#30D158", width=1.5, dash="dot"),
    ))

    fig.add_trace(go.Scatter(
        x=df["snapshot_date"],
        y=df["cash"],
        mode="lines",
        name="Cash",
        line=dict(color="#FF9F0A", width=1.5, dash="dot"),
    ))

    fig.update_layout(
        **_dark_layout(),
        title="Portfolio Value Over Time",
        xaxis_title=None,
        yaxis_title="Value (USD)",
        yaxis_tickformat="$,.0f",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


# ── Model performance ─────────────────────────────────────────────────────────

def model_metrics_chart(df: pd.DataFrame) -> go.Figure:
    """Multi-line chart of model evaluation metrics over time.

    Shows accuracy, precision, recall, specificity, F1 — one line each.

    Args:
        df: DataFrame with columns: run_date, accuracy, precision_score,
            recall, specificity, f1_score.
    """
    if df.empty:
        return _empty_chart("No model runs recorded yet")

    df = df.copy()
    df["run_date"] = pd.to_datetime(df["run_date"])

    metrics = [
        ("accuracy", "#0A84FF", "Accuracy"),
        ("precision_score", "#30D158", "Precision"),
        ("recall", "#FF9F0A", "Recall"),
        ("specificity", "#BF5AF2", "Specificity"),
        ("f1_score", "#FF375F", "F1 Score"),
    ]

    fig = go.Figure()
    for col, color, name in metrics:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["run_date"],
                y=df[col],
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=1.5),
                marker=dict(size=5),
            ))

    fig.update_layout(
        **_dark_layout(),
        title="Daily Model Performance",
        xaxis_title=None,
        yaxis_title="Score",
        yaxis_range=[0, 1],
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


def model_roc_auc_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of ROC AUC over time."""
    if df.empty or "roc_auc" not in df.columns:
        return _empty_chart("No ROC AUC data")

    df = df.copy()
    df["run_date"] = pd.to_datetime(df["run_date"])
    df = df.dropna(subset=["roc_auc"])

    fig = go.Figure(go.Bar(
        x=df["run_date"],
        y=df["roc_auc"],
        marker_color="#0A84FF",
        name="ROC AUC",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Random (0.5)")

    fig.update_layout(
        **_dark_layout(),
        title="Daily ROC AUC",
        xaxis_title=None,
        yaxis_title="ROC AUC",
        yaxis_range=[0, 1],
    )
    return fig


# ── Positions ─────────────────────────────────────────────────────────────────

def positions_chart(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of open positions by market value."""
    if df.empty:
        return _empty_chart("No open positions")

    market_val_col = "live_market_value" if "live_market_value" in df.columns else "market_value"
    df = df.copy().sort_values(market_val_col, ascending=True)

    # Color by unrealized PnL
    pnl_col = "unrealized_pnl_live" if "unrealized_pnl_live" in df.columns else "unrealized_pnl"
    colors = []
    for _, row in df.iterrows():
        pnl = row.get(pnl_col, 0) or 0
        colors.append("#30D158" if pnl >= 0 else "#FF375F")

    fig = go.Figure(go.Bar(
        x=df[market_val_col],
        y=df["ticker"],
        orientation="h",
        marker_color=colors,
        text=df[market_val_col].apply(lambda v: f"${v:,.0f}" if v else ""),
        textposition="inside",
    ))

    fig.update_layout(
        **_dark_layout(),
        title="Open Positions by Market Value",
        xaxis_title="Market Value (USD)",
        xaxis_tickformat="$,.0f",
        height=max(300, len(df) * 35 + 100),
        showlegend=False,
    )
    return fig


# ── Announcement signal distribution ─────────────────────────────────────────

def prediction_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Histogram of predicted probabilities for recent announcements."""
    if df.empty or "predicted_probability" not in df.columns:
        return _empty_chart("No predictions yet")

    fig = go.Figure(go.Histogram(
        x=df["predicted_probability"],
        nbinsx=20,
        marker_color="#0A84FF",
        name="Score distribution",
    ))

    fig.update_layout(
        **_dark_layout(),
        title="Prediction Score Distribution",
        xaxis_title="Predicted Probability",
        yaxis_title="Count",
        bargap=0.1,
    )
    return fig


# ── Portfolio composition donut ───────────────────────────────────────────────

def portfolio_composition_chart(positions_df: pd.DataFrame, cash: float, total_value: float) -> go.Figure:
    """Donut chart showing cash vs each stock position as share of total portfolio.

    Args:
        positions_df: Enriched positions DataFrame with live_market_value or market_value.
        cash: Current cash balance.
        total_value: Total portfolio value (cash + equity).
    """
    if total_value <= 0:
        return _empty_chart("No portfolio data")

    labels = []
    values = []
    colors = []

    # Cash slice
    labels.append("Cash")
    values.append(max(cash, 0))
    colors.append("#FF9F0A")

    # One slice per position
    palette = [
        "#0A84FF", "#30D158", "#BF5AF2", "#FF375F", "#5AC8FA",
        "#FFD60A", "#FF6961", "#77DD77", "#AEC6CF", "#CFCFC4",
        "#B39EB5", "#FFB347", "#87CEEB", "#DDA0DD", "#98FB98",
        "#F08080", "#20B2AA", "#FF7F50", "#9370DB", "#3CB371",
    ]

    mv_col = "live_market_value" if "live_market_value" in positions_df.columns else "market_value"

    for i, (_, row) in enumerate(positions_df.iterrows()):
        mv = row.get(mv_col) or 0
        if mv > 0:
            labels.append(row["ticker"])
            values.append(mv)
            colors.append(palette[i % len(palette)])

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#1a1a2e", width=2)),
        textinfo="label+percent",
        hovertemplate="%{label}<br>$%{value:,.0f}<br>%{percent}<extra></extra>",
        sort=False,
    ))

    fig.update_layout(
        **_dark_layout(),
        title="Portfolio Composition",
        showlegend=False,
        annotations=[dict(
            text=f"${total_value:,.0f}",
            x=0.5, y=0.5,
            font_size=14,
            font_color="#FAFAFA",
            showarrow=False,
        )],
        height=350,
    )
    return fig


# ── Positions progress bars (TP progress) ─────────────────────────────────────

def tp_progress_chart(positions_df: pd.DataFrame) -> go.Figure:
    """Horizontal bullet chart showing current return vs 30% TP target per position."""
    if positions_df.empty or "unrealized_pnl_pct" not in positions_df.columns:
        return _empty_chart("No position data")

    df = positions_df.dropna(subset=["unrealized_pnl_pct"]).copy()
    if df.empty:
        return _empty_chart("No live price data available")

    df = df.sort_values("unrealized_pnl_pct", ascending=True)

    fig = go.Figure()

    # Background bar (0 → 30% = TP target)
    fig.add_trace(go.Bar(
        x=[30] * len(df),
        y=df["ticker"],
        orientation="h",
        marker_color="rgba(255,255,255,0.05)",
        showlegend=False,
        hoverinfo="skip",
    ))

    # Current return bar
    colors = ["#30D158" if v >= 0 else "#FF375F" for v in df["unrealized_pnl_pct"]]
    fig.add_trace(go.Bar(
        x=df["unrealized_pnl_pct"].clip(-30, 35),
        y=df["ticker"],
        orientation="h",
        marker_color=colors,
        text=df["unrealized_pnl_pct"].apply(lambda v: f"{v:+.1f}%"),
        textposition="outside",
        name="Current Return",
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))

    # TP line at 30%
    fig.add_vline(x=30, line_dash="dash", line_color="#FFD60A",
                  annotation_text="TP 30%", annotation_position="top")
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.3)")

    fig.update_layout(
        **_dark_layout(),
        title="Return vs Take-Profit Target",
        xaxis_title="Return (%)",
        xaxis_ticksuffix="%",
        barmode="overlay",
        height=max(280, len(df) * 35 + 100),
        showlegend=False,
    )
    return fig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dark_layout() -> dict:
    """Base Plotly layout matching our dark Streamlit theme."""
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA", size=12),
        margin=dict(l=20, r=20, t=50, b=20),
    )


def _empty_chart(message: str) -> go.Figure:
    """Return a blank chart with a centered message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="#888"),
    )
    fig.update_layout(
        **_dark_layout(),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=200,
    )
    return fig
