"""Streamlit dashboard for the Macro Regime Nowcaster.

Panels:
    1. Current regime banner (colour-coded)
    2. Regime probability time-series chart (stacked area)
    3. Latent factor time series (4 panels)
    4. Historical regime timeline (shaded bars + NBER recessions)
    5. Asset allocation pie chart
    6. LLM narrative summary card
    7. Key macro data table

Run with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on the path when run directly
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from loguru import logger

from src.utils.logging_config import setup_logging

setup_logging(level="WARNING")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Macro Regime Nowcaster",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Colour palette for regimes
# ---------------------------------------------------------------------------
REGIME_COLORS = {
    "expansion": "#2ecc71",
    "recovery":  "#3498db",
    "slowdown":  "#f39c12",
    "recession": "#e74c3c",
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Controls")
    start_date = st.date_input("Start Date", value=pd.Timestamp("2000-01-01"))
    end_date   = st.date_input("End Date",   value=pd.Timestamp.today())
    n_factors  = st.slider("Latent Factors", min_value=1, max_value=8, value=4)
    n_regimes  = st.slider("Regimes",        min_value=2, max_value=6, value=4)
    run_button = st.button("🔄 Run Nowcast", type="primary")

# ---------------------------------------------------------------------------
# Main title
# ---------------------------------------------------------------------------
st.title("📊 Macro Regime Nowcaster")
st.caption(
    "Real-time economic regime detection · Dynamic Factor Model · Markov-Switching"
)

# ---------------------------------------------------------------------------
# Helper: try to load cached results or show instructions
# ---------------------------------------------------------------------------

def _show_instructions() -> None:
    st.info(
        """
        **No nowcast data available.**

        To get started:
        1. Copy `.env.example` → `.env` and add your `FRED_API_KEY`
        2. Run `make fetch-data` to download FRED data
        3. Run `make train` to estimate model parameters
        4. Click **Run Nowcast** in the sidebar
        """,
        icon="ℹ️",
    )


@st.cache_data(ttl=3600, show_spinner="Running nowcast pipeline…")
def _run_nowcast(start: str, end: str, n_fac: int, n_reg: int):
    """Cached nowcast execution."""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        return None, "FRED_API_KEY not set — add it to your .env file."

    try:
        from src.data.fred_client import FREDClient
        from src.data.data_pipeline import DataPipeline
        from src.models.nowcaster import Nowcaster

        client = FREDClient(api_key=api_key)
        pipeline = DataPipeline(
            fred_client=client,
            start_date=start,
        )
        nowcaster = Nowcaster(pipeline=pipeline, n_factors=n_fac, n_regimes=n_reg)
        result = nowcaster.run(end_date=end)
        factors = nowcaster._last_factors
        regime_probs = nowcaster._rsm.get_regime_probabilities()
        return {"result": result, "factors": factors, "regime_probs": regime_probs}, None
    except Exception as exc:  # noqa: BLE001
        logger.exception("Nowcast run failed")
        return None, str(exc)


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
data = None
error_msg = None

if run_button:
    data, error_msg = _run_nowcast(
        str(start_date), str(end_date), n_factors, n_regimes
    )

if error_msg:
    st.error(f"❌ {error_msg}")
elif data is None:
    _show_instructions()
else:
    result = data["result"]
    factors: pd.DataFrame = data["factors"]
    regime_probs: pd.DataFrame = data["regime_probs"]

    # ------------------------------------------------------------------
    # 1. Regime banner
    # ------------------------------------------------------------------
    regime = result.current_regime
    color = REGIME_COLORS.get(regime, "#95a5a6")
    st.markdown(
        f"""
        <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
            <h2 style="color:white; margin:0;">Current Regime: {regime.upper()}</h2>
            <p style="color:white; margin:4px 0 0 0; font-size:1.1em;">
                GDP Nowcast: {result.gdp_nowcast:.2f}% annualised
                (90% CI: [{result.gdp_ci_lower:.2f}%, {result.gdp_ci_upper:.2f}%])
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col1, col2 = st.columns([3, 1])

    # ------------------------------------------------------------------
    # 2. Regime probability time series
    # ------------------------------------------------------------------
    with col1:
        st.subheader("Regime Probabilities")
        fig_probs = go.Figure()
        for regime_name in regime_probs.columns:
            fig_probs.add_trace(
                go.Scatter(
                    x=regime_probs.index,
                    y=regime_probs[regime_name],
                    name=regime_name.title(),
                    mode="lines",
                    stackgroup="one",
                    fillcolor=REGIME_COLORS.get(regime_name, "#95a5a6"),
                    line=dict(width=0.5),
                )
            )
        fig_probs.update_layout(
            yaxis_title="Probability",
            xaxis_title="Date",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=300,
        )
        st.plotly_chart(fig_probs, use_container_width=True)

    # ------------------------------------------------------------------
    # 5. Asset allocation pie chart
    # ------------------------------------------------------------------
    with col2:
        st.subheader("Asset Allocation")
        from src.allocation.regime_allocator import RegimeAllocator
        allocator = RegimeAllocator()
        allocation = allocator.get_allocation(result.regime_probabilities)
        fig_pie = px.pie(
            names=list(allocation.keys()),
            values=list(allocation.values()),
            color=list(allocation.keys()),
            color_discrete_map={
                "equities": "#2ecc71",
                "bonds": "#3498db",
                "commodities": "#f39c12",
                "cash": "#bdc3c7",
            },
        )
        fig_pie.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    # ------------------------------------------------------------------
    # 3. Latent factor time series
    # ------------------------------------------------------------------
    st.subheader("Latent Factors")
    factor_cols = factors.columns.tolist()
    n_cols = min(len(factor_cols), 4)
    factor_chart_cols = st.columns(n_cols)
    for i, col_name in enumerate(factor_cols[:n_cols]):
        with factor_chart_cols[i]:
            fig_f = go.Figure()
            fig_f.add_trace(
                go.Scatter(
                    x=factors.index,
                    y=factors[col_name],
                    mode="lines",
                    line=dict(color="#3498db"),
                )
            )
            fig_f.update_layout(
                title=col_name.replace("_", " ").title(),
                height=200,
                margin=dict(t=30, b=10, l=10, r=10),
                xaxis=dict(showticklabels=False),
            )
            st.plotly_chart(fig_f, use_container_width=True)

    # ------------------------------------------------------------------
    # 4. Regime timeline
    # ------------------------------------------------------------------
    st.subheader("Regime Timeline")
    dominant_regime = regime_probs.idxmax(axis=1)
    regime_numeric = dominant_regime.map(
        {r: i for i, r in enumerate(regime_probs.columns)}
    )
    fig_timeline = go.Figure()
    fig_timeline.add_trace(
        go.Scatter(
            x=regime_probs.index,
            y=regime_numeric,
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(52,152,219,0.3)",
        )
    )
    fig_timeline.update_layout(
        yaxis=dict(
            tickvals=list(range(len(regime_probs.columns))),
            ticktext=[r.title() for r in regime_probs.columns],
            title="Regime",
        ),
        xaxis_title="Date",
        height=200,
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # ------------------------------------------------------------------
    # 6. LLM narrative card
    # ------------------------------------------------------------------
    st.subheader("Narrative Summary")
    with st.expander("Generate LLM Narrative (requires OPENAI_API_KEY)", expanded=False):
        if st.button("Generate Narrative"):
            with st.spinner("Generating narrative…"):
                try:
                    from src.agent.narrative_agent import NarrativeAgent
                    agent = NarrativeAgent()
                    report = agent.generate(result.to_dict())
                    st.markdown(f"**Summary:** {report.summary}")
                    st.markdown("**Key Drivers:**")
                    for driver in report.key_drivers:
                        st.markdown(f"- {driver}")
                    st.markdown("**Risk Flags:**")
                    for flag in report.risk_flags:
                        st.markdown(f"- {flag}")
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"Narrative generation failed: {exc}")

    # ------------------------------------------------------------------
    # 7. Key macro data table
    # ------------------------------------------------------------------
    st.subheader("Latest Regime Probabilities")
    st.dataframe(
        regime_probs.tail(12).style.format("{:.1%}"),
        use_container_width=True,
    )
