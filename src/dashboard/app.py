"""Streamlit dashboard for the Macro Regime Nowcaster.

Panels:
    1. Colour-coded regime banner with key metrics
    2. Ensemble recession probability gauge + breakdown
    3. Regime probability time-series chart (stacked area, NBER shaded)
    4. Latent factor time series (4 panels)
    5. Regime timeline with NBER recession shading
    6. Asset allocation pie chart (from RegimeAllocator)
    7. LLM narrative summary card (NarrativeAgent + FedScraper)
    8. Recent regime probability data table

Run with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on the path when run directly
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from loguru import logger

from src.utils.logging_config import setup_logging

setup_logging(level="WARNING")

# ---------------------------------------------------------------------------
# NBER recession dates for shading
# ---------------------------------------------------------------------------
NBER_RECESSIONS = [
    ("1980-01-01", "1980-07-31"),
    ("1981-07-01", "1982-11-30"),
    ("1990-07-01", "1991-03-31"),
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]

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
# Colour palette
# ---------------------------------------------------------------------------
REGIME_COLORS = {
    "expansion": "#2ecc71",
    "recession": "#e74c3c",
}

ASSET_COLORS = {
    "equities": "#2ecc71",
    "bonds": "#3498db",
    "commodities": "#f39c12",
    "cash": "#bdc3c7",
}

SIGNAL_COLORS = {
    "rsm": "#9b59b6",
    "probit": "#3498db",
    "cfnai": "#e67e22",
    "ensemble": "#e74c3c",
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Controls")
    start_date = st.date_input("Start Date", value=pd.Timestamp("2000-01-01"))
    end_date = st.date_input("End Date", value=pd.Timestamp.today())
    n_factors = st.slider("Latent Factors", min_value=1, max_value=8, value=4)
    run_button = st.button("🔄 Run Nowcast", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("**Architecture**")
    st.markdown(
        "DFM → RSM + Probit + CFNAI → Ensemble → Allocation"
    )
    st.markdown(
        "- **RSM weight:** 0.25\n"
        "- **Probit weight:** 0.50\n"
        "- **CFNAI weight:** 0.25"
    )

# ---------------------------------------------------------------------------
# Main title
# ---------------------------------------------------------------------------
st.title("📊 Macro Regime Nowcaster")
st.caption(
    "Real-time economic regime detection · Dynamic Factor Model · "
    "Markov-Switching · Ensemble Recession Probability"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_nber_shading(fig: go.Figure, x_min=None, x_max=None) -> None:
    """Add NBER recession shading rectangles to a Plotly figure."""
    for start, end in NBER_RECESSIONS:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        if x_max and s > pd.Timestamp(x_max):
            continue
        if x_min and e < pd.Timestamp(x_min):
            continue
        fig.add_vrect(
            x0=s, x1=e,
            fillcolor="rgba(200,200,200,0.25)",
            layer="below",
            line_width=0,
            annotation_text="",
        )


def _show_instructions() -> None:
    st.info(
        """
        **No nowcast data available.**

        To get started:
        1. Copy `.env.example` → `.env` and add your `FRED_API_KEY`
        2. Run `make fetch-data` to download FRED data
        3. Click **🔄 Run Nowcast** in the sidebar

        The pipeline will fetch data from FRED, fit the Dynamic Factor Model,
        run ensemble recession detection, and display results here.
        """,
        icon="ℹ️",
    )


@st.cache_data(ttl=3600, show_spinner="Running nowcast pipeline…")
def _run_nowcast(start: str, end: str, n_fac: int):
    """Cached nowcast execution.  Returns all artefacts needed for display."""
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
        pipeline = DataPipeline(fred_client=client, start_date=start)
        nowcaster = Nowcaster(
            pipeline=pipeline,
            n_factors=n_fac,
            n_regimes=2,
            use_ensemble=True,
        )
        result = nowcaster.run(end_date=end)
        factors = nowcaster._last_factors
        regime_probs = nowcaster.get_ensemble_probabilities()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        return {
            "result": result,
            "factors": factors,
            "regime_probs": regime_probs,
            "timestamp": timestamp,
        }, None
    except Exception as exc:  # noqa: BLE001
        logger.exception("Nowcast run failed")
        return None, str(exc)


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
data = None
error_msg = None

if run_button:
    data, error_msg = _run_nowcast(str(start_date), str(end_date), n_factors)
    if data is not None:
        st.session_state["nowcast_data"] = data
    if error_msg:
        st.session_state.pop("nowcast_data", None)

# Restore from session state so data survives button reruns
if data is None and "nowcast_data" in st.session_state:
    data = st.session_state["nowcast_data"]

if error_msg:
    st.error(f"❌ {error_msg}")
elif data is None:
    _show_instructions()
else:
    result = data["result"]
    factors: pd.DataFrame = data["factors"]
    regime_probs: pd.DataFrame = data["regime_probs"]
    timestamp: str = data["timestamp"]

    # ==================================================================
    # 1. Regime banner
    # ==================================================================
    regime = result.current_regime
    p_recession = result.recession_probability
    color = REGIME_COLORS.get(regime, "#95a5a6")

    st.markdown(
        f"""
        <div style="background-color:{color}; padding:24px; border-radius:12px;
                    text-align:center; margin-bottom:16px;">
            <h1 style="color:white; margin:0; font-size:2.2em;">
                {regime.upper()}
            </h1>
            <p style="color:rgba(255,255,255,0.9); margin:6px 0 0 0; font-size:1.15em;">
                GDP Nowcast: {result.gdp_nowcast:.2f}% ann.
                &nbsp;|&nbsp;
                Recession Probability: {p_recession:.1%}
                &nbsp;|&nbsp;
                {timestamp}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ==================================================================
    # 2. Key metrics row
    # ==================================================================
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("GDP Nowcast", f"{result.gdp_nowcast:.2f}%",
              f"CI: [{result.gdp_ci_lower:.1f}%, {result.gdp_ci_upper:.1f}%]")
    m2.metric("Recession Prob", f"{p_recession:.1%}")
    m3.metric("Current Regime", regime.title())
    m4.metric("Model Signals", f"{len(result.ensemble_detail)} active")

    st.markdown("---")

    # ==================================================================
    # 3. Ensemble breakdown + Asset allocation (side by side)
    # ==================================================================
    col_ens, col_alloc = st.columns([3, 2])

    with col_ens:
        st.subheader("Ensemble Signal Breakdown")
        detail = result.ensemble_detail
        if detail:
            signal_names = []
            signal_vals = []
            signal_colors = []
            for key in ["rsm", "probit", "cfnai"]:
                if key in detail:
                    label = {"rsm": "RSM (Markov)", "probit": "Probit", "cfnai": "CFNAI"}[key]
                    signal_names.append(label)
                    signal_vals.append(detail[key])
                    signal_colors.append(SIGNAL_COLORS[key])

            # Add ensemble bar
            if "ensemble" in detail:
                signal_names.append("Ensemble")
                signal_vals.append(detail["ensemble"])
                signal_colors.append(SIGNAL_COLORS["ensemble"])

            fig_ens = go.Figure()
            fig_ens.add_trace(go.Bar(
                x=signal_vals,
                y=signal_names,
                orientation="h",
                marker_color=signal_colors,
                text=[f"{v:.1%}" for v in signal_vals],
                textposition="auto",
            ))
            fig_ens.add_vline(x=0.5, line_dash="dash", line_color="grey",
                              annotation_text="50% threshold")
            fig_ens.update_layout(
                xaxis=dict(range=[0, 1], title="P(Recession)", tickformat=".0%"),
                yaxis=dict(autorange="reversed"),
                height=250,
                margin=dict(l=10, r=10, t=10, b=30),
            )
            st.plotly_chart(fig_ens, use_container_width=True)
        else:
            st.info("Ensemble detail not available")

    with col_alloc:
        st.subheader("Portfolio Allocation")
        from src.allocation.regime_allocator import RegimeAllocator
        allocator = RegimeAllocator()
        allocation = allocator.get_allocation_from_nowcast(result)

        fig_pie = px.pie(
            names=list(allocation.keys()),
            values=list(allocation.values()),
            color=list(allocation.keys()),
            color_discrete_map=ASSET_COLORS,
        )
        fig_pie.update_traces(
            textinfo="label+percent",
            textposition="inside",
        )
        fig_pie.update_layout(
            height=250,
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Show allocation table
        alloc_df = pd.DataFrame(
            {"Weight": [f"{v:.1%}" for v in allocation.values()]},
            index=[k.title() for k in allocation.keys()],
        )
        st.dataframe(alloc_df, use_container_width=True)

    st.markdown("---")

    # ==================================================================
    # 4. Regime probability time series with NBER shading
    # ==================================================================
    st.subheader("Regime Probabilities Over Time")

    if isinstance(regime_probs, pd.DataFrame) and len(regime_probs) > 0:
        fig_probs = go.Figure()

        for col_name in regime_probs.columns:
            fig_probs.add_trace(
                go.Scatter(
                    x=regime_probs.index,
                    y=regime_probs[col_name],
                    name=col_name.title(),
                    mode="lines",
                    stackgroup="one",
                    fillcolor=REGIME_COLORS.get(col_name, "#95a5a6"),
                    line=dict(width=0.5, color=REGIME_COLORS.get(col_name, "#95a5a6")),
                )
            )

        _add_nber_shading(fig_probs)

        fig_probs.update_layout(
            yaxis=dict(title="Probability", range=[0, 1], tickformat=".0%"),
            xaxis=dict(
                title="Date",
                range=[regime_probs.index.min(), regime_probs.index.max()],
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=350,
            margin=dict(l=10, r=10, t=10, b=30),
        )
        st.plotly_chart(fig_probs, use_container_width=True)
    else:
        st.warning("No regime probabilities to display")

    # ==================================================================
    # 5. Latent factor time series
    # ==================================================================
    st.subheader("Latent Factors (DFM)")
    factor_cols = factors.columns.tolist()
    n_cols = min(len(factor_cols), 4)

    # Determine the actual data range from factor values.
    # The Kalman smoother fills early rows with near-zero values when
    # few series are available; trim those by finding the first row
    # where any factor deviates meaningfully from zero (|z| > 0.05).
    factors_plot = factors[factor_cols[:n_cols]].copy()
    meaningful = factors_plot.abs().max(axis=1) > 0.05
    if meaningful.any():
        factors_plot = factors_plot.loc[meaningful.idxmax():]

    factor_chart_cols = st.columns(n_cols)
    for i, col_name in enumerate(factor_cols[:n_cols]):
        with factor_chart_cols[i]:
            series = factors_plot[col_name].dropna()
            # Clip to ±3σ for display only (model uses unclipped values)
            series_display = series.clip(-3.0, 3.0)
            fig_f = go.Figure()
            fig_f.add_trace(
                go.Scatter(
                    x=series_display.index,
                    y=series_display,
                    mode="lines",
                    line=dict(color="#3498db", width=1.5),
                )
            )
            # Zero line
            fig_f.add_hline(y=0, line_dash="dot", line_color="grey", opacity=0.5)
            _add_nber_shading(fig_f)

            clean_name = col_name.replace("_", " ").title()
            fig_f.update_layout(
                title=dict(text=clean_name, font=dict(size=13)),
                height=200,
                margin=dict(t=35, b=30, l=10, r=10),
                xaxis=dict(
                    showticklabels=True,
                    tickformat="%Y",
                    dtick="M60",
                    tickangle=-45,
                    tickfont=dict(size=9),
                    range=[series_display.index.min(), series_display.index.max()],
                ),
                yaxis=dict(title="z-score", range=[-3.5, 3.5]),
            )
            st.plotly_chart(fig_f, use_container_width=True)

    st.markdown("---")

    # ==================================================================
    # 6. Regime timeline with NBER shading
    # ==================================================================
    st.subheader("Historical Regime Classification")
    if isinstance(regime_probs, pd.DataFrame) and "recession" in regime_probs.columns:
        fig_timeline = go.Figure()

        # Recession probability as a filled area
        fig_timeline.add_trace(
            go.Scatter(
                x=regime_probs.index,
                y=regime_probs["recession"],
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(231, 76, 60, 0.3)",
                line=dict(color="#e74c3c", width=1.5),
                name="P(Recession)",
            )
        )
        # 50% threshold
        fig_timeline.add_hline(
            y=0.5, line_dash="dash", line_color="grey",
            annotation_text="50% threshold",
        )
        _add_nber_shading(fig_timeline)

        fig_timeline.update_layout(
            yaxis=dict(title="P(Recession)", range=[0, 1], tickformat=".0%"),
            xaxis=dict(
                title="Date",
                range=[regime_probs.index.min(), regime_probs.index.max()],
            ),
            height=250,
            margin=dict(l=10, r=10, t=10, b=30),
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    st.markdown("---")

    # ==================================================================
    # 7. LLM narrative card (with FedScraper)
    # ==================================================================
    st.subheader("📝 Narrative Analysis")
    with st.expander("Generate LLM Narrative", expanded=False):
        st.markdown(
            "Generates a macro narrative by combining the quantitative nowcast with "
            "recent Federal Reserve communications (FOMC minutes, statements, Beige Book)."
        )
        fetch_fed = st.checkbox("Fetch latest Fed documents", value=True)
        gen_button = st.button("Generate Narrative", type="secondary")

        if gen_button:
            fed_docs = []
            if fetch_fed:
                with st.spinner("Scraping Federal Reserve communications…"):
                    try:
                        from src.agent.fed_scraper import FedScraper
                        scraper = FedScraper(request_delay=1.0)
                        fed_docs = scraper.fetch_all_recent(n_each=2)
                        st.success(f"Fetched {len(fed_docs)} Fed documents")
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Fed scraping failed: {exc}")

            with st.spinner("Generating narrative…"):
                try:
                    from src.agent.narrative_agent import NarrativeAgent
                    agent = NarrativeAgent()
                    report = agent.generate(
                        nowcast_result=result.to_dict(),
                        fed_documents=fed_docs if fed_docs else None,
                    )

                    # If parsing extracted structured fields, show them;
                    # otherwise fall back to rendering the raw LLM markdown.
                    if report.key_drivers or report.risk_flags:
                        st.markdown(f"### Summary\n{report.summary}")
                        st.markdown("### Key Drivers")
                        for driver in report.key_drivers:
                            st.markdown(f"- {driver}")
                        st.markdown("### Risk Flags")
                        for flag in report.risk_flags:
                            st.markdown(f"- {flag}")
                        st.markdown(
                            f"**Fed Alignment:** {report.corroboration_score.title()}"
                        )
                    else:
                        # Parsing didn't extract bullets — show full response
                        st.markdown(report.raw_response)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Narrative generation failed: {exc}")

    # ==================================================================
    # 8. Data tables
    # ==================================================================
    st.subheader("📋 Recent Data")
    tab1, tab2, tab3 = st.tabs(["Regime Probabilities", "Factor Values", "Allocation Weights"])

    with tab1:
        if isinstance(regime_probs, pd.DataFrame) and len(regime_probs) > 0:
            st.dataframe(
                regime_probs.tail(24).style.format("{:.1%}"),
                use_container_width=True,
            )
        else:
            st.info("No regime probabilities to display")

    with tab2:
        if isinstance(factors, pd.DataFrame) and len(factors) > 0:
            st.dataframe(
                factors.tail(24).style.format("{:.3f}"),
                use_container_width=True,
            )
        else:
            st.info("No factor data to display")

    with tab3:
        # Build allocation time series
        if isinstance(regime_probs, pd.DataFrame) and len(regime_probs) > 0:
            alloc_ts = allocator.get_allocation_dataframe(regime_probs.tail(24))
            st.dataframe(
                alloc_ts.style.format("{:.1%}"),
                use_container_width=True,
            )

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center; color:#7f8c8d; font-size:0.85em;'>"
        f"Macro Regime Nowcaster · Last updated: {timestamp} · "
        f"2-regime ensemble (RSM + Probit + CFNAI)"
        f"</div>",
        unsafe_allow_html=True,
    )
