"""Interactive dashboard for the Sales Prediction project.

This is a presentation layer only. Every prediction comes from the persisted Phase 5 pipeline through
`src/predict.py`, and the dataset is read with the project's own `src/data_preprocessing.py`. No model is
trained, tuned or modified here, and neither the raw CSV nor the model artefact is ever written to.

Run with:   streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
for _path in (PROJECT_ROOT, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import data_preprocessing as dp  # noqa: E402
import predict as predict_api  # noqa: E402

DATA_PATH = PROJECT_ROOT / "dataset" / "advertising.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "final_sales_prediction_pipeline.joblib"

# --------------------------------------------------------------------------------------
# Established project results (Phases 1-5). Displayed only; never recomputed or invented here.
# Sources are named beside each block so every figure on screen is traceable to a notebook.
# --------------------------------------------------------------------------------------

# Phase 5, section 7 - the one-time evaluation on the 40 held-out rows.
FINAL_TEST = {"RMSE": 1.2173, "MAE": 0.9354, "R²": 0.9520, "MSE": 1.4819}

# Phase 4, section 11 - cross-validation on the 160 training rows. Never mixed with the above.
CV_METRICS = {"RMSE": 1.2808, "RMSE std": 0.1936, "MAE": 0.9393, "R²": 0.9341}

# Phase 5, section 8 - every model scored on the same 40-row hold-out.
TEST_RMSE_COMPARISON = pd.DataFrame({
    "Model": ["Gradient Boosting (Phase 3)", "Random Forest (Phase 3)",
              "Gradient Boosting (tuned) — FINAL", "Lasso (α=0.01)", "Linear Regression",
              "Ridge (α=1.0)", "Baseline (training mean)"],
    "Test RMSE": [1.1451, 1.1798, 1.2173, 1.7050, 1.7052, 1.7074, 5.6482],
    "Test MAE": [0.8679, 0.9025, 0.9354, 1.2725, 1.2748, 1.2734, 4.9315],
    "Test R²": [0.9576, 0.9550, 0.9520, 0.9059, 0.9059, 0.9057, -0.0324],
})

# Phase 5, section 7 - share of the 40 test predictions inside each absolute-error band.
ERROR_COVERAGE = pd.DataFrame({
    "Band": ["± 0.5", "± 1.0", "± 1.5", "± 2.0"],
    "Coverage": [27.5, 62.5, 90.0, 92.5],
    "Observations": [11, 25, 36, 37],
})

# Phase 5, sections 7 and 9.
RESIDUAL_FACTS = {
    "Mean error (bias)": -0.2422,
    "Median absolute error": 0.7970,
    "Largest absolute error": 4.2553,
    "Largest under-prediction": 2.9502,
    "Residual skewness": -0.3114,
}

# Phase 1, section 13 - Pearson correlations with Sales.
PHASE1_CORRELATIONS = {"TV": 0.901, "Radio": 0.350, "Newspaper": 0.158}

DATASET_FACTS = {"Observations": 200, "Columns": 4, "Missing cells": 0, "Duplicate rows": 0,
                 "Training rows": 160, "Test rows": 40}

# --------------------------------------------------------------------------------------
# Visual identity. Categorical slots 1-3 of the validated dark palette; all-pairs checked
# against this app's card surface (#151f36): CVD ΔE 9.4, normal-vision ΔE 20.9, contrast >= 3:1.
# --------------------------------------------------------------------------------------
C_TV = "#3987e5"
C_RADIO = "#d95926"
C_NEWS = "#199e70"
C_TARGET = "#9085e9"
C_MUTED = "#3f5175"
C_SURFACE = "#151f36"
C_TEXT = "#e8edf7"
C_TEXT_DIM = "#9aabc9"
C_GRID = "#25324f"

FEATURE_COLORS = {"TV": C_TV, "Radio": C_RADIO, "Newspaper": C_NEWS}

PAGES = ["Overview", "Predict Sales", "Data Analysis", "Model Performance",
         "Model Details", "About Project"]

SMALL_DATA_CAVEAT = (
    "This model was trained and evaluated on a small 200-row advertising dataset. "
    "Reported metrics should not be interpreted as guaranteed performance on future "
    "advertising campaigns."
)


# --------------------------------------------------------------------------------------
# Cached loaders - the model is loaded once per session, never retrained
# --------------------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading the persisted model…")
def load_model():
    """Load the Phase 5 pipeline once. Returns (pipeline, metadata).

    Uses the existing prediction API so the dashboard and the CLI share one loading path.
    """
    return predict_api.load_artifact(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Read the raw dataset through the project's validated loader. Read-only."""
    return dp.load_raw_data(DATA_PATH)


@st.cache_data(show_spinner=False)
def dataset_summary() -> pd.DataFrame:
    df = load_dataset()
    summary = df.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    summary.insert(0, "count", df.count())
    return summary.round(3)


@st.cache_data(show_spinner=False)
def correlation_matrix() -> pd.DataFrame:
    return load_dataset().corr().round(3)


@st.cache_data(show_spinner=False)
def training_ranges() -> dict:
    """Min/max of each predictor in the 160 training rows, read from the artefact's metadata."""
    return predict_api.training_ranges(MODEL_PATH)


@st.cache_data(show_spinner=False)
def newspaper_outliers() -> pd.DataFrame:
    """The two rows Phase 1 flagged by the IQR rule. Retained in every phase; shown for context."""
    df = load_dataset()
    q1, q3 = df["Newspaper"].quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    return df[df["Newspaper"] > upper]


# --------------------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------------------

CSS = """
<style>
  html, body, [class*="css"] { font-size: 16px; }

  .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1220px; }

  h1, h2, h3 { letter-spacing: -0.015em; }

  .hero-title {
    font-size: 2.6rem; font-weight: 750; line-height: 1.12; margin: 0 0 .35rem 0;
    color: #f4f7fd;
  }
  .hero-sub { font-size: 1.16rem; color: #9aabc9; margin: 0 0 1.6rem 0; font-weight: 400; }

  .section-title {
    font-size: 1.32rem; font-weight: 700; color: #f4f7fd;
    margin: 2.1rem 0 .3rem 0; padding-bottom: .5rem;
    border-bottom: 1px solid #25324f;
  }
  .section-note { color: #9aabc9; font-size: .98rem; margin: .55rem 0 1rem 0; line-height: 1.6; }

  .card {
    background: #151f36; border: 1px solid #25324f; border-radius: 12px;
    padding: 1.15rem 1.25rem; height: 100%;
  }
  .card-label {
    font-size: .82rem; text-transform: uppercase; letter-spacing: .085em;
    color: #9aabc9; font-weight: 650; margin-bottom: .45rem;
  }
  .card-value { font-size: 1.95rem; font-weight: 720; color: #f4f7fd; line-height: 1.15; }
  .card-value.is-text { font-size: 1.12rem; font-weight: 650; line-height: 1.4; word-break: break-word; }
  .card-foot { font-size: .88rem; color: #9aabc9; margin-top: .3rem; }

  .result-card {
    background: linear-gradient(135deg, #14294a 0%, #151f36 100%);
    border: 1px solid #3987e5; border-radius: 16px;
    padding: 2rem 1.5rem; text-align: center; margin: .5rem 0 1rem 0;
  }
  .result-label {
    font-size: .92rem; text-transform: uppercase; letter-spacing: .1em;
    color: #9aabc9; font-weight: 650;
  }
  .result-value {
    font-size: 4.1rem; font-weight: 780; color: #6ea8f0;
    line-height: 1.05; margin: .3rem 0 .1rem 0;
  }
  .result-unit { font-size: 1rem; color: #9aabc9; }

  .pipeline-row { display: flex; flex-wrap: wrap; gap: .45rem; align-items: center; margin: .3rem 0 .6rem 0; }
  .pipe-step {
    background: #151f36; border: 1px solid #25324f; border-radius: 8px;
    padding: .5rem .85rem; font-size: .95rem; color: #e8edf7; font-weight: 550;
  }
  .pipe-step.is-live { border-color: #3987e5; color: #6ea8f0; }
  .pipe-arrow { color: #3f5175; font-size: 1.05rem; }

  .note {
    background: #151f36; border-left: 3px solid #3f5175; border-radius: 6px;
    padding: .85rem 1.05rem; color: #c6d3e8; font-size: .97rem; line-height: 1.65;
    margin: .6rem 0;
  }
  .note.accent { border-left-color: #3987e5; }

  .kv { font-size: 1rem; line-height: 2.0; color: #e8edf7; }
  .kv b { color: #9aabc9; font-weight: 600; }

  div[data-testid="stMetricValue"] { font-size: 1.85rem; }
  section[data-testid="stSidebar"] { background: #0d1626; }
  section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
  div[role="radiogroup"] label { font-size: 1.02rem; padding: .12rem 0; }
  .stPlotlyChart { border-radius: 10px; }
</style>
"""


def style_fig(fig: go.Figure, height: int = 380, legend: bool = False) -> go.Figure:
    """Apply the shared chart styling: recessive grid, readable type, transparent surface."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=14, t=48, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=14, color=C_TEXT),
        title=dict(font=dict(size=17, color="#f4f7fd"), x=0, xanchor="left", y=0.96),
        hoverlabel=dict(bgcolor=C_SURFACE, bordercolor=C_GRID, font_size=14),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                    font=dict(size=13, color=C_TEXT_DIM), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=C_GRID, zerolinecolor=C_GRID, linecolor=C_GRID,
                     tickfont=dict(size=13, color=C_TEXT_DIM),
                     title_font=dict(size=14, color=C_TEXT_DIM))
    fig.update_yaxes(gridcolor=C_GRID, zerolinecolor=C_GRID, linecolor=C_GRID,
                     tickfont=dict(size=13, color=C_TEXT_DIM),
                     title_font=dict(size=14, color=C_TEXT_DIM))
    return fig


def card(label: str, value: str, foot: str = "", text_value: bool = False) -> str:
    """One summary card. `text_value=True` uses the smaller type for long non-numeric values."""
    foot_html = f'<div class="card-foot">{foot}</div>' if foot else ""
    cls = "card-value is-text" if text_value else "card-value"
    return (f'<div class="card"><div class="card-label">{label}</div>'
            f'<div class="{cls}">{value}</div>{foot_html}</div>')


def section(title: str, note: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="section-note">{note}</div>', unsafe_allow_html=True)


def note_box(text: str, accent: bool = False) -> None:
    st.markdown(f'<div class="note{" accent" if accent else ""}">{text}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Page: Overview
# --------------------------------------------------------------------------------------


def page_overview() -> None:
    st.markdown('<div class="hero-title">Sales Prediction Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Advertising Sales Prediction using Machine Learning</div>',
                unsafe_allow_html=True)

    cols = st.columns(3)
    facts = [
        ("Dataset observations", "200", "advertising campaigns"),
        ("Input features", "3", "TV · Radio · Newspaper"),
        ("Target", "Sales", "continuous, 1.6 – 27.0"),
    ]
    for col, (label, value, foot) in zip(cols, facts):
        col.markdown(card(label, value, foot), unsafe_allow_html=True)

    st.write("")
    cols = st.columns(3)
    facts = [
        ("Training rows", "160", "80% of the dataset"),
        ("Test rows", "40", "held out until Phase 5"),
        ("Final model", "Gradient Boosting", "tuned by cross-validation"),
    ]
    for col, (label, value, foot) in zip(cols, facts):
        col.markdown(card(label, value, foot), unsafe_allow_html=True)

    section("What this project does")
    st.markdown(
        """
The task is to **estimate the sales a campaign achieves from the money spent advertising it**, across three
platforms: **TV**, **Radio** and **Newspaper**. Those three budgets are the only inputs; `Sales` is the
quantity predicted.

The dataset records 200 campaigns, each pairing the three budgets with the sales that followed. A Gradient
Boosting model learns the relationship between them from 160 of those rows, and the remaining 40 were kept
untouched until the very end so the reported performance measures genuinely unseen data.

One limitation is worth stating at the top rather than in a footnote: the dataset contains **no audience,
geographic, seasonal or time information**, so the "target audience segmentation" element of the original
task statement cannot be studied here. The model works with advertising expenditure and platform choice, and
nothing else.
        """
    )

    section("ML pipeline")
    steps = ["Dataset", "Audit", "Preprocessing", "Model Training", "Cross-Validation",
             "Final Evaluation", "Prediction"]
    html = '<div class="pipeline-row">'
    for i, step in enumerate(steps):
        live = " is-live" if step == "Prediction" else ""
        html += f'<span class="pipe-step{live}">{step}</span>'
        if i < len(steps) - 1:
            html += '<span class="pipe-arrow">→</span>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Each stage is a notebook in <code>notebooks/</code>. This dashboard is '
        'the final stage: it consumes the pipeline persisted at the end of the project and performs no '
        'training of its own.</div>',
        unsafe_allow_html=True,
    )

    section("Headline result")
    cols = st.columns(4)
    for col, (label, value, foot) in zip(cols, [
        ("Test RMSE", f"{FINAL_TEST['RMSE']:.4f}", "sales units"),
        ("Test MAE", f"{FINAL_TEST['MAE']:.4f}", "sales units"),
        ("Test R²", f"{FINAL_TEST['R²']:.4f}", "variance explained"),
        ("vs baseline", "−78.4%", "RMSE reduction"),
    ]):
        col.markdown(card(label, value, foot), unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Measured once on the 40 held-out rows. See '
        '<b>Model Performance</b> for how this differs from the cross-validation estimate.</div>',
        unsafe_allow_html=True,
    )

    st.warning(SMALL_DATA_CAVEAT, icon="⚠️")


# --------------------------------------------------------------------------------------
# Page: Predict Sales
# --------------------------------------------------------------------------------------


def page_predict() -> None:
    st.markdown('<div class="hero-title">Predict Sales</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Enter an advertising budget for each platform to generate a prediction '
        'from the persisted model.</div>',
        unsafe_allow_html=True,
    )

    ranges = training_ranges()

    section("Advertising budgets", "Defaults are the median spend observed in the training data.")

    cols = st.columns(3)
    inputs = {}
    defaults = {"TV": 150.0, "Radio": 23.0, "Newspaper": 26.0}
    helps = {
        "TV": "Budget spent on the TV platform.",
        "Radio": "Budget spent on the Radio platform.",
        "Newspaper": "Budget spent on the Newspaper platform.",
    }
    for col, feature in zip(cols, ("TV", "Radio", "Newspaper")):
        low, high = ranges[feature]
        with col:
            inputs[feature] = st.number_input(
                f"{feature} Advertising Spend",
                min_value=0.0,
                value=defaults[feature],
                step=1.0,
                format="%.1f",
                help=f"{helps[feature]} Observed in training: {low:.1f} – {high:.1f}.",
                key=f"input_{feature}",
            )
            st.caption(f"Training range · **{low:.1f} – {high:.1f}**")

    st.markdown(
        '<div class="section-note">Budgets cannot be negative — the prediction API rejects negative '
        'values, and these inputs are floored at 0 to match it.</div>',
        unsafe_allow_html=True,
    )

    outside = [f for f in ("TV", "Radio", "Newspaper")
               if not (ranges[f][0] <= inputs[f] <= ranges[f][1])]

    predict_clicked = st.button("Predict Sales", type="primary", use_container_width=True)

    if predict_clicked:
        st.session_state["last_inputs"] = dict(inputs)

    if not st.session_state.get("last_inputs"):
        note_box(
            "Set the three budgets above and select <b>Predict Sales</b>. The prediction is produced by "
            "the pipeline saved at the end of Phase 5 — nothing is trained when you click.",
            accent=True,
        )
        return

    values = st.session_state["last_inputs"]

    try:
        prediction = predict_api.predict_sales(
            values["TV"], values["Radio"], values["Newspaper"], model_path=MODEL_PATH
        )
    except (predict_api.ModelArtifactError, TypeError, ValueError) as exc:
        st.error(f"Could not generate a prediction: {exc}")
        return

    if outside:
        st.warning(
            "These inputs are outside the range observed during model training "
            f"({', '.join(outside)}). The prediction can still be generated, but it represents "
            "extrapolation beyond the training data.",
            icon="⚠️",
        )

    section("Result")
    st.markdown(
        f'<div class="result-card">'
        f'<div class="result-label">Predicted Sales</div>'
        f'<div class="result-value">{prediction:.2f}</div>'
        f'<div class="result-unit">sales units — the same scale as the dataset\'s Sales column</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    for col, feature in zip(cols, ("TV", "Radio", "Newspaper")):
        col.markdown(card(f"{feature} spend", f"{values[feature]:,.1f}"), unsafe_allow_html=True)

    st.write("")
    fig = go.Figure()
    features = ["TV", "Radio", "Newspaper"]
    fig.add_trace(go.Bar(
        y=features,
        x=[values[f] for f in features],
        orientation="h",
        marker=dict(color=[FEATURE_COLORS[f] for f in features],
                    line=dict(color=C_SURFACE, width=2)),
        text=[f"{values[f]:,.1f}" for f in features],
        textposition="outside",
        textfont=dict(size=14, color=C_TEXT),
        hovertemplate="<b>%{y}</b><br>Spend: %{x:,.1f}<extra></extra>",
        width=0.55,
    ))
    fig.update_layout(title="Advertising budgets you entered")
    fig.update_xaxes(title="Advertising spend (same units as the dataset)",
                     range=[0, max(values.values()) * 1.22 + 1])
    fig.update_yaxes(title="")
    st.plotly_chart(style_fig(fig, height=300), use_container_width=True)
    st.markdown(
        '<div class="section-note">This chart shows the three budgets you entered, nothing more. It is '
        '<b>not</b> a feature-importance chart and does not indicate how much each platform contributed '
        'to the prediction.</div>',
        unsafe_allow_html=True,
    )

    note_box(
        "Prediction generated by the persisted Phase 5 production pipeline. It is an estimate from a model "
        "fitted to 160 observations — a typical miss on held-out data was about "
        f"<b>{FINAL_TEST['MAE']:.2f}</b> sales units, and it is not a guaranteed outcome.",
        accent=True,
    )


# --------------------------------------------------------------------------------------
# Page: Data Analysis
# --------------------------------------------------------------------------------------


def page_data_analysis() -> None:
    df = load_dataset()

    st.markdown('<div class="hero-title">Data Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Exploratory analysis of the advertising dataset, computed live from '
        '<code>dataset/advertising.csv</code>.</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    for col, (label, value, foot) in zip(cols, [
        ("Dataset shape", f"{df.shape[0]} × {df.shape[1]}", "rows × columns"),
        ("Missing cells", f"{int(df.isna().sum().sum())}", "complete dataset"),
        ("Duplicate rows", f"{int(df.duplicated().sum())}", "every row distinct"),
        ("Columns", f"{df.shape[1]}", " · ".join(df.columns)),
    ]):
        col.markdown(card(label, value, foot), unsafe_allow_html=True)

    section("Summary statistics")
    st.dataframe(dataset_summary(), use_container_width=True)

    section("Sales distribution", "The target variable across all 200 campaigns.")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["Sales"], nbinsx=22,
        marker=dict(color=C_TARGET, line=dict(color=C_SURFACE, width=2)),
        hovertemplate="Sales %{x}<br>%{y} campaigns<extra></extra>",
    ))
    fig.add_vline(x=df["Sales"].mean(), line=dict(color=C_TEXT_DIM, width=2, dash="dash"),
                  annotation_text=f"mean {df['Sales'].mean():.2f}",
                  annotation_font=dict(size=13, color=C_TEXT_DIM))
    fig.update_layout(title="Distribution of Sales", bargap=0.06)
    fig.update_xaxes(title="Sales")
    fig.update_yaxes(title="Number of campaigns")
    st.plotly_chart(style_fig(fig, height=360), use_container_width=True)

    section("Advertising spend distributions",
            "The three predictors. TV spans a far wider range than Radio, which is why the pipeline "
            "standardises them before modelling.")
    fig = go.Figure()
    for feature in ("TV", "Radio", "Newspaper"):
        fig.add_trace(go.Box(
            y=df[feature], name=feature,
            marker=dict(color=FEATURE_COLORS[feature]),
            line=dict(width=2), fillcolor="rgba(0,0,0,0)",
            boxpoints="outliers",
            hovertemplate=f"<b>{feature}</b><br>%{{y}}<extra></extra>",
        ))
    fig.update_layout(title="Spend by platform")
    fig.update_yaxes(title="Advertising spend")
    fig.update_xaxes(title="")
    st.plotly_chart(style_fig(fig, height=390, legend=True), use_container_width=True)

    section("Advertising spend vs Sales",
            "Each point is one campaign. The dashed line is a least-squares fit, shown as a visual guide "
            "to the trend — it is not the deployed model.")

    tabs = st.tabs(["TV vs Sales", "Radio vs Sales", "Newspaper vs Sales"])
    for tab, feature in zip(tabs, ("TV", "Radio", "Newspaper")):
        with tab:
            r = df[feature].corr(df["Sales"])
            slope, intercept = np.polyfit(df[feature], df["Sales"], 1)
            xs = np.linspace(df[feature].min(), df[feature].max(), 100)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df[feature], y=df["Sales"], mode="markers", name="Campaigns",
                marker=dict(size=9, color=FEATURE_COLORS[feature], opacity=0.72,
                            line=dict(color=C_SURFACE, width=1.5)),
                hovertemplate=f"{feature} %{{x}}<br>Sales %{{y}}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=xs, y=slope * xs + intercept, mode="lines", name="Least-squares fit",
                line=dict(color=C_TEXT_DIM, width=2, dash="dash"), hoverinfo="skip",
            ))
            fig.update_layout(title=f"{feature} vs Sales  ·  r = {r:.3f}")
            fig.update_xaxes(title=f"{feature} advertising spend")
            fig.update_yaxes(title="Sales")
            st.plotly_chart(style_fig(fig, height=430, legend=True), use_container_width=True)

    section("Correlation", "Pearson correlation between every pair of numeric columns.")
    corr = correlation_matrix()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0, "#8c3b2e"], [0.5, "#1c2740"], [1, "#2f6fb8"]],
        zmid=0, zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text:.3f}",
        textfont=dict(size=15, color=C_TEXT),
        hovertemplate="%{y} ↔ %{x}<br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(title=dict(text="r", font=dict(size=13, color=C_TEXT_DIM)),
                      tickfont=dict(size=12, color=C_TEXT_DIM), thickness=14),
    ))
    fig.update_layout(title="Correlation matrix")
    st.plotly_chart(style_fig(fig, height=430), use_container_width=True)

    cols = st.columns(3)
    for col, (feature, value) in zip(cols, PHASE1_CORRELATIONS.items()):
        strength = {"TV": "strong", "Radio": "moderate", "Newspaper": "weak"}[feature]
        col.markdown(card(f"{feature} ↔ Sales", f"{value:.3f}", f"{strength} correlation"),
                     unsafe_allow_html=True)

    note_box(
        "These are <b>correlations</b>, established in the Phase 1 audit: TV ≈ 0.901, Radio ≈ 0.350, "
        "Newspaper ≈ 0.158. A correlation describes how two quantities move together in this "
        "observational data — it does <b>not</b> establish that spending more on a platform causes sales "
        "to rise. Budgets were set by someone for reasons the dataset does not record, and no experiment "
        "is involved."
    )

    section("Data quality audit", "Findings from Phase 1, carried through every later phase unchanged.")
    outliers = newspaper_outliers()
    cols = st.columns([1, 1])
    with cols[0]:
        st.markdown(
            f"""
<div class="kv">
<b>Missing cells:</b> {int(df.isna().sum().sum())} of {df.size}<br>
<b>Exact duplicate rows:</b> {int(df.duplicated().sum())}<br>
<b>Negative values:</b> 0<br>
<b>Constant columns:</b> none<br>
<b>Newspaper outliers (IQR):</b> {len(outliers)} · retained
</div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.dataframe(outliers, use_container_width=True)

    note_box(
        "The two flagged rows were identified by the standard <b>IQR rule</b> — values above "
        "Q3 + 1.5 × IQR, which for Newspaper is a spend above 93.63. Both were <b>retained</b> in every "
        "phase. They are plausible advertising budgets rather than data errors, and statistical "
        "unusualness alone is not evidence that an observation is invalid. Phase 4 tested this directly: "
        "removing the one that falls in the training set changed cross-validated RMSE by less than the "
        "fold-to-fold noise, and not even consistently in one direction."
    )

    with st.expander("View the full dataset (200 rows)"):
        st.dataframe(df, use_container_width=True, height=420)


# --------------------------------------------------------------------------------------
# Page: Model Performance
# --------------------------------------------------------------------------------------


def page_performance() -> None:
    st.markdown('<div class="hero-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">How the final model scored, and how that figure was produced.</div>',
        unsafe_allow_html=True,
    )

    section("Final test-set performance", "40 held-out observations, evaluated exactly once.")
    cols = st.columns(4)
    for col, (label, value, foot) in zip(cols, [
        ("RMSE", f"{FINAL_TEST['RMSE']:.4f}", "root mean squared error"),
        ("MAE", f"{FINAL_TEST['MAE']:.4f}", "mean absolute error"),
        ("R²", f"{FINAL_TEST['R²']:.4f}", "variance explained"),
        ("MSE", f"{FINAL_TEST['MSE']:.4f}", "mean squared error"),
    ]):
        col.markdown(card(label, value, foot), unsafe_allow_html=True)

    section("Cross-validation performance (Phase 4)",
            "5-fold cross-validation on the 160 training rows. This is how the model was selected — "
            "it is a separate measurement from the test result above and the two are never combined.")
    cols = st.columns(3)
    for col, (label, value, foot) in zip(cols, [
        ("CV RMSE", f"{CV_METRICS['RMSE']:.4f} ± {CV_METRICS['RMSE std']:.4f}", "mean ± std across folds"),
        ("CV MAE", f"{CV_METRICS['MAE']:.4f}", "training-set estimate"),
        ("CV R²", f"{CV_METRICS['R²']:.4f}", "training-set estimate"),
    ]):
        col.markdown(card(label, value, foot), unsafe_allow_html=True)

    note_box(
        f"The final test RMSE ({FINAL_TEST['RMSE']:.4f}) came in <b>better</b> than the cross-validated "
        f"estimate ({CV_METRICS['RMSE']:.4f}), by 0.064 — comfortably inside the CV standard deviation of "
        f"{CV_METRICS['RMSE std']:.4f}, so the two measurements are consistent with each other.",
        accent=True,
    )

    section("Test RMSE comparison",
            "Every model fitted on the same 160 training rows and scored on the same 40 held-out rows.")
    comp = TEST_RMSE_COMPARISON.sort_values("Test RMSE")
    colors = [C_TV if "FINAL" in m else C_MUTED for m in comp["Model"]]
    fig = go.Figure(go.Bar(
        x=comp["Test RMSE"], y=comp["Model"], orientation="h",
        marker=dict(color=colors, line=dict(color=C_SURFACE, width=2)),
        text=[f"{v:.4f}" for v in comp["Test RMSE"]],
        textposition="outside", textfont=dict(size=14, color=C_TEXT),
        hovertemplate="<b>%{y}</b><br>Test RMSE %{x:.4f}<extra></extra>",
        width=0.62,
    ))
    fig.update_layout(title="Test RMSE comparison  ·  lower is better")
    fig.update_xaxes(title="Test RMSE (sales units)", range=[0, 6.6])
    fig.update_yaxes(title="", autorange="reversed")
    st.plotly_chart(style_fig(fig, height=430), use_container_width=True)

    st.dataframe(
        TEST_RMSE_COMPARISON.set_index("Model").style.format(
            {"Test RMSE": "{:.4f}", "Test MAE": "{:.4f}", "Test R²": "{:.4f}"}),
        use_container_width=True,
    )

    st.warning(
        "The tuned final model performed worse on this particular hold-out set than the untuned Phase 3 "
        "Gradient Boosting model. The final model was retained because model selection was completed "
        "using training-set cross-validation before the test set was used.",
        icon="⚠️",
    )
    note_box(
        "Why this matters: switching to the untuned model now, on the strength of these 40 rows, would be "
        "choosing a model <i>using the test set</i> — which is exactly what a held-out set exists to "
        "prevent. Phase 4 had already shown the two configurations sit within 0.023 RMSE of each other "
        "across 25 repeated cross-validation folds, against a fold-to-fold spread of roughly 0.19–0.25, so "
        "neither ordering establishes one as genuinely better. Following the rule costs about 0.07 RMSE "
        "units of reported performance and buys a number that means what it claims to."
    )

    section("Prediction-error coverage",
            "The share of the 40 test predictions that landed within a given distance of the actual value.")
    fig = go.Figure(go.Bar(
        x=ERROR_COVERAGE["Band"], y=ERROR_COVERAGE["Coverage"],
        marker=dict(color=C_TV, line=dict(color=C_SURFACE, width=2)),
        text=[f"{c:.1f}%<br><span style='font-size:12px'>{n}/40</span>"
              for c, n in zip(ERROR_COVERAGE["Coverage"], ERROR_COVERAGE["Observations"])],
        textposition="outside", textfont=dict(size=14, color=C_TEXT),
        hovertemplate="Within %{x} sales units<br>%{y:.1f}% of test observations<extra></extra>",
        width=0.55,
    ))
    fig.update_layout(title="Prediction-error coverage on the 40-row hold-out")
    fig.update_xaxes(title="Absolute error band (sales units)")
    fig.update_yaxes(title="Share of test observations (%)", range=[0, 112])
    st.plotly_chart(style_fig(fig, height=400), use_container_width=True)

    note_box(
        "These are <b>prediction-error coverage</b> figures — the proportion of held-out campaigns whose "
        "predicted Sales fell within that many units of the actual value. They are <b>not</b> "
        "classification accuracy: this is a regression problem and there are no classes. They describe one "
        "40-row sample and are not guarantees about future predictions."
    )

    section("Residual behaviour", "Error characteristics on the same 40 held-out rows.")
    cols = st.columns(len(RESIDUAL_FACTS))
    for col, (label, value) in zip(cols, RESIDUAL_FACTS.items()):
        col.markdown(card(label, f"{value:+.4f}" if "bias" in label or "skew" in label else f"{value:.4f}"),
                     unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">The negative bias means a mild tendency to over-predict — small next to '
        "the target's standard deviation of 5.63, and on 40 rows not distinguishable from zero. Residuals "
        "showed no obvious funnel or curvature.</div>",
        unsafe_allow_html=True,
    )

    st.warning(SMALL_DATA_CAVEAT, icon="⚠️")


# --------------------------------------------------------------------------------------
# Page: Model Details
# --------------------------------------------------------------------------------------


def page_model_details() -> None:
    _, metadata = load_model()

    st.markdown('<div class="hero-title">Model Details</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Read from the persisted artefact\'s own metadata, not hardcoded.</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    specs = [
        ("Final model", metadata.get("model_type", "—"), "regression", True),
        ("Training rows", f"{metadata.get('training_rows', '—')}", "used to fit", False),
        ("Test rows", f"{metadata.get('test_rows', '—')}", "held out", False),
        ("Target", metadata.get("target_name", "—"), "predicted quantity", False),
    ]
    for col, (label, value, foot, is_text) in zip(cols, specs):
        col.markdown(card(label, value, foot, text_value=is_text), unsafe_allow_html=True)

    section("Hyperparameters", "Selected in Phase 4 by grid search scored with 5-fold cross-validation.")
    hyper = metadata.get("hyperparameters", {})
    hyper_df = pd.DataFrame({"Hyperparameter": list(hyper.keys()),
                             "Value": [str(v) for v in hyper.values()]})
    cols = st.columns([1.1, 1])
    with cols[0]:
        st.dataframe(hyper_df, use_container_width=True, hide_index=True)
    with cols[1]:
        st.markdown(
            f"""
<div class="kv">
<b>Preprocessing:</b> StandardScaler, inside the pipeline<br>
<b>Fitted on:</b> the {metadata.get('training_rows', 160)} training rows only<br>
<b>Features:</b> {' · '.join(metadata.get('feature_names', []))}<br>
<b>Target:</b> {metadata.get('target_name', 'Sales')}<br>
<b>Selected in:</b> {metadata.get('selected_in', '—')}
</div>
            """,
            unsafe_allow_html=True,
        )

    section("Training feature ranges",
            "The span of each predictor in the training data. Inputs outside these ranges are "
            "extrapolation, and the Predict page flags them.")
    ranges = metadata.get("training_feature_ranges", {})
    ranges_df = pd.DataFrame(
        [{"Feature": k, "Minimum": v[0], "Maximum": v[1]} for k, v in ranges.items()]
    )
    st.dataframe(ranges_df, use_container_width=True, hide_index=True)

    section("Artifact")
    size_bytes = MODEL_PATH.stat().st_size if MODEL_PATH.is_file() else 0
    cols = st.columns(3)
    specs = [
        ("Path", "models/<br>final_sales_prediction_pipeline.joblib", "complete pipeline", True),
        ("Size", f"{size_bytes:,} bytes", f"{size_bytes / 1024:.1f} KiB", False),
        ("Contents", "Pipeline + metadata", "preprocessing included", True),
    ]
    for col, (label, value, foot, is_text) in zip(cols, specs):
        col.markdown(card(label, value, foot, text_value=is_text), unsafe_allow_html=True)

    st.markdown(
        '<div class="section-note">The artefact stores the <b>whole</b> pipeline — the fitted scaler as '
        "well as the model. Saving the estimator alone would produce something that silently mis-predicts, "
        "because it expects standardised inputs.</div>",
        unsafe_allow_html=True,
    )

    section("Environment recorded at save time")
    env_keys = ["python_version", "sklearn_version", "joblib_version", "pandas_version", "numpy_version"]
    env_df = pd.DataFrame(
        [{"Component": k.replace("_version", "").replace("sklearn", "scikit-learn"),
          "Version": metadata.get(k, "—")} for k in env_keys]
    )
    cols = st.columns([1, 1])
    with cols[0]:
        st.dataframe(env_df, use_container_width=True, hide_index=True)
    with cols[1]:
        st.markdown(
            f"""
<div class="kv">
<b>Dataset:</b> {metadata.get('dataset_filename', 'advertising.csv')}<br>
<b>Dataset SHA-256:</b><br>
<span style="font-size:.84rem; color:#9aabc9; word-break:break-all;">
{metadata.get('dataset_sha256', '—')}</span><br>
<b>Split seed:</b> {metadata.get('random_state', 42)} · test_size {metadata.get('test_size', 0.2)}
</div>
            """,
            unsafe_allow_html=True,
        )

    section("Limitations")
    st.markdown(
        """
- The dataset contains only **200 observations**, and the final test set only **40**. Every metric is an
  estimate from a small benchmark dataset, with wide uncertainty around it.
- The dataset carries **no time, market, audience, geography, seasonality or campaign context**. Whatever
  role those factors play is invisible to this model.
- **Correlation does not establish causation.** The model describes association in observational data; it
  cannot say what would happen to sales if a budget were changed.
- **Predictions outside the training feature range represent extrapolation**, and should be treated with
  more caution than the headline metrics suggest.
- **Tree-based models regress toward the mean at extreme inputs.** Gradient boosting cannot predict beyond
  the range of target values it saw in training, so very high or very low campaigns are pulled inward.
- Results reflect this dataset's particular split. A different 40-row hold-out would give noticeably
  different figures — as this project demonstrated more than once.
        """
    )


# --------------------------------------------------------------------------------------
# Page: About Project
# --------------------------------------------------------------------------------------


def page_about() -> None:
    st.markdown('<div class="hero-title">About this Project</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">CodSoft Data Science Internship — Task 4: Sales Prediction using '
        'Python</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns([1, 1])
    with cols[0]:
        section("Project")
        st.markdown(
            """
**Project** · CodSoft Data Science Internship — Task 4
**Task** · Sales Prediction using Python
**Problem** · Predict Sales using advertising expenditure across TV, Radio and Newspaper.
**Dataset** · Advertising dataset, 200 observations, 4 columns
            """
        )
        section("Machine learning")
        st.markdown(
            """
- Gradient Boosting Regression *(final model)*
- Random Forest Regression
- Linear Regression
- Ridge Regression
- Lasso Regression
- 5-fold cross-validation
- Hyperparameter tuning by grid search
            """
        )
    with cols[1]:
        section("Technology")
        st.markdown(
            """
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Joblib
            """
        )
        section("Dataset")
        st.markdown(
            """
The **Advertising** dataset (200 rows × 4 columns), sourced from the Kaggle dataset the CodSoft task
links to and verified against the linked notebook's own published outputs before use. The raw file has
been checksum-verified as unchanged at every phase of the project.
            """
        )

    section("Project pipeline")
    steps = ["Data Audit", "Preprocessing", "Model Training", "Validation",
             "Final Evaluation", "Persistence", "Prediction API", "Dashboard"]
    html = '<div class="pipeline-row">'
    for i, step in enumerate(steps):
        live = " is-live" if step == "Dashboard" else ""
        html += f'<span class="pipe-step{live}">{step}</span>'
        if i < len(steps) - 1:
            html += '<span class="pipe-arrow">→</span>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    section("How the phases fit together")
    phase_df = pd.DataFrame({
        "Phase": ["1 · Dataset Audit", "2 · Preprocessing", "3 · Model Training",
                  "4 · Validation & Tuning", "5 · Final Evaluation", "6 · Dashboard"],
        "What it produced": [
            "Verified dataset, EDA, leakage audit",
            "Reusable pipeline, locked 160/40 split",
            "Baseline plus five models, first comparison",
            "Cross-validated selection and tuning",
            "One-time test evaluation, persisted pipeline",
            "This application",
        ],
        "Test set used?": ["No", "No", "Scored once per model", "No", "Scored once", "No"],
    })
    st.dataframe(phase_df, use_container_width=True, hide_index=True)

    note_box(
        "This dashboard is a <b>presentation layer</b>. It loads the pipeline persisted in Phase 5 and "
        "calls the existing prediction API — it does not train, tune or modify any model, and it never "
        "writes to the dataset or the artefact."
    )

    st.warning(SMALL_DATA_CAVEAT, icon="⚠️")


# --------------------------------------------------------------------------------------
# App shell
# --------------------------------------------------------------------------------------

PAGE_RENDERERS = {
    "Overview": page_overview,
    "Predict Sales": page_predict,
    "Data Analysis": page_data_analysis,
    "Model Performance": page_performance,
    "Model Details": page_model_details,
    "About Project": page_about,
}


def main() -> None:
    st.set_page_config(
        page_title="Sales Prediction Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Sales Prediction")
        st.caption("CodSoft Data Science Internship · Task 4")
        st.divider()
        if "page" not in st.session_state:
            st.session_state.page = PAGES[0]
        st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
        st.divider()

        try:
            _, metadata = load_model()
            st.caption(
                f"**Model loaded**  \n{metadata.get('model_type', 'pipeline')}  \n"
                f"Test RMSE {FINAL_TEST['RMSE']:.4f} · R² {FINAL_TEST['R²']:.4f}"
            )
        except predict_api.ModelArtifactError as exc:
            st.error(f"Model not loaded: {exc}")

    try:
        PAGE_RENDERERS[st.session_state.page]()
    except predict_api.ModelArtifactError as exc:
        st.error(
            f"The persisted model could not be loaded: {exc}\n\n"
            "Run `notebooks/05_final_evaluation_persistence.ipynb` to create the artefact."
        )
    except FileNotFoundError as exc:
        st.error(f"A required project file is missing: {exc}")


if __name__ == "__main__":
    main()
