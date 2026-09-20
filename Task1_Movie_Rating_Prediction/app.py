"""Interactive dashboard for the Movie Rating Prediction project.

This is a presentation layer only. Every prediction comes from the persisted pipeline through
`src.predict.predict_rating`, and all dataset statistics are computed with the project's own
`src.data_preprocessing` functions. No model is trained or modified here, and the raw CSV is only read.

Run with:   streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "dataset" / "IMDb Movies India.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "final_movie_rating_pipeline.joblib"
FIG_DIR = PROJECT_ROOT / "visualizations"

# --------------------------------------------------------------------------------------
# Established project results (Phases 1–6). Displayed only; never recomputed here.
# --------------------------------------------------------------------------------------
FINAL_METRICS = {"MAE": 0.8862, "MSE": 1.3453, "RMSE": 1.1599, "R²": 0.3046}
BASELINE_METRICS = {"MAE": 1.1243, "MSE": 1.9361, "RMSE": 1.3914, "R²": -0.0008}
IMPROVEMENT = {"MAE": 21.2, "RMSE": 16.6}
FINAL_EXTRA = {"Median absolute error": 0.6966, "Maximum absolute error": 5.1441,
               "Within ±1 rating point": "64.6%", "Off by more than ±2 points": "8.8%"}
SPLIT = {"Training rows": "6,335", "Test rows": "1,584", "Group overlap": "0"}
RATING_GROUP_ERRORS = pd.DataFrame({
    "Rating group": ["Low (< 4)", "Medium (4 – < 7)", "High (≥ 7)"],
    "Films": [152, 1076, 356], "MAE": [2.039, 0.642, 1.133], "RMSE": [2.195, 0.833, 1.354]})
CV_RESULTS = pd.DataFrame({
    "Model": ["Gradient Boosting (tuned)", "Random Forest", "Gradient Boosting (initial)", "Linear Regression"],
    "CV RMSE": [1.155, 1.173, 1.176, 1.223], "± std": [0.036, 0.034, 0.032, 0.024],
    "CV R²": [0.298, 0.276, 0.272, 0.212]})
PHASE3_RESULTS = pd.DataFrame({
    "Model": ["Gradient Boosting", "Random Forest", "Linear Regression", "Baseline"],
    "MAE": [0.9044, 0.9101, 0.9424, 1.1243], "RMSE": [1.1791, 1.1832, 1.2197, 1.3914],
    "R²": [0.2813, 0.2764, 0.2311, -0.0008]})
ABLATION = pd.DataFrame({
    "Feature set": ["Year + Duration", "+ Genre", "+ Director", "+ Actors (full)"],
    "CV RMSE": [1.284, 1.211, 1.184, 1.155]})
IMPORTANCE = pd.DataFrame({
    "Feature group": ["Year", "Genre", "Actors", "Director", "Duration"],
    "RMSE increase when shuffled": [0.20, 0.13, 0.10, 0.04, 0.03]})
MODEL_PARAMS = {"n_estimators": 600, "learning_rate": 0.05, "max_depth": 4,
                "min_samples_leaf": 10, "subsample": 0.8, "random_state": 42}

PAGES = ["🏠 Overview", "🎯 Predict Rating", "📊 Data Analysis", "📈 Model Performance",
         "🧠 Model Details", "ℹ️ About Project"]

ACCENT = "#8b5cf6"      # UI accent (chrome only)
SERIES = "#3987e5"      # primary data colour
SERIES_2 = "#d95926"    # secondary data colour
GOOD = "#22c55e"

st.set_page_config(page_title="Movie Rating Predictor", page_icon="🎬", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
  .stApp { background: radial-gradient(1200px 600px at 20% -10%, #241b3d 0%, #0d0b14 55%); color: #ececf1; }
  section[data-testid="stSidebar"] { background: #14111f; border-right: 1px solid #2a2440; }
  h1, h2, h3 { color: #ffffff !important; letter-spacing: -0.02em; }
  h1 { font-size: 2.6rem !important; font-weight: 800 !important; }
  h2 { font-size: 1.6rem !important; font-weight: 700 !important; margin-top: 0.4rem !important; }
  p, li, label, .stMarkdown { font-size: 1.02rem; line-height: 1.6; }
  .hero-sub { color: #b9b4d0; font-size: 1.15rem; margin: -0.4rem 0 0.4rem 0; }
  .card { background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.09);
          border-radius: 14px; padding: 1.1rem 1.25rem; height: 100%; }
  .kpi-label { color: #a7a1c2; font-size: 0.86rem; text-transform: uppercase; letter-spacing: 0.08em; }
  .kpi-value { color: #ffffff; font-size: 2.05rem; font-weight: 750; line-height: 1.15; margin-top: 0.25rem; }
  .kpi-note { color: #8f89ab; font-size: 0.85rem; margin-top: 0.2rem; }
  .flow { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
          border-radius: 14px; padding: 1rem 1.25rem; font-family: ui-monospace, Consolas, monospace;
          color: #d6d2e8; white-space: pre; overflow-x: auto; font-size: 0.95rem; line-height: 1.5; }
  .result-card { background: linear-gradient(135deg, rgba(139,92,246,0.22), rgba(57,135,229,0.14));
                 border: 1px solid rgba(139,92,246,0.45); border-radius: 18px;
                 padding: 1.6rem 1.4rem; text-align: center; }
  .result-label { color: #cfc9e6; letter-spacing: 0.16em; font-size: 0.82rem; text-transform: uppercase; }
  .result-value { font-size: 4rem; font-weight: 800; color: #ffffff; line-height: 1.05; margin: 0.3rem 0; }
  .result-scale { color: #b9b4d0; font-size: 1rem; }
  .pill { display: inline-block; padding: 0.24rem 0.7rem; border-radius: 999px; font-size: 0.82rem;
          border: 1px solid rgba(255,255,255,0.16); color: #d6d2e8; margin: 0.15rem 0.25rem 0.15rem 0; }
  .status { color: #22c55e; font-weight: 600; }
  .muted { color: #9a94b8; font-size: 0.9rem; }
  .stButton > button, div[data-testid="stFormSubmitButton"] > button {
      background: linear-gradient(135deg, #8b5cf6, #5b6ef5); color: #fff; border: 0;
      border-radius: 10px; padding: 0.65rem 1.1rem; font-weight: 650; font-size: 1.02rem; }
  .stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
      filter: brightness(1.08); color: #fff; }
  div[data-testid="stForm"] { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.09);
                              border-radius: 16px; padding: 1.2rem 1.3rem; }
  div[data-testid="stImage"] img { border-radius: 10px; }
  div[data-testid="stForm"] input, div[data-testid="stForm"] div[data-baseweb="input"],
  div[data-testid="stForm"] div[data-baseweb="base-input"] {
      background: rgba(255,255,255,0.05) !important; border-radius: 9px !important; }
  div[data-testid="stForm"] div[data-baseweb="input"] { border: 1px solid rgba(255,255,255,0.14) !important; }
  div[data-testid="stForm"] label { color: #cfc9e6 !important; font-weight: 600; }
  div[data-testid="stSidebarNav"] { display: none; }
  .stTabs [data-baseweb="tab"] { font-size: 1rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Loaders (cached — the model is loaded once per session, never retrained)
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the saved model pipeline…")
def get_artifact():
    """Load the persisted pipeline through the project's own prediction module."""
    from src.predict import load_pipeline
    return load_pipeline()


@st.cache_data(show_spinner="Loading the dataset…")
def get_data():
    """Read the raw CSV and build the cleaned frames with the project's preprocessing functions."""
    from src import data_preprocessing as dp
    raw = dp.load_raw_data(DATA_PATH)          # encoding handled inside the project module
    cleaned = dp.clean_movies(raw)
    rated, _ = dp.build_modeling_frame(raw)
    return raw, cleaned, rated


def predict(movie: dict) -> float:
    from src.predict import predict_rating
    return predict_rating(movie)


def dark_layout(fig, height=380, legend=True):
    fig.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d6d2e8", size=13), margin=dict(l=10, r=10, t=50, b=10),
        showlegend=legend, title_font=dict(size=16, color="#ffffff"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)")
    return fig


def kpi(label, value, note=""):
    st.markdown(
        f'<div class="card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)


def goto(page: str):
    st.session_state.page = page


# --------------------------------------------------------------------------------------
# Page 1 — Overview
# --------------------------------------------------------------------------------------
def page_overview():
    st.markdown("# 🎬 Movie Rating Prediction")
    st.markdown('<p class="hero-sub">Data Science &amp; Machine Learning Project</p>', unsafe_allow_html=True)
    st.markdown(
        "An end-to-end machine learning system that predicts movie ratings from movie metadata using a "
        "Gradient Boosting Regressor trained on the IMDb Movies India dataset.")
    st.write("")

    c = st.columns(3)
    with c[0]: kpi("Dataset", "15,509", "total movie records")
    with c[1]: kpi("Rated movies", "7,919", "usable labelled rows")
    with c[2]: kpi("Model", "Gradient Boosting", "regressor, tuned")
    st.write("")
    c = st.columns(3)
    with c[0]: kpi("Final RMSE", f"{FINAL_METRICS['RMSE']:.4f}", f"baseline {BASELINE_METRICS['RMSE']:.4f}")
    with c[1]: kpi("Final MAE", f"{FINAL_METRICS['MAE']:.4f}", f"baseline {BASELINE_METRICS['MAE']:.4f}")
    with c[2]: kpi("Final R²", f"{FINAL_METRICS['R²']:.4f}", "share of variance explained")
    st.write("")
    c = st.columns(3)
    with c[0]: kpi("RMSE improvement", f"{IMPROVEMENT['RMSE']}%", "versus the mean baseline")
    with c[1]: kpi("MAE improvement", f"{IMPROVEMENT['MAE']}%", "versus the mean baseline")
    with c[2]: kpi("Within ±1 point", FINAL_EXTRA["Within ±1 rating point"], "of the actual rating")

    st.caption("All figures are the Phase 5 final holdout results, measured once on 1,584 unseen films.")
    st.divider()

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("## How the System Works")
        st.markdown(
            '<div class="flow">   Movie Details\n'
            '        ↓\n'
            '   Data Preprocessing\n'
            '        ↓\n'
            '   Feature Engineering\n'
            '        ↓\n'
            '   Gradient Boosting Model\n'
            '        ↓\n'
            '   Predicted Rating</div>', unsafe_allow_html=True)
        st.write("")
        st.button("🎯 Predict a Movie", on_click=goto, args=("🎯 Predict Rating",), type="primary")
    with right:
        st.markdown("## What This Demonstrates")
        st.markdown(
            "- A **leakage-safe** pipeline: group-aware splitting and fold-local preprocessing\n"
            "- **Honest evaluation**: one final test on data untouched during development\n"
            "- **Model selection by cross-validation**, not a single lucky split\n"
            "- A **persisted pipeline** that turns raw movie details into a prediction\n"
            "- Documented **limitations**, including where the model is weakest")
        st.markdown(
            f'<span class="pill">Year</span><span class="pill">Duration</span><span class="pill">Genre</span>'
            f'<span class="pill">Director</span><span class="pill">Actors 1–3</span>', unsafe_allow_html=True)
        st.caption("Votes and Rating are never model inputs.")


# --------------------------------------------------------------------------------------
# Page 2 — Predict Rating
# --------------------------------------------------------------------------------------
def rating_band(value: float) -> tuple[str, str]:
    if value < 4:
        return "Low predicted rating range", "#ef4444"
    if value < 7:
        return "Moderate predicted rating range", "#eab308"
    if value < 8:
        return "High predicted rating range", "#3987e5"
    return "Very high predicted rating range", GOOD


def gauge_chart(value: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": " / 10", "font": {"size": 44, "color": "#ffffff"}},
        gauge={
            "axis": {"range": [0, 10], "tickwidth": 1, "tickcolor": "#8f89ab",
                     "tickmode": "array", "tickvals": [0, 2, 4, 6, 8, 10]},
            "bar": {"color": ACCENT, "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.04)", "borderwidth": 0,
            "steps": [{"range": [0, 4], "color": "rgba(239,68,68,0.18)"},
                      {"range": [4, 7], "color": "rgba(234,179,8,0.16)"},
                      {"range": [7, 8], "color": "rgba(57,135,229,0.20)"},
                      {"range": [8, 10], "color": "rgba(34,197,94,0.20)"}],
            "threshold": {"line": {"color": "#ffffff", "width": 3}, "thickness": 0.8, "value": value}}))
    return dark_layout(fig, height=300, legend=False)


def page_predict():
    st.markdown("# 🎯 Predict a Movie Rating")
    st.markdown('<p class="hero-sub">Enter movie details and the saved pipeline returns a predicted rating.</p>',
                unsafe_allow_html=True)

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Movie Name", placeholder="e.g. My Demo Film")
            year = st.number_input("Release Year *", min_value=1900, max_value=2030, value=2024, step=1,
                                   help="Required — the most influential feature in the model.")
            duration = st.number_input("Duration (minutes)", min_value=0, max_value=400, value=120, step=1,
                                       help="Set to 0 if unknown; the pipeline handles a missing duration.")
            genre = st.text_input("Genre", value="Drama, Thriller",
                                  help="Comma-separated, e.g. 'Action, Comedy, Crime'.")
        with c2:
            director = st.text_input("Director", placeholder="e.g. Example Director")
            actor1 = st.text_input("Actor 1", placeholder="Lead actor")
            actor2 = st.text_input("Actor 2", placeholder="Optional")
            actor3 = st.text_input("Actor 3", placeholder="Optional")
        submitted = st.form_submit_button("🔮 Predict Rating", type="primary", width="stretch")

    if not submitted:
        st.info("Fill in what you know and press **Predict Rating**. Only the release year is required — "
                "unknown directors, unknown actors and missing fields are all handled.")
        return

    if not year:
        st.error("Release year is required. Please enter a year between 1900 and 2030.")
        return

    movie = {"Name": name.strip() or "Untitled demo movie",
             "Year": int(year),
             "Duration": f"{int(duration)} min" if duration and duration > 0 else None,
             "Genre": genre.strip() or None,
             "Director": director.strip() or None,
             "Actor 1": actor1.strip() or None,
             "Actor 2": actor2.strip() or None,
             "Actor 3": actor3.strip() or None}

    try:
        value = predict(movie)
    except ValueError as err:
        st.error(f"Could not make a prediction: {err}")
        return
    except FileNotFoundError:
        st.error("The saved model file could not be found. Expected it at `models/final_movie_rating_pipeline.joblib`.")
        return
    except Exception:  # pragma: no cover - user-facing safety net
        st.error("Something went wrong while predicting. Please check the inputs and try again.")
        return

    band, colour = rating_band(value)
    st.write("")
    left, right = st.columns([1, 1.1])
    with left:
        st.markdown(
            f'<div class="result-card"><div class="result-label">Model Predicted Rating</div>'
            f'<div class="result-value">⭐ {value:.2f}</div>'
            f'<div class="result-scale">out of 10</div>'
            f'<div style="margin-top:0.7rem;color:{colour};font-weight:650">{band}</div></div>',
            unsafe_allow_html=True)
        st.caption("This is model output for the details you entered — not an actual or official IMDb rating.")
    with right:
        st.plotly_chart(gauge_chart(value), width="stretch",
                        config={"displayModeBar": False})

    st.caption("Bands are ranges of model output — below 4 low, 4 to under 7 moderate, 7 to under 8 high, "
               "8 and above very high. They describe the prediction, not guaranteed film quality.")
    st.divider()

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("### 🎬 Movie Information")
        actors = [a for a in [movie["Actor 1"], movie["Actor 2"], movie["Actor 3"]] if a]
        info = pd.DataFrame({
            "Field": ["Name", "Year", "Duration", "Genre", "Director", "Actors"],
            "Value": [movie["Name"], str(movie["Year"]), movie["Duration"] or "not provided",
                      movie["Genre"] or "not provided", movie["Director"] or "not provided",
                      ", ".join(actors) if actors else "not provided"]})
        st.dataframe(info, hide_index=True, width="stretch")
    with c2:
        st.markdown("### 🧩 Features used by the model")
        st.markdown("Year · Duration · Genre · Director · Actor information")
        st.markdown("### 🚫 Not used as prediction input")
        st.markdown("Rating · Votes")
        st.caption("The target Rating is what the model predicts. Votes are excluded from the primary prediction "
                   "model according to the project's modeling design.")


# --------------------------------------------------------------------------------------
# Page 3 — Data Analysis
# --------------------------------------------------------------------------------------
def page_data():
    st.markdown("# 📊 Data Analysis")
    st.markdown('<p class="hero-sub">Computed live from the raw dataset with the project\'s own '
                'preprocessing functions.</p>', unsafe_allow_html=True)
    try:
        raw, cleaned, rated = get_data()
    except FileNotFoundError:
        st.error("The dataset could not be found. Expected it at `dataset/IMDb Movies India.csv`.")
        return

    genre_tokens = rated["Genre_list"].dropna().explode()
    c = st.columns(3)
    with c[0]: kpi("Total rows", f"{len(raw):,}", f"{raw.shape[1]} columns")
    with c[1]: kpi("Rated movies", f"{len(rated):,}", f"{len(rated) / len(raw):.1%} of all rows")
    with c[2]: kpi("Missing values", f"{int(raw.isna().sum().sum()):,}", "across all columns")
    st.write("")
    c = st.columns(3)
    with c[0]: kpi("Genres", f"{genre_tokens.nunique()}", "individual genre labels")
    with c[1]: kpi("Directors", f"{raw['Director'].nunique():,}", "distinct names")
    with c[2]: kpi("Actors", f"{pd.concat([raw['Actor 1'], raw['Actor 2'], raw['Actor 3']]).nunique():,}",
                   "across the three actor columns")

    st.divider()
    t1, t2, t3, t4 = st.tabs(["⭐ Ratings", "🧩 Missing data", "🎭 Genres & people", "🔗 Relationships"])

    with t1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(rated, x="Rating", nbins=45, color_discrete_sequence=[SERIES],
                               title="Rating distribution (rated movies)")
            fig.add_vline(x=rated["Rating"].mean(), line_color=SERIES_2, line_width=2,
                          annotation_text=f"mean {rated['Rating'].mean():.2f}")
            fig.update_layout(xaxis_title="IMDb rating", yaxis_title="Number of movies")
            st.plotly_chart(dark_layout(fig, legend=False), width="stretch")
        with c2:
            dec = rated.assign(Decade=(rated["Year"] // 10 * 10).astype("Int64").astype(str) + "s")
            fig = px.box(dec.sort_values("Year"), x="Decade", y="Rating",
                         color_discrete_sequence=[SERIES], title="Rating by release decade")
            fig.update_layout(xaxis_title="Release decade", yaxis_title="IMDb rating")
            st.plotly_chart(dark_layout(fig, legend=False), width="stretch")
        st.caption(f"Mean {rated['Rating'].mean():.2f} · median {rated['Rating'].median():.2f} · "
                   f"std {rated['Rating'].std():.2f} · range {rated['Rating'].min():.1f}–{rated['Rating'].max():.1f}")

    with t2:
        miss = (raw.isna().mean() * 100).round(2).sort_values()
        fig = px.bar(x=miss.values, y=miss.index, orientation="h", color_discrete_sequence=[SERIES],
                     title="Missing values per column (% of all rows)", text=[f"{v:.1f}%" for v in miss.values])
        fig.update_layout(xaxis_title="Missing (%)", yaxis_title="Column")
        st.plotly_chart(dark_layout(fig, height=420, legend=False), width="stretch")
        st.markdown(
            "- `Rating` is missing for about half the rows — those films cannot be training examples.\n"
            "- `Duration` is the most incomplete feature (53.3% overall, 26.1% among rated films).\n"
            "- Missing durations are imputed with the **median of the film's release decade**, learned from "
            "training data, plus a 'was missing' indicator.")

    with t3:
        c1, c2 = st.columns(2)
        with c1:
            counts = genre_tokens.value_counts()
            fig = px.bar(x=counts.values, y=counts.index, orientation="h", color_discrete_sequence=[SERIES],
                         title="Movies per genre (rated movies)")
            fig.update_layout(xaxis_title="Number of movies", yaxis_title="Genre",
                              yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(dark_layout(fig, height=520, legend=False), width="stretch")
        with c2:
            dcounts = rated["Director"].value_counts()
            bins = pd.cut(dcounts, [1, 2, 3, 5, 10, 20, 1000], right=False,
                          labels=["1", "2", "3–4", "5–9", "10–19", "20+"]).value_counts().reindex(
                ["1", "2", "3–4", "5–9", "10–19", "20+"])
            fig = px.bar(x=bins.index.astype(str), y=bins.values, color_discrete_sequence=[SERIES],
                         title="How many rated movies each director has", text=bins.values)
            fig.update_layout(xaxis_title="Movies per director", yaxis_title="Number of directors")
            st.plotly_chart(dark_layout(fig, height=520, legend=False), width="stretch")

        med = (rated[["Genre_list", "Rating"]].dropna(subset=["Genre_list"]).explode("Genre_list")
               .groupby("Genre_list")["Rating"].agg(["median", "size"]))
        med = med[med["size"] >= 50].sort_values("median")
        fig = px.bar(med, x="median", y=med.index, orientation="h", color_discrete_sequence=[SERIES],
                     title="Median rating by genre (genres with at least 50 rated movies)",
                     hover_data={"size": True})
        fig.update_layout(xaxis_title="Median IMDb rating", yaxis_title="Genre")
        st.plotly_chart(dark_layout(fig, height=520, legend=False), width="stretch")
        st.caption("A movie counts in each of its genres, so the groups overlap.")

    with t4:
        rated_num = rated.assign(
            Votes_num=pd.to_numeric(rated["Votes_eda"], errors="coerce"),
            Duration=pd.to_numeric(rated["Duration_min"], errors="coerce"),
            Year_num=pd.to_numeric(rated["Year"], errors="coerce"))
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(rated_num, x="Year_num", y="Rating", opacity=0.3,
                             color_discrete_sequence=[SERIES], title="Rating vs release year")
            fig.update_layout(xaxis_title="Release year", yaxis_title="IMDb rating")
            st.plotly_chart(dark_layout(fig, legend=False), width="stretch")
        with c2:
            sub = rated_num.dropna(subset=["Duration"])
            fig = px.scatter(sub, x="Duration", y="Rating", opacity=0.3,
                             color_discrete_sequence=[SERIES], title="Rating vs duration")
            fig.update_layout(xaxis_title="Duration (minutes)", yaxis_title="IMDb rating")
            st.plotly_chart(dark_layout(fig, legend=False), width="stretch")
        st.caption("Year and duration each have only a weak linear association with rating "
                   "(Pearson r ≈ −0.17 and ≈ −0.03 on the rated rows). Votes are shown nowhere as a model "
                   "feature — they are excluded from the primary model by design.")

    with st.expander("Preview the raw data (read-only)"):
        st.dataframe(raw.head(25), width="stretch")
        st.caption("The CSV is never modified by this dashboard or by the project's notebooks.")


# --------------------------------------------------------------------------------------
# Page 4 — Model Performance
# --------------------------------------------------------------------------------------
def show_figure(filename: str, caption: str):
    path = FIG_DIR / filename
    if path.exists():
        st.image(str(path), width="stretch")
        st.caption(caption)
    else:
        st.warning(f"Chart not found: `visualizations/{filename}`")


def page_performance():
    st.markdown("# 📈 Model Performance")
    st.markdown('<p class="hero-sub">Final evaluation on a held-out, group-aware test set of 1,584 films, '
                'measured once.</p>', unsafe_allow_html=True)

    c = st.columns(4)
    for col, (label, key) in zip(c, [("MAE", "MAE"), ("MSE", "MSE"), ("RMSE", "RMSE"), ("R²", "R²")]):
        with col:
            kpi(label, f"{FINAL_METRICS[key]:.4f}", f"baseline {BASELINE_METRICS[key]:.4f}")
    st.write("")
    c = st.columns(4)
    with c[0]: kpi("RMSE improvement", f"{IMPROVEMENT['RMSE']}%", "versus mean baseline")
    with c[1]: kpi("MAE improvement", f"{IMPROVEMENT['MAE']}%", "versus mean baseline")
    with c[2]: kpi("Within ±1 point", FINAL_EXTRA["Within ±1 rating point"], "of the actual rating")
    with c[3]: kpi("Off by > ±2 points", FINAL_EXTRA["Off by more than ±2 points"], "of test films")

    st.caption("R² is the share of rating variance explained — it is **not** an accuracy percentage. "
               f"Median absolute error {FINAL_EXTRA['Median absolute error']}, maximum "
               f"{FINAL_EXTRA['Maximum absolute error']}.")

    with st.expander("How the final evaluation was set up"):
        st.markdown(
            f"- **Training rows:** {SPLIT['Training rows']} · **test rows:** {SPLIT['Test rows']} · "
            f"**group overlap:** {SPLIT['Group overlap']}\n"
            "- Films sharing a normalised `Name` + `Year` were kept on the same side of the split, so "
            "near-duplicate records could not appear in both training and test data.\n"
            "- All preprocessing statistics were fitted on training rows only.\n"
            "- The test set was used **once**, after model selection was complete.")

    st.divider()
    tabs = st.tabs(["🎯 Final evaluation", "🔁 Cross-validation", "🧪 Earlier comparison", "⚠️ Limitations"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            show_figure("17_final_actual_vs_predicted.png",
                        "Each point is a test film. The dashed diagonal is a perfect prediction; the spread "
                        "around it is the error.")
        with c2:
            show_figure("18_final_residual_distribution.png",
                        "Residual = actual − predicted. Centred near zero overall, with a long tail of large "
                        "misses.")
        c1, c2 = st.columns(2)
        with c1:
            show_figure("19_final_error_by_rating_group.png",
                        "Error by rating band: mid-range films are predicted far more accurately than extremes.")
        with c2:
            show_figure("20_final_prediction_calibration.png",
                        "Mean predicted vs mean actual rating per band. The gap between the lines is the "
                        "compression toward the middle.")
        st.dataframe(RATING_GROUP_ERRORS, hide_index=True, width="stretch")
        show_figure("21_final_top_errors.png",
                    "The 20 largest errors — dominated by films rated very low or very high.")

    with tabs[1]:
        st.markdown("### Model selection evidence (5-fold group-aware cross-validation)")
        st.markdown("These are **development** results on the training rows, used to choose the model. They are "
                    "not the final test numbers and are not directly interchangeable with them.")
        fig = px.bar(CV_RESULTS.sort_values("CV RMSE", ascending=False), x="CV RMSE", y="Model",
                     orientation="h", error_x="± std", color_discrete_sequence=[SERIES],
                     title="Mean cross-validation RMSE (lower is better)")
        fig.update_layout(xaxis_title="CV RMSE", yaxis_title="")
        st.plotly_chart(dark_layout(fig, height=340, legend=False), width="stretch")
        st.dataframe(CV_RESULTS, hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            show_figure("13_cv_error_bars.png", "Mean CV RMSE with fold-to-fold spread.")
        with c2:
            show_figure("14_feature_group_ablation.png", "RMSE as each feature group is added.")
        c1, c2 = st.columns(2)
        with c1:
            show_figure("15_feature_importance.png",
                        "Permutation importance on validation folds — how much RMSE worsens when a feature "
                        "group is shuffled.")
        with c2:
            show_figure("16_prediction_range_analysis.png",
                        "Actual vs predicted rating spread, showing the compression toward the middle.")

    with tabs[2]:
        st.markdown("### First-pass models on the same test films (before tuning)")
        st.dataframe(PHASE3_RESULTS, hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            show_figure("09_model_comparison.png", "Baseline, Linear Regression, Random Forest and Gradient "
                                                   "Boosting on the first split.")
        with c2:
            show_figure("10_actual_vs_predicted.png", "Actual vs predicted for the three first-pass models.")
        show_figure("11_residual_distribution.png", "Residual distributions of the first-pass models.")

    with tabs[3]:
        st.markdown("### Model Limitations")
        st.markdown(
            "- **Overall R² is modest** (≈ 0.30): most of the variation in ratings is not explained by this "
            "metadata.\n"
            "- **Predictions are compressed toward the middle** of the rating range (prediction standard "
            "deviation 0.82 versus 1.39 for actual ratings).\n"
            "- **Low-rated movies tend to be over-predicted** and **high-rated movies under-predicted**.\n"
            "- **Extreme ratings are harder to predict**: MAE 2.04 below 4 and 1.13 at 7 or above, against "
            "0.64 in the mid-range.\n"
            "- The model is trained on the **IMDb Movies India dataset**; other film industries or later "
            "years are outside its scope.\n"
            "- **Unseen directors and actors are handled safely**, but carry no history, so those predictions "
            "are less informative.\n"
            "- **Votes are not used** in the primary model, by design.\n"
            "- Predictions are **estimates, not guaranteed ratings**, and the system is not production-ready.")


# --------------------------------------------------------------------------------------
# Page 5 — Model Details
# --------------------------------------------------------------------------------------
def page_model_details():
    st.markdown("# 🧠 Model Details")
    st.markdown('<p class="hero-sub">The persisted pipeline behind every prediction in this dashboard.</p>',
                unsafe_allow_html=True)

    try:
        meta = get_artifact()["metadata"]
    except Exception:
        meta = {}
        st.warning("The saved pipeline could not be read, so live metadata is unavailable. "
                   "The documented configuration is shown below.")

    c = st.columns(3)
    with c[0]: kpi("Model", "Gradient Boosting", "Regressor")
    with c[1]: kpi("Training rows", f"{meta.get('training_rows', 7919):,}", "all rated movies")
    with c[2]: kpi("Engineered features", f"{meta.get('n_features_out', 413)}", "after preprocessing")

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Hyperparameters")
        st.dataframe(pd.DataFrame({"Parameter": list(MODEL_PARAMS), "Value": [str(v) for v in MODEL_PARAMS.values()]}),
                     hide_index=True, width="stretch")
        st.markdown("### Prediction target")
        st.markdown("`Rating` — the IMDb rating on a 1–10 scale.")
    with c2:
        st.markdown("### Feature groups")
        fig = px.bar(IMPORTANCE.sort_values("RMSE increase when shuffled"),
                     x="RMSE increase when shuffled", y="Feature group", orientation="h",
                     color_discrete_sequence=[SERIES], title="Permutation importance (validation folds)")
        fig.update_layout(xaxis_title="RMSE increase when shuffled", yaxis_title="")
        st.plotly_chart(dark_layout(fig, height=320, legend=False), width="stretch")
        st.caption("Shows what the predictions depend on — not a causal effect on ratings.")

    st.divider()
    st.markdown("### Feature Engineering")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "- **Genre** → multi-hot encoding over individual genre labels, plus an explicit "
            "'genre missing' indicator\n"
            "- **Director** → count of training films, plus one-hot columns for directors with "
            "**≥ 10** training films\n"
            "- **Actors** → pooled counts across all three slots, plus multi-hot columns for actors with "
            "**≥ 20** training films")
    with c2:
        st.markdown(
            "- **Duration** → imputed with the median of the film's release decade, plus a "
            "'was missing' indicator\n"
            "- **Year** → used directly as a numeric feature\n"
            "- **Excluded** → `Votes` (modeling assumption), `Rating` (target), `Name` (identifier)")
    st.info("Preprocessing is **fitted inside the saved pipeline** and reused at prediction time. The dashboard "
            "never refits it, and unseen directors, actors or genre combinations cannot break a prediction.")

    st.markdown("### Feature-group ablation")
    fig = px.line(ABLATION, x="Feature set", y="CV RMSE", markers=True, color_discrete_sequence=[SERIES],
                  title="Cross-validation RMSE as feature groups are added")
    fig.update_traces(line_width=3, marker_size=10)
    fig.update_layout(xaxis_title="", yaxis_title="CV RMSE (lower is better)")
    st.plotly_chart(dark_layout(fig, height=340, legend=False), width="stretch")

    if meta:
        with st.expander("Artifact metadata (read from the saved pipeline)"):
            st.dataframe(pd.Series({k: str(v) for k, v in meta.items()}, name="value").to_frame(),
                         width="stretch")


# --------------------------------------------------------------------------------------
# Page 6 — About
# --------------------------------------------------------------------------------------
def page_about():
    st.markdown("# ℹ️ About This Project")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown(
            "**Project:** Movie Rating Prediction  \n"
            "**Internship:** CodSoft Data Science Virtual Internship  \n"
            "**Dataset:** IMDb Movies India  \n"
            "**Problem:** Predict movie ratings from available movie metadata.")
        st.caption("This is CodSoft Task 2 — Movie Rating Prediction with Python. The local folder is named "
                   "Task1_Movie_Rating_Prediction as the author's own numbering of three selected projects.")
        st.markdown("### Technologies Used")
        for t in ["Python", "Pandas", "NumPy", "Matplotlib", "Seaborn", "Scikit-learn", "Streamlit", "Plotly",
                  "Joblib"]:
            st.markdown(f'<span class="pill">{t}</span>', unsafe_allow_html=True)
    with c2:
        st.markdown("### Workflow")
        st.markdown(
            '<div class="flow">Data Audit\n   ↓\nEDA\n   ↓\nPreprocessing\n   ↓\nFeature Engineering\n   ↓\n'
            'Model Training\n   ↓\nCross Validation\n   ↓\nModel Improvement\n   ↓\nFinal Evaluation\n   ↓\n'
            'Model Persistence\n   ↓\nInteractive Prediction Dashboard</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### Project Phases")
    st.dataframe(pd.DataFrame({
        "Phase": ["1 Dataset audit", "2 Preprocessing", "3 First models", "4 Model improvement",
                  "5 Final evaluation", "6 Model persistence", "7 Documentation & dashboard"],
        "Notebook / file": ["notebooks/01_dataset_audit.ipynb", "notebooks/02_data_preprocessing.ipynb",
                            "notebooks/03_model_training.ipynb", "notebooks/04_model_improvement.ipynb",
                            "notebooks/05_final_evaluation.ipynb", "notebooks/06_model_persistence.ipynb",
                            "README.md · app.py"],
        "Outcome": ["Data-quality findings", "Leakage-safe cleaning and features",
                    "Group-aware split, first comparison", "Cross-validation, tuning, ablation",
                    "Final holdout metrics", "Saved pipeline + prediction API",
                    "Documentation and this dashboard"]}),
        hide_index=True, width="stretch")
    st.caption("The dashboard is a presentation layer over the finished pipeline; it does not train or modify "
               "any model.")


# --------------------------------------------------------------------------------------
# Sidebar + routing
# --------------------------------------------------------------------------------------
def main():
    if "page" not in st.session_state:
        st.session_state.page = PAGES[0]

    with st.sidebar:
        st.markdown("## 🎬 Movie Rating Predictor")
        st.caption("Data Science & ML project demo")
        st.write("")
        st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
        st.write("")
        st.divider()
        try:
            meta = get_artifact()["metadata"]
            status = '<span class="status">● Model Ready</span>'
            rows = f"{meta.get('training_rows', 7919):,} rated movies"
        except Exception:
            status = '<span style="color:#ef4444;font-weight:600">● Model Unavailable</span>'
            rows = "artifact not loaded"
        st.markdown(
            f'<div class="muted"><b>Model</b><br>Gradient Boosting<br><br>'
            f'<b>Dataset</b><br>IMDb Movies India<br><br>'
            f'<b>Trained on</b><br>{rows}<br><br>{status}</div>', unsafe_allow_html=True)

    if not MODEL_PATH.exists():
        st.error("The saved model file is missing. Expected `models/final_movie_rating_pipeline.joblib`. "
                 "Run `notebooks/06_model_persistence.ipynb` to recreate it.")
        return

    {"🏠 Overview": page_overview, "🎯 Predict Rating": page_predict, "📊 Data Analysis": page_data,
     "📈 Model Performance": page_performance, "🧠 Model Details": page_model_details,
     "ℹ️ About Project": page_about}[st.session_state.page]()


main()
