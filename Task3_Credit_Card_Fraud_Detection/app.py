"""Interactive dashboard for the Credit Card Fraud Detection project (CodSoft Task 5).

A presentation layer only. Every prediction comes from the locked Phase 5 pipeline through
``src/predict.py``; every performance figure is read from the result files the notebooks wrote
(``results/``, ``models/final_model_metadata.json``). Nothing is trained, tuned, re-thresholded or
re-evaluated here, and neither the raw CSV nor the model artefact is ever written to.

Run with:   streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import data_preprocessing as dp  # noqa: E402
import predict as predict_api  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
VIZ_DIR = PROJECT_ROOT / "visualizations"
MODEL_CARD_PATH = PROJECT_ROOT / "models" / "final_model_metadata.json"

C_LEGIT, C_FRAUD, C_MODEL, C_INK = "#3b5bdb", "#e8590c", "#9c36b5", "#212529"
FEATURES = list(predict_api.FEATURE_COLUMNS)
# Short button labels for the Phase 5 demonstration examples (full names appear on hover).
EXAMPLE_LABELS = {"Fraud the model detects": "Fraud · detected", "Fraud the model misses": "Fraud · missed",
                  "Legitimate transaction": "Legitimate", "Legitimate transaction flagged (false alarm)": "False alarm"}
COMPONENTS = [f"V{i}" for i in range(1, 29)]

# Phase 3 holdout baseline, copied from notebooks/03_model_training.ipynb (section 15.1). Phase 3 did not
# write a results file, so these historical values are the only figures on this page not read from disk.
PHASE3_HOLDOUT = {
    "RF, original (Phase 3 default)": {"pr_auc": 0.787618, "precision": 0.971831, "recall": 0.726316, "f1": 0.831325},
    "RF + SMOTE (highest Phase 3 PR-AUC)": {"pr_auc": 0.811967, "precision": 0.912500, "recall": 0.768421, "f1": 0.834286},
    "LR, original (Phase 3 default)": {"pr_auc": 0.691967, "precision": 0.846154, "recall": 0.578947, "f1": 0.687500},
}

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon=":material/credit_card:", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-size: 17px; }
      .block-container { padding-top: 2.2rem; max-width: 1280px; }
      h1 { font-size: 2.3rem !important; } h2 { font-size: 1.65rem !important; } h3 { font-size: 1.3rem !important; }
      .card { border: 1px solid #dee2e6; border-radius: 10px; padding: 16px 18px; background: #ffffff; height: 100%; }
      .card .label { color: #495057; font-size: 0.92rem; margin-bottom: 4px; }
      .card .value { color: #212529; font-size: 1.85rem; font-weight: 700; line-height: 1.15; }
      .card .note { color: #6c757d; font-size: 0.85rem; margin-top: 4px; }
      .verdict { border-radius: 12px; padding: 20px 24px; margin: 6px 0 14px 0; border: 2px solid; }
      .verdict.fraud { border-color: #e8590c; background: #fff4e6; }
      .verdict.legit { border-color: #3b5bdb; background: #edf2ff; }
      .verdict .big { font-size: 2.1rem; font-weight: 800; color: #212529; }
      .verdict .sub { font-size: 1.05rem; color: #343a40; }
      .muted { color: #6c757d; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------------------
# Cached loaders - read once, never written
# --------------------------------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading the locked model...")
def load_model() -> dict:
    return predict_api.load_artifact()


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict:
    path = RESULTS_DIR / name if name != "model_card" else MODEL_CARD_PATH
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Loading the dataset...")
def load_raw_dataset() -> pd.DataFrame:
    """The raw 284,807-row CSV, checksum-verified by the project's own loader. Read-only."""
    return dp.load_data(drop_duplicates=False)


@st.cache_data(show_spinner=False)
def dataset_summary() -> dict:
    raw = load_raw_dataset()
    counts = raw["Class"].value_counts().sort_index()
    return {"rows": len(raw), "columns": raw.shape[1], "legit": int(counts[0]), "fraud": int(counts[1]),
            "fraud_pct": float(counts[1] / len(raw) * 100), "duplicates": int(raw.duplicated().sum())}


@st.cache_data(show_spinner=False)
def class_correlations() -> pd.Series:
    raw = load_raw_dataset()
    return raw.corr(numeric_only=True)["Class"].drop("Class")


@st.cache_data(show_spinner=False)
def hourly_counts() -> pd.DataFrame:
    raw = load_raw_dataset()
    hours = (raw["Time"] // 3600).astype(int)
    return raw.groupby([hours, "Class"]).size().unstack(fill_value=0)


@st.cache_data(show_spinner=False)
def per_class_histogram(feature: str, bins: int = 60, log_amount: bool = False) -> dict:
    raw = load_raw_dataset()
    values = np.log1p(raw[feature]) if log_amount else raw[feature]
    lo, hi = np.percentile(values, [0.1, 99.9])
    lo = min(lo, np.percentile(values[raw["Class"] == 1], 1))
    hi = max(hi, np.percentile(values[raw["Class"] == 1], 99))
    edges = np.linspace(lo, hi, bins + 1)
    out = {"centers": (edges[:-1] + edges[1:]) / 2}
    for cls in (0, 1):
        v = values[raw["Class"] == cls]
        hist, _ = np.histogram(v[(v >= lo) & (v <= hi)], bins=edges, density=True)
        out[cls] = hist
    return out


def card(label: str, value: str, note: str = "") -> str:
    return f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div>' + (
        f'<div class="note">{note}</div>' if note else "") + "</div>"


def cards(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, note) in zip(cols, items):
        col.markdown(card(label, value, note), unsafe_allow_html=True)


def show_image(filename: str, caption: str) -> None:
    path = VIZ_DIR / filename
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.warning(f"Figure {filename} not found - run the notebooks to regenerate it.")


def base_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), template="plotly_white",
                      font=dict(size=15), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


# --------------------------------------------------------------------------------------------------
# Page 1 - Overview
# --------------------------------------------------------------------------------------------------


def page_overview() -> None:
    final = load_json("phase5_final_metrics.json")
    m = final["metrics"]
    s = dataset_summary()

    st.title("Credit Card Fraud Detection")
    st.markdown("**CodSoft Data Science Internship — Task 5** · supervised binary classification on real, "
                "anonymised card transactions")
    st.write(
        "The task: identify fraudulent credit card transactions. The data holds two days of transactions by European "
        "cardholders (September 2013). Fraud is extremely rare, so the model has to find a handful of fraudulent "
        "payments among tens of thousands of legitimate ones without flagging genuine customers."
    )

    st.subheader("The dataset")
    cards([
        ("Transactions", f"{s['rows']:,}", f"{s['columns']} columns: Time, V1–V28, Amount, Class"),
        ("Legitimate", f"{s['legit']:,}", "Class = 0"),
        ("Fraudulent", f"{s['fraud']:,}", "Class = 1"),
        ("Fraud share", f"{s['fraud_pct']:.3f}%", f"about 1 in {s['rows'] // s['fraud']:,} transactions"),
    ])

    st.subheader("The final model — untouched test set")
    cards([
        ("PR-AUC", f"{m['pr_auc']:.3f}", "primary metric (average precision)"),
        ("Precision", f"{m['precision']:.3f}", f"{m['tp']} of {m['tp'] + m['fp']} alerts were fraud"),
        ("Recall", f"{m['recall']:.3f}", f"{m['tp']} of {m['tp'] + m['fn']} frauds caught"),
        ("F1-score", f"{m['f1']:.3f}", f"threshold {final['threshold']:.2f}"),
    ])
    st.caption(f"Random forest (100 trees, min_samples_leaf=5, max_features=0.3) · StandardScaler · no resampling · "
               f"evaluated once on {final['test_rows']:,} held-out transactions after the model was locked.")

    st.subheader("Why class imbalance changes everything")
    left, right = st.columns([1.1, 1])
    with left:
        fig = go.Figure(go.Bar(x=["Legitimate", "Fraudulent"], y=[s["legit"], s["fraud"]],
                               marker_color=[C_LEGIT, C_FRAUD], text=[f"{s['legit']:,}", f"{s['fraud']:,}"],
                               textposition="outside"))
        fig.update_yaxes(type="log", title="transactions (log scale)")
        st.plotly_chart(base_layout(fig, 380).update_layout(title="Class distribution (log scale)"), width="stretch")
    with right:
        st.markdown(
            f"""
            * Only **{s['fraud_pct']:.3f}%** of transactions are fraud — **{s['legit'] / s['fraud']:,.0f} legitimate
              for every fraudulent one**.
            * A model that calls *everything* legitimate is **{s['legit'] / s['rows'] * 100:.2f}% accurate** and catches
              **no fraud at all** — so accuracy is not a useful score here.
            * The project therefore judges models on **precision** (how many alerts are real), **recall** (how much
              fraud is caught), **F1**, and **PR-AUC**, which summarises the precision/recall trade-off across all
              thresholds.
            """
        )


# --------------------------------------------------------------------------------------------------
# Page 2 - Fraud prediction
# --------------------------------------------------------------------------------------------------


def _set_fields(values: dict) -> None:
    for f in FEATURES:
        st.session_state[f"field_{f}"] = repr(float(values[f])) if f in values else ""


def page_predict() -> None:
    demo = load_json("phase5_demo_examples.json")
    threshold = predict_api.model_threshold()
    load_model()

    st.title("Fraud Prediction")
    st.write("Enter the 30 features of one transaction and analyse it with the locked model. "
             f"A transaction is classified as **fraud** when its fraud probability is **≥ {threshold:.2f}** "
             "(the threshold selected in Phase 4 from training data).")

    for f in FEATURES:
        st.session_state.setdefault(f"field_{f}", "")

    with st.container(border=True):
        st.markdown("**Load a demonstration example** — real transactions from the held-out test split of the Kaggle "
                    "dataset, chosen after the final evaluation (the first test row in each outcome category). They are "
                    "shown for demonstration only; they played no part in building the model.")
        cols = st.columns(len(demo["examples"]) + 1)
        for col, ex in zip(cols, demo["examples"]):
            if col.button(EXAMPLE_LABELS.get(ex["name"], ex["name"]), key=f"ex_{ex['name']}", help=ex["name"],
                          width="stretch"):
                _set_fields(ex["features"])
                st.session_state["loaded_example"] = ex
                st.session_state.pop("last_result", None)
        if cols[-1].button("Clear all fields", key="clear", width="stretch"):
            _set_fields({})
            st.session_state.pop("loaded_example", None)
            st.session_state.pop("last_result", None)
        if "loaded_example" in st.session_state:
            ex = st.session_state["loaded_example"]
            st.caption(f"Loaded: **{ex['name']}** — true class in the dataset: "
                       f"{'fraud' if ex['true_class'] == 1 else 'legitimate'}.")

    tab_form, tab_paste = st.tabs(["Enter features", "Paste a full transaction"])
    with tab_form:
        st.markdown("##### Transaction information")
        c1, c2 = st.columns(2)
        c1.text_input("Time — seconds since the first transaction in the dataset", key="field_Time")
        c2.text_input("Amount — transaction amount", key="field_Amount")
        st.markdown("##### PCA components V1–V28")
        st.caption("Anonymised principal components supplied by the dataset owners; their original meaning is not "
                   "published.")
        grid = st.columns(4)
        for i, comp in enumerate(COMPONENTS):
            grid[i % 4].text_input(comp, key=f"field_{comp}")
        submitted = st.button("Analyze Transaction", type="primary", key="analyze", width="stretch")
    with tab_paste:
        st.write("Paste either a JSON object with all 30 features, or 30 comma-separated numbers in the order "
                 "Time, V1, …, V28, Amount.")
        pasted = st.text_area("Transaction", key="pasted", height=110)
        submitted_paste = st.button("Analyze pasted transaction", key="analyze_paste", width="stretch")

    transaction = None
    if submitted:
        transaction = {f: st.session_state.get(f"field_{f}", "") for f in FEATURES}
    elif submitted_paste:
        text = (pasted or "").strip()
        if not text:
            st.error("Nothing to analyse - paste a transaction first.")
        elif text.startswith("{"):
            try:
                transaction = json.loads(text)
            except json.JSONDecodeError as exc:
                st.error(f"That is not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno}).")
        else:
            transaction = [part.strip() for part in text.replace("\n", ",").split(",") if part.strip() != ""]

    if transaction is not None:
        try:
            st.session_state["last_result"] = predict_api.predict_transaction(transaction)
        except predict_api.InvalidTransactionError as exc:
            st.session_state.pop("last_result", None)
            st.error(f"**The transaction could not be analysed.** {exc}")

    result = st.session_state.get("last_result")
    if result:
        fraud = result["predicted_class"] == 1
        st.markdown(
            f'<div class="verdict {"fraud" if fraud else "legit"}"><div class="big">Prediction: '
            f'{"FRAUD" if fraud else "LEGITIMATE"}</div><div class="sub">Fraud probability '
            f'<b>{result["fraud_probability"]:.4f}</b> · threshold <b>{result["threshold"]:.2f}</b> · decision rule: '
            f'fraud if probability ≥ threshold</div></div>',
            unsafe_allow_html=True,
        )
        fig = go.Figure(go.Bar(x=[result["fraud_probability"]], y=["P(fraud)"], orientation="h",
                               marker_color=C_FRAUD if fraud else C_LEGIT, width=0.5))
        fig.add_vline(x=result["threshold"], line_dash="dash", line_color=C_INK,
                      annotation_text=f"threshold {result['threshold']:.2f}", annotation_position="top")
        fig.update_xaxes(range=[0, 1], title="fraud probability")
        st.plotly_chart(base_layout(fig, 200).update_layout(showlegend=False, margin=dict(t=40)), width="stretch")
        st.info("This is a model score, not a verdict. A transaction flagged here is not proven fraudulent, and one "
                "passed is not proven genuine: on the untouched test set the model caught 69 of 95 frauds and raised "
                "4 false alarms. Real fraud decisions involve investigation, context and costs this model does not see.")


# --------------------------------------------------------------------------------------------------
# Page 3 - Data analysis
# --------------------------------------------------------------------------------------------------


def page_analysis() -> None:
    s = dataset_summary()
    st.title("Data Analysis")
    st.write(f"Exploratory views of the raw dataset ({s['rows']:,} transactions, read directly from "
             "`dataset/creditcard.csv`). Nothing here trains or evaluates a model.")
    cards([
        ("Legitimate", f"{s['legit']:,}", f"{100 - s['fraud_pct']:.3f}%"),
        ("Fraudulent", f"{s['fraud']:,}", f"{s['fraud_pct']:.3f}%"),
        ("Imbalance", f"{s['legit'] / s['fraud']:.0f} : 1", "legitimate per fraud"),
        ("Exact duplicate rows", f"{s['duplicates']:,}", "dropped before the split in Phase 2"),
    ])

    tab1, tab2, tab3, tab4 = st.tabs(["Transaction amount", "Time", "Feature distributions", "Correlation"])
    with tab1:
        h = per_class_histogram("Amount", bins=60, log_amount=True)
        fig = go.Figure()
        for cls, name, color in ((0, "Legitimate", C_LEGIT), (1, "Fraudulent", C_FRAUD)):
            fig.add_trace(go.Scatter(x=h["centers"], y=h[cls], mode="lines", name=name, line=dict(color=color, width=3),
                                     fill="tozeroy", opacity=0.6))
        fig.update_xaxes(title="log1p(Amount) — log scale used for display only")
        fig.update_yaxes(title="density within class")
        st.plotly_chart(base_layout(fig).update_layout(title="Amount by class (each class normalised separately)"),
                        width="stretch")
        st.write("Amounts are heavily right-skewed (Phase 1: skewness 16.98). Fraud has a *lower* median amount than "
                 "legitimate traffic (9.25 vs 22.00) but a higher mean (122.21 vs 88.29).")
    with tab2:
        hc = hourly_counts()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.08)
        fig.add_trace(go.Bar(x=hc.index, y=hc[0], name="Legitimate", marker_color=C_LEGIT), row=1, col=1)
        fig.add_trace(go.Bar(x=hc.index, y=hc[1], name="Fraudulent", marker_color=C_FRAUD), row=2, col=1)
        fig.update_yaxes(title_text="legitimate / hour", row=1, col=1)
        fig.update_yaxes(title_text="fraud / hour", row=2, col=1)
        fig.update_xaxes(title_text="hours since the first transaction", row=2, col=1)
        st.plotly_chart(base_layout(fig, 520).update_layout(title="Transactions per elapsed hour (~48-hour window)"),
                        width="stretch")
        st.write("`Time` counts seconds from the first transaction in the file — not a clock time. Legitimate volume "
                 "shows a day/night rhythm; fraud is spread more evenly, so its *share* rises overnight.")
    with tab3:
        feature = st.selectbox("Component", COMPONENTS, index=COMPONENTS.index("V14"))
        h = per_class_histogram(feature)
        fig = go.Figure()
        for cls, name, color in ((0, "Legitimate", C_LEGIT), (1, "Fraudulent", C_FRAUD)):
            fig.add_trace(go.Scatter(x=h["centers"], y=h[cls], mode="lines", name=name, line=dict(color=color, width=3),
                                     fill="tozeroy", opacity=0.6))
        fig.update_xaxes(title=f"{feature} (display window covers the bulk of both classes)")
        fig.update_yaxes(title="density within class")
        st.plotly_chart(base_layout(fig).update_layout(title=f"{feature} by class"), width="stretch")
        st.write("V14, V17, V12 and V10 show the clearest separation between the classes (Phase 1). This is a "
                 "descriptive view, not a measure of feature importance.")
    with tab4:
        corr = class_correlations().reindex(class_correlations().abs().sort_values().index)
        fig = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation="h",
                               marker_color=[C_FRAUD if v > 0 else C_LEGIT for v in corr.values]))
        fig.update_xaxes(title="Pearson correlation with Class (point-biserial)")
        st.plotly_chart(base_layout(fig, 760).update_layout(title="Linear association of each feature with fraud"),
                        width="stretch")
        st.write("Marginal, linear associations only — not causation and not importance. The V-components are "
                 "mutually uncorrelated by construction (PCA).")
        show_image("06_correlation_heatmap.png", "Phase 1 correlation heatmap")


# --------------------------------------------------------------------------------------------------
# Page 4 - Model performance
# --------------------------------------------------------------------------------------------------


def page_performance() -> None:
    final = load_json("phase5_final_metrics.json")
    m, boot = final["metrics"], final["bootstrap_95"]
    st.title("Model Performance")
    st.write(f"**Final test results** of the locked model — evaluated **once** on {final['test_rows']:,} held-out "
             f"transactions ({m['support_fraud']} fraudulent) that played no part in training, tuning or threshold "
             "selection. Read from `results/phase5_final_metrics.json`; nothing is recomputed here.")
    cards([
        ("PR-AUC", f"{m['pr_auc']:.4f}", f"95% bootstrap {boot['pr_auc']['low']:.3f}–{boot['pr_auc']['high']:.3f}"),
        ("Precision", f"{m['precision']:.4f}", f"{boot['precision']['low']:.3f}–{boot['precision']['high']:.3f}"),
        ("Recall", f"{m['recall']:.4f}", f"{boot['recall']['low']:.3f}–{boot['recall']['high']:.3f}"),
        ("F1-score", f"{m['f1']:.4f}", f"{boot['f1']['low']:.3f}–{boot['f1']['high']:.3f}"),
    ])
    st.write("")
    cards([
        ("ROC-AUC", f"{m['roc_auc']:.4f}", "secondary - optimistic under imbalance"),
        ("Accuracy", f"{m['accuracy']:.4f}", "all-legitimate baseline: 0.9983"),
        ("False positive rate", f"{m['false_positive_rate']:.5f}", f"{m['fp']} of {m['support_legitimate']:,} legitimate"),
        ("False negative rate", f"{m['false_negative_rate']:.4f}", f"{m['fn']} of {m['support_fraud']} frauds missed"),
    ])

    st.subheader("Confusion matrix")
    left, right = st.columns([1, 1])
    with left:
        z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
        share = [[m["tn"] / m["support_legitimate"], m["fp"] / m["support_legitimate"]],
                 [m["fn"] / m["support_fraud"], m["tp"] / m["support_fraud"]]]
        text = [[f"True Negative<br><b>{m['tn']:,}</b>", f"False Positive<br><b>{m['fp']:,}</b>"],
                [f"False Negative<br><b>{m['fn']:,}</b>", f"True Positive<br><b>{m['tp']:,}</b>"]]
        fig = go.Figure(go.Heatmap(z=share, x=["Predicted legitimate", "Predicted fraud"],
                                   y=["Actual legitimate", "Actual fraud"], text=text, texttemplate="%{text}",
                                   colorscale="Greys", zmin=0, zmax=1, showscale=False, textfont=dict(size=17)))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(base_layout(fig, 400).update_layout(title="Shading = share of the actual class"),
                        width="stretch")
    with right:
        st.markdown(
            f"""
            * **{m['tp']} frauds caught** and **{m['fn']} missed** out of {m['support_fraud']}.
            * **{m['fp']} false alarms** among {m['support_legitimate']:,} legitimate transactions.
            * A **false positive** is a genuine customer's payment flagged for review; a **false negative** is a
              fraudulent payment that goes through. Which costs more depends on business context this dataset does
              not contain, so the threshold was chosen by maximum out-of-fold F1 rather than by an assumed cost.
            """
        )

    st.subheader("Precision-recall and ROC curves")
    c1, c2 = st.columns(2)
    with c1:
        show_image("26_final_precision_recall_curve.png", "Final test precision-recall curve with the locked operating point")
    with c2:
        show_image("27_final_roc_curve.png", "Final test ROC curve")
    st.write("PR-AUC is the headline metric: its precision axis counts every false alarm against the alerts raised. "
             "ROC-AUC measures the false-positive rate against *all* 56,651 legitimate transactions, so even hundreds "
             "of false alarms would barely move it.")

    st.subheader("Three evaluations — kept separate")
    cv = final["phase4_cv"]
    rows = [{"evaluation": f"Phase 3 holdout baseline — {k}", **v} for k, v in PHASE3_HOLDOUT.items()]
    rows.append({"evaluation": "Phase 4 training-only CV — locked configuration (mean of 5 folds)",
                 **{k: cv[k]["mean"] for k in ("pr_auc", "precision", "recall", "f1")}})
    rows.append({"evaluation": "Phase 5 FINAL TEST — locked model",
                 **{k: m[k] for k in ("pr_auc", "precision", "recall", "f1")}})
    table = pd.DataFrame(rows).set_index("evaluation")
    st.dataframe(table.style.format("{:.4f}"), width="stretch")
    st.caption("Phase 3 scored untuned baselines on the same test set; Phase 4 selected the model with training-only "
               "cross-validation; Phase 5 is the single unbiased evaluation of the locked model. They are not a "
               "contest - Phase 3's test numbers were never used to choose the model.")


# --------------------------------------------------------------------------------------------------
# Page 5 - Model details
# --------------------------------------------------------------------------------------------------


def page_details() -> None:
    card_ = load_json("model_card")
    cand = load_json("phase4_candidate.json")
    cv = cand["cv_metrics_at_0_5"]
    st.title("Model Details")

    st.subheader("The locked pipeline")
    params = card_["classifier_params"]
    st.markdown(
        f"""
        | Component | Setting |
        |---|---|
        | Preprocessing | `StandardScaler` on all 30 features (`Time`, `V1`–`V28`, `Amount`), fitted on the 226,980 training rows |
        | Imbalance handling | none — original class distribution, `class_weight=None` (selected in Phase 4) |
        | Classifier | random forest, `n_estimators={params['n_estimators']}`, `min_samples_leaf={params['min_samples_leaf']}`, `max_features={params['max_features']}`, `max_depth={params['max_depth']}`, `random_state=42` |
        | Decision rule | fraud if P(fraud) ≥ **{card_['threshold']:.2f}** |
        | Artefact | `{card_['artifact']}` · {card_['artifact_bytes']:,} bytes · SHA-256 `{card_['artifact_sha256'][:16]}…` |
        """
    )
    st.caption("Prediction flow: dashboard → src/predict.py → saved pipeline (scaler → forest) → fraud probability → "
               "locked threshold → decision. The dashboard contains no preprocessing or model code of its own.")

    st.subheader("How it was chosen — training data only")
    st.markdown(
        f"""
        * **Stratified 5-fold cross-validation** on the training split (`random_state=42`); scaler and any resampler
          re-fitted inside every fold, validation folds never resampled.
        * **Compared:** 3 scalings × 5 imbalance strategies for logistic regression, 5 strategies for random forest, then
          hyperparameter grids for both — 305 cross-validated fits in total.
        * **Selection rule, declared in advance:** highest mean CV PR-AUC, then the one-standard-error rule — among
          configurations statistically indistinguishable from the best, choose the simplest.
        * **Threshold rule:** maximum F1 on out-of-fold training predictions → **{cand['candidate_operating_threshold']:.2f}**.
        * **Test set:** locked away through all of this; used exactly once, in Phase 5, after the model was saved.
        """
    )
    cards([
        ("CV PR-AUC", f"{cv['pr_auc']['mean']:.4f}", f"± {cv['pr_auc']['std']:.4f} across 5 folds"),
        ("CV precision @0.5", f"{cv['precision']['mean']:.4f}", f"± {cv['precision']['std']:.4f}"),
        ("CV recall @0.5", f"{cv['recall']['mean']:.4f}", f"± {cv['recall']['std']:.4f}"),
        ("CV F1 @0.5", f"{cv['f1']['mean']:.4f}", f"± {cv['f1']['std']:.4f}"),
    ])
    st.caption("Training-only cross-validation estimates - not test results.")

    with st.expander("Top configurations from the Phase 4 cross-validation", expanded=False):
        summary = pd.read_csv(RESULTS_DIR / "phase4_cv_summary.csv", index_col=0)
        top = summary.sort_values("pr_auc_mean", ascending=False).head(10)[
            ["model", "imbalance", "pr_auc_mean", "pr_auc_std", "precision_mean", "recall_mean", "f1_mean"]]
        st.dataframe(top.style.format({c: "{:.4f}" for c in top.columns if c.endswith(("mean", "std"))}),
                     width="stretch")
    show_image("22_precision_recall_threshold_analysis.png", "Phase 4: out-of-fold threshold analysis for the candidate")
    show_image("17_cv_model_comparison.png", "Phase 4: every cross-validated configuration (mean ± std CV PR-AUC)")

    st.subheader("Why PR-AUC, and not accuracy")
    st.write("With 0.17% fraud, a model that never predicts fraud is 99.83% accurate. Accuracy therefore says almost "
             "nothing about fraud detection. PR-AUC summarises how well fraud is ranked above legitimate traffic across "
             "every threshold, and its no-skill level is the fraud rate (≈ 0.0017), not 0.5.")


# --------------------------------------------------------------------------------------------------
# Page 6 - About
# --------------------------------------------------------------------------------------------------


def page_about() -> None:
    st.title("About the Project")
    st.markdown(
        """
        **CodSoft Data Science Internship — Task 5: Credit Card Fraud Detection** (local project folder
        `Task3_Credit_Card_Fraud_Detection`).

        **Problem statement (CodSoft):** build a machine learning model to identify fraudulent credit card
        transactions; preprocess and normalise the data, handle class imbalance, split into training and testing
        sets, train a classifier such as logistic regression or random forests, evaluate with precision, recall and
        F1, and consider oversampling or undersampling.

        **Dataset provenance:** Kaggle `mlg-ulb/creditcardfraud` (Machine Learning Group, Université Libre de
        Bruxelles), linked from the CodSoft task document. 284,807 transactions, 31 columns. `V1`–`V28` are
        anonymised PCA components; only `Time`, `Amount` and `Class` are in original form. Licence: DbCL 1.0.
        """
    )
    st.subheader("Workflow")
    st.markdown(
        """
        | Phase | What was done |
        |---|---|
        | 1 | Provenance, audit and exploratory analysis of the raw dataset (checksummed, never modified) |
        | 2 | Duplicate removal, stratified 80/20 split, scaling pipeline, imbalance framework |
        | 3 | Baseline logistic regression and random forest under five imbalance strategies |
        | 4 | Training-only cross-validation: scaling, imbalance strategy, tuning, threshold, candidate selection |
        | 5 | Model locked, trained, saved, and evaluated once on the untouched test set |
        | 6 | This dashboard, built on the saved model |
        """
    )
    st.subheader("Technologies")
    st.write("Python 3.13 · pandas · NumPy · SciPy · scikit-learn · imbalanced-learn · joblib · Matplotlib · seaborn · "
             "Plotly · Streamlit · Jupyter")
    st.subheader("Limitations")
    st.markdown(
        """
        * Two days of transactions from September 2013; fraud patterns and customer behaviour change over time.
        * Anonymised features: no domain interpretation of individual components is possible.
        * 95 frauds in the test set: every fraud-class metric carries real uncertainty (recall 95% bootstrap interval ≈ 0.63–0.81).
        * The threshold balances precision and recall (maximum F1); a real deployment would set it from actual costs.
        * This is a study and demonstration, not a production fraud system: no monitoring, drift detection or
          real-time infrastructure.
        """
    )
    st.subheader("Future improvements")
    st.markdown(
        """
        Probability calibration · cost-sensitive threshold selection · time-ordered validation · drift monitoring ·
        model explanations (e.g. SHAP) · real-time scoring service with monitoring and a retraining policy.
        """
    )


PAGES = [
    st.Page(page_overview, title="Overview", icon=":material/home:", url_path="overview", default=True),
    st.Page(page_predict, title="Fraud Prediction", icon=":material/search:", url_path="prediction"),
    st.Page(page_analysis, title="Data Analysis", icon=":material/bar_chart:", url_path="analysis"),
    st.Page(page_performance, title="Model Performance", icon=":material/monitoring:", url_path="performance"),
    st.Page(page_details, title="Model Details", icon=":material/tune:", url_path="details"),
    st.Page(page_about, title="About Project", icon=":material/info:", url_path="about"),
]

def main() -> None:
    st.sidebar.markdown("### Credit Card Fraud Detection")
    st.sidebar.caption("CodSoft Task 5 · locked random forest · threshold "
                       f"{predict_api.model_threshold():.2f}")
    st.navigation(PAGES).run()


# `streamlit run app.py` (and AppTest.from_file) execute this file as __main__. Importing it - as the
# dashboard tests do to render one page at a time - defines the pages without starting navigation.
if __name__ == "__main__":
    main()
