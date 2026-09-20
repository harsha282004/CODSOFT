# Movie Rating Prediction

Predicting the IMDb rating of Indian movies from the information known about a film — its year, runtime,
genres, director and lead cast — using regression.

**CodSoft Data Science Internship — Movie Rating Prediction.**
This folder is named `Task1_Movie_Rating_Prediction` as my own project numbering; it corresponds to
**CodSoft Task 2: Movie Rating Prediction with Python**.

**Status:** complete. Final results come from a held-out test set of 1,584 films that was never used during
development.

---

## Overview

| | |
|---|---|
| **Task** | Regression — predict a film's rating (1–10 scale) |
| **Dataset** | IMDb India movies, 15,509 rows × 10 columns |
| **Usable labelled rows** | 7,919 (rows that have a rating) |
| **Final model** | Gradient Boosting Regressor (tuned) |
| **Final test performance** | MAE **0.886** · RMSE **1.160** · R² **0.305** |
| **Improvement over mean baseline** | 21.2% MAE, 16.6% RMSE |
| **Deliverable** | A saved pipeline (`models/final_movie_rating_pipeline.joblib`) plus a prediction API |

---

## Problem Statement

Given the metadata available about a movie before or independently of audience response, estimate the rating it
receives. The challenge is that this metadata is sparse and highly categorical: thousands of distinct directors and
actors, multi-label genres, and missing values in most columns.

A second challenge is **evaluating honestly**. The dataset contains repeated `Name` + `Year` records, and
frequency-based features can easily leak information from validation data into training. Both issues are addressed
explicitly in this project.

## Objectives

1. Audit the raw dataset and document what is actually in it.
2. Build a reproducible, leakage-safe cleaning and feature pipeline.
3. Compare regression models against a meaningful baseline.
4. Select a model using group-aware cross-validation, not a single lucky split.
5. Report one final, unbiased evaluation on untouched data.
6. Package the result as a reusable prediction pipeline.

---

## Dataset

- **File:** `dataset/IMDb Movies India.csv` (kept unmodified throughout; its SHA-256 is verified in every notebook)
- **Size:** 15,509 rows × 10 columns
- **Columns:** `Name`, `Year`, `Duration`, `Genre`, `Rating`, `Votes`, `Director`, `Actor 1`, `Actor 2`, `Actor 3`
- **Encoding:** not valid UTF-8 — it must be read with `encoding="latin-1"`

Key findings from the audit (`notebooks/01_dataset_audit.ipynb`):

| Finding | Detail |
|---|---|
| Target availability | Only **7,919 of 15,509** rows have a `Rating`; the rest cannot be used as training examples |
| Values stored as text | `Year` as `"(2019)"`, `Duration` as `"109 min"`, `Votes` as `"25,732"` — all need parsing |
| Missing metadata | `Duration` 53.3% missing overall (26.1% among rated rows); `Genre` 12.1%; actors 10–20% |
| Duplicates | 6 exact duplicate rows; 36 `Name` + `Year` repeats that differ in other fields |
| Data quality | One `Votes` value is `"$5.16M"` — not a vote count; one title is blank |
| Genres | 485 distinct genre strings built from just **24** individual genre labels |
| High cardinality | 5,938 directors and 10,288 distinct actors across the three actor columns |

---

## Technology Stack

Python 3.13 · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · joblib · Jupyter

Exact pinned versions are in [`requirements.txt`](requirements.txt).

---

## Project Workflow

```
            Raw IMDb India dataset (read-only)
                          ↓
      Phase 1  Dataset audit ................. notebooks/01
                          ↓
      Phase 2  Data cleaning & feature design . notebooks/02 + src/data_preprocessing.py
                          ↓
      Phase 3  Group-aware 80/20 split ........ notebooks/03
               First model comparison
                          ↓
      Phase 4  5-fold group-aware CV .......... notebooks/04
               Tuning · ablation · importance
                          ↓
      Phase 5  FINAL holdout evaluation ....... notebooks/05
                          ↓
      Phase 6  Model persistence .............. notebooks/06 + src/predict.py
               Reusable prediction pipeline
```

| Stage | Phases | Data used |
|---|---|---|
| **Development** | 1–4 | Training rows only (6,335) |
| **Final evaluation** | 5 | The untouched test set (1,584 films), used exactly once |
| **Deployment-style artifact** | 6 | All 7,919 rated rows, after evaluation was complete |

---

## Data Preprocessing

All cleaning lives in [`src/data_preprocessing.py`](src/data_preprocessing.py) and is split into two layers, which
is what makes the pipeline leakage-safe:

**1. Row-wise cleaning** — each row is processed on its own, so it is safe to run before splitting:

- `Year`: `"(2019)"` → `2019` (strict pattern; anything else becomes missing, never guessed)
- `Duration`: `"109 min"` → `109`
- `Votes`: `"25,732"` → `25732`, kept **for analysis only**; `"$5.16M"` is flagged rather than interpreted
- `Genre`: split into individual genre labels
- Exact duplicate rows removed; rows without a `Rating` excluded (missing ratings are never imputed)

**2. Fitted transformers** — these learn from data, so they are fitted on training rows only:

| Column | Strategy |
|---|---|
| `Duration` | Imputed with the **median of the film's release decade**, plus a "was missing" indicator. Chosen because missingness and typical runtime both vary strongly by era (8% missing for 2010s films vs 30–50% for older ones) |
| `Genre` | Multi-hot encoding over individual genres, plus an explicit "genre missing" indicator |
| `Director` | Frequency count (films per director) + one-hot columns for directors with **≥ 10** training films |
| `Actor 1–3` | Pooled frequency counts across all three slots + multi-hot columns for actors with **≥ 20** training films |

**Deliberately excluded:** `Votes` (accumulated audience activity may not be available when a prediction is made —
a documented modelling assumption, not a leakage claim), `Rating` (the target), and `Name` (an identifier).

### Preventing leakage

- **Group-aware splitting.** Records sharing a normalised `Name` + `Year` always stay on the same side of a split, so
  near-duplicate films cannot appear in both training and evaluation. Verified group overlap: **0**.
- **Fold-local fitting.** Every statistic — imputation values, genre vocabulary, director and actor counts — is
  learned inside the training portion of each fold. Directors that appear only in validation data are provably
  absent from the fitted statistics.
- **No target encoding.** No feature is derived from `Rating`.

---

## Feature Engineering

Adding feature groups one at a time, measured with 5-fold group cross-validation and the tuned model
(`notebooks/04_model_improvement.ipynb`):

| Features | CV RMSE |
|---|---|
| Year + Duration | 1.284 |
| + Genre | 1.211 |
| + Director | 1.184 |
| + Actors (full set) | **1.155** |

Every step improved RMSE on all five folds.

Permutation importance on the validation folds — how much RMSE worsens when a feature group is shuffled:

| Feature group | RMSE increase when shuffled |
|---|---|
| Year | +0.20 |
| Genre | +0.13 |
| Actors | +0.10 |
| Director | +0.04 |
| Duration | +0.03 |

These show what the model's **predictions depend on**. They are not causal statements about what makes a film
well rated.

---

## Model Development

Five-fold group-aware cross-validation on the training rows only (`notebooks/04_model_improvement.ipynb`).
**These are development results, used for model selection — not the project's final performance.**

| Model | CV RMSE | CV MAE | CV R² |
|---|---|---|---|
| **Gradient Boosting (tuned)** | **1.155 ± 0.036** | 0.901 ± 0.026 | 0.298 ± 0.013 |
| Random Forest | 1.173 ± 0.034 | — | 0.276 ± 0.012 |
| Gradient Boosting (initial settings) | 1.176 ± 0.032 | — | 0.272 ± 0.007 |
| Linear Regression | 1.223 ± 0.024 | — | 0.212 ± 0.008 |

Tuning was deliberately small: 12 sampled Gradient Boosting configurations and 10 Random Forest configurations,
scored on RMSE with the same group-aware folds.

## Model Selection

Random Forest and the initial Gradient Boosting settings were **statistically indistinguishable** in
cross-validation, so neither was declared better on a single-split result. The tuned Gradient Boosting model was
selected because it:

- had the lowest mean CV RMSE and MAE and the highest R²
- beat every alternative on **all five folds**, by more than the fold-to-fold noise
- showed a much smaller overfitting gap than Random Forest (train-to-validation RMSE gap 0.16 vs 0.38)

**Final configuration:**

```python
GradientBoostingRegressor(
    n_estimators=600, learning_rate=0.05, max_depth=4,
    min_samples_leaf=10, subsample=0.8, random_state=42,
)
```

---

## Final Holdout Results

Measured once, in `notebooks/05_final_evaluation.ipynb`, on the **1,584-film test set** that was excluded from all
development, tuning and selection. Training rows: 6,335. Group overlap: 0.

| Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| **Gradient Boosting (tuned) — final** | **0.8862** | **1.3453** | **1.1599** | **0.3046** |
| Baseline (predicting the training mean) | 1.1243 | 1.9361 | 1.3914 | −0.0008 |

**Improvement over the baseline: 21.2% MAE, 16.6% RMSE.**

Additional final-test detail: median absolute error **0.697**, maximum absolute error **5.144**, **64.6%** of films
predicted within 1 rating point, **8.8%** off by more than 2.

### Earlier single-split comparison (Phase 3)

First-pass models on the same test films, before tuning — shown for context, not as the final result:

| Model (Phase 3, single split) | MAE | RMSE | R² |
|---|---|---|---|
| Gradient Boosting | 0.9044 | 1.1791 | 0.2813 |
| Random Forest | 0.9101 | 1.1832 | 0.2764 |
| Linear Regression | 0.9424 | 1.2197 | 0.2311 |
| Baseline | 1.1243 | 1.3914 | −0.0008 |

Cross-validation means (model development) and holdout scores (final evaluation) answer different questions and are
**not interchangeable**: the first averages performance across several validation splits of the training data, the
second measures one model on films it had never seen.

---

## Error Analysis

Performance by actual rating (final test set):

| Rating group | Films | MAE | RMSE |
|---|---|---|---|
| Low (< 4) | 152 | 2.039 | 2.195 |
| Medium (4 – < 7) | 1,076 | **0.642** | 0.833 |
| High (≥ 7) | 356 | 1.133 | 1.354 |

**Predictions are compressed toward the middle of the scale.** Actual ratings span 1.6–9.7 with a standard deviation
of 1.391, while predictions span 3.46–9.24 with a standard deviation of 0.819. The model over-predicts films that
were rated very poorly and under-predicts films that were rated very highly — the calibration chart below shows this
directly.

Other observed patterns (associations, not proven causes): error falls steadily as a director's number of training
films rises, and recent releases are harder to predict than older ones.

---

## Prediction Pipeline

The saved artifact **`models/final_movie_rating_pipeline.joblib`** (~0.33 MB) contains:

1. the fitted preprocessing transformer,
2. the fitted Gradient Boosting model,
3. metadata (hyperparameters, thresholds, feature configuration, training row count, library versions, source-data
   checksum).

A caller never has to reproduce the preprocessing:

```
raw movie fields → fitted transformer → 413 features → fitted model → predicted rating
```

**Important distinction:** the artifact was retrained on **all 7,919 rated rows** *after* the final evaluation was
complete, so it uses every labelled observation. The reported performance above comes from Phase 5's untouched
1,584-film test set, where the model had been trained on 6,335 rows. The artifact itself is not re-evaluated,
because no unseen data remains for it.

Unseen directors, unseen actors and new genre combinations are handled safely: frequency features become 0 and no
name-specific column is set, rather than the prediction failing.

---

## Project Structure

```
Task1_Movie_Rating_Prediction/
│
├── dataset/
│   └── IMDb Movies India.csv          # raw data, never modified
│
├── models/
│   └── final_movie_rating_pipeline.joblib   # preprocessing + model + metadata
│
├── notebooks/
│   ├── 01_dataset_audit.ipynb         # what is actually in the data
│   ├── 02_data_preprocessing.ipynb    # cleaning and feature design decisions
│   ├── 03_model_training.ipynb        # first split, first model comparison
│   ├── 04_model_improvement.ipynb     # group CV, tuning, ablation, importance
│   ├── 05_final_evaluation.ipynb      # final holdout evaluation
│   └── 06_model_persistence.ipynb     # artifact creation and pipeline tests
│
├── src/
│   ├── data_preprocessing.py          # cleaning + leakage-safe transformers
│   └── predict.py                     # load_pipeline / predict_rating / predict_movies
│
├── visualizations/                    # 21 generated charts (01–21)
│
├── requirements.txt
└── README.md
```

---

## Installation

Windows (PowerShell or Command Prompt), from the repository root:

```bat
python -m venv .venv
.\.venv\Scripts\activate
pip install -r Task1_Movie_Rating_Prediction\requirements.txt
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r Task1_Movie_Rating_Prediction/requirements.txt
```

---

## Usage

### Run the notebooks

```bat
cd Task1_Movie_Rating_Prediction
jupyter notebook          ... or:  jupyter lab
```

Run them in order — each phase builds on the previous one:

| Notebook | Purpose |
|---|---|
| `notebooks/01_dataset_audit.ipynb` | Dataset audit and data-quality findings |
| `notebooks/02_data_preprocessing.ipynb` | Cleaning decisions and feature design |
| `notebooks/03_model_training.ipynb` | Group-aware split, baseline and first models |
| `notebooks/04_model_improvement.ipynb` | Cross-validation, tuning, feature analysis |
| `notebooks/05_final_evaluation.ipynb` | Final holdout evaluation (test set used once) |
| `notebooks/06_model_persistence.ipynb` | Saves the artifact and tests the prediction pipeline |

Every notebook re-reads the raw CSV and verifies its checksum; none of them modify it.

### Predict with the saved pipeline

Run from inside the `Task1_Movie_Rating_Prediction` folder:

```python
from src.predict import predict_rating

# Demonstration input only — invented metadata, not a real film
movie = {
    "Name": "Example Movie",          # identifier only; not used as a feature
    "Year": "2024",                   # "(2024)", "2024" or 2024 all work
    "Duration": "120 min",            # "120 min" or "120" both work
    "Genre": "Drama, Thriller",
    "Director": "Example Director",
    "Actor 1": "Actor One",
    "Actor 2": "Actor Two",
    "Actor 3": "Actor Three",         # optional
}

print(predict_rating(movie))
```

Batch prediction adds a `Predicted Rating` column to a DataFrame of films:

```python
import pandas as pd
from src.predict import predict_movies

results = predict_movies(pd.DataFrame([movie, movie]))
print(results[["Name", "Predicted Rating"]])
```

Other helpers: `load_pipeline()` returns the artifact (`{"pipeline": ..., "metadata": ...}`), and
`describe_artifact()` returns its metadata. `Year` is the only required field; everything else may be omitted.
`Rating` and `Votes` are ignored even if supplied.

Any predicted value shown in examples or notebooks is **model output for demonstration input** — not an actual or
claimed IMDb rating.

---

## Visualizations

All 21 charts live in [`visualizations/`](visualizations/). A selection:

**Cross-validation model comparison** — mean RMSE with fold-to-fold spread, showing how close the tree models are.

![Cross-validation RMSE with error bars](visualizations/13_cv_error_bars.png)

**Feature-group ablation** — how much each group of features contributes.

![Feature-group ablation](visualizations/14_feature_group_ablation.png)

**Final test: actual vs predicted** — the diagonal is a perfect prediction.

![Final actual vs predicted](visualizations/17_final_actual_vs_predicted.png)

**Final test: error by rating group** — mid-range films are predicted far more accurately than extremes.

![Error by rating group](visualizations/19_final_error_by_rating_group.png)

**Final test: prediction calibration** — the gap between the two lines is the compression toward the middle.

![Prediction calibration](visualizations/20_final_prediction_calibration.png)

Also available: missing-value and genre profiles (01–08), the first model comparison and residuals (09–11), CV and
importance charts (12, 15, 16), the final residual distribution (18) and the largest errors (21).

---

## Limitations

1. **R² is about 0.30.** The model explains roughly a third of the variation in ratings; most of it is not captured
   by this metadata.
2. **Predictions are compressed toward the middle** (prediction standard deviation 0.82 vs 1.39 for actual ratings).
3. **Extreme ratings have much higher error** — MAE 2.04 for films rated below 4 versus 0.64 for mid-range films.
4. **Metadata alone is limited.** Script, budget, marketing, release scale and reception are not in the dataset.
5. **`Votes` was excluded by design**, as a documented modelling assumption about what is known at prediction time.
6. **Scope:** Indian films up to roughly 2021–22. Applying the model outside that scope is extrapolation.
7. **Unseen directors and actors carry no history**, so predictions for them rely on year, duration and genre alone,
   and are less accurate.
8. **Not production-ready.** Typical error is about 0.9 rating points; this is a statistical estimate, not a reliable
   forecast for an individual film.
9. **The artifact is version-sensitive.** It is a joblib/pickle-based scikit-learn object tied to the pinned
   versions; `load_pipeline()` warns when the scikit-learn version differs. Loading it also requires `src/` to be
   importable, because it references the project's transformer classes — `load_pipeline()` handles that, so prefer
   it over calling `joblib.load` directly.

## Future Improvements

None of the following have been implemented; they are directions suggested by the limitations above.

- Run a clearly separated secondary experiment that **includes `Votes`**, to quantify what excluding it costs.
- Add richer metadata if it can be sourced (budget, language, release scale, production house).
- Explore text features from plot summaries or synopses.
- Try models and objectives that handle the rating extremes better, since that is where the error concentrates.
- Evaluate on a newer, out-of-time dataset to test whether performance holds for recent releases.
- Wrap the prediction pipeline in a small web interface or API for demonstration.

---

## Internship Context

Completed as part of the **CodSoft Data Science Internship**. This project corresponds to **CodSoft Task 2 — Movie
Rating Prediction with Python**; the folder name `Task1_Movie_Rating_Prediction` reflects my own numbering of the
three projects I selected.

Reproducibility notes: `random_state=42` throughout; every notebook runs top to bottom from a fresh kernel and was
verified to produce identical results across runs; the raw CSV's SHA-256 checksum is asserted in every notebook and
is unchanged since the initial audit.
