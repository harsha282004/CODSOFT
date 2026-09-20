# Sales Prediction Using Python

Predicting sales from advertising spend across TV, Radio and Newspaper.

**CodSoft Data Science Internship — Sales Prediction Using Python.**
This folder is named `Task2_Sales_Prediction` as my own project numbering; it corresponds to
**CodSoft Task 4: Sales Prediction Using Python**.

**Status:** Phases 1–3 complete — dataset audit, exploratory data analysis, a reusable preprocessing
pipeline, and an initial model comparison. The Phase 3 results are a first comparison, not final
performance: no hyperparameter has been tuned and no model has been saved. Phase 4 handles improvement and
final selection.

---

## Objective

Sales prediction involves forecasting the amount of a product that customers may purchase, taking into
account factors such as advertising expenditure, target audience segmentation and advertising platform
selection.

For this dataset specifically, that means estimating `Sales` from the advertising budget allocated to three
platforms. The dataset contains no audience or demographic fields, so the audience-segmentation aspect of
the task statement cannot be studied here — this is stated plainly rather than implied otherwise.

---

## Dataset

The dataset is the one the CodSoft source document links to, traced end to end rather than substituted:

| | |
|---|---|
| **Source document** | `DATA SCIENCE.pdf`, page 9 — "TASK 4 · SALES PREDICTION USING PYTHON · DATASET CLICK HERE" |
| **Link behind "CLICK HERE"** | `https://www.kaggle.com/code/ashydv/sales-prediction-simple-linear-regression/input` |
| **Dataset behind that notebook** | Kaggle dataset `ashydv/advertising-dataset` (from the notebook's `datasetDataSources` metadata) |
| **Dataset URL** | https://www.kaggle.com/datasets/ashydv/advertising-dataset |
| **Filename** | `dataset/advertising.csv` |
| **Downloaded** | 2026-09-20 |
| **Size** | 4,062 bytes |
| **SHA-256** | `137f755ad6fd3bc6471085f7631a2fba6c04cb8acd71eaf5cc9af6839d43fdd5` |
| **Shape** | 200 rows × 4 columns |
| **Origin** | the *Advertising* dataset from *An Introduction to Statistical Learning* (ISLR) |
| **Licence** | as published on Kaggle by the dataset author |

**Verification.** The downloaded file was checked against the saved outputs of the linked Kaggle notebook
before use — identical shape `(200, 4)`, identical first five rows, identical `describe()` table.

### Columns

| Column | Type | Role | Description |
|---|---|---|---|
| `TV` | float64 | predictor | advertising budget spent on the TV platform |
| `Radio` | float64 | predictor | advertising budget spent on the Radio platform |
| `Newspaper` | float64 | predictor | advertising budget spent on the Newspaper platform |
| `Sales` | float64 | **target** | sales recorded for that observation |

**Target: `Sales`** — the only column measuring an amount purchased; the other three are the advertising
expenditure inputs the task statement names.

The raw file is treated as read-only. The notebook re-computes its SHA-256 at the end of every run and
asserts it against the value recorded at download time.

---

## Phase 1 Status

Dataset audit and exploratory data analysis completed — see
[notebooks/01_dataset_audit.ipynb](notebooks/01_dataset_audit.ipynb).

Key findings:

- **Complete data** — 0 missing values out of 800 cells.
- **No duplicates** — 0 exact duplicate rows, 0 duplicated feature combinations.
- **All numeric** — four `float64` columns, no categorical variables, no ID column, no date column. The data
  is cross-sectional, not a time series.
- **Target** — `Sales` ranges 1.6–27.0 (mean 15.13, median 16.00, sd 5.28), near-symmetric (skew −0.07) and
  flatter than normal (kurtosis −0.64). No transformation indicated.
- **Associations with `Sales`** — `TV` r = 0.901, `Radio` r = 0.350, `Newspaper` r = 0.158. These are
  associations observed in the data, not evidence of causation.
- **No meaningful multicollinearity** — the strongest predictor pair is `Radio`–`Newspaper` at r = 0.354.
- **Outliers** — 2 rows (1.0%), both unusually high `Newspaper` budgets, flagged by both the IQR and the
  3-standard-deviation criteria. Plausible values, kept.
- **Leakage** — no feature is derived from `Sales`, no future information, no ID used as a predictor and no
  duplicate rows that could cross a train/test boundary. The remaining risks are procedural and belong to
  Phase 2: fit any preprocessing inside a pipeline on training folds only, and touch the test set once.

No model has been trained, no value imputed and no row removed.

---

## Phase 2 — Data Preprocessing

Preprocessing and feature engineering completed — see
[notebooks/02_data_preprocessing.ipynb](notebooks/02_data_preprocessing.ipynb), which imports the
implementation from [src/data_preprocessing.py](src/data_preprocessing.py) rather than repeating it. The
same module will be used by training, validation, final evaluation, persistence and prediction, so those
phases cannot drift apart from one another.

### Data flow

```
advertising.csv
      ↓
Raw data validation        (schema, dtypes, missing, finite, row count, checksum)
      ↓
Separate X / y             (X = TV, Radio, Newspaper   ·   y = Sales)
      ↓
Train/test split           (80 / 20, random_state=42 — BEFORE anything is learned)
      ↓
Fit preprocessor on X_train only
      ↓
Transform X_train  →  Transform X_test   (same fitted object, no refit)
      ↓
Model-ready datasets  →  Phase 3 model training
```

### Decisions

| | |
|---|---|
| **Predictors** | `TV`, `Radio`, `Newspaper` — 3 numeric features, fixed order |
| **Target** | `Sales`, excluded from `X` by construction |
| **Split** | 80 / 20 → **160 train / 40 test**, `random_state = 42` |
| **Split ordering** | performed before any learned preprocessing, so no test statistic can reach the transformer |
| **Outliers** | **retained** — all 200 observations kept; the two unusual `Newspaper` budgets are plausible spends, not data errors, and statistical unusualness alone is not evidence of invalidity. Their influence on a fitted model is a Phase 3 diagnostic, not a preprocessing decision. |
| **Missing values** | **no imputation** — there are none, and a silent imputer would later mask a genuinely absent input at prediction time. Missing or non-finite input is validated and raised on instead. |
| **Encoding** | none — no categorical, text, ID or date column exists |
| **Feature engineering** | none — no derived feature, and nothing derived from the target |
| **Scaling** | `StandardScaler` inside a `ColumnTransformer` keyed on column names, **fitted on `X_train` only** |
| **Transformed features** | 3, original names preserved, no missing and no infinite values |

**Why scaling.** Ridge and Lasso penalise coefficients and so depend on predictor scale, and `TV` spans
0.7–296.4 while `Radio` spans 0.0–49.6. Ordinary least squares is unaffected by a linear rescaling and
tree-based models are invariant to it, so one standardised pipeline shared by every candidate is simpler and
safer than maintaining two competing preprocessing paths, at no cost to the models that do not need it.

### Leakage prevention

Four independent checks, all passing in the notebook:

1. The scaler's learned `mean_` is `[150.02, 22.88, 29.95]` — the **training** mean. The full-dataset mean is
   `[147.04, 23.26, 30.55]`. They differ, so the fit provably excluded the test rows.
2. Transforming the test set leaves `mean_`, `scale_` and `n_samples_seen_` unchanged; `n_samples_seen_`
   stays at 160. `transform_features()` never calls `fit`.
3. The transformed training set is centred on exactly 0; the transformed **test** set is not (means ≈ −0.18,
   0.13, 0.15) — exactly what an unseen sample should look like.
4. No predictor is a function of the target, no row is shared between the splits, and a frame containing
   `Sales` is rejected rather than silently transformed.

For cross-validation in Phase 3, the preprocessor must be composed into a `Pipeline` with the estimator so
that it is refitted **inside** each fold.

### Reproducibility

Repeated calls with `random_state = 42` produce byte-identical `X_train`, `X_test`, `y_train`, `y_test` and
identical transformed matrices, so every later phase evaluates on the same hold-out set. A different seed
produces a different split, confirming the check is meaningful. Nine validation cases (missing column,
unexpected column, absent target, non-numeric column, missing value, infinite value, wrong row count, target
inside `X`, absent file) are each rejected with a specific error.

The raw dataset checksum is re-verified at the end of the notebook and is unchanged from Phase 1.

---

## Phase 3 — Initial Model Training

A first comparison of model families on the fixed Phase 2 hold-out — see
[notebooks/03_model_training.ipynb](notebooks/03_model_training.ipynb). Every configuration was fixed in
advance and scored once. **No tuning, no search, and no model artefact saved.**

### Setup

- **Split:** the Phase 2 hold-out reused unchanged — 160 train / 40 test, `random_state = 42`. The notebook
  rebuilds it from `src/data_preprocessing.py` and asserts it matches Phase 2 before training anything.
- **Preprocessing:** the Phase 2 `ColumnTransformer` composed with each estimator in a `sklearn.Pipeline`, so
  the scaler is fitted inside `fit` on training rows only and merely applied at predict time.
- **Baseline:** predict the mean of `y_train` (15.3306) for every test observation. This is a reference
  point, not a machine learning model. The mean is taken from the training rows only — using the full
  dataset's mean would leak the test set into the reference.
- **Metrics:** MAE, MSE, RMSE and R². This is regression, so accuracy is not a meaningful metric and is not
  reported.

### Results on the fixed 40-row hold-out

| Model | Test MAE | Test RMSE | Test R² | Train RMSE | Gap |
|---|---|---|---|---|---|
| Gradient Boosting (200 stages, lr 0.05, depth 3) | 0.8679 | **1.1451** | 0.9576 | 0.4438 | 0.701 |
| Random Forest (300 trees) | 0.9025 | **1.1798** | 0.9550 | 0.4716 | 0.708 |
| Lasso (α = 0.01) | 1.2725 | 1.7050 | 0.9059 | 1.6360 | 0.069 |
| Linear Regression | 1.2748 | 1.7052 | 0.9059 | 1.6359 | 0.069 |
| Ridge (α = 1.0) | 1.2734 | 1.7074 | 0.9057 | 1.6362 | 0.071 |
| Baseline (train mean) | 4.9315 | 5.6482 | −0.0324 | 5.1768 | 0.471 |

Ordered by **lowest test RMSE on the fixed Phase 3 hold-out**. This describes this particular 40-row sample
with these fixed configurations; it is not a general ranking of the algorithms.

### Observations

- **Every model beats the naive baseline by a wide margin** — roughly 70–80% lower test RMSE. The advertising
  budgets carry real signal about sales.
- **The three linear models are indistinguishable from one another**, spanning 1.7050–1.7074 RMSE. With three
  weakly correlated predictors and 160 training rows there is nothing for a regularisation penalty to fix, so
  this is the expected outcome. Lasso at α = 0.01 drove no coefficient to zero.
- **Both tree ensembles form a clearly better cluster** at ~1.15–1.18, about 31% below the linear group,
  which suggests the relationship is not purely additive.
- **Generalisation gaps split by family:** ~0.07 for the linear models (little capacity to memorise) against
  ~0.70 for both ensembles, whose train R² exceeds 0.99. The gap indicates unused headroom to constrain in
  Phase 4 — it does not disqualify them, since they predict *unseen* data considerably better.
- **Error profile of the leading model:** median absolute error 0.78; 65% of test predictions within 1 sales
  unit, 95% within 2; largest miss 4.01. These are descriptive statistics of one 40-row sample, not
  guarantees about future predictions.
- **Retained outliers:** of the two `Newspaper` rows Phase 1 flagged, one fell in each split. The test-set one
  had the 4th largest error of 40 (1.30) — above the median but far from extreme, and neither of the two
  worst errors was a flagged row. Phase 2's decision to retain them stands.
- **Linear diagnostics:** residuals roughly centred with no strong curvature or funnel, but left-skewed with
  a −7.19 tail. The correlation between predicted value and absolute residual is 0.103, too weak to either
  confirm or refute the possible non-constant variance Phase 1 noted.
- **Reproducibility:** the split, the baseline and every reported metric reproduce exactly across fresh
  kernels. Four of five models refit bit-identically; Random Forest agrees to ~1e-14 because `n_jobs=-1`
  averages its 300 trees in a nondeterministic summation order. With `n_jobs=1` it is bit-identical too.

### Initial candidate for Phase 4

**Gradient Boosting Regressor** — lowest test RMSE (1.1451), MAE (0.8679) and highest R² (0.9576).

This is an initial candidate, not a final selection. It leads Random Forest by 0.0347 RMSE units — about 3%
— on a 40-row sample, which is inside the noise; a different split could reverse the order. Both should be
carried into Phase 4 as live candidates, where cross-validation on the **training** set will do the
comparison properly. The hold-out has now been used for its one permitted purpose and should not drive
further decisions.

---

## Project Structure

```
Task2_Sales_Prediction/
├── dataset/                      # raw data, never modified
│   └── advertising.csv
├── notebooks/
│   ├── 01_dataset_audit.ipynb        # Phase 1 — audit & EDA
│   ├── 02_data_preprocessing.ipynb   # Phase 2 — preprocessing & validation
│   └── 03_model_training.ipynb       # Phase 3 — baseline & initial models
├── src/
│   └── data_preprocessing.py         # reusable pipeline, imported by every later phase
├── visualizations/
│   ├── 02_sales_distribution.png
│   ├── 03_feature_vs_sales.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_preprocessing_effect.png
│   ├── 06_model_comparison.png
│   ├── 07_model_mae_comparison.png
│   ├── 08_actual_vs_predicted.png
│   ├── 09_residual_distribution.png
│   └── 10_residuals_vs_predicted.png
├── models/                           # saved model artefacts (Phase 4 onwards — empty)
├── README.md
└── requirements.txt
```

Two chart slots from the planned sequence were deliberately left unused rather than filled with empty
figures: `01_missing_values.png` (there is no missingness to plot) and `05_sales_by_category.png` (there are
no categorical variables). Both findings are reported in the notebook instead.

---

## Reproducing

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_dataset_audit.ipynb        # Phase 1
jupyter notebook notebooks/02_data_preprocessing.ipynb   # Phase 2
jupyter notebook notebooks/03_model_training.ipynb       # Phase 3
```

All three notebooks resolve their paths relative to the repository, contain no absolute paths and run top to
bottom from a fresh kernel. To use the preprocessing directly:

```python
import sys; sys.path.insert(0, "src")
import data_preprocessing as dp

data = dp.prepare_modeling_data()   # validated, split 160/40, scaler fitted on X_train only
data.X_train_prepared, data.y_train, data.X_test_prepared, data.y_test
```
