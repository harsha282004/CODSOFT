# Sales Prediction Using Python

Predicting sales from advertising spend across TV, Radio and Newspaper.

**CodSoft Data Science Internship — Sales Prediction Using Python.**
This folder is named `Task2_Sales_Prediction` as my own project numbering; it corresponds to
**CodSoft Task 4: Sales Prediction Using Python**.

**Status:** Phases 1–2 complete — dataset audit, exploratory data analysis, and a reusable preprocessing
pipeline. No model has been trained yet; model training begins in Phase 3.

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

## Project Structure

```
Task2_Sales_Prediction/
├── dataset/                      # raw data, never modified
│   └── advertising.csv
├── notebooks/
│   ├── 01_dataset_audit.ipynb        # Phase 1 — audit & EDA
│   └── 02_data_preprocessing.ipynb   # Phase 2 — preprocessing & validation
├── src/
│   └── data_preprocessing.py         # reusable pipeline, imported by every later phase
├── visualizations/
│   ├── 02_sales_distribution.png
│   ├── 03_feature_vs_sales.png
│   ├── 04_correlation_heatmap.png
│   └── 05_preprocessing_effect.png
├── models/                           # saved model artefacts (Phase 3 onwards)
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
```

Both notebooks resolve their paths relative to the repository, contain no absolute paths and run top to
bottom from a fresh kernel. To use the preprocessing directly:

```python
import sys; sys.path.insert(0, "src")
import data_preprocessing as dp

data = dp.prepare_modeling_data()   # validated, split 160/40, scaler fitted on X_train only
data.X_train_prepared, data.y_train, data.X_test_prepared, data.y_test
```
