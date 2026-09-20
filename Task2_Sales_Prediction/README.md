# Sales Prediction Using Python

Predicting sales from advertising spend across TV, Radio and Newspaper.

**CodSoft Data Science Internship — Sales Prediction Using Python.**
This folder is named `Task2_Sales_Prediction` as my own project numbering; it corresponds to
**CodSoft Task 4: Sales Prediction Using Python**.

**Status:** Phases 1–6 complete — dataset audit, exploratory data analysis, a reusable preprocessing
pipeline, an initial model comparison, cross-validated model selection with tuning, a final one-time
evaluation on the held-out test set with the fitted pipeline persisted, and an interactive dashboard.

**Final result:** test **RMSE 1.2173 · MAE 0.9354 · R² 0.9520** on 40 held-out rows — an estimate from a
small benchmark dataset, not a performance guarantee. See the caveats in
[Phase 5](#phase-5--final-evaluation--model-persistence).

**Try it:** `streamlit run app.py`

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

## Phase 4 — Model Validation, Cross-Validation & Tuning

Model selection done properly, on the training set only — see
[notebooks/04_model_validation_tuning.ipynb](notebooks/04_model_validation_tuning.ipynb). **The 40-row test
set was not touched anywhere in this phase**; its fingerprint is recorded at the start of the notebook and
verified unchanged at the end.

### Methodology

- **Cross-validation:** 5-fold `KFold`, `shuffle=True`, `random_state=42`, on the 160 training rows. Each
  fold trains on 128 rows and validates on 32.
- **Leakage control:** every model is a `Pipeline` whose first step is the Phase 2 preprocessor, so the
  `StandardScaler` is refitted **inside each fold**. Demonstrated empirically in section 15: a fold's scaler
  has `n_samples_seen_ = 128` and matches that fold's mean, not the full training mean.
- **Candidates:** Linear Regression, Ridge (α=1.0), Lasso (α=0.01), Random Forest, Gradient Boosting, plus a
  mean baseline.
- **Metrics:** MAE, RMSE, R², with per-fold detail and training scores for gap analysis.
- **Tuning:** exhaustive `GridSearchCV` on the two nonlinear candidates,
  `scoring="neg_root_mean_squared_error"`, `cv=5`, same folds. 216 Gradient Boosting configurations (1,080
  fits) and 72 Random Forest configurations (360 fits).

### Cross-validation results (160 training rows)

| Model | CV RMSE | CV MAE | CV R² | Train RMSE | Gap |
|---|---|---|---|---|---|
| Gradient Boosting (tuned) | **1.2808** ± 0.1936 | 0.9393 | 0.9341 | 0.4755 | 0.805 |
| Random Forest (tuned) | 1.3043 ± 0.2240 | 0.9500 | 0.9324 | 0.4942 | 0.810 |
| Random Forest (Phase 3 config) | 1.3044 ± 0.2354 | 0.9481 | 0.9324 | 0.4940 | 0.810 |
| Gradient Boosting (Phase 3 config) | 1.3603 ± 0.1695 | 1.0396 | 0.9248 | 0.3586 | 1.002 |
| Lasso (α=0.01) | 1.6789 ± 0.2787 | 1.2777 | 0.8797 | 1.6267 | 0.052 |
| Ridge (α=1.0) | 1.6805 ± 0.2810 | 1.2791 | 0.8796 | 1.6271 | 0.053 |
| Linear Regression | 1.6808 ± 0.2825 | 1.2782 | 0.8793 | 1.6266 | 0.054 |
| Baseline (train mean) | 5.2061 ± 0.4784 | 4.3316 | −0.0712 | 5.1697 | 0.036 |

**Cross-validation reverses the Phase 3 ordering.** On the Phase 3 hold-out, Gradient Boosting beat Random
Forest by 0.0347 RMSE; across the five training folds, Random Forest beats Gradient Boosting by 0.0559 and
wins 3 of 5 folds. Neither ordering is trustworthy: the two models are ~0.05 apart while each varies by
0.17–0.24 between folds. The Phase 3 ranking was an artefact of which 40 rows landed in the test set.

### Tuned hyperparameters

| Gradient Boosting | | Random Forest | |
|---|---|---|---|
| `learning_rate` | 0.02 | `n_estimators` | 200 |
| `max_depth` | 4 | `max_depth` | 10 |
| `min_samples_leaf` | 2 | `min_samples_leaf` | 1 |
| `n_estimators` | 200 | `max_features` | 1.0 |
| `subsample` | 0.8 | `max_samples` | None |
| **CV RMSE** | **1.2808** (from 1.3603, −5.8%) | **CV RMSE** | **1.3043** (from 1.3044, −0.0%) |

**Tuning the Random Forest achieved nothing** — the search reproduced the default configuration. That is a
legitimate result, reported as such rather than dressed up as a 0.0001 improvement. For the forest, the only
parameter that mattered was `max_features`: restricting splits to 1 of 3 features (`"sqrt"`) averaged 2.106
RMSE against 1.419 for using all 3.

### Overfitting analysis

| Model | Train RMSE | CV RMSE | Gap | Gap as % of CV RMSE |
|---|---|---|---|---|
| Gradient Boosting (Phase 3) | 0.3586 | 1.3603 | 1.0017 | 74% |
| Random Forest | 0.4940 | 1.3044 | 0.8104 | 62% |
| Linear / Ridge / Lasso | ~1.627 | ~1.680 | ~0.053 | 3% |

Both ensembles fit training noise substantially: 62–74% of their apparent accuracy does not survive unseen
data, with train R² above 0.99. The linear models' 3% gaps reflect having almost no capacity to memorise.

A large gap is not on its own grounds for rejection — the ensembles still beat the linear models by ~0.38
RMSE on held-out folds. It indicates unused headroom, and constraining Gradient Boosting (slower learning
rate, `subsample=0.8`) did reduce its gap from 1.002 to 0.805.

### Stability analysis

Repeating the cross-validation over five different partitions (`RepeatedKFold`, 25 held-out evaluations):

| Model | Repeated CV RMSE | std | min fold | max fold |
|---|---|---|---|---|
| Random Forest (tuned) | 1.3007 | 0.2443 | 0.8443 | 1.9550 |
| Random Forest (Phase 3) | 1.3062 | 0.2457 | 0.8317 | 1.9588 |
| Gradient Boosting (tuned) | 1.3067 | 0.2469 | 0.9457 | 1.9576 |
| Gradient Boosting (Phase 3) | 1.3301 | 0.1886 | 1.0098 | 1.7657 |
| Linear Regression | 1.6830 | 0.2474 | 1.2862 | 2.2569 |

- **All four ensemble variants are indistinguishable** — a 0.029 spread against standard deviations of
  0.19–0.25, with individual folds ranging from 0.83 to 1.96.
- **Most of the tuning gain did not survive a change of partition.** Gradient Boosting's advantage was
  0.0795 on the original folds but 0.0234 across 25 — roughly 70% of it was the search fitting that specific
  fold partition, the selection bias inherent in picking the best of 216 configurations scored on the same
  folds.
- **The ensemble-versus-linear gap is the one stable finding** (~1.30 vs ~1.68 in every partition tested).

### Outlier sensitivity

Of the two `Newspaper` outliers Phase 1 flagged, **row 101 is in the training set and row 16 is in the test
set**. Since the test set is off-limits in this phase, only row 101 could be removed — a 160-vs-159 row
comparison.

| Model | CV RMSE (160) | CV RMSE (159) | Change |
|---|---|---|---|
| Gradient Boosting (tuned) | 1.2808 | 1.3200 | +0.0392 (worse) |
| Random Forest (tuned) | 1.3043 | 1.2504 | −0.0538 (better) |
| Random Forest (Phase 3) | 1.3044 | 1.2479 | −0.0564 (better) |
| Gradient Boosting (Phase 3) | 1.3603 | 1.3320 | −0.0283 (better) |
| Linear Regression | 1.6808 | 1.6662 | −0.0146 (better) |

**Decision: keep all observations.** The direction is inconsistent — removal helps four models and hurts the
leading one — and every change is around an order of magnitude smaller than the fold-to-fold noise. Phase 2's
reasoning stands: the value is a plausible advertising budget, not a data error.

### Recommendation for Phase 5

**Tuned Gradient Boosting** as the primary model, with **tuned Random Forest** as an equally defensible
alternative. The basis is a tie-break on the pre-specified selection metric (5-fold CV RMSE), not a
demonstrated advantage — under repeated cross-validation the tuned Random Forest is nominally ahead instead.
These two models cannot be separated by this dataset.

### Deferred to Phase 5

Final test-set evaluation (scored **once**, without re-selecting afterwards), model persistence, the
prediction script and the dashboard. **No test-set result, no saved model and no dashboard exist yet.**

---

## Phase 5 — Final Evaluation & Model Persistence

The one-time evaluation of the locked model on the 40-row test set reserved since Phase 2 — see
[notebooks/05_final_evaluation_persistence.ipynb](notebooks/05_final_evaluation_persistence.ipynb).

**Phase 4 vs Phase 5, kept strictly separate:**

| | |
|---|---|
| **Phase 4** | Model **selection**, using 5-fold cross-validation on the 160 **training** rows only. The test set was never touched. |
| **Phase 5** | One-time final **evaluation** on the previously untouched 40-row hold-out. No selection, no tuning, no preprocessing changes. |

The test set's fingerprint was recorded in Phase 4 and re-verified here before anything was fitted — the 40
rows evaluated are byte-for-byte those set aside.

### Final model and selection rationale

Selected in **Phase 4** by 5-fold CV RMSE on the training set, not by anything measured in Phase 5:

```python
GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.02,
    max_depth=4,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42,
)
```

wrapped in the Phase 2 preprocessing pipeline (`ColumnTransformer` + `StandardScaler`, fitted on the 160
training rows only — verified: the scaler's `n_samples_seen_` is 160 and its mean matches the training mean,
not the full-dataset mean).

Phase 4 recommended this configuration as primary while stating plainly that it and the tuned Random Forest
**cannot be separated by this dataset**.

### Cross-validation estimate vs final test result

| Quantity | Measured on | RMSE | MAE | R² |
|---|---|---|---|---|
| **Phase 4 CV** (5-fold) | 160 training rows | 1.2808 ± 0.1936 | 0.9393 | 0.9341 |
| **Phase 5 TEST** (once) | 40 held-out rows | **1.2173** | **0.9354** | **0.9520** |

The test figure is 0.064 better than the CV estimate — well inside the CV standard deviation of 0.19, so the
two are consistent. MSE was 1.4819.

### Comparison on the same 40-row hold-out

| Model | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| Gradient Boosting (Phase 3, untuned) | 0.8679 | 1.1451 | 0.9576 |
| Random Forest (Phase 3) | 0.9025 | 1.1798 | 0.9550 |
| **Gradient Boosting (tuned) — FINAL** | **0.9354** | **1.2173** | **0.9520** |
| Lasso (α=0.01) | 1.2725 | 1.7050 | 0.9059 |
| Linear Regression | 1.2748 | 1.7052 | 0.9059 |
| Ridge (α=1.0) | 1.2734 | 1.7074 | 0.9057 |
| Baseline (training mean) | 4.9315 | 5.6482 | −0.0324 |

Against the naive baseline the final model cuts test RMSE by **78.4%** and MAE by **81.0%**, and it beats
every linear model by about 29%.

**An honest note.** The tuned final model scored *worse* on this hold-out (1.2173) than the untuned Phase 3
configuration (1.1451). It was still reported as final, because it was selected in advance by
cross-validation and switching on the strength of these 40 rows would be selecting on the test set — the
error this project's structure exists to avoid. Phase 4 showed the two configurations sit within 0.023 RMSE
of each other across 25 repeated CV folds, against fold standard deviations of ~0.19–0.25, so neither result
establishes one as better. Following the pre-registered rule costs 0.07 RMSE units of reported performance
and buys a number that means what it says.

### Residual and error findings

- **Bias −0.2422** — a mild tendency to over-predict, small against the target's standard deviation of 5.63.
- **Median absolute error 0.7970**; largest absolute error **4.2553** (row 150); largest under-prediction
  +2.9502 (row 66).
- **Prediction-error coverage** (not classification accuracy — this is regression): **27.5%** of test
  predictions within ±0.5 sales units, **62.5%** within ±1.0, **90.0%** within ±1.5, **92.5%** within ±2.0.
- 3 of 40 observations miss by more than 2.0; 1 misses by more than 3.0.
- Residuals are mildly left-skewed (−0.3114) with **no obvious funnel or curvature**. The prediction range
  (3.63–23.26) is slightly narrower than the actual range (5.3–24.7) — the mild regression toward the mean
  characteristic of tree ensembles, which cannot extrapolate beyond the target values seen in training.
- The retained `Newspaper` outlier in the test set (row 16) had an absolute error of 0.8993, ranking 18th of
  40 — middle of the distribution. Keeping it cost nothing measurable.
- No causal claim is made: the model captures association in observational data.

### Artifact

| | |
|---|---|
| **Path** | `models/final_sales_prediction_pipeline.joblib` |
| **Size** | 452,631 bytes (442.0 KiB) |
| **Contents** | `{"pipeline": <fitted Pipeline>, "metadata": {...}}` — the **complete** pipeline, preprocessing included |
| **Metadata** | model type, hyperparameters, feature names, target, training rows, CV and test metrics, training feature ranges, dataset checksum, and scikit-learn / joblib / Python / pandas / numpy versions |
| **Environment** | Python 3.13.0, scikit-learn 1.9.1, joblib 1.6.0 |

Saving the whole pipeline matters: the estimator alone would silently mis-predict, since it expects
standardised inputs.

Validated by reloading **in a fresh Python process** with no access to the notebook's state — predictions
were bit-identical to those taken before saving (max difference 0.000e+00).

### Prediction API

[`src/predict.py`](src/predict.py) — no retraining, no dataset needed, artefact never modified:

```python
import sys; sys.path.insert(0, "src")
from predict import predict_sales

predict_sales(tv=150.0, radio=25.0, newspaper=30.0)   # -> 14.0096
```

```bash
python src/predict.py --tv 150 --radio 25 --newspaper 30
python src/predict.py --smoke-test
```

Also provides `predict_batch()`, `model_metadata()` and `training_ranges()`. Inputs are validated: missing
values (`None`/NaN), non-numeric strings, booleans, infinities and negative budgets are all rejected with
specific errors — **10 of 10** invalid-input cases rejected in testing. Floats, integers and numeric strings
give identical predictions. The CLI exits non-zero with a clear message on bad input, and flags inputs
outside the training ranges (TV 0.7–296.4, Radio 0.0–49.6, Newspaper 0.3–100.9).

### Reproducibility

**15 of 15** validation checks pass. The notebook runs from a fresh kernel with zero errors, zero cell
stderr and zero unexecuted cells; refitting reproduces every final metric bit-identically; the split matches
Phases 2–4; and the raw dataset checksum is unchanged at
`137f755ad6fd3bc6471085f7631a2fba6c04cb8acd71eaf5cc9af6839d43fdd5`.

### Limitations

These are estimates from a **small benchmark dataset**, and should be read that way:

- The full dataset is **200 observations**; the test set is **40**. Confidence intervals around every figure
  above are wide, and a different 40-row split would give noticeably different numbers — as this project
  demonstrated twice over: the hold-out ranked Gradient Boosting above Random Forest and the untuned
  configuration above the tuned one, and cross-validation reversed both.
- **The model is not perfect.** It missed one test observation by 4.26 sales units.
- **These figures do not transfer to future advertising campaigns.** The data has no time dimension, no
  market or audience information, and no stated units. Nothing here supports extrapolating to campaigns,
  products or markets unlike those in the dataset.
- **No causation.** The model describes association between advertising budgets and sales; it cannot say
  what would happen to sales if a budget were changed.

### Deferred

The dashboard/application phase, which will consume the artefact through `src/predict.py`. It must not
refit, re-tune or re-evaluate on the test set — the hold-out has now been used, and further measurement
against it would no longer be unbiased.

---

## Phase 6 — Streamlit Dashboard

An interactive dashboard over the finished pipeline — [`app.py`](app.py).

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run it from the `Task2_Sales_Prediction` directory. All paths resolve relative to the file, so no
machine-specific configuration is needed.

**The dashboard is a presentation layer.** It loads the pipeline persisted in Phase 5 and calls the existing
`src/predict.py` API. It **never retrains, tunes or modifies any model**, and it never writes to the dataset
or the artefact — both were checksum-verified as unchanged after the dashboard was built.

### Pages

| Page | What it shows |
|---|---|
| **Overview** | Summary cards (200 observations, 3 features, target `Sales`, 160 train / 40 test, Gradient Boosting), what the project does, the ML pipeline from Dataset → Audit → Preprocessing → Model Training → Cross-Validation → Final Evaluation → Prediction, and the headline test result |
| **Predict Sales** | The interactive page — three budget inputs, out-of-range warnings, a prominent prediction, and a chart of the entered budgets |
| **Data Analysis** | Live analysis of `dataset/advertising.csv`: summary statistics, Sales distribution, per-platform spend distributions, each predictor against Sales, a correlation matrix, and the Phase 1 data-quality audit |
| **Model Performance** | Final test metrics and Phase 4 CV metrics shown **separately**, the test-RMSE comparison across all models, prediction-error coverage, and residual behaviour |
| **Model Details** | Hyperparameters, preprocessing, training ranges, artefact information and recorded environment — read from the artefact's own metadata rather than hardcoded — plus the limitations section |
| **About Project** | Task, problem, technology, ML methods, the full phase pipeline and dataset provenance |

### Prediction workflow

1. Enter a budget for **TV**, **Radio** and **Newspaper**. Defaults are the median training spend, and each
   input shows the range observed during training.
2. Inputs are floored at 0 — negative budgets are rejected, consistent with the prediction API.
3. Select **Predict Sales**. The app calls `predict_sales()` from `src/predict.py`; no inference code is
   duplicated in `app.py`.
4. If any budget falls outside the training range, a warning states that the prediction is **extrapolation
   beyond the training data**. The prediction is still produced — it is flagged, not blocked.
5. The result is displayed prominently alongside the entered budgets and a bar chart of them. That chart
   shows the inputs only; it is explicitly **not** a feature-importance chart.

### Model loading

The pipeline is loaded once per session with `@st.cache_resource`, through `predict_api.load_artifact()` —
so the dashboard and the CLI share one loading path and no page duplicates it. Dataset reads and derived
statistics are cached with `@st.cache_data`. Nothing is trained at startup.

### Honest reporting carried into the UI

The dashboard surfaces the project's methodological caveats rather than hiding them:

- The small-dataset warning appears on the Overview, Model Performance and About pages.
- CV metrics and test metrics are shown in **separate** sections and never combined into one figure.
- Error bands are labelled **prediction-error coverage**, explicitly not classification accuracy.
- The Model Performance page states that the tuned final model scored worse on this hold-out than the
  untuned Phase 3 model, and explains why it was retained anyway.
- Correlations are described as correlations, with causation explicitly ruled out.

### Dependencies added

```
streamlit==1.64.0
plotly==7.1.0
```

Both were already installed at these versions; no existing pin was changed, and `joblib==1.6.0` remains.
A minimal theme lives in [`.streamlit/config.toml`](.streamlit/config.toml).

### Testing

68 automated checks via `streamlit.testing.v1.AppTest`, all passing: every page renders without exception,
predictions match the API exactly, multiple valid predictions work, negative and invalid inputs are
rejected, out-of-range inputs warn (and in-range inputs do not), and each page shows its documented figures.
The server was also started for real and served HTTP 200, and the CLI prediction API was confirmed still
working.

---

## Project Structure

```
Task2_Sales_Prediction/
├── app.py                        # Phase 6 — Streamlit dashboard
├── .streamlit/
│   └── config.toml               # dashboard theme
├── dataset/                      # raw data, never modified
│   └── advertising.csv
├── notebooks/
│   ├── 01_dataset_audit.ipynb        # Phase 1 — audit & EDA
│   ├── 02_data_preprocessing.ipynb   # Phase 2 — preprocessing & validation
│   ├── 03_model_training.ipynb       # Phase 3 — baseline & initial models
│   ├── 04_model_validation_tuning.ipynb  # Phase 4 — CV, tuning & stability
│   └── 05_final_evaluation_persistence.ipynb  # Phase 5 — final test & persistence
├── src/
│   ├── data_preprocessing.py         # reusable pipeline, imported by every later phase
│   └── predict.py                    # prediction API for the persisted model (used by the dashboard)
├── visualizations/
│   ├── 02_sales_distribution.png
│   ├── 03_feature_vs_sales.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_preprocessing_effect.png
│   ├── 06_model_comparison.png
│   ├── 07_model_mae_comparison.png
│   ├── 08_actual_vs_predicted.png
│   ├── 09_residual_distribution.png
│   ├── 10_residuals_vs_predicted.png
│   ├── 11_cv_model_comparison.png
│   ├── 12_cv_rmse_distribution.png
│   ├── 13_hyperparameter_comparison.png
│   ├── 14_train_vs_cv_performance.png
│   ├── 15_outlier_sensitivity.png
│   ├── 16_final_actual_vs_predicted.png
│   ├── 17_final_residual_distribution.png
│   ├── 18_final_residuals_vs_predicted.png
│   └── 19_final_prediction_errors.png
├── models/
│   └── final_sales_prediction_pipeline.joblib   # fitted pipeline + metadata
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
jupyter notebook notebooks/04_model_validation_tuning.ipynb  # Phase 4 (~7 min: 1,440 model fits)
jupyter notebook notebooks/05_final_evaluation_persistence.ipynb  # Phase 5
streamlit run app.py                                     # Phase 6 dashboard
```

All five notebooks resolve their paths relative to the repository, contain no absolute paths and run top to
bottom from a fresh kernel. To use the preprocessing directly:

```python
import sys; sys.path.insert(0, "src")
import data_preprocessing as dp

data = dp.prepare_modeling_data()   # validated, split 160/40, scaler fitted on X_train only
data.X_train_prepared, data.y_train, data.X_test_prepared, data.y_test
```
