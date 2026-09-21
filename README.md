# CODSOFT — Data Science Internship

**Three end-to-end machine-learning projects completed for the CodSoft Data Science Virtual Internship:
movie rating prediction, sales prediction and credit card fraud detection.**

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-3.0.6-150458?logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.64.0-FF4B4B?logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter&logoColor=white)

Each project goes from a raw dataset to a locked, persisted model. The work runs in phases: dataset audit,
preprocessing, baseline models, cross-validated selection and tuning, one final evaluation on held-out data,
model persistence, a prediction API and an interactive Streamlit dashboard. Every metric in this README is
taken from the project's own documentation and result files.

> **Folder numbers are not CodSoft task numbers.** `Task1_…`, `Task2_…` and `Task3_…` are local numbering.
> They correspond to **CodSoft Tasks 2, 4 and 5**. See [CodSoft Task Mapping](#4-codsoft-task-mapping).

---

## Table of Contents

| | | |
|---|---|---|
| [1. Internship Overview](#1-internship-overview) | [9. Project 3 — Credit Card Fraud Detection](#9-project-3--credit-card-fraud-detection) | [17. Installation and Environment](#17-installation-and-environment) |
| [2. Repository Overview](#2-repository-overview) | [10. Cross-Project Technical Summary](#10-cross-project-technical-summary) | [18. Running the Projects](#18-running-the-projects) |
| [3. Project Status](#3-project-status) | [11. Machine Learning Techniques Used](#11-machine-learning-techniques-used) | [19. Git LFS and Dataset Handling](#19-git-lfs-and-dataset-handling) |
| [4. CodSoft Task Mapping](#4-codsoft-task-mapping) | [12. Data Preprocessing and Feature Engineering](#12-data-preprocessing-and-feature-engineering) | [20. Project Documentation](#20-project-documentation) |
| [5. Projects at a Glance](#5-projects-at-a-glance) | [13. Model Development and Evaluation](#13-model-development-and-evaluation) | [21. Key Learning Outcomes](#21-key-learning-outcomes) |
| [6. Technology Stack](#6-technology-stack) | [14. Validation and Reproducibility](#14-validation-and-reproducibility) | [22. Limitations and Practical Considerations](#22-limitations-and-practical-considerations) |
| [7. Project 1 — Movie Rating Prediction](#7-project-1--movie-rating-prediction) | [15. Interactive Dashboards](#15-interactive-dashboards) | [23. Future Improvements](#23-future-improvements) |
| [8. Project 2 — Sales Prediction](#8-project-2--sales-prediction) | [16. Repository Structure](#16-repository-structure) | [24. Internship Completion Summary](#24-internship-completion-summary) |

---

## 1. Internship Overview

This repository contains the Data Science projects completed for the **CodSoft Data Science Virtual
Internship**. Three tasks were selected from the CodSoft Data Science task list, and each one was built as a
self-contained project with its own data, notebooks, source code, saved model, dashboard and documentation.

Between them, the three tasks cover:

- **Regression on messy, high-cardinality metadata**: predicting IMDb ratings of Indian films.
- **Regression on a small, clean benchmark dataset**: predicting sales from advertising spend.
- **Binary classification under severe class imbalance**: detecting fraudulent card transactions.

The CodSoft task statements asked for specific things: regression for the movie and sales tasks, and for the
fraud task preprocessing, normalisation, imbalance handling, a train/test split, logistic regression or random
forests, and precision/recall/F1 evaluation. Each project README records how it addresses its task statement.

---

## 2. Repository Overview

- **All three selected projects are complete.**
- **Each project is organised independently.** It has its own `dataset/`, `notebooks/`, `src/`, `models/`,
  `visualizations/`, `app.py`, `requirements.txt` and `README.md`. The fraud project also has `results/`,
  `docs/` and `tests/`.
- **Each project has a detailed README.** That README is the authoritative source for the project. This root
  README only summarises it.
- **The raw datasets are treated as read-only.** Each project verifies its dataset's SHA-256 checksum inside
  the notebooks.

---

## 3. Project Status

| Local Project | CodSoft Task | Project | Status |
|---|---:|---|---|
| [Task1_Movie_Rating_Prediction](Task1_Movie_Rating_Prediction/) | Task 2 | Movie Rating Prediction | ✅ Completed |
| [Task2_Sales_Prediction](Task2_Sales_Prediction/) | Task 4 | Sales Prediction | ✅ Completed |
| [Task3_Credit_Card_Fraud_Detection](Task3_Credit_Card_Fraud_Detection/) | Task 5 | Credit Card Fraud Detection | ✅ Completed |

Completed phases in each project:

| Project | Audit | Preprocessing | Baselines | CV + tuning | Final evaluation | Persistence + API | Dashboard |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Movie Rating Prediction | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Sales Prediction | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Credit Card Fraud Detection | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 4. CodSoft Task Mapping

The local folder numbers reflect my own ordering of the three selected projects. **They are not the official
CodSoft task numbers.** The folders are deliberately left with their current names.

| Local folder | Official CodSoft task |
|---|---|
| `Task1_Movie_Rating_Prediction` | **CodSoft Task 2**: Movie Rating Prediction with Python |
| `Task2_Sales_Prediction` | **CodSoft Task 4**: Sales Prediction Using Python |
| `Task3_Credit_Card_Fraud_Detection` | **CodSoft Task 5**: Credit Card Fraud Detection |

In short, local **Task 1 → CodSoft Task 2**, local **Task 2 → CodSoft Task 4** and local
**Task 3 → CodSoft Task 5**.

---

## 5. Projects at a Glance

| Project | Problem type | Dataset | Main models | Evaluation focus | Dashboard | Documentation |
|---|---|---|---|---|---|---|
| **Movie Rating Prediction** | Supervised regression | `IMDb Movies India.csv`: 15,509 rows × 10 columns (7,919 rated) | Mean baseline, Linear Regression, Random Forest, **Gradient Boosting** (final) | MAE, MSE, RMSE, R² on a group-aware holdout | Streamlit, 6 pages | [README](Task1_Movie_Rating_Prediction/README.md) |
| **Sales Prediction** | Supervised regression | `advertising.csv`: 200 rows × 4 columns | Mean baseline, Linear, Ridge, Lasso, Random Forest, **Gradient Boosting** (final) | MAE, MSE, RMSE, R² on a 40-row holdout | Streamlit, 6 pages | [README](Task2_Sales_Prediction/README.md) |
| **Credit Card Fraud Detection** | Imbalanced binary classification | `creditcard.csv`: 284,807 rows × 31 columns | Logistic Regression, **Random Forest** (final), each under 5 imbalance strategies | PR-AUC, precision, recall, F1, confusion matrix | Streamlit, 6 pages | [README](Task3_Credit_Card_Fraud_Detection/README.md) |

**Headline results** (each measured once on held-out test data):

| Project | Final model | Headline test metrics |
|---|---|---|
| Movie Rating Prediction | Gradient Boosting (tuned) | MAE **0.8862** · RMSE **1.1599** · R² **0.3046** (1,584 films) |
| Sales Prediction | Gradient Boosting (tuned) | MAE **0.9354** · RMSE **1.2173** · R² **0.9520** (40 observations) |
| Credit Card Fraud Detection | Random Forest | Precision **0.9452** · Recall **0.7263** · F1 **0.8214** · PR-AUC **0.7932** (56,746 transactions) |

Do not compare these numbers across projects. The problems, datasets and metrics are different.

---

## 6. Technology Stack

All three projects use Python 3.13 with the same pinned versions of their shared libraries.

| Category | Tools |
|---|---|
| Language | Python 3.13 |
| Data handling | pandas 3.0.6, NumPy 2.5.3 |
| Machine learning | scikit-learn 1.9.1; imbalanced-learn 0.14.2 (fraud project) |
| Statistics / I/O | SciPy 1.18.1 and pyarrow 25.0.1 (fraud project) |
| Static visualisation | Matplotlib 3.11.2, Seaborn 0.13.2 |
| Interactive visualisation | Plotly 7.1.0 |
| Dashboards | Streamlit 1.64.0 |
| Persistence | joblib 1.6.0 |
| Notebooks | Jupyter 1.1.1 |
| Version control | Git, plus Git LFS for the fraud dataset |

The exact versions for each project are pinned in that project's `requirements.txt`.

---

## 7. Project 1 — Movie Rating Prediction

📁 [`Task1_Movie_Rating_Prediction`](Task1_Movie_Rating_Prediction/) · **CodSoft Task 2**

### Problem

Predict a film's IMDb `Rating` (a continuous value on the 1–10 scale) from its metadata: release year,
runtime, genres, director and three lead actors.

### Dataset

| | |
|---|---|
| File | `dataset/IMDb Movies India.csv` (read as `latin-1`; the file is not valid UTF-8) |
| Size | 15,509 rows × 10 columns |
| Columns | `Name`, `Year`, `Duration`, `Genre`, `Rating`, `Votes`, `Director`, `Actor 1`, `Actor 2`, `Actor 3` |
| Target | `Rating` (only 7,919 rows, 51.1%, have one) |
| Model inputs | `Year`, `Duration`, `Genre`, `Director`, `Actor 1`–`Actor 3` |
| Not used as features | `Name` (identifier), `Votes` (a documented modelling assumption, because vote counts accumulate after release) |

### Approach

- **Preprocessing:** strict parsers for `"(YYYY)"`, `"N min"` and vote strings. Unrated rows are excluded,
  and missing ratings are never imputed. Six exact duplicates are removed from the modelling frame.
- **Feature engineering:** missing `Duration` values are filled with the median for the film's release decade,
  plus a missing-value indicator. Genres are multi-hot encoded. Directors and actors get frequency counts, and
  names with enough training films (directors with at least 10, actors with at least 20) get their own one-hot
  columns. No target encoding is used anywhere.
- **Leakage-safe design:** row-wise cleaning is kept separate from the fitted transformers. Every statistic,
  vocabulary and name count is learned inside `fit`, on training rows only.
- **Group-aware split:** `GroupShuffleSplit` (80/20, `random_state=42`) over normalised **`Name` + `Year`**
  groups. This keeps near-duplicate films from landing on both sides of the split. The result is 6,335
  training rows and 1,584 test rows, with 0 overlapping groups.
- **Models:** a mean baseline, Linear Regression (with `StandardScaler`), Random Forest and Gradient Boosting.
- **Validation and tuning:** 5-fold `GroupKFold` on the training rows only, with `RandomizedSearchCV` over
  12 Gradient Boosting and 10 Random Forest candidates. The phase also ran a feature-group ablation, a
  threshold sensitivity analysis and permutation importance.
- **Selection:** tuned Gradient Boosting had the lowest mean CV RMSE, beat every alternative on all five
  folds, and overfit much less than Random Forest.

### Final model

```python
GradientBoostingRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=42,
)
```

### Final evaluation (Phase 5, test set used once)

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| **Gradient Boosting (tuned)** | **0.8862** | **1.3453** | **1.1599** | **0.3046** |
| Baseline (training mean) | 1.1243 | 1.9361 | 1.3914 | −0.0008 |

That is 21.2% lower MAE and 16.6% lower RMSE than the baseline, and 64.6% of test films were predicted
within ±1 rating point.

### Persistence, API and dashboard

- **Persistence:** in Phase 6 the locked configuration is refitted on all 7,919 rated rows and saved as one
  preprocessing-and-model pipeline, `models/final_movie_rating_pipeline.joblib`. No metric is computed in
  Phase 6, so the reported performance is the Phase 5 holdout result.
- **API:** `src/predict.py` provides `predict_rating`, `predict_movies` and `describe_artifact`. `Year` is the
  only required field, and `Rating`/`Votes` are ignored if supplied.
- **Dashboard:** `app.py` has six pages: Overview, Predict Rating, Data Analysis, Model Performance, Model
  Details and About Project.

### Main limitations

R² is about 0.30, so most of the variation in ratings is not explained by this metadata. Predictions are
compressed toward the middle of the scale, which means low-rated films are over-predicted and high-rated
films under-predicted. Directors and actors the model has not seen carry no history.

➡️ [Detailed Movie Rating Prediction Documentation](Task1_Movie_Rating_Prediction/README.md)

---

## 8. Project 2 — Sales Prediction

📁 [`Task2_Sales_Prediction`](Task2_Sales_Prediction/) · **CodSoft Task 4**

### Problem

Predict `Sales` from the advertising spend on **TV**, **Radio** and **Newspaper**. The dataset has no
audience, demographic or time fields, so the audience-segmentation part of the CodSoft task statement cannot
be studied with it. The project README states this explicitly.

### Dataset

| | |
|---|---|
| File | `dataset/advertising.csv` |
| Provenance | The Kaggle dataset `ashydv/advertising-dataset`, traced from the link in the CodSoft task document. It is the *Advertising* dataset from *An Introduction to Statistical Learning*. |
| Shape | **200 rows × 4 columns** |
| Features | `TV`, `Radio`, `Newspaper` |
| Target | `Sales` |
| Quality | No missing values, no duplicates. Two high-`Newspaper` IQR outliers were **kept**, because they are plausible budgets. |

### Approach

- **EDA:** `TV` correlates strongly with `Sales` (r = 0.901), `Radio` moderately (0.350) and `Newspaper`
  weakly (0.158). The project describes these as correlations, not causal effects.
- **Preprocessing:** a `ColumnTransformer` with a `StandardScaler`, fitted on the training rows only. There is
  no imputation; missing or non-finite input raises an error instead. No feature engineering.
- **Split:** random 80/20 split, giving **160 training rows and 40 test rows** (`random_state=42`).
- **Models:** a mean baseline, Linear Regression, Ridge, Lasso, Random Forest and Gradient Boosting.
- **Cross-validation:** 5-fold `KFold` on the 160 training rows, with the scaler refitted inside every fold.
- **Tuning:** exhaustive `GridSearchCV` over 216 Gradient Boosting and 72 Random Forest configurations. It was
  followed by a repeated-CV stability check and an outlier sensitivity analysis.

### Final model and evaluation

The final model is a **tuned `GradientBoostingRegressor`** (`n_estimators=200`, `learning_rate=0.02`,
`max_depth=4`, `min_samples_leaf=2`, `subsample=0.8`, `random_state=42`). It was selected by training-set
cross-validation and evaluated once on the 40-row holdout.

| Metric | Final test value |
|---|---:|
| MAE | **0.9354** |
| MSE | **1.4819** |
| RMSE | **1.2173** |
| R² | **0.9520** |

> ⚠️ **Tuning did not help on the test set.** Tuned Gradient Boosting improved the cross-validated RMSE over
> the Phase 3 configuration (1.2808 vs 1.3603). **On the 40-row holdout it scored worse (RMSE 1.2173) than the
> untuned Phase 3 Gradient Boosting (RMSE 1.1451).** The tuned model was kept anyway: selection had already
> been finalised on training-set CV, and switching models because of the test result would amount to
> selecting on the test set. Phase 4 had also shown that these configurations cannot be told apart on this
> dataset. Across 25 repeated CV folds they were within 0.023 RMSE of each other, while fold-to-fold standard
> deviations were about 0.19–0.25.

### Persistence, API and dashboard

- **Persistence:** the full fitted pipeline (scaler and model) is saved to
  `models/final_sales_prediction_pipeline.joblib` together with its metadata. Reloading it in a fresh process
  gives bit-identical predictions.
- **API and CLI:** `src/predict.py` provides `predict_sales`, `predict_batch` and `model_metadata`, validates
  its inputs, and flags inputs outside the training ranges.
- **Dashboard:** `app.py` has six pages: Overview, Predict Sales, Data Analysis, Model Performance, Model
  Details and About Project.

### Main limitations

The dataset has only 200 rows and the test set only 40, so a different split would give noticeably
different numbers. There is no time, market or audience context. Predictions outside the training ranges are
extrapolation.

➡️ [Detailed Sales Prediction Documentation](Task2_Sales_Prediction/README.md)

---

## 9. Project 3 — Credit Card Fraud Detection

📁 [`Task3_Credit_Card_Fraud_Detection`](Task3_Credit_Card_Fraud_Detection/) · **CodSoft Task 5**

This is the most extensive project in the repository.

![Fraud detection dashboard overview](Task3_Credit_Card_Fraud_Detection/docs/images/dashboard-overview.png)

### Problem

Classify each card transaction as **fraudulent** (`Class = 1`) or **legitimate** (`Class = 0`) from its 30
numeric attributes.

### Dataset

| | |
|---|---|
| File | `dataset/creditcard.csv`, stored with **Git LFS** |
| Provenance | Kaggle `mlg-ulb/creditcardfraud` (ULB Machine Learning Group), the dataset linked from the CodSoft task document |
| Raw size | **284,807 rows × 31 columns**, no missing values |
| Features | `Time`, anonymised PCA components `V1`–`V28`, `Amount` |
| Target | `Class` |
| Raw imbalance | 284,315 legitimate vs 492 fraud (0.173%), about 578 : 1 |

### Duplicates and split

- **1,081 extra duplicate copies were removed before the split**, so no identical row can appear in both train
  and test. That leaves **283,726 rows: 283,253 legitimate and 473 fraud**.
- **Stratified 80/20 split, `random_state=42`:**

| Split | Rows | Legitimate | Fraud |
|---|---:|---:|---:|
| Train | 226,980 | 226,602 | 378 |
| Test | 56,746 | 56,651 | 95 |

### Why accuracy is not enough

A model that predicts "legitimate" for every test transaction is **99.8326% accurate and catches 0 of 95
frauds**. The final model's accuracy (99.9471%) is only 0.11 percentage points higher. For that reason the
project reports accuracy but never uses it to make a decision. It relies on **PR-AUC, precision, recall, F1
and the confusion matrix**. ROC-AUC is kept as a secondary metric, because it measures false alarms against
all legitimate transactions and so barely moves even when there are hundreds of false alarms.

### Approach

- **Preprocessing:** a `StandardScaler` inside the pipeline, fitted on training data only. Three scaling
  strategies were compared, and cross-validation kept the standard one.
- **Imbalance strategies:** original distribution, class weights, random oversampling, SMOTE and random
  undersampling. All run inside an `imblearn` pipeline, so only the training folds are ever resampled.
- **Baselines (Phase 3):** Logistic Regression and Random Forest under each of the five strategies. These
  results are kept only as a historical baseline and were never used to choose the final model.
- **Cross-validation (Phase 4):** 5-fold `StratifiedKFold` on the **training rows only**. The test set was
  fingerprinted and stayed isolated until the final evaluation.
- **Tuning:** 24 Logistic Regression and 12 Random Forest configurations. The primary metric was mean CV
  PR-AUC, and the final choice used a **one-standard-error rule** declared before any results were seen.
- **Threshold analysis:** the threshold with the highest out-of-fold F1 on training predictions was
  **0.50**.

### Final model

| Component | Setting |
|---|---|
| Algorithm | Random Forest: 100 trees, `min_samples_leaf=5`, `max_features=0.3`, `max_depth=None`, `random_state=42` |
| Preprocessing | `StandardScaler` on all 30 features |
| Imbalance handling | No resampling, `class_weight=None` |
| Decision threshold | **0.50** (fraud if P(fraud) ≥ 0.50) |
| Training-only CV PR-AUC | 0.838580 ± 0.044807 |

### Final test results (Phase 5, evaluated once)

| Precision | Recall | F1 | PR-AUC | ROC-AUC | Accuracy |
|---:|---:|---:|---:|---:|---:|
| **0.945205** | **0.726316** | **0.821429** | **0.793217** | 0.936863 | 0.999471 |

| Confusion matrix | Predicted legitimate | Predicted fraud |
|---|---:|---:|
| **Actual legitimate** | TN = 56,647 | FP = 4 |
| **Actual fraud** | FN = 26 | TP = 69 |

The model caught **69 of 95 frauds** and flagged **4 of 56,651 legitimate transactions**. The test PR-AUC is
below the CV estimate, which Phase 4 anticipated. It is still inside the candidate's CV fold range
(0.768–0.880). Nothing was changed after this evaluation.

### Persistence, CLI and dashboard

- **Persistence:** `models/final_credit_card_fraud_pipeline.joblib` stores the scaler, the forest **and the
  0.50 threshold** together, so prediction always uses the rule that was evaluated. A readable companion
  file, `models/final_model_metadata.json`, describes it.
- **API and CLI:** `src/predict.py` provides `predict_transaction`, `predict_batch` and a CLI
  (`--input`, `--json`, `--smoke-test`) with strict input validation.
- **Dashboard:** `app.py` has six pages: Overview, Fraud Prediction, Data Analysis, Model Performance, Model
  Details and About Project. It reads every metric from the saved result files.
- **Leakage prevention:** duplicates were removed before the split, and every transformation and resampler
  runs inside a pipeline fitted per fold. The threshold was chosen on out-of-fold training predictions, and the
  test set was opened only after the artefact had been saved and hashed.

### Main limitations

The data covers two days of European transactions from 2013. The features are anonymised, and no temporal
or drift validation was done. The positive class is small (95 test frauds). The probabilities are not
calibrated, and the project is explicitly **not a production system**.

➡️ [Detailed Credit Card Fraud Detection Documentation](Task3_Credit_Card_Fraud_Detection/README.md)

---

## 10. Cross-Project Technical Summary

| Technology | Movie Rating | Sales | Fraud Detection |
|---|:---:|:---:|:---:|
| Python 3.13 | ✅ | ✅ | ✅ |
| pandas | ✅ | ✅ | ✅ |
| NumPy | ✅ | ✅ | ✅ |
| Matplotlib | ✅ | ✅ | ✅ |
| Seaborn | ✅ | ✅ | ✅ |
| scikit-learn | ✅ | ✅ | ✅ |
| imbalanced-learn | — | — | ✅ |
| SciPy / pyarrow | — | — | ✅ |
| Jupyter | ✅ | ✅ | ✅ |
| joblib | ✅ | ✅ | ✅ |
| Streamlit | ✅ | ✅ | ✅ |
| Plotly | ✅ | ✅ | ✅ |
| Git LFS | — | — | ✅ |

### How the projects progress

| Project | Focus |
|---|---|
| **Movie Rating** | Regression, feature engineering for sparse and high-cardinality metadata, group-aware evaluation |
| **Sales** | Regression, EDA, a reusable preprocessing pipeline, comparison of six approaches, exhaustive tuning, stability analysis |
| **Credit Card Fraud** | Imbalanced binary classification, leakage prevention, stratified cross-validation, threshold analysis, a persisted model with its decision rule, a tested dashboard |

---

## 11. Machine Learning Techniques Used

| Technique | Where it was used |
|---|---|
| Mean baseline (`DummyRegressor` / training mean) | Movie Rating, Sales |
| Linear Regression | Movie Rating, Sales |
| Ridge and Lasso regression | Sales |
| Logistic Regression (including L1-regularised variants in tuning) | Fraud Detection |
| Random Forest | All three (final model for Fraud Detection) |
| Gradient Boosting | Movie Rating (final), Sales (final) |
| Class weights, random oversampling, SMOTE, random undersampling | Fraud Detection |
| `GroupShuffleSplit` / `GroupKFold` | Movie Rating |
| `KFold` / `RepeatedKFold` | Sales |
| Stratified split / `StratifiedKFold` | Fraud Detection |
| `RandomizedSearchCV` | Movie Rating |
| `GridSearchCV` | Sales, Fraud Detection |
| Permutation importance and feature-group ablation | Movie Rating |
| Paired bootstrap / bootstrap confidence intervals | Movie Rating (model comparison), Fraud Detection (test metrics) |
| Out-of-fold threshold selection (max F1) | Fraud Detection |
| One-standard-error model selection rule | Fraud Detection |

---

## 12. Data Preprocessing and Feature Engineering

| | Movie Rating | Sales | Fraud Detection |
|---|---|---|---|
| **Main data challenge** | Text-encoded numbers, 53% missing `Duration`, multi-label genres, thousands of directors and actors | Very small dataset, two outliers | 578 : 1 class imbalance, duplicate rows, anonymised features |
| **Missing values** | Decade-median imputation for `Duration` plus indicators; categorical fields left missing and flagged, never invented | None present; missing input is rejected at prediction time | None present |
| **Duplicates** | 6 exact duplicates removed; `Name` + `Year` repeats kept and handled by the group-aware split | None present | 1,081 extra copies removed before the split |
| **Outliers** | Short runtimes investigated and kept | 2 `Newspaper` outliers kept (sensitivity-tested) | Kept, because extreme values carry the fraud signal |
| **Encoding / features** | Multi-hot genres, frequency counts, threshold one-hot for people | None needed | None needed (PCA components used as provided) |
| **Scaling** | `StandardScaler` for Linear Regression only | `StandardScaler` in the shared pipeline | `StandardScaler` (chosen over two alternatives by CV) |

In all three projects, **every fitted statistic comes from training data only**, and preprocessing is
refitted inside each cross-validation fold through a scikit-learn or imbalanced-learn `Pipeline`.

---

## 13. Model Development and Evaluation

All three projects follow the same evaluation protocol:

1. **Baseline first.** Every model is measured against a naive reference.
2. **Selection on training data only.** Cross-validation on the training split picks the model and its
   hyperparameters (and, for fraud, the threshold).
3. **One final test evaluation.** The locked model is scored once on held-out data, and nothing is changed
   afterwards.
4. **Honest reporting.** CV estimates and test results are reported separately. Inconvenient results are kept
   in the documentation: the tuned sales model scored worse on the test set, the Random Forest searches found
   nothing better than the defaults, and the fraud test PR-AUC came in below CV.

| Project | Selection metric | Final test metrics reported |
|---|---|---|
| Movie Rating | Mean 5-fold group CV RMSE | MAE, MSE, RMSE, R² |
| Sales | Mean 5-fold CV RMSE | MAE, MSE, RMSE, R² |
| Fraud Detection | Mean 5-fold stratified CV PR-AUC with the one-standard-error rule | Precision, recall, F1, PR-AUC, ROC-AUC, accuracy, confusion matrix |

---

## 14. Validation and Reproducibility

Each project treats reproducibility as a requirement to verify, not something to assume.

| Practice | Movie Rating | Sales | Fraud Detection |
|---|:---:|:---:|:---:|
| Fixed random seeds (`random_state=42`) | ✅ | ✅ | ✅ |
| Train/test separation before fitting | ✅ | ✅ | ✅ |
| Preprocessing fitted on training data only | ✅ | ✅ | ✅ |
| Cross-validation for selection | ✅ | ✅ | ✅ |
| Explicit leakage checks | ✅ | ✅ | ✅ |
| Raw-dataset SHA-256 checks in notebooks | ✅ | ✅ | ✅ |
| Persisted model artefact | ✅ | ✅ | ✅ |
| Fresh-process reload reproduces predictions | ✅ | ✅ | ✅ |
| Prediction smoke tests | ✅ | ✅ | ✅ |
| Dashboard `AppTest` checks | — | ✅ (68 documented) | ✅ (42, in `tests/`) |
| Git LFS for the dataset | — | — | ✅ |

**Documented validation checks**

- **Movie Rating:** 92 automated notebook checks, all passing (Phase 3: 15, Phase 4: 23, Phase 5: 22,
  Phase 6: 32).
- **Sales:** 68 Streamlit `AppTest` checks, all passing. Nine deliberately invalid preprocessing inputs are
  each rejected with a specific error.
- **Credit Card Fraud Detection:**

| Phase | Result |
|---|---|
| Phase 1: Dataset audit | 22/22 |
| Phase 2: Preprocessing | 65/65 |
| Phase 3: Baselines | 22/22 |
| Phase 4: Cross-validation and tuning | 29/29 |
| Phase 5: Final evaluation | 18/18 integrity + 9/9 smoke + 12/12 CLI |
| Phase 6: Dashboard | 42/42 AppTest |
| Documentation Audit | PASS: repository documentation audited for links, metrics, structure, encoding and consistency |

> These counts are **project validation checks** (integrity, leakage, determinism and interface tests).
> They are not model accuracy metrics.

The fraud project also found that parallel Random Forest predictions were not bit-for-bit repeatable. It fixed
this with `FixedOrderRandomForestClassifier`, which adds the trees up in a fixed order. Three independent
trainings then produced the same artefact SHA-256.

---

## 15. Interactive Dashboards

Each project includes a **Streamlit** dashboard (`app.py`) that runs locally. None of the project READMEs
describes a public deployment, and none is claimed here.

| Dashboard | Pages | Location |
|---|---|---|
| Movie Rating Prediction | Overview · Predict Rating · Data Analysis · Model Performance · Model Details · About Project | [`Task1_Movie_Rating_Prediction/`](Task1_Movie_Rating_Prediction/) |
| Sales Prediction | Overview · Predict Sales · Data Analysis · Model Performance · Model Details · About Project | [`Task2_Sales_Prediction/`](Task2_Sales_Prediction/) |
| Credit Card Fraud Detection | Overview · Fraud Prediction · Data Analysis · Model Performance · Model Details · About Project | [`Task3_Credit_Card_Fraud_Detection/`](Task3_Credit_Card_Fraud_Detection/) |

All three dashboards are **presentation layers only**:

- They load the **persisted model artefact** through the project's `src/predict.py`.
- They **never retrain or refit** a model at prediction time.
- They never write to the dataset or the model file.

---

## 16. Repository Structure

```
CODSOFT/
├── Task1_Movie_Rating_Prediction/        # CodSoft Task 2
│   ├── .streamlit/                       # dashboard theme
│   ├── dataset/                          # IMDb Movies India.csv
│   ├── notebooks/                        # 01–06 phase notebooks
│   ├── src/                              # data_preprocessing.py, predict.py
│   ├── models/                           # final_movie_rating_pipeline.joblib
│   ├── visualizations/                   # generated charts
│   ├── app.py                            # Streamlit dashboard
│   ├── requirements.txt
│   └── README.md
│
├── Task2_Sales_Prediction/               # CodSoft Task 4
│   ├── .streamlit/
│   ├── dataset/                          # advertising.csv
│   ├── notebooks/                        # 01–05 phase notebooks
│   ├── src/                              # data_preprocessing.py, predict.py
│   ├── models/                           # final_sales_prediction_pipeline.joblib
│   ├── visualizations/
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
│
├── Task3_Credit_Card_Fraud_Detection/    # CodSoft Task 5
│   ├── .streamlit/
│   ├── dataset/                          # creditcard.csv (Git LFS)
│   ├── notebooks/                        # 01–05 phase notebooks
│   ├── src/                              # data_preprocessing.py, model_training.py, predict.py
│   ├── models/                           # final pipeline + final_model_metadata.json
│   ├── results/                          # Phase 4 CV results, Phase 5 final metrics
│   ├── visualizations/
│   ├── docs/                             # dashboard screenshots
│   ├── tests/                            # test_dashboard.py (Streamlit AppTest)
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
│
├── .gitattributes                        # Git LFS rule for creditcard.csv
├── .gitignore
└── README.md                             # this file
```

---

## 17. Installation and Environment

There is **no root-level `requirements.txt`**. Each project has its own, and Python 3.13 is the documented
version. The shared libraries are pinned to the same versions in all three files. The fraud project adds
`imbalanced-learn`, `scipy` and `pyarrow`.

The examples below use **Windows PowerShell**, the environment the projects were developed on.

```powershell
# 1. Install Git LFS once per machine (needed for the fraud dataset), then clone
git lfs install
git clone https://github.com/harsha282004/CODSOFT.git
cd CODSOFT

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Enter the project you want to run
cd Task3_Credit_Card_Fraud_Detection

# 4. Install that project's requirements
pip install -r requirements.txt
```

Then follow that project's README. The saved models are joblib/pickle scikit-learn objects tied to the
pinned versions (in particular scikit-learn 1.9.1), so installing the listed versions is recommended.

---

## 18. Running the Projects

Each command runs from inside its project folder. Every example below starts at the repository root.

### Jupyter notebooks

```powershell
jupyter notebook
```

Run the notebooks in `notebooks/` in numerical order.

### Command-line prediction

**Movie Rating:** prints the artifact metadata and one demonstration prediction.

```powershell
cd Task1_Movie_Rating_Prediction
python src\predict.py
```

**Sales:** a single prediction, or a set of representative scenarios.

```powershell
cd Task2_Sales_Prediction
python src\predict.py --tv 150 --radio 25 --newspaper 30
python src\predict.py --smoke-test
```

**Credit Card Fraud:** end-to-end checks, or scoring a JSON file that holds one transaction's 30 features.

```powershell
cd Task3_Credit_Card_Fraud_Detection
python src\predict.py --smoke-test
python src\predict.py --input transaction.json
```

### Streamlit dashboards

```powershell
cd Task1_Movie_Rating_Prediction      # or Task2_Sales_Prediction / Task3_Credit_Card_Fraud_Detection
streamlit run app.py
```

The dashboard opens at `http://localhost:8501` by default.

### Dashboard tests (fraud project)

```powershell
cd Task3_Credit_Card_Fraud_Detection
python tests\test_dashboard.py
```

---

## 19. Git LFS and Dataset Handling

The fraud dataset, **`Task3_Credit_Card_Fraud_Detection/dataset/creditcard.csv`** (about 144 MiB), is larger
than GitHub's 100 MB file limit, so it is stored with **Git LFS**. The repository's `.gitattributes` has one
LFS rule:

```
Task3_Credit_Card_Fraud_Detection/dataset/creditcard.csv filter=lfs diff=lfs merge=lfs -text
```

If you cloned the repository before installing Git LFS, the CSV will be a small pointer file. Fetch the real
file with:

```powershell
git lfs install
git lfs pull
```

The fraud project README explains how to check the file's SHA-256, and every fraud notebook checks it and
stops if it differs. The other two datasets are small and are stored as ordinary Git files. None of the three
datasets was created for this repository; each one's source is documented in its project README.

---

## 20. Project Documentation

| Project | Detailed README | Covers |
|---|---|---|
| Movie Rating Prediction | [Task1_Movie_Rating_Prediction/README.md](Task1_Movie_Rating_Prediction/README.md) | Dataset audit, preprocessing, leakage prevention, feature engineering, CV/tuning/ablation, error analysis, API, dashboard |
| Sales Prediction | [Task2_Sales_Prediction/README.md](Task2_Sales_Prediction/README.md) | Dataset provenance, EDA, preprocessing, model comparison, tuning and stability, final evaluation, API, dashboard |
| Credit Card Fraud Detection | [Task3_Credit_Card_Fraud_Detection/README.md](Task3_Credit_Card_Fraud_Detection/README.md) | Provenance, imbalance handling, CV and threshold selection, final evaluation, error analysis, leakage prevention, reproducibility, dashboard |

---

## 21. Key Learning Outcomes

- **Data auditing:** checking provenance, parsing and missingness before any modelling, and verifying checksums.
- **Preprocessing:** separating row-wise cleaning from fitted transformers, and imputing only where it is justified.
- **Feature engineering:** multi-hot encoding, frequency encoding and threshold encoding for high-cardinality fields.
- **Regression and classification:** building both kinds of model, each measured against a naive baseline.
- **Model comparison:** telling real differences from noise with paired bootstraps, fold-level comparisons and repeated CV.
- **Cross-validation:** grouped, plain and stratified folds, with preprocessing refitted in each fold.
- **Hyperparameter tuning:** randomized and grid search, and recognising the selection bias they introduce.
- **Imbalanced classification:** comparing class weights, oversampling, SMOTE and undersampling inside CV.
- **Precision, recall and F1:** reasoning about false positives and false negatives.
- **PR-AUC vs ROC-AUC:** why PR-AUC is more informative when the positive class is rare.
- **Threshold analysis:** choosing a decision threshold on out-of-fold training predictions, never on the test set.
- **Leakage prevention:** split first, fit inside pipelines, keep group-aware and stratified splits intact.
- **Model persistence:** saving the full pipeline and its metadata (and, for fraud, the threshold) as one artefact.
- **Reproducibility:** fixed seeds, fresh-process verification, deterministic artefacts.
- **Streamlit development:** building multi-page dashboards on persisted models, tested with `AppTest`.
- **Git LFS:** version-controlling a dataset above GitHub's file-size limit.
- **Documentation:** writing results, caveats and limitations clearly enough to be checked.

---

## 22. Limitations and Practical Considerations

These are summarised from the individual project READMEs. See each README for the full list.

### Dataset limitations

- **Movie Rating:** the metadata is sparse and leaves out budget, script, marketing and reviews. Some ratings
  rest on as few as 5 votes. The data covers Indian films up to about 2021–22.
- **Sales:** only 200 observations, with no time, market, audience or geography fields and no stated units.
- **Fraud:** two days of European transactions from 2013. The features are anonymised PCA components, and the
  test set has only 95 frauds.

### Model limitations

- **Movie Rating:** R² is about 0.30, predictions are compressed toward the mean, and errors are large for
  extreme ratings and for directors and actors the model has not seen.
- **Sales:** tree ensembles regress toward the mean at extreme inputs, and the model describes association,
  not causation.
- **Fraud:** the Random Forest probabilities are not calibrated, and most missed frauds look like legitimate
  traffic on the available features.

### Evaluation limitations

- **Movie Rating:** there is one final evaluation, so the test score has no spread. The tuning search was
  small, with two winning values at the edge of the searched range.
- **Sales:** a different 40-row split would give noticeably different numbers. The holdout twice produced
  model orderings that cross-validation did not confirm.
- **Fraud:** no time-ordered or drift validation was done, and the max-F1 threshold weights false alarms and
  missed fraud equally.

### Deployment limitations

- None of the projects is deployed as a hosted service, and none is claimed to be production-ready.
- The saved artefacts are tied to the pinned library versions.
- The fraud project has no real-time serving, monitoring, drift detection or retraining infrastructure.

---

## 23. Future Improvements

These items come from the project READMEs. **None of them is implemented.**

| Project | Documented future work |
|---|---|
| Movie Rating | A separate experiment that includes `Votes`; richer metadata or plot-text features; better handling of rating extremes; a wider search with nested CV; repeated group-aware CV; out-of-time evaluation; a small web API |
| Sales | `TV × Radio` interaction terms; nested CV; bootstrap confidence intervals; influence diagnostics; prediction intervals in the dashboard; a larger, richer dataset |
| Fraud | Probability calibration; cost-sensitive threshold selection; temporal validation; drift monitoring; SHAP explanations; a real-time inference service; an automated retraining policy |

---

## 24. Internship Completion Summary

This repository contains **three completed CodSoft Data Science projects**:

| Local folder | CodSoft task | Project | Status |
|---|---|---|---|
| [Task1_Movie_Rating_Prediction](Task1_Movie_Rating_Prediction/) | Task 2 | Movie Rating Prediction | ✅ Completed |
| [Task2_Sales_Prediction](Task2_Sales_Prediction/) | Task 4 | Sales Prediction | ✅ Completed |
| [Task3_Credit_Card_Fraud_Detection](Task3_Credit_Card_Fraud_Detection/) | Task 5 | Credit Card Fraud Detection | ✅ Completed |

Each project goes through the same stages: auditing the data, preprocessing it without leakage, comparing
models against a baseline, selecting on training data only, evaluating once on held-out data, saving the model
and presenting it in a Streamlit dashboard. Each project's README documents its results and its limitations.

---

<div align="center">

**CodSoft Data Science Virtual Internship**

Local Task 1 → CodSoft Task 2 · Local Task 2 → CodSoft Task 4 · Local Task 3 → CodSoft Task 5

</div>
