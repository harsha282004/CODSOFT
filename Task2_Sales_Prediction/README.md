# Sales Prediction Using Python

Predicting sales from advertising spend across TV, Radio and Newspaper.

**CodSoft Data Science Internship — Sales Prediction Using Python.**
This folder is named `Task2_Sales_Prediction` as my own project numbering; it corresponds to
**CodSoft Task 4: Sales Prediction Using Python**.

**Status:** Phase 1 complete — dataset audit and exploratory data analysis. No model trained yet.

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

## Project Structure

```
Task2_Sales_Prediction/
├── dataset/                      # raw data, never modified
│   └── advertising.csv
├── notebooks/
│   └── 01_dataset_audit.ipynb    # Phase 1 — audit & EDA
├── src/                          # reusable code (Phase 2 onwards)
├── visualizations/
│   ├── 02_sales_distribution.png
│   ├── 03_feature_vs_sales.png
│   └── 04_correlation_heatmap.png
├── models/                       # saved model artefacts (Phase 2 onwards)
├── README.md
└── requirements.txt
```

Two chart slots from the planned sequence were deliberately left unused rather than filled with empty
figures: `01_missing_values.png` (there is no missingness to plot) and `05_sales_by_category.png` (there are
no categorical variables). Both findings are reported in the notebook instead.

---

## Reproducing Phase 1

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_dataset_audit.ipynb
```

The notebook resolves its paths relative to the repository, contains no absolute paths and runs top to
bottom from a fresh kernel.
