"""Baseline model construction and evaluation for the Credit Card Fraud Detection project.

This module turns the Phase 2 preprocessing framework (``data_preprocessing.py``) into trainable
pipelines and scores them. It is imported by the Phase 3 notebook and is intended to be reused
unchanged by Phase 4, so that cross-validation, tuning and threshold analysis are built on the
same pipeline definitions the baselines were measured with.

Design decisions:

* **Two model families only** - logistic regression and random forest, the two the CodSoft brief
  names. Configurations are library defaults plus the settings needed for determinism and
  convergence (see ``LOGISTIC_REGRESSION_PARAMS`` / ``RANDOM_FOREST_PARAMS``). Nothing is tuned.
* **Five imbalance strategies** - see ``STRATEGIES``. Resampling always sits *inside* an
  ``imblearn.pipeline.Pipeline`` after preprocessing, so it acts on the data passed to ``fit``
  (the training split) and is skipped entirely at ``predict`` time. Class weighting is a
  classifier argument; it changes the loss, not the rows.
* **Default decision threshold** - predictions come from each estimator's own ``predict``, which
  for both families is equivalent to ``P(fraud) > 0.5`` (strict). Threshold selection is a Phase 4
  concern and must be done on training data, never on the test set.
* **Fraud is the positive class** - every precision / recall / F1 call passes ``pos_label=1``
  explicitly, and every ranking metric (ROC-AUC, average precision) is computed from
  probabilities, never from hard labels.
* **Fitted models are not kept by default** - a fully grown random forest trained on ~450k
  resampled rows is large, and the evaluation only needs its scores. Pass ``keep_model=True``
  when a later phase genuinely needs the fitted object.

No model is persisted to disk by this module.

Phase 4 additions (bottom of the module) provide training-only cross-validation with out-of-fold
predictions, fold summaries with descriptive intervals, probability-based search scorers and
threshold analysis. None of them accepts test data.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import data_preprocessing as dp

__all__ = [
    "RANDOM_STATE",
    "DEFAULT_THRESHOLD",
    "POSITIVE_LABEL",
    "MODELS",
    "STRATEGIES",
    "MODEL_LABELS",
    "STRATEGY_LABELS",
    "LOGISTIC_REGRESSION_PARAMS",
    "RANDOM_FOREST_PARAMS",
    "ModelName",
    "Strategy",
    "ExperimentResult",
    "FixedOrderRandomForestClassifier",
    "build_classifier",
    "build_pipeline",
    "compute_metrics",
    "run_experiment",
    "results_table",
    # Phase 4
    "N_SPLITS",
    "THRESHOLD_GRID",
    "SCORING",
    "FOLD_METRICS",
    "CVResult",
    "make_cv",
    "fold_table",
    "cross_validate_pipeline",
    "summarise_folds",
    "threshold_table",
    "select_threshold_max_f1",
    "RowRecordingScorer",
    "search_results_frame",
]

RANDOM_STATE = dp.RANDOM_STATE

#: The probability cut-off implied by ``predict`` for both model families. Not tuned.
DEFAULT_THRESHOLD = 0.5

#: Fraud. Passed explicitly to every metric so no call can silently score the majority class.
POSITIVE_LABEL = 1

ModelName = Literal["logistic_regression", "random_forest"]
Strategy = Literal["original", "class_weight", "oversample", "smote", "undersample"]

MODELS: tuple[ModelName, ...] = ("logistic_regression", "random_forest")
STRATEGIES: tuple[Strategy, ...] = ("original", "class_weight", "oversample", "smote", "undersample")

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}
STRATEGY_LABELS = {
    "original": "Original distribution",
    "class_weight": "Class weights (balanced)",
    "oversample": "Random oversampling",
    "smote": "SMOTE",
    "undersample": "Random undersampling",
}

#: Library defaults (L2 penalty, C=1.0, lbfgs) plus two explicit settings:
#: ``max_iter=1000`` - lbfgs converged in 14 iterations on the original training split and 37
#: after SMOTE, so the default 100 would already suffice; 1000 is headroom so that no strategy
#: can fail to converge silently. Convergence warnings are not suppressed anywhere.
#: ``random_state`` - lbfgs is deterministic, but the argument is fixed for completeness.
LOGISTIC_REGRESSION_PARAMS = {"max_iter": 1000, "random_state": RANDOM_STATE}

#: Library defaults (100 trees, gini, max_features="sqrt", unlimited depth, min_samples_leaf=1)
#: plus ``random_state`` and ``n_jobs``. scikit-learn draws every tree's seed from
#: ``random_state`` before any parallel work starts, so results are identical for any n_jobs;
#: parallelism is therefore safe and is used because a single 100-tree fit on the original
#: training split takes ~84 s even with all cores. Reproducibility is verified empirically in
#: the Phase 3 notebook rather than assumed.
RANDOM_FOREST_PARAMS = {"n_estimators": 100, "random_state": RANDOM_STATE, "n_jobs": -1}


class FixedOrderRandomForestClassifier(RandomForestClassifier):
    """A random forest that trains in parallel but sums tree probabilities in a fixed order.

    ``RandomForestClassifier.predict_proba`` with ``n_jobs != 1`` adds each tree's probabilities
    into a shared array in whatever order the threads finish. Floating-point addition is not
    associative, so the result can differ in the last bit from call to call - measured in Phase 4:
    20 of 20 repeated calls on one fitted forest differed by up to 2.2e-16. A one-ulp change is
    enough to flip a score that sits exactly on a decision threshold, or to reorder tied scores
    inside PR-AUC.

    Training is unaffected (every tree is grown from its own pre-drawn seed, so ``fit`` stays
    parallel and deterministic). Only prediction is made sequential, which fixes the summation
    order to the order of ``estimators_``. With ``min_samples_leaf=1`` and no conflicting labels
    every leaf is pure, per-tree probabilities are exactly 0 or 1, and the sums were already exact
    - which is why the Phase 3 forests reproduced bit-for-bit; this class returns identical values
    for them.
    """

    def predict_proba(self, X):
        n_jobs = self.n_jobs
        self.n_jobs = 1
        try:
            return super().predict_proba(X)
        finally:
            self.n_jobs = n_jobs


def build_classifier(model: ModelName, *, class_weight: str | None = None, **overrides):
    """An unfitted estimator: the documented baseline configuration plus any ``overrides``.

    With no overrides this is exactly the Phase 3 baseline. Phase 4 passes tuned values (``C``,
    ``l1_ratio``, ``solver``, ``min_samples_leaf`` ...) through ``overrides``.
    """
    if model == "logistic_regression":
        params = {**LOGISTIC_REGRESSION_PARAMS, "class_weight": class_weight, **overrides}
        return LogisticRegression(**params)
    if model == "random_forest":
        params = {**RANDOM_FOREST_PARAMS, "class_weight": class_weight, **overrides}
        return FixedOrderRandomForestClassifier(**params)
    raise ValueError(f"Unknown model {model!r}; expected one of {MODELS}.")


def build_pipeline(
    model: ModelName,
    strategy: Strategy,
    *,
    scaling: dp.ScalingStrategy = "standard",
    classifier_params: dict | None = None,
) -> Pipeline:
    """Preprocessing -> [resampler] -> classifier, as one ``imblearn`` pipeline.

    The resampler step is present only for strategies that change rows. ``imblearn`` applies it
    during ``fit`` and skips it during ``predict`` / ``predict_proba``, so it can only ever see
    the data ``fit`` receives - the training split, or a training fold under cross-validation.

    ``classifier_params`` overrides the baseline classifier settings (Phase 4 tuning). An explicit
    ``class_weight`` in it takes precedence over the one implied by ``strategy``.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; expected one of {STRATEGIES}.")

    steps = [("preprocess", dp.build_preprocessor(scaling))]

    resampler_key = {"original": "none", "class_weight": "class_weight"}.get(strategy, strategy)
    sampler = dp.build_resampler(resampler_key, random_state=RANDOM_STATE)
    if sampler is not None:
        steps.append(("resample", sampler))

    overrides = dict(classifier_params or {})
    class_weight = overrides.pop("class_weight", "balanced" if strategy == "class_weight" else None)
    steps.append(("classifier", build_classifier(model, class_weight=class_weight, **overrides)))
    return Pipeline(steps)


def compute_metrics(y_true, y_pred, y_score) -> dict:
    """Every Phase 3 metric, with fraud as the positive class.

    ``y_pred`` are hard labels (for accuracy, precision, recall, F1 and the confusion matrix);
    ``y_score`` are fraud probabilities (for ROC-AUC and average precision). Keeping the two
    arguments separate makes it impossible to compute a ranking metric from thresholded labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score, dtype=float)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score, pos_label=POSITIVE_LABEL),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


@dataclass
class ExperimentResult:
    """Everything one (model, strategy) run produced. Scores are kept; the model is optional."""

    model: ModelName
    strategy: Strategy
    params: dict
    fit_seconds: float
    resample_seconds: float
    classifier_training_rows: int
    classifier_training_counts: dict
    sampler_strategy: dict | None
    test_proba: np.ndarray
    test_pred: np.ndarray
    train_metrics: dict
    test_metrics: dict
    pipeline: Pipeline | None = field(default=None, repr=False)

    @property
    def label(self) -> str:
        return f"{MODEL_LABELS[self.model]} | {STRATEGY_LABELS[self.strategy]}"


def run_experiment(
    model: ModelName,
    strategy: Strategy,
    split: dp.SplitData,
    *,
    scaling: dp.ScalingStrategy = "standard",
    keep_model: bool = False,
) -> ExperimentResult:
    """Fit one pipeline on the training split, then score it on train and test.

    The test split is used exactly twice - ``predict_proba`` and ``predict`` - after the fit is
    complete. It is never passed to ``fit`` and never reaches the resampler.
    """
    pipeline = build_pipeline(model, strategy, scaling=scaling)

    # Measure the resampling step on its own, on the training split, so the cost of SMOTE and
    # friends is visible separately from the classifier - and so the classifier's actual
    # training-set size can be reported. This does not influence the pipeline fitted below.
    resample_seconds = 0.0
    counts = split.y_train.value_counts().sort_index()
    n_rows = len(split.y_train)
    if "resample" in pipeline.named_steps:
        pre = dp.build_preprocessor(scaling).fit(split.X_train)
        sampler = dp.build_resampler(
            {"oversample": "oversample", "smote": "smote", "undersample": "undersample"}[strategy],
            random_state=RANDOM_STATE,
        )
        start = time.perf_counter()
        _, y_resampled = sampler.fit_resample(pre.transform(split.X_train), split.y_train)
        resample_seconds = time.perf_counter() - start
        counts = pd.Series(y_resampled).value_counts().sort_index()
        n_rows = len(y_resampled)
        del y_resampled, pre, sampler

    start = time.perf_counter()
    pipeline.fit(split.X_train, split.y_train)
    fit_seconds = time.perf_counter() - start

    sampler_strategy = None
    if "resample" in pipeline.named_steps:
        sampler_strategy = {int(k): int(v) for k, v in pipeline.named_steps["resample"].sampling_strategy_.items()}

    train_proba = pipeline.predict_proba(split.X_train)[:, 1]
    train_pred = pipeline.predict(split.X_train)
    test_proba = pipeline.predict_proba(split.X_test)[:, 1]
    test_pred = pipeline.predict(split.X_test)

    return ExperimentResult(
        model=model,
        strategy=strategy,
        params=pipeline.named_steps["classifier"].get_params(),
        fit_seconds=fit_seconds,
        resample_seconds=resample_seconds,
        classifier_training_rows=int(n_rows),
        classifier_training_counts={int(k): int(v) for k, v in counts.items()},
        sampler_strategy=sampler_strategy,
        test_proba=test_proba,
        test_pred=np.asarray(test_pred),
        train_metrics=compute_metrics(split.y_train, train_pred, train_proba),
        test_metrics=compute_metrics(split.y_test, test_pred, test_proba),
        pipeline=pipeline if keep_model else None,
    )


def results_table(results: list[ExperimentResult], which: Literal["test", "train"] = "test") -> pd.DataFrame:
    """One row per experiment with the metrics of the requested split."""
    rows = []
    for r in results:
        metrics = r.test_metrics if which == "test" else r.train_metrics
        rows.append({"model": MODEL_LABELS[r.model], "strategy": STRATEGY_LABELS[r.strategy], **metrics})
    return pd.DataFrame(rows).set_index(["model", "strategy"])


# =========================================================================================== #
# Phase 4 - training-only cross-validation, out-of-fold predictions and threshold analysis.
#
# Every function below receives training data only. None of them accepts, loads or derives the
# test split: that is what keeps the Phase 5 holdout untouched while models are being selected.
# =========================================================================================== #

N_SPLITS = 5

#: Thresholds examined in Phase 4. A coarse, fixed grid: selecting an operating point from a
#: fine grid would fit the threshold to out-of-fold noise.
THRESHOLD_GRID = tuple(round(float(t), 2) for t in np.arange(0.05, 0.951, 0.05))

#: Scorers for ``GridSearchCV`` / ``RandomizedSearchCV``. Ranking metrics use ``predict_proba``
#: explicitly, so every Phase 4 number is computed exactly as Phases 3 and 5 compute theirs (a
#: decision-function scorer can break ties differently once probabilities saturate at 1.0).
SCORING = {
    "pr_auc": make_scorer(average_precision_score, response_method="predict_proba", pos_label=POSITIVE_LABEL),
    "roc_auc": make_scorer(roc_auc_score, response_method="predict_proba"),
    "precision": make_scorer(precision_score, pos_label=POSITIVE_LABEL, zero_division=0),
    "recall": make_scorer(recall_score, pos_label=POSITIVE_LABEL, zero_division=0),
    "f1": make_scorer(f1_score, pos_label=POSITIVE_LABEL, zero_division=0),
}

FOLD_METRICS = ("pr_auc", "precision", "recall", "f1", "roc_auc", "accuracy")


def make_cv(n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    """The one cross-validation splitter every Phase 4 experiment shares."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def fold_table(cv: StratifiedKFold, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Row and fraud counts for each fold, with a train/validation overlap count."""
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]
        rows.append({
            "fold": fold,
            "train rows": len(train_idx),
            "validation rows": len(valid_idx),
            "train fraud": int(y_tr.sum()),
            "validation fraud": int(y_va.sum()),
            "train fraud %": y_tr.mean() * 100,
            "validation fraud %": y_va.mean() * 100,
            "overlap": len(np.intersect1d(train_idx, valid_idx)),
        })
    return pd.DataFrame(rows).set_index("fold")


@dataclass
class CVResult:
    """Everything one cross-validated configuration produced. Training data only."""

    name: str
    fold_metrics: pd.DataFrame            # validation fold, default threshold 0.5
    train_fold_metrics: pd.DataFrame      # the fold's own (un-resampled) training rows
    oof_proba: pd.Series                  # exactly one out-of-fold probability per training row
    fold_of_row: pd.Series                # the fold each training row was validated in
    fit_seconds: list
    scaler_samples_seen: list             # per fold - must equal that fold's training size
    fold_train_sizes: list
    sampler_strategies: list              # per fold, or None when nothing is resampled
    fold_train_counts: list
    warnings: list

    def summary(self) -> pd.Series:
        return summarise_folds(self.fold_metrics)


def _scalers_in(pipeline: Pipeline) -> list:
    """Every fitted StandardScaler inside the pipeline's preprocessing step."""
    found = []
    for _, transformer, _ in pipeline.named_steps["preprocess"].transformers_:
        steps = transformer.named_steps.values() if hasattr(transformer, "named_steps") else [transformer]
        found.extend(step for step in steps if isinstance(step, StandardScaler))
    return found


def cross_validate_pipeline(
    name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    *,
    record_train_metrics: bool = True,
) -> CVResult:
    """Fit a fresh clone of ``pipeline`` on each training fold; score the untouched validation fold.

    Preprocessing and any resampler live inside the pipeline, so both are re-fitted from scratch
    on every fold's training rows and never see that fold's validation rows. The validation fold
    keeps its natural class distribution. Every training row receives exactly one out-of-fold
    probability, from the single model that did not train on it.
    """
    oof = pd.Series(np.nan, index=X.index, dtype=float)
    fold_of_row = pd.Series(0, index=X.index, dtype=int)
    fold_rows, train_rows = [], []
    fit_seconds, scaler_seen, train_sizes, sampler_strats, train_counts, caught = [], [], [], [], [], []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[valid_idx], y.iloc[valid_idx]
        model = clone(pipeline)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            start = time.perf_counter()
            model.fit(X_tr, y_tr)
            fit_seconds.append(time.perf_counter() - start)
        caught.extend((fold, x.category.__name__, str(x.message)) for x in w)

        proba = model.predict_proba(X_va)[:, 1]
        pred = model.predict(X_va)
        if not oof.iloc[valid_idx].isna().all():
            raise AssertionError("a training row received a second out-of-fold prediction")
        oof.iloc[valid_idx] = proba
        fold_of_row.iloc[valid_idx] = fold
        fold_rows.append({"fold": fold, **compute_metrics(y_va, pred, proba)})

        if record_train_metrics:
            tr_proba = model.predict_proba(X_tr)[:, 1]
            train_rows.append({"fold": fold, **compute_metrics(y_tr, model.predict(X_tr), tr_proba)})

        scaler_seen.append([int(np.max(s.n_samples_seen_)) for s in _scalers_in(model)])
        train_sizes.append(len(train_idx))
        train_counts.append({int(k): int(v) for k, v in y_tr.value_counts().sort_index().items()})
        sampler = model.named_steps.get("resample")
        sampler_strats.append(
            None if sampler is None else {int(k): int(v) for k, v in sampler.sampling_strategy_.items()}
        )
        del model

    return CVResult(
        name=name,
        fold_metrics=pd.DataFrame(fold_rows).set_index("fold"),
        train_fold_metrics=pd.DataFrame(train_rows).set_index("fold") if train_rows else pd.DataFrame(),
        oof_proba=oof,
        fold_of_row=fold_of_row,
        fit_seconds=fit_seconds,
        scaler_samples_seen=scaler_seen,
        fold_train_sizes=train_sizes,
        sampler_strategies=sampler_strats,
        fold_train_counts=train_counts,
        warnings=caught,
    )


def summarise_folds(fold_metrics: pd.DataFrame, metrics=FOLD_METRICS, confidence: float = 0.95) -> pd.Series:
    """Mean, sample std, min, max and a t-based interval for the mean, per metric.

    The interval treats the fold scores as a sample of size ``n_splits``. It is **descriptive
    only**: CV folds share most of their training data, so fold scores are not independent and
    the interval is likely too narrow (there is no unbiased estimator of k-fold CV variance).
    """
    n = len(fold_metrics)
    t = stats.t.ppf(0.5 + confidence / 2, df=n - 1)
    out = {}
    for m in metrics:
        values = fold_metrics[m].to_numpy(dtype=float)
        mean, sd = values.mean(), values.std(ddof=1)
        half = t * sd / np.sqrt(n)
        out.update({f"{m}_mean": mean, f"{m}_std": sd, f"{m}_min": values.min(), f"{m}_max": values.max(),
                    f"{m}_ci_low": mean - half, f"{m}_ci_high": mean + half})
    return pd.Series(out)


def threshold_table(y_true, proba, thresholds=THRESHOLD_GRID) -> pd.DataFrame:
    """Precision, recall, F1 and error counts at each threshold, from one score vector.

    Predictions are ``proba >= threshold``. Precision is NaN (not 0) when nothing is flagged, so
    an empty operating point cannot masquerade as a measured one.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        rows.append({"threshold": t, "precision": precision, "recall": recall, "f1": f1,
                     "tp": tp, "fp": fp, "fn": fn, "tn": tn, "flagged": tp + fp})
    return pd.DataFrame(rows).set_index("threshold")


def select_threshold_max_f1(table: pd.DataFrame) -> float:
    """The grid threshold with the highest F1; ties go to the LOWER threshold (more recall).

    Declared in advance: with no cost model in the data, F1 is the transparent default that
    balances missed fraud and false alarms through recall and precision. The tie-break favours
    recall because a missed fraud is usually the costlier error.
    """
    best = table["f1"].max()
    return float(table.index[np.isclose(table["f1"], best)].min())


class RowRecordingScorer:
    """Wraps a scorer and records the index labels of every row it is asked to score.

    Used as direct evidence that a hyperparameter search scored training rows only.
    """

    def __init__(self, scorer):
        self.scorer = scorer
        self.seen: set = set()

    def __call__(self, estimator, X, y, **kwargs):
        self.seen.update(X.index.tolist())
        return self.scorer(estimator, X, y, **kwargs)


def search_results_frame(search) -> pd.DataFrame:
    """Tidy view of a fitted search: parameters, mean/std of each scorer, ranked by CV PR-AUC."""
    res = pd.DataFrame(search.cv_results_)
    params = [c for c in res.columns if c.startswith("param_")]
    scores = [c for c in res.columns if c.startswith(("mean_test_", "std_test_", "mean_train_", "std_train_"))]
    splits = [f"split{i}_test_pr_auc" for i in range(search.n_splits_)]
    out = res[params + scores + splits + ["mean_fit_time"]].copy()
    out.columns = [c.replace("param_", "").replace("classifier__", "") for c in out.columns]
    return out.sort_values("mean_test_pr_auc", ascending=False, kind="stable").reset_index(drop=True)
