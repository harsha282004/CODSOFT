"""Reusable preprocessing for the Credit Card Fraud Detection project.

This module is the single definition of how the raw transaction data becomes model-ready
features. It is imported by the Phase 2 notebook and is intended to be imported unchanged by
later phases (model training, validation, final evaluation, persistence and prediction), so
that every one of them sees exactly the same split and exactly the same transformation.

Design decisions, all of which follow from the Phase 1 audit of ``dataset/creditcard.csv``:

* **No imputation.** The dataset has zero missing values, zero infinities and no disguised
  placeholders. An imputer here would fit a statistic that is never used, and at prediction
  time it would silently invent a value for a genuinely absent input instead of reporting the
  problem. Non-finite input is validated and raised on, not filled.
* **No categorical encoding.** All 30 predictors are numeric ``float64``; there is nothing to
  encode. This follows from the PCA transformation applied to the original data.
* **No feature engineering.** No feature is derived from ``Class``, which would be leakage.
* **Duplicate rows dropped before the split.** See ``DUPLICATE_DECISION``.
* **Outliers retained.** See ``OUTLIER_DECISION``.
* **Scaling fitted on the training split only**, inside a ``ColumnTransformer`` that later
  phases wrap in a ``Pipeline``. See ``build_preprocessor`` for the justification of *which*
  columns are scaled.
* **Resampling is a training-fold concern.** ``build_resampler`` returns a sampler but never
  applies one; the samplers are meant to sit inside an ``imblearn`` pipeline so that they act
  on the training fold only and never on validation or test data.

No classifier is defined or trained here. The raw CSV is treated as read-only: nothing in this
module writes to ``dataset/``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

__all__ = [
    "TARGET_COLUMN",
    "TIME_COLUMN",
    "AMOUNT_COLUMN",
    "COMPONENT_COLUMNS",
    "FEATURE_COLUMNS",
    "EXPECTED_COLUMNS",
    "EXPECTED_RAW_ROWS",
    "EXPECTED_DEDUPLICATED_ROWS",
    "RAW_DATASET_SHA256",
    "RANDOM_STATE",
    "TEST_SIZE",
    "DUPLICATE_DECISION",
    "OUTLIER_DECISION",
    "ScalingStrategy",
    "ResamplingStrategy",
    "SplitData",
    "project_root",
    "default_dataset_path",
    "sha256_of_file",
    "verify_raw_dataset_integrity",
    "load_data",
    "validate_raw_dataframe",
    "split_features_target",
    "split_data",
    "build_preprocessor",
    "get_feature_names",
    "build_resampler",
    "validate_split",
    "validate_preprocessor",
    "validate_resampling",
]

# --------------------------------------------------------------------------------------- #
# Dataset constants, all verified against the file in Phase 1 rather than assumed.
# --------------------------------------------------------------------------------------- #

TARGET_COLUMN = "Class"
TIME_COLUMN = "Time"
AMOUNT_COLUMN = "Amount"
COMPONENT_COLUMNS = [f"V{i}" for i in range(1, 29)]

#: Predictors, in the order they appear in the raw file.
FEATURE_COLUMNS = [TIME_COLUMN, *COMPONENT_COLUMNS, AMOUNT_COLUMN]

#: Full raw layout, target last.
EXPECTED_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]

EXPECTED_RAW_ROWS = 284_807
EXPECTED_DEDUPLICATED_ROWS = 283_726

#: SHA-256 of the raw CSV as downloaded in Phase 1 from Kaggle ``mlg-ulb/creditcardfraud``.
RAW_DATASET_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"

RANDOM_STATE = 42
TEST_SIZE = 0.2

DUPLICATE_DECISION = """\
The 1,081 exact duplicate rows found in Phase 1 (1,062 legitimate, 19 fraudulent) are dropped
before the train/test split, keeping the first occurrence of each.

The reason is not that duplicates are dirty. Phase 1 established that they carry no conflicting
labels and are plausible collisions of small transactions in a finite-precision PCA space. The
reason is split contamination: an identical row appearing in both the training and the test set
means the model is scored on a row it has already memorised, which inflates the test result
without any corresponding gain in real performance. Dropping them before the split removes that
possibility entirely.

Dropping after the split would not work - it would leave the copies that had already crossed the
boundary. Hence: deduplicate, then split.
"""

OUTLIER_DECISION = """\
Outliers are retained. Phase 1 found that the 1.5 x IQR rule flags 138,473 rows (48.62% of the
dataset) on at least one feature, including 477 of the 492 frauds (96.95%), and that the flags
concentrate sharply in the minority class (V14 flags 87.40% of fraud against 4.83% of legitimate
rows). In fraud detection, extremity in the feature space is part of the signal being searched
for; removing it would remove the thing a model is supposed to learn.

No capping, winsorising or trimming is applied anywhere in this module.
"""

#: Which columns a scaling strategy standardises.
ScalingStrategy = Literal["standard", "minimal", "log_amount"]

#: Which resampler ``build_resampler`` returns. "none" and "class_weight" both return ``None``
#: because neither changes the training rows - class weighting is a classifier argument applied
#: in a later phase, not a preprocessing step.
ResamplingStrategy = Literal["none", "class_weight", "oversample", "smote", "undersample"]


@dataclass(frozen=True)
class SplitData:
    """The stratified train/test split. Held as one object so no phase can mix up its parts."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        return len(self.X_test)

    def class_counts(self) -> pd.DataFrame:
        """Per-split class counts and percentages, for reporting."""
        rows = {}
        for name, y in (("train", self.y_train), ("test", self.y_test)):
            counts = y.value_counts().sort_index()
            rows[name] = {
                "rows": len(y),
                "legitimate": int(counts.get(0, 0)),
                "fraudulent": int(counts.get(1, 0)),
                "fraud_%": float(counts.get(1, 0)) / len(y) * 100,
                "ratio_0_to_1": float(counts.get(0, 0)) / max(int(counts.get(1, 0)), 1),
            }
        return pd.DataFrame(rows).T


# --------------------------------------------------------------------------------------- #
# Paths and raw-file integrity
# --------------------------------------------------------------------------------------- #


def project_root() -> Path:
    """The ``Task3_Credit_Card_Fraud_Detection`` directory, resolved from this file."""
    return Path(__file__).resolve().parent.parent


def default_dataset_path() -> Path:
    """Location of the raw CSV downloaded in Phase 1."""
    return project_root() / "dataset" / "creditcard.csv"


def sha256_of_file(path: Path | str, chunk: int = 1 << 20) -> str:
    """Stream a file through SHA-256 without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_raw_dataset_integrity(path: Path | str | None = None) -> str:
    """Return the raw CSV's digest, raising if it no longer matches the Phase 1 value.

    Any phase can call this before and after its work to prove it did not touch the raw data.
    """
    path = Path(path) if path is not None else default_dataset_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. It is not recreated automatically - re-download "
            f"it from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud as Phase 1 did."
        )
    digest = sha256_of_file(path)
    if digest != RAW_DATASET_SHA256:
        raise ValueError(
            f"Raw dataset checksum mismatch at {path}.\n"
            f"  expected {RAW_DATASET_SHA256}\n"
            f"  found    {digest}\n"
            "The file is not the one audited in Phase 1. Stop and investigate before continuing."
        )
    return digest


# --------------------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------------------- #


def validate_raw_dataframe(df: pd.DataFrame) -> None:
    """Raise if the loaded frame is not the dataset every later phase expects."""
    actual = list(df.columns)
    if actual != EXPECTED_COLUMNS:
        missing = [c for c in EXPECTED_COLUMNS if c not in actual]
        extra = [c for c in actual if c not in EXPECTED_COLUMNS]
        raise ValueError(
            f"Unexpected columns. missing={missing or 'none'} extra={extra or 'none'} "
            f"(order matters: expected {EXPECTED_COLUMNS[:3]}...{EXPECTED_COLUMNS[-2:]})"
        )
    if len(df) != EXPECTED_RAW_ROWS:
        raise ValueError(f"Expected {EXPECTED_RAW_ROWS:,} raw rows, found {len(df):,}.")

    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric columns found: {non_numeric}")

    if int(df.isna().sum().sum()) != 0:
        raise ValueError("Missing values found; Phase 1 established there are none.")
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("Non-finite values found; Phase 1 established there are none.")

    observed = set(df[TARGET_COLUMN].unique())
    if observed != {0, 1}:
        raise ValueError(f"Target must be binary {{0, 1}}, found {sorted(observed)}.")


def load_data(
    path: Path | str | None = None,
    *,
    drop_duplicates: bool = True,
    verify_checksum: bool = True,
) -> pd.DataFrame:
    """Load the raw dataset, validate it, and optionally drop exact duplicate rows.

    Parameters
    ----------
    path
        Location of ``creditcard.csv``. Defaults to the Phase 1 download.
    drop_duplicates
        Drop exact duplicate rows, keeping the first occurrence. Default ``True`` - see
        ``DUPLICATE_DECISION``. Set ``False`` to reproduce the raw Phase 1 frame.
    verify_checksum
        Confirm the file is byte-for-byte the one audited in Phase 1.

    Returns
    -------
    A new DataFrame. The file on disk is only ever read.
    """
    path = Path(path) if path is not None else default_dataset_path()
    if verify_checksum:
        verify_raw_dataset_integrity(path)

    df = pd.read_csv(path)
    validate_raw_dataframe(df)

    if drop_duplicates:
        df = df.drop_duplicates(keep="first").reset_index(drop=True)
        if len(df) != EXPECTED_DEDUPLICATED_ROWS:
            raise ValueError(
                f"Expected {EXPECTED_DEDUPLICATED_ROWS:,} rows after deduplication, "
                f"found {len(df):,}."
            )
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictors from the target.

    ``Class`` is selected out by name, so it cannot end up inside ``X`` by accident - and the
    assertion below makes that failure loud rather than silent.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    if TARGET_COLUMN in X.columns:
        raise AssertionError("Target column leaked into the feature matrix.")
    if len(X) != len(y):
        raise AssertionError(f"X has {len(X):,} rows but y has {len(y):,}.")
    return X, y


def split_data(
    df: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> SplitData:
    """Stratified train/test split.

    Stratification is not optional here. With a fraud rate near 0.17%, an unstratified draw can
    leave either split with a badly distorted positive rate purely by chance, which would make
    every downstream metric a lottery.
    """
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y, shuffle=True
    )
    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


# --------------------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------------------- #


def _log1p_frame(values):
    """``log1p`` for the optional Amount transform. Module-level so the pipeline can pickle."""
    return np.log1p(values)


def build_preprocessor(strategy: ScalingStrategy = "standard") -> ColumnTransformer:
    """Build the (unfitted) preprocessing transformer.

    **What needs scaling, and why it is not "all of it by reflex".**

    Phase 1 measured the scales directly. ``Time`` spans 172,792 and ``Amount`` spans 25,691,
    against a total ``V1``-``V28`` range of about 234 - roughly 737x and 110x the component
    range respectively. For any gradient- or distance-based model those two columns would
    dominate the objective through magnitude alone. **Scaling ``Time`` and ``Amount`` is
    therefore not a judgement call; it is required, and every strategy below does it.**

    The genuinely open question is ``V1``-``V28``. They are already centred on zero, but they
    are *not* unit-variance: their standard deviations fall monotonically from 1.9587 (``V1``)
    to 0.3301 (``V28``), a spread of about 5.9x. Two defensible readings:

    * *Leave them.* That 5.9x spread is the PCA variance ordering and is real information about
      how much of the original data each component captures. It is also small next to the 737x
      mismatch that actually needed fixing.
    * *Standardise them.* L2-regularised logistic regression penalises all coefficients equally,
      so a low-variance component needs a larger coefficient for the same effect and is
      penalised harder for it. That is an arbitrary bias introduced by the units, not by the
      data.

    Neither can be settled without fitting a model, so both are implemented and the choice is
    deferred to a Phase 3 comparison by cross-validation **on the training split only**.
    ``"standard"`` is the default because regularised logistic regression is the primary model
    the CodSoft brief names, and uniform treatment under the penalty is the safer default there.
    Tree-based models are scale-invariant and are unaffected either way.

    Parameters
    ----------
    strategy
        ``"standard"``  - standardise all 30 features (default).
        ``"minimal"``   - standardise ``Time`` and ``Amount``; pass ``V1``-``V28`` through.
        ``"log_amount"``- ``log1p(Amount)`` first, then standardise all 30. ``Amount`` has a
                          skewness of 16.98; standardising alone recentres it but leaves the
                          shape untouched. ``log1p`` is safe here because Phase 1 confirmed
                          ``Amount >= 0`` with no negatives.

    Returns
    -------
    An **unfitted** ``ColumnTransformer``. Fit it on training data only. A ``RobustScaler``
    would be another reasonable candidate given the heavy tails; it is deliberately not added
    here to keep the Phase 3 comparison small.
    """
    if strategy == "minimal":
        return ColumnTransformer(
            transformers=[
                ("scale_time_amount", StandardScaler(), [TIME_COLUMN, AMOUNT_COLUMN]),
                ("passthrough_components", "passthrough", COMPONENT_COLUMNS),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    if strategy == "standard":
        return ColumnTransformer(
            transformers=[("scale_all", StandardScaler(), FEATURE_COLUMNS)],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    if strategy == "log_amount":
        amount_pipeline = Pipeline(
            [
                ("log1p", FunctionTransformer(_log1p_frame, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("scale_time", StandardScaler(), [TIME_COLUMN]),
                ("scale_components", StandardScaler(), COMPONENT_COLUMNS),
                ("log_scale_amount", amount_pipeline, [AMOUNT_COLUMN]),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    raise ValueError(
        f"Unknown scaling strategy {strategy!r}; expected 'standard', 'minimal' or 'log_amount'."
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Output column names of a fitted preprocessor, in output order."""
    return list(preprocessor.get_feature_names_out())


# --------------------------------------------------------------------------------------- #
# Class imbalance
# --------------------------------------------------------------------------------------- #


def build_resampler(strategy: ResamplingStrategy = "none", *, random_state: int = RANDOM_STATE):
    """Return an (unfitted) resampler, or ``None`` when the strategy does not resample.

    **This function never applies anything.** It hands back a sampler for the caller to place
    inside an ``imblearn.pipeline.Pipeline``. That placement is what makes resampling safe: an
    imblearn pipeline applies its samplers during ``fit`` only, and skips them entirely during
    ``predict`` / ``score``. Under cross-validation that means each training fold is resampled
    independently and no validation fold ever contains a synthetic or duplicated row.

    Resampling data by hand before cross-validation defeats this and is the single most common
    way this dataset is mishandled: duplicated or interpolated copies of the same fraud land in
    both the training and the validation fold, and the reported recall becomes fiction.

    Parameters
    ----------
    strategy
        ``"none"``         - original training distribution. Returns ``None``.
        ``"class_weight"`` - handled by the classifier's own ``class_weight`` argument in a
                             later phase, not by touching the rows. Returns ``None``.
        ``"oversample"``   - ``RandomOverSampler``: duplicates existing minority rows.
        ``"smote"``        - ``SMOTE``: interpolates new minority points between neighbours.
        ``"undersample"``  - ``RandomUnderSampler``: discards majority rows.
    """
    if strategy in ("none", "class_weight"):
        return None

    # Imported lazily so that the module remains usable for loading and splitting even if
    # imbalanced-learn is not installed.
    if strategy == "oversample":
        from imblearn.over_sampling import RandomOverSampler

        return RandomOverSampler(random_state=random_state)

    if strategy == "smote":
        from imblearn.over_sampling import SMOTE

        return SMOTE(random_state=random_state)

    if strategy == "undersample":
        from imblearn.under_sampling import RandomUnderSampler

        return RandomUnderSampler(random_state=random_state)

    raise ValueError(
        f"Unknown resampling strategy {strategy!r}; expected one of 'none', 'class_weight', "
        "'oversample', 'smote', 'undersample'."
    )


# --------------------------------------------------------------------------------------- #
# Validation helpers - each returns (check name, passed) pairs
# --------------------------------------------------------------------------------------- #


def validate_split(
    split: SplitData,
    df: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    tolerance_pp: float = 0.05,
) -> list[tuple[str, bool]]:
    """Structural checks on the split, including stratification and row overlap."""
    y_all = df[TARGET_COLUMN]
    full_rate = float((y_all == 1).mean() * 100)
    train_rate = float((split.y_train == 1).mean() * 100)
    test_rate = float((split.y_test == 1).mean() * 100)

    expected_test = round(len(df) * test_size)
    train_index = set(split.X_train.index)
    test_index = set(split.X_test.index)

    return [
        ("X and y row counts match in train", len(split.X_train) == len(split.y_train)),
        ("X and y row counts match in test", len(split.X_test) == len(split.y_test)),
        ("target has exactly two classes", set(y_all.unique()) == {0, 1}),
        ("train + test equals the full row count", split.n_train + split.n_test == len(df)),
        ("test size matches the requested fraction", abs(split.n_test - expected_test) <= 1),
        ("train contains both classes", set(split.y_train.unique()) == {0, 1}),
        ("test contains both classes", set(split.y_test.unique()) == {0, 1}),
        (
            f"train fraud rate within {tolerance_pp} pp of the full dataset",
            abs(train_rate - full_rate) <= tolerance_pp,
        ),
        (
            f"test fraud rate within {tolerance_pp} pp of the full dataset",
            abs(test_rate - full_rate) <= tolerance_pp,
        ),
        ("no row index appears in both splits", len(train_index & test_index) == 0),
        ("target column absent from the feature matrix", TARGET_COLUMN not in split.X_train.columns),
        ("feature columns identical in train and test", list(split.X_train.columns) == list(split.X_test.columns)),
    ]


def validate_preprocessor(
    preprocessor: ColumnTransformer,
    split: SplitData,
    X_train_t: np.ndarray,
    X_test_t: np.ndarray,
) -> list[tuple[str, bool]]:
    """Checks on a *fitted* preprocessor and its outputs, including the leakage test."""
    checks: list[tuple[str, bool]] = [
        ("transformed train row count unchanged", X_train_t.shape[0] == split.n_train),
        ("transformed test row count unchanged", X_test_t.shape[0] == split.n_test),
        ("train and test have the same feature count", X_train_t.shape[1] == X_test_t.shape[1]),
        ("no NaN introduced by preprocessing", not np.isnan(X_train_t).any() and not np.isnan(X_test_t).any()),
        ("no infinity introduced by preprocessing", np.isfinite(X_train_t).all() and np.isfinite(X_test_t).all()),
        ("no feature count inflation (30 in, 30 out)", X_train_t.shape[1] == len(FEATURE_COLUMNS)),
    ]

    # The leakage test: every fitted StandardScaler must have seen exactly the training rows.
    scalers = [
        step
        for _, transformer, _ in preprocessor.transformers_
        for step in (
            transformer.named_steps.values() if isinstance(transformer, Pipeline) else [transformer]
        )
        if isinstance(step, StandardScaler)
    ]
    checks.append(("at least one scaler was fitted", len(scalers) > 0))
    checks.append(
        (
            "every scaler saw exactly len(X_train) samples (no test rows)",
            all(int(np.max(s.n_samples_seen_)) == split.n_train for s in scalers),
        )
    )
    return checks


def validate_resampling(
    y_train_original: pd.Series,
    y_train_resampled,
    y_test_before,
    y_test_after,
) -> list[tuple[str, bool]]:
    """Checks that resampling touched the training data only."""
    before = pd.Series(y_test_before).value_counts().sort_index()
    after = pd.Series(y_test_after).value_counts().sort_index()
    resampled = pd.Series(y_train_resampled).value_counts().sort_index()
    original = y_train_original.value_counts().sort_index()

    return [
        ("test row count unchanged by resampling", len(y_test_before) == len(y_test_after)),
        ("test class counts identical before and after", before.equals(after)),
        ("training distribution actually changed", not resampled.equals(original)),
        ("resampled training data still has both classes", set(pd.Series(y_train_resampled).unique()) == {0, 1}),
    ]
