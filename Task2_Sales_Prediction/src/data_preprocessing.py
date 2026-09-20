"""Reusable preprocessing for the Sales Prediction project.

This module is the single definition of how raw data becomes model-ready features. It is imported by the
Phase 2 notebook and is intended to be imported unchanged by later phases (training, validation, final
evaluation, persistence and prediction), so that every one of them sees exactly the same split and the same
transformation.

Design decisions, all of which follow from the Phase 1 audit of ``dataset/advertising.csv``:

* **No imputation.** The dataset has zero missing values. Adding an imputer would fit a statistic that is
  never used, and at prediction time it would silently invent a value for a genuinely absent input instead of
  reporting the problem. Missing or non-finite input is therefore validated and raised on, not filled.
* **No categorical encoding.** All three predictors are continuous ``float64``; there is nothing to encode.
* **No feature engineering.** The three advertising budgets are used as they are. No feature is derived from
  the target, which would be leakage.
* **All 200 observations retained.** The two unusually high ``Newspaper`` budgets found in Phase 1 are
  plausible advertising spends, not data errors. See ``OUTLIER_DECISION``.
* **Standardisation included.** See ``build_preprocessor`` for the justification.

The raw CSV is treated as read-only. Nothing in this module writes to ``dataset/``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

__all__ = [
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "EXPECTED_COLUMNS",
    "EXPECTED_ROWS",
    "RAW_DATASET_SHA256",
    "RANDOM_STATE",
    "TEST_SIZE",
    "OUTLIER_DECISION",
    "SplitData",
    "ModelingData",
    "project_root",
    "default_dataset_path",
    "sha256_of_file",
    "verify_raw_dataset_integrity",
    "load_raw_data",
    "validate_raw_dataframe",
    "split_features_target",
    "validate_features",
    "make_train_test_split",
    "build_preprocessor",
    "get_feature_names",
    "transform_features",
    "validate_transformed",
    "prepare_modeling_data",
]

# --------------------------------------------------------------------------------------------------
# Dataset contract — established and verified in Phase 1 (notebooks/01_dataset_audit.ipynb)
# --------------------------------------------------------------------------------------------------

RAW_DATASET_FILENAME = "advertising.csv"

#: Advertising budget per platform. These are the only model inputs.
FEATURE_COLUMNS: tuple[str, ...] = ("TV", "Radio", "Newspaper")

#: The quantity to predict.
TARGET_COLUMN = "Sales"

#: Every column the raw file is expected to contain, in file order.
EXPECTED_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS + (TARGET_COLUMN,)

EXPECTED_ROWS = 200

#: SHA-256 of the raw file as downloaded on 2026-09-20 from the Kaggle dataset
#: ``ashydv/advertising-dataset``, which is what the CodSoft task links to.
RAW_DATASET_SHA256 = "137f755ad6fd3bc6471085f7631a2fba6c04cb8acd71eaf5cc9af6839d43fdd5"

#: Fixed so that every phase evaluates on an identical hold-out set.
RANDOM_STATE = 42

#: 80 / 20 split -> 160 training rows, 40 test rows.
TEST_SIZE = 0.2

OUTLIER_DECISION = (
    "All 200 observations are retained. Phase 1 flagged exactly two rows (1.0%) as unusual, both of them "
    "high Newspaper budgets (114.0 and 100.9). They are statistically unusual, not invalid: both sit in a "
    "realistic range for an advertising budget, neither is negative, a sentinel code or a duplicate, and "
    "nothing in the data indicates a recording error. Statistical unusualness alone is not evidence that an "
    "observation is wrong, and with only 200 rows, discarding real data costs more than it gains. They are "
    "therefore neither dropped, winsorised, clipped nor replaced. Whether they exert undue influence on a "
    "fitted model is a question for model diagnostics in a later phase, not a preprocessing decision."
)


# --------------------------------------------------------------------------------------------------
# Paths and integrity
# --------------------------------------------------------------------------------------------------


def project_root() -> Path:
    """Return the ``Task2_Sales_Prediction`` directory, resolved from this file's location.

    Resolving from ``__file__`` rather than the working directory keeps callers working whether they run
    from the project root, from ``notebooks/`` or from anywhere else in the repository.
    """
    return Path(__file__).resolve().parent.parent


def default_dataset_path() -> Path:
    """Return the path of the raw dataset, ``dataset/advertising.csv``."""
    return project_root() / "dataset" / RAW_DATASET_FILENAME


def sha256_of_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of the file at ``path``."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_raw_dataset_integrity(path: str | Path | None = None) -> str:
    """Check that the raw dataset is byte-for-byte unchanged and return its checksum.

    Raises:
        FileNotFoundError: if the dataset is missing.
        ValueError: if the checksum differs from :data:`RAW_DATASET_SHA256`. The file is never repaired or
            overwritten — a mismatch is reported so the cause can be investigated.
    """
    path = Path(path) if path is not None else default_dataset_path()
    if not path.is_file():
        raise FileNotFoundError(f"Raw dataset not found at {path}")

    actual = sha256_of_file(path)
    if actual != RAW_DATASET_SHA256:
        raise ValueError(
            f"Raw dataset checksum mismatch for {path}.\n"
            f"  expected: {RAW_DATASET_SHA256}\n"
            f"  actual  : {actual}\n"
            "The raw file must remain unchanged. Investigate the cause; do not overwrite or repair it."
        )
    return actual


# --------------------------------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------------------------------


def validate_raw_dataframe(df: pd.DataFrame, *, expect_row_count: bool = True) -> None:
    """Validate a freshly loaded raw DataFrame against the Phase 1 dataset contract.

    Checks the presence of every expected column, the absence of unexpected columns, numeric dtypes,
    the presence of the target, missing values, non-finite values and (optionally) the row count.

    Raises:
        ValueError: with a message naming the specific failure.
    """
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Raw dataset is missing expected column(s): {missing_cols}. "
            f"Expected exactly {list(EXPECTED_COLUMNS)}, found {list(df.columns)}."
        )

    unexpected_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if unexpected_cols:
        raise ValueError(
            f"Raw dataset contains unexpected column(s): {unexpected_cols}. "
            f"Expected exactly {list(EXPECTED_COLUMNS)}."
        )

    if TARGET_COLUMN not in df.columns:  # defensive; covered by the check above
        raise ValueError(f"Target column {TARGET_COLUMN!r} is absent from the raw dataset.")

    non_numeric = [c for c in EXPECTED_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Column(s) expected to be numeric are not: {non_numeric}.")

    na_counts = df[list(EXPECTED_COLUMNS)].isna().sum()
    if int(na_counts.sum()) > 0:
        raise ValueError(
            "Raw dataset contains missing values, which contradicts the Phase 1 audit:\n"
            f"{na_counts[na_counts > 0].to_string()}"
        )

    non_finite = {
        c: int((~np.isfinite(df[c].to_numpy(dtype=float))).sum())
        for c in EXPECTED_COLUMNS
        if int((~np.isfinite(df[c].to_numpy(dtype=float))).sum()) > 0
    }
    if non_finite:
        raise ValueError(f"Raw dataset contains non-finite values (inf / -inf): {non_finite}.")

    if expect_row_count and len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Raw dataset has {len(df)} rows, expected {EXPECTED_ROWS}. "
            "The raw file may have been altered."
        )


def load_raw_data(path: str | Path | None = None, *, verify_checksum: bool = True) -> pd.DataFrame:
    """Load ``dataset/advertising.csv`` exactly as it is on disk and validate its schema.

    No value is altered: no dtype override, no ``na_values``, no renaming, no index column, no sorting. The
    returned DataFrame is a fresh object, so mutating it cannot affect the file or any other caller.

    Args:
        path: dataset location. Defaults to :func:`default_dataset_path`.
        verify_checksum: also assert the file is byte-for-byte the audited raw dataset. Disable only when
            deliberately loading a different file (e.g. a test fixture).

    Returns:
        The raw data as a ``pandas.DataFrame``.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the checksum or the schema does not match the expected contract.
    """
    path = Path(path) if path is not None else default_dataset_path()
    if not path.is_file():
        raise FileNotFoundError(f"Raw dataset not found at {path}")

    if verify_checksum:
        verify_raw_dataset_integrity(path)

    df = pd.read_csv(path)
    validate_raw_dataframe(df, expect_row_count=verify_checksum)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a raw DataFrame into predictors ``X`` and target ``y``.

    ``X`` contains only :data:`FEATURE_COLUMNS`, in that fixed order, so column order can never drift
    between training and prediction. The target is excluded from ``X`` by construction and is never used to
    build a predictor.

    Returns:
        ``(X, y)`` where ``X`` is a DataFrame of the three budgets and ``y`` is the ``Sales`` Series.
    """
    validate_raw_dataframe(df, expect_row_count=False)

    X = df[list(FEATURE_COLUMNS)].copy()
    y = df[TARGET_COLUMN].copy()

    if TARGET_COLUMN in X.columns:  # defensive; unreachable given the selection above
        raise ValueError(f"Target {TARGET_COLUMN!r} leaked into the feature matrix.")

    return X, y


def validate_features(X: pd.DataFrame, *, context: str = "X") -> None:
    """Validate a feature matrix before it is fitted or transformed.

    Checks exact column set and order, numeric dtypes, absence of the target, and absence of missing or
    non-finite values. Missing input is reported rather than imputed — see the module docstring.

    Raises:
        ValueError: with a message naming the specific failure.
    """
    if TARGET_COLUMN in X.columns:
        raise ValueError(f"{context} must not contain the target column {TARGET_COLUMN!r}.")

    if tuple(X.columns) != FEATURE_COLUMNS:
        raise ValueError(
            f"{context} has columns {list(X.columns)}, expected exactly {list(FEATURE_COLUMNS)} in order."
        )

    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(f"{context} column(s) are not numeric: {non_numeric}.")

    na_counts = X.isna().sum()
    if int(na_counts.sum()) > 0:
        raise ValueError(
            f"{context} contains missing values. This pipeline deliberately does not impute: the audited "
            "dataset has none, so a missing input signals a real problem that should be reported rather "
            f"than filled in.\n{na_counts[na_counts > 0].to_string()}"
        )

    values = X.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{context} contains non-finite values (NaN / inf).")


# --------------------------------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitData:
    """The train/test split, kept together so later phases cannot accidentally mix parts of it."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    random_state: int
    test_size: float

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        return len(self.X_test)

    def summary(self) -> pd.DataFrame:
        """Return a small table of split sizes, convenient for display in a notebook."""
        total = self.n_train + self.n_test
        return pd.DataFrame(
            [
                {"Split": "train", "Rows": self.n_train, "Share": self.n_train / total},
                {"Split": "test", "Rows": self.n_test, "Share": self.n_test / total},
                {"Split": "total", "Rows": total, "Share": 1.0},
            ]
        )


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> SplitData:
    """Split predictors and target into training and test sets.

    The split happens **before** any learned preprocessing, so no statistic from the test rows can reach the
    transformer. The data is cross-sectional with no group or time column (Phase 1), so a plain random split
    is appropriate; no stratification applies to a continuous target.

    The same ``random_state`` always produces the same split, which is what lets every later phase evaluate
    on an identical hold-out set.
    """
    validate_features(X, context="X")

    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    if len(X_train) + len(X_test) != len(X):
        raise ValueError("Split sizes do not sum to the number of input rows.")
    if tuple(X_train.columns) != tuple(X_test.columns):
        raise ValueError("X_train and X_test have different predictor columns.")

    return SplitData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        random_state=random_state,
        test_size=test_size,
    )


# --------------------------------------------------------------------------------------------------
# Preprocessing pipeline
# --------------------------------------------------------------------------------------------------


def build_preprocessor(*, scale: bool = True) -> ColumnTransformer:
    """Build the (unfitted) preprocessing transformer for the three numeric predictors.

    Architecture::

        ColumnTransformer
          └── "numeric" -> Pipeline([("scaler", StandardScaler())]) on ["TV", "Radio", "Newspaper"]

    A ``ColumnTransformer`` keyed on explicit column names is used even though every column is numeric,
    because it pins the transformation to names rather than positions: a caller that supplies columns in a
    different order, or an extra column, is rejected rather than silently mis-transformed. ``remainder`` is
    left at its default ``"drop"``, so nothing outside :data:`FEATURE_COLUMNS` can ever reach a model.

    **Why standardisation.** Later phases compare model families with different needs. Ridge and Lasso
    penalise coefficients, so their results depend on predictor scale, and ``TV`` spans 0.7–296.4 while
    ``Radio`` spans 0–49.6 — without scaling the penalty would fall unevenly across the three budgets.
    Ordinary least squares is unaffected by a linear rescaling (its fitted values and metrics are identical
    either way), and tree-based models are invariant to monotone rescaling, so standardisation is harmless
    for them. One scaled pipeline shared by every model is therefore simpler and safer than maintaining two
    competing preprocessing paths, at no cost to the models that do not need it.

    ``StandardScaler`` learns only a mean and a standard deviation, both of them fitted on the training rows
    alone. It is a per-column linear map, so it cannot mix information between rows or between columns.

    Args:
        scale: set ``False`` to obtain the same validated column selection without standardisation. Provided
            for a deliberate comparison in a later phase; the project default is ``True``.

    Returns:
        An unfitted ``ColumnTransformer``. Fit it on ``X_train`` only.
    """
    steps: list[tuple[str, object]] = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    else:
        steps.append(("passthrough", "passthrough"))

    numeric_pipeline = Pipeline(steps=steps)

    return ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, list(FEATURE_COLUMNS))],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return the output feature names of a fitted preprocessor."""
    return list(preprocessor.get_feature_names_out())


def transform_features(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    *,
    context: str = "X",
) -> pd.DataFrame:
    """Apply an **already fitted** preprocessor to ``X`` and return a named DataFrame.

    This never fits. Calling it on test data, on a validation fold or on a single future observation applies
    exactly the parameters learned from the training rows.

    The original index is preserved so transformed rows can still be aligned with ``y``.
    """
    validate_features(X, context=context)

    transformed = preprocessor.transform(X)
    return pd.DataFrame(transformed, columns=get_feature_names(preprocessor), index=X.index)


def validate_transformed(
    transformed: pd.DataFrame,
    *,
    expected_rows: int | None = None,
    expected_features: Sequence[str] | None = None,
    context: str = "transformed data",
) -> None:
    """Validate a transformed feature matrix: shape, names, missing values and finiteness.

    Raises:
        ValueError: with a message naming the specific failure.
    """
    if expected_rows is not None and len(transformed) != expected_rows:
        raise ValueError(f"{context} has {len(transformed)} rows, expected {expected_rows}.")

    if expected_features is not None and list(transformed.columns) != list(expected_features):
        raise ValueError(
            f"{context} has features {list(transformed.columns)}, expected {list(expected_features)}."
        )

    na_total = int(transformed.isna().sum().sum())
    if na_total:
        raise ValueError(f"{context} contains {na_total} missing value(s).")

    if not np.isfinite(transformed.to_numpy(dtype=float)).all():
        raise ValueError(f"{context} contains non-finite values.")

    if TARGET_COLUMN in transformed.columns:
        raise ValueError(f"{context} contains the target column {TARGET_COLUMN!r}.")


# --------------------------------------------------------------------------------------------------
# End-to-end preparation
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelingData:
    """Everything a later phase needs to train and evaluate, prepared without leakage.

    ``preprocessor`` is fitted on ``X_train`` only. ``X_train_prepared`` and ``X_test_prepared`` were both
    produced by that one fitted object.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_prepared: pd.DataFrame
    X_test_prepared: pd.DataFrame
    preprocessor: ColumnTransformer
    feature_names: list[str]
    random_state: int
    test_size: float

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        return len(self.X_test)


def prepare_modeling_data(
    path: str | Path | None = None,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    scale: bool = True,
    verify_checksum: bool = True,
) -> ModelingData:
    """Run the full Phase 2 data flow and return model-ready data.

    Flow::

        advertising.csv -> validate -> split X / y -> train/test split
                        -> fit preprocessor on X_train ONLY
                        -> transform X_train -> transform X_test

    The preprocessor is fitted strictly after the split and strictly on the training rows, so no information
    from the test set influences the transformation. The returned ``preprocessor`` is the same fitted object
    used for both matrices and should be reused — never refitted — for validation folds and future
    predictions.

    No model is trained here.
    """
    df = load_raw_data(path, verify_checksum=verify_checksum)
    X, y = split_features_target(df)
    split = make_train_test_split(X, y, test_size=test_size, random_state=random_state)

    preprocessor = build_preprocessor(scale=scale)
    preprocessor.fit(split.X_train)  # fitted on training rows only

    feature_names = get_feature_names(preprocessor)
    X_train_prepared = transform_features(preprocessor, split.X_train, context="X_train")
    X_test_prepared = transform_features(preprocessor, split.X_test, context="X_test")

    validate_transformed(
        X_train_prepared,
        expected_rows=split.n_train,
        expected_features=feature_names,
        context="X_train_prepared",
    )
    validate_transformed(
        X_test_prepared,
        expected_rows=split.n_test,
        expected_features=feature_names,
        context="X_test_prepared",
    )

    return ModelingData(
        X_train=split.X_train,
        X_test=split.X_test,
        y_train=split.y_train,
        y_test=split.y_test,
        X_train_prepared=X_train_prepared,
        X_test_prepared=X_test_prepared,
        preprocessor=preprocessor,
        feature_names=feature_names,
        random_state=split.random_state,
        test_size=split.test_size,
    )
