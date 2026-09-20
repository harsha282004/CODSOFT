"""Prediction interface for the final Sales Prediction pipeline.

This module loads the persisted pipeline produced by Phase 5 and turns advertising budgets into a sales
prediction. It never trains, never refits and never modifies the artefact, and it does not need the original
dataset — everything required for prediction lives inside the saved pipeline.

Usage as a library::

    from predict import predict_sales
    predict_sales(tv=150.0, radio=25.0, newspaper=30.0)   # -> 15.87

Usage from the command line::

    python src/predict.py --tv 150 --radio 25 --newspaper 30
    python src/predict.py --smoke-test

The artefact stores the **whole** pipeline — the Phase 2 `ColumnTransformer` (with its `StandardScaler`
fitted on the 160 training rows) followed by the tuned `GradientBoostingRegressor`. Calling `predict` on it
applies exactly the transformation used at training time, so no preprocessing has to be reproduced here.

A caution that belongs with every number this module returns: the model was fitted on 160 observations of a
200-row benchmark dataset. Its predictions are estimates from a small sample and should not be treated as
reliable forecasts for advertising campaigns unlike those in that data.
"""

from __future__ import annotations

import argparse
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import pandas as pd

__all__ = [
    "DEFAULT_MODEL_PATH",
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "ModelArtifactError",
    "load_artifact",
    "model_metadata",
    "training_ranges",
    "predict_sales",
    "predict_batch",
]

#: Order matters: the pipeline expects these three columns, with these names.
FEATURE_COLUMNS: tuple[str, ...] = ("TV", "Radio", "Newspaper")

TARGET_COLUMN = "Sales"

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "final_sales_prediction_pipeline.joblib"


class ModelArtifactError(RuntimeError):
    """Raised when the persisted pipeline is missing or not in the expected form."""


# --------------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _load_cached(path_str: str) -> tuple[Any, dict]:
    path = Path(path_str)
    if not path.is_file():
        raise ModelArtifactError(
            f"Model artefact not found at {path}.\n"
            "Run notebooks/05_final_evaluation_persistence.ipynb to create it."
        )

    try:
        payload = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - surfaced with context below
        raise ModelArtifactError(f"Could not load the model artefact at {path}: {exc}") from exc

    # Phase 5 saves {"pipeline": ..., "metadata": {...}}. A bare pipeline is also accepted so that the
    # module keeps working if the artefact is ever re-saved without the metadata wrapper.
    if isinstance(payload, Mapping) and "pipeline" in payload:
        pipeline = payload["pipeline"]
        metadata = dict(payload.get("metadata", {}))
    else:
        pipeline = payload
        metadata = {}

    if not hasattr(pipeline, "predict"):
        raise ModelArtifactError(
            f"The object loaded from {path} has no predict() method; it is not a usable pipeline."
        )

    return pipeline, metadata


def load_artifact(model_path: str | Path | None = None) -> tuple[Any, dict]:
    """Load the persisted pipeline and its metadata.

    The result is cached per path, so repeated predictions do not re-read the file. The artefact is opened
    read-only and never written back.

    Returns:
        ``(pipeline, metadata)``.

    Raises:
        ModelArtifactError: if the file is missing, unreadable, or does not contain a usable pipeline.
    """
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    return _load_cached(str(path.resolve()))


def model_metadata(model_path: str | Path | None = None) -> dict:
    """Return the metadata recorded alongside the pipeline when it was persisted."""
    return dict(load_artifact(model_path)[1])


def training_ranges(model_path: str | Path | None = None) -> dict[str, tuple[float, float]]:
    """Return the min/max of each predictor in the training data, if the artefact recorded them.

    Useful for flagging inputs that fall outside the range the model actually saw. Returns an empty dict
    when the artefact carries no such record.
    """
    ranges = model_metadata(model_path).get("training_feature_ranges", {})
    return {k: (float(v[0]), float(v[1])) for k, v in ranges.items()}


# --------------------------------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------------------------------


def _validate_spend(value: Any, name: str) -> float:
    """Coerce one advertising budget to a finite, non-negative float, or raise a clear error.

    Rejects ``None``, NaN, infinities, booleans and anything non-numeric. Numeric strings such as ``"150"``
    are accepted for CLI convenience. Negative budgets are rejected because a negative advertising spend is
    not a meaningful input, not merely an unusual one.
    """
    if value is None:
        raise ValueError(f"{name} is required but was None. Provide a numeric advertising budget.")

    # bool is a subclass of int; True/False are almost certainly a mistake here.
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got a boolean ({value!r}).")

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{name} is required but was an empty string.")
        try:
            value = float(stripped)
        except ValueError:
            raise TypeError(f"{name} must be numeric, got the non-numeric string {value!r}.") from None

    try:
        number = float(value)
    except (TypeError, ValueError):
        raise TypeError(
            f"{name} must be a number, got {type(value).__name__} ({value!r})."
        ) from None

    if math.isnan(number):
        raise ValueError(
            f"{name} is missing (NaN). This model does not impute missing budgets — supply a value or "
            "handle the gap before calling."
        )
    if math.isinf(number):
        raise ValueError(f"{name} must be finite, got {number}.")
    if number < 0:
        raise ValueError(f"{name} must be zero or positive, got {number}. Budgets cannot be negative.")

    return number


def _as_frame(tv: float, radio: float, newspaper: float) -> pd.DataFrame:
    """Build the single-row frame the pipeline expects, with columns in the trained order."""
    return pd.DataFrame([[tv, radio, newspaper]], columns=list(FEATURE_COLUMNS))


# --------------------------------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------------------------------


def predict_sales(
    tv: float,
    radio: float,
    newspaper: float,
    *,
    model_path: str | Path | None = None,
) -> float:
    """Predict sales for one combination of advertising budgets.

    Args:
        tv: budget allocated to the TV platform.
        radio: budget allocated to the Radio platform.
        newspaper: budget allocated to the Newspaper platform.
        model_path: artefact to use. Defaults to ``models/final_sales_prediction_pipeline.joblib``.

    Returns:
        The predicted ``Sales`` value, in the same units as the training data.

    Raises:
        TypeError: if an input is not numeric.
        ValueError: if an input is missing, non-finite or negative.
        ModelArtifactError: if the persisted pipeline cannot be loaded.

    The three budgets are the only inputs. ``Sales`` is the quantity being predicted and is never read as an
    input.
    """
    values = {
        "TV": _validate_spend(tv, "TV"),
        "Radio": _validate_spend(radio, "Radio"),
        "Newspaper": _validate_spend(newspaper, "Newspaper"),
    }

    pipeline, _ = load_artifact(model_path)
    frame = _as_frame(values["TV"], values["Radio"], values["Newspaper"])
    return float(pipeline.predict(frame)[0])


def predict_batch(
    rows: Iterable[Mapping[str, Any]] | pd.DataFrame,
    *,
    model_path: str | Path | None = None,
) -> pd.DataFrame:
    """Predict sales for several budget combinations at once.

    Args:
        rows: a DataFrame with ``TV``/``Radio``/``Newspaper`` columns, or an iterable of mappings with those
            keys.
        model_path: artefact to use.

    Returns:
        A copy of the inputs with a ``Predicted Sales`` column appended. The input is not modified.

    Every value goes through the same validation as :func:`predict_sales`, so one bad cell raises rather
    than silently producing a number.
    """
    frame = pd.DataFrame(rows).copy() if not isinstance(rows, pd.DataFrame) else rows.copy()

    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Expected {list(FEATURE_COLUMNS)}.")

    if len(frame) == 0:
        raise ValueError("No rows to predict.")

    validated = pd.DataFrame(
        [
            {c: _validate_spend(row[c], f"{c} (row {idx})") for c in FEATURE_COLUMNS}
            for idx, row in frame.iterrows()
        ],
        index=frame.index,
        columns=list(FEATURE_COLUMNS),
    )

    pipeline, _ = load_artifact(model_path)
    frame["Predicted Sales"] = pipeline.predict(validated)
    return frame


# --------------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------------

#: Representative budget combinations for the smoke test. These are illustrative inputs chosen to span the
#: training ranges — they are NOT rows from the dataset and do not correspond to real observed campaigns.
SMOKE_TEST_CASES: tuple[tuple[str, float, float, float], ...] = (
    ("low spend across all platforms", 10.0, 5.0, 5.0),
    ("medium spend across all platforms", 150.0, 25.0, 30.0),
    ("high TV, low other", 290.0, 5.0, 5.0),
    ("high Radio, low other", 10.0, 48.0, 5.0),
    ("high Newspaper, low other", 10.0, 5.0, 110.0),
    ("high spend across all platforms", 290.0, 48.0, 110.0),
)


def _run_smoke_test(model_path: str | Path | None = None) -> int:
    """Predict the representative cases and print them. Returns a process exit code."""
    pipeline, metadata = load_artifact(model_path)

    print(f"artefact : {Path(model_path or DEFAULT_MODEL_PATH).name}")
    if metadata:
        print(f"model    : {metadata.get('model_type', 'unknown')}")
        print(f"trained  : {metadata.get('training_rows', '?')} rows, "
              f"scikit-learn {metadata.get('sklearn_version', '?')}")
    print()

    ranges = training_ranges(model_path)
    header = f"{'case':36s} {'TV':>7s} {'Radio':>7s} {'News':>7s} {'predicted Sales':>16s}"
    print(header)
    print("-" * len(header))

    for label, tv, radio, newspaper in SMOKE_TEST_CASES:
        prediction = predict_sales(tv, radio, newspaper, model_path=model_path)
        flag = ""
        if ranges:
            outside = [
                name for name, value in (("TV", tv), ("Radio", radio), ("Newspaper", newspaper))
                if name in ranges and not (ranges[name][0] <= value <= ranges[name][1])
            ]
            if outside:
                flag = f"  <- outside training range: {', '.join(outside)}"
        print(f"{label:36s} {tv:7.1f} {radio:7.1f} {newspaper:7.1f} {prediction:16.4f}{flag}")

    print()
    print("These are illustrative budget combinations, not rows from the dataset, and the predictions are")
    print("estimates from a model fitted to 160 observations of a 200-row benchmark dataset.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description="Predict sales from TV, Radio and Newspaper advertising budgets.",
    )
    parser.add_argument("--tv", type=str, help="TV advertising budget")
    parser.add_argument("--radio", type=str, help="Radio advertising budget")
    parser.add_argument("--newspaper", type=str, help="Newspaper advertising budget")
    parser.add_argument("--model-path", type=str, default=None,
                        help="path to the persisted pipeline (defaults to models/…joblib)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="predict a set of representative budget combinations and exit")
    args = parser.parse_args(argv)

    try:
        if args.smoke_test:
            return _run_smoke_test(args.model_path)

        if args.tv is None or args.radio is None or args.newspaper is None:
            parser.error("--tv, --radio and --newspaper are all required (or use --smoke-test)")

        prediction = predict_sales(args.tv, args.radio, args.newspaper, model_path=args.model_path)
        print(f"{prediction:.4f}")
        return 0

    except (ModelArtifactError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
