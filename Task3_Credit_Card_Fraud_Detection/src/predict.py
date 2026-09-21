"""Prediction interface for the locked Credit Card Fraud Detection model.

This module loads the artefact produced by Phase 5 (``models/final_credit_card_fraud_pipeline.joblib``)
and scores transactions with it. It never trains, refits or modifies the artefact, and it does not need the
dataset: everything required for prediction lives inside the saved pipeline.

Usage as a library::

    from predict import predict_transaction
    result = predict_transaction({"Time": 406.0, "V1": -2.31, ..., "V28": -0.14, "Amount": 0.0})
    # {"fraud_probability": 0.83, "predicted_class": 1, "label": "FRAUD", "threshold": 0.5}

Usage from the command line::

    python src/predict.py --input transaction.json        # one JSON object with the 30 features
    python src/predict.py --json '{"Time": 0, "V1": ...}'
    python src/predict.py --smoke-test

What the artefact contains
--------------------------
A dictionary with the whole scikit-learn/imbalanced-learn pipeline (the Phase 2 ``StandardScaler`` fitted on
the 226,980 training rows, followed by the Phase 4 random forest), the **locked decision threshold**, the
expected feature order and provenance metadata. The threshold is read from the artefact - it is not
hard-coded here - so the prediction rule cannot drift away from what Phase 5 evaluated.

Decision rule: a transaction is classified as fraud when ``P(fraud) >= threshold``. That is the rule the
threshold was selected under in Phase 4 (Rule 3, maximum out-of-fold F1).

A caution that belongs with every result: the model was trained on two days of anonymised European card
transactions from September 2013. A fraud probability is a model score, not proof that a transaction is
fraudulent, and it is not a production fraud decision.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

# The pickled forest class lives in src/model_training.py; make it importable however this module is
# loaded (as ``predict``, as ``src.predict`` or as a script).
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

__all__ = [
    "FEATURE_COLUMNS",
    "DEFAULT_MODEL_PATH",
    "ModelArtifactError",
    "InvalidTransactionError",
    "load_artifact",
    "model_threshold",
    "model_metadata",
    "validate_transaction",
    "predict_transaction",
    "predict_batch",
]

#: The 30 predictors, in the order the pipeline was trained on.
FEATURE_COLUMNS: tuple[str, ...] = ("Time", *(f"V{i}" for i in range(1, 29)), "Amount")

DEFAULT_MODEL_PATH = _SRC_DIR.parent / "models" / "final_credit_card_fraud_pipeline.joblib"


class ModelArtifactError(RuntimeError):
    """The persisted model is missing, unreadable or not in the expected form."""


class InvalidTransactionError(ValueError):
    """The input is not a valid transaction. The message says exactly what is wrong."""


# --------------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _load_cached(path_str: str) -> dict:
    path = Path(path_str)
    if not path.is_file():
        raise ModelArtifactError(
            f"Model artefact not found at {path}. "
            "Run notebooks/05_final_model_evaluation.ipynb to create it."
        )
    try:
        payload = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise ModelArtifactError(f"Could not load the model artefact at {path}: {exc}") from exc

    if not isinstance(payload, Mapping) or not {"pipeline", "threshold", "feature_columns"} <= set(payload):
        raise ModelArtifactError(
            f"{path} is not a Phase 5 artefact: expected a dictionary with 'pipeline', 'threshold' and "
            "'feature_columns'."
        )
    pipeline = payload["pipeline"]
    if not hasattr(pipeline, "predict_proba"):
        raise ModelArtifactError(f"The pipeline in {path} has no predict_proba(); it cannot score fraud.")
    if tuple(payload["feature_columns"]) != FEATURE_COLUMNS:
        raise ModelArtifactError(f"The artefact at {path} expects a different feature layout.")
    threshold = float(payload["threshold"])
    if not 0.0 < threshold < 1.0:
        raise ModelArtifactError(f"The artefact at {path} stores an invalid threshold: {threshold!r}.")
    return dict(payload)


def load_artifact(model_path: str | Path | None = None) -> dict:
    """Load (and cache) the Phase 5 artefact. It is opened read-only and never written back."""
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    return _load_cached(str(path.resolve()))


def model_threshold(model_path: str | Path | None = None) -> float:
    """The locked decision threshold stored in the artefact."""
    return float(load_artifact(model_path)["threshold"])


def model_metadata(model_path: str | Path | None = None) -> dict:
    """Provenance and configuration recorded when the artefact was created."""
    return dict(load_artifact(model_path).get("metadata", {}))


# --------------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------------


def _to_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or value is None:
        raise InvalidTransactionError(f"'{name}' must be a number, got {value!r}.")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise InvalidTransactionError(f"'{name}' is empty; a numeric value is required.")
        try:
            value = float(text)
        except ValueError:
            raise InvalidTransactionError(f"'{name}' must be a number, got {value!r}.") from None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise InvalidTransactionError(f"'{name}' must be a number, got {value!r}.") from None
    if math.isnan(number):
        raise InvalidTransactionError(f"'{name}' is NaN; a real numeric value is required.")
    if math.isinf(number):
        raise InvalidTransactionError(f"'{name}' is infinite; a finite numeric value is required.")
    return number


def validate_transaction(transaction: Mapping[str, Any] | Sequence[Any] | pd.Series) -> pd.DataFrame:
    """Check one transaction and return it as a one-row frame in training column order.

    Accepts a mapping (or ``pandas.Series``) keyed by feature name, or a plain sequence of exactly 30
    values in the order of ``FEATURE_COLUMNS``. Raises ``InvalidTransactionError`` with a specific message
    for missing or unexpected features, a wrong feature count, non-numeric values, NaN or infinity, and a
    negative ``Amount`` or ``Time`` (the dataset contains neither).
    """
    if isinstance(transaction, pd.Series):
        transaction = transaction.to_dict()

    if isinstance(transaction, Mapping):
        keys = [str(k) for k in transaction]
        missing = [c for c in FEATURE_COLUMNS if c not in transaction]
        unexpected = [k for k in keys if k not in FEATURE_COLUMNS]
        if missing:
            raise InvalidTransactionError(
                f"Missing {len(missing)} feature(s): {', '.join(missing[:10])}"
                + (" ..." if len(missing) > 10 else "") + f". All 30 are required: Time, V1-V28, Amount."
            )
        if unexpected:
            raise InvalidTransactionError(
                f"Unexpected feature(s): {', '.join(unexpected[:10])}. Only Time, V1-V28 and Amount are accepted "
                "(the target column 'Class' must not be supplied)."
            )
        values = {c: _to_float(transaction[c], c) for c in FEATURE_COLUMNS}
    elif isinstance(transaction, (str, bytes)) or not isinstance(transaction, Sequence) and not isinstance(
        transaction, np.ndarray
    ):
        raise InvalidTransactionError(
            "A transaction must be a mapping of feature name to value, or a sequence of 30 numbers "
            f"(got {type(transaction).__name__})."
        )
    else:
        seq = list(np.asarray(transaction, dtype=object).ravel())
        if len(seq) != len(FEATURE_COLUMNS):
            raise InvalidTransactionError(
                f"Expected {len(FEATURE_COLUMNS)} values (Time, V1-V28, Amount), got {len(seq)}."
            )
        values = {c: _to_float(v, c) for c, v in zip(FEATURE_COLUMNS, seq)}

    if values["Amount"] < 0:
        raise InvalidTransactionError("'Amount' cannot be negative.")
    if values["Time"] < 0:
        raise InvalidTransactionError("'Time' (seconds since the first transaction) cannot be negative.")
    return pd.DataFrame([values], columns=list(FEATURE_COLUMNS))


# --------------------------------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------------------------------


def _decide(probability: float, threshold: float) -> tuple[int, str]:
    predicted = int(probability >= threshold)
    return predicted, ("FRAUD" if predicted == 1 else "LEGITIMATE")


def predict_transaction(
    transaction: Mapping[str, Any] | Sequence[Any] | pd.Series,
    model_path: str | Path | None = None,
) -> dict:
    """Score one transaction with the locked pipeline and the locked threshold.

    Returns ``{"fraud_probability", "predicted_class", "label", "threshold"}``.
    """
    frame = validate_transaction(transaction)
    artifact = load_artifact(model_path)
    probability = float(artifact["pipeline"].predict_proba(frame)[0, 1])
    threshold = float(artifact["threshold"])
    predicted, label = _decide(probability, threshold)
    return {"fraud_probability": probability, "predicted_class": predicted, "label": label, "threshold": threshold}


def predict_batch(frame: pd.DataFrame, model_path: str | Path | None = None) -> pd.DataFrame:
    """Score many transactions at once. Returns probability, class and label per row (index preserved).

    The frame must contain the 30 feature columns; any other column (including ``Class``) is rejected so
    the target can never be passed to the model by accident.
    """
    if not isinstance(frame, pd.DataFrame):
        raise InvalidTransactionError("predict_batch expects a pandas DataFrame.")
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    unexpected = [c for c in frame.columns if c not in FEATURE_COLUMNS]
    if missing:
        raise InvalidTransactionError(f"Missing column(s): {', '.join(missing)}.")
    if unexpected:
        raise InvalidTransactionError(f"Unexpected column(s): {', '.join(map(str, unexpected))}.")
    data = frame[list(FEATURE_COLUMNS)]
    try:
        numeric = data.astype(float)
    except (TypeError, ValueError) as exc:
        raise InvalidTransactionError(f"All feature values must be numeric: {exc}") from None
    if not np.isfinite(numeric.to_numpy()).all():
        raise InvalidTransactionError("The frame contains NaN or infinite values.")

    artifact = load_artifact(model_path)
    threshold = float(artifact["threshold"])
    probability = artifact["pipeline"].predict_proba(numeric)[:, 1]
    predicted = (probability >= threshold).astype(int)
    return pd.DataFrame(
        {"fraud_probability": probability, "predicted_class": predicted,
         "label": np.where(predicted == 1, "FRAUD", "LEGITIMATE")},
        index=frame.index,
    )


# --------------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------------

_DEMO_EXAMPLES = _SRC_DIR.parent / "results" / "phase5_demo_examples.json"


def _run_smoke_test(model_path: str | Path | None = None) -> int:
    """Valid and invalid inputs end to end. Returns a process exit code (0 = all passed)."""
    checks: list[tuple[str, bool]] = []
    artifact = load_artifact(model_path)
    checks.append(("artefact loads with a threshold", 0 < float(artifact["threshold"]) < 1))

    if _DEMO_EXAMPLES.is_file():
        examples = json.loads(_DEMO_EXAMPLES.read_text(encoding="utf-8"))["examples"]
        for ex in examples:
            result = predict_transaction(ex["features"], model_path)
            checks.append((f"example '{ex['name']}' scores as recorded in Phase 5",
                           abs(result["fraud_probability"] - ex["fraud_probability"]) <= 1e-12))
    else:
        checks.append(("demo examples available (results/phase5_demo_examples.json)", False))

    valid = {c: 0.0 for c in FEATURE_COLUMNS}
    bad_inputs = {
        "missing feature": {k: v for k, v in valid.items() if k != "V7"},
        "NaN value": {**valid, "V3": float("nan")},
        "infinite value": {**valid, "Amount": float("inf")},
        "wrong feature count": [0.0] * 29,
        "malformed value": {**valid, "V1": "abc"},
        "target column supplied": {**valid, "Class": 1},
        "not a transaction": "hello",
    }
    for name, bad in bad_inputs.items():
        try:
            predict_transaction(bad, model_path)
            checks.append((f"rejects {name}", False))
        except InvalidTransactionError:
            checks.append((f"rejects {name}", True))

    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    failed = sum(not ok for _, ok in checks)
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 0 if failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a credit card transaction with the locked Phase 5 model.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="path to a JSON file holding one transaction object")
    group.add_argument("--json", help="one transaction as a JSON object string")
    group.add_argument("--smoke-test", action="store_true", help="run the end-to-end input/prediction checks")
    parser.add_argument("--model", type=Path, default=None, help="alternative artefact path")
    args = parser.parse_args(argv)

    try:
        if args.smoke_test:
            return _run_smoke_test(args.model)
        raw = args.input.read_text(encoding="utf-8") if args.input else args.json
        try:
            transaction = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise InvalidTransactionError(f"Input is not valid JSON: {exc}") from None
        print(json.dumps(predict_transaction(transaction, args.model), indent=2))
        return 0
    except (InvalidTransactionError, ModelArtifactError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
