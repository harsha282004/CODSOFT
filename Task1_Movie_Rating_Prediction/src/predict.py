"""Reusable prediction interface for the Movie Rating Prediction model.

The saved artifact holds the **fitted preprocessing transformer and the fitted model together**, so
callers never reproduce the preprocessing themselves:

    raw movie fields -> fitted transformer -> feature matrix -> fitted model -> predicted rating

Typical use:

    from predict import predict_rating, predict_movies

    predict_rating({"Name": "Example", "Year": "(2024)", "Duration": "120 min",
                    "Genre": "Drama, Thriller", "Director": "Some Director",
                    "Actor 1": "A", "Actor 2": "B", "Actor 3": "C"})

    predict_movies(dataframe_of_movies)   # adds a "Predicted Rating" column

The model configuration is the one locked in Phase 5; this module only loads and applies it.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
MODEL_FILENAME = "final_movie_rating_pipeline.joblib"
MODEL_PATH = PROJECT_ROOT / "models" / MODEL_FILENAME

# Unpickling the artifact needs the transformer classes importable under their original module name
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import data_preprocessing as dp  # noqa: E402  (import after sys.path setup)

#: Fields accepted from the caller. Only the model-feature ones reach the model.
INPUT_FIELDS = ("Name", "Year", "Duration", "Genre", "Director", "Actor 1", "Actor 2", "Actor 3")
#: Year is required: it is the single most influential feature. Everything else may be missing.
REQUIRED_FIELDS = ("Year",)
#: Never used for prediction, even when supplied.
IGNORED_FIELDS = ("Rating", "Votes")
#: Rating range observed in the training data (for reference; predictions are NOT clipped to it).
OBSERVED_RATING_RANGE = (1.1, 10.0)

_CACHE: dict[Path, dict] = {}


class ArtifactError(RuntimeError):
    """Raised when the saved pipeline cannot be found or is not usable."""


def load_pipeline(path: str | Path | None = None, use_cache: bool = True) -> dict:
    """Load the saved artifact.

    Returns a dict with:
      - ``pipeline``: fitted sklearn Pipeline (preprocessing + model)
      - ``metadata``: configuration and provenance information

    The path is resolved relative to the project, not the current working directory.
    """
    path = Path(path) if path is not None else MODEL_PATH
    if use_cache and path in _CACHE:
        return _CACHE[path]
    if not path.exists():
        raise ArtifactError(f"Model artifact not found at {path}. Run notebooks/06_model_persistence.ipynb to create it.")

    artifact = joblib.load(path)
    for key in ("pipeline", "metadata"):
        if key not in artifact:
            raise ArtifactError(f"Artifact at {path} is missing the '{key}' entry")

    trained_with = artifact["metadata"].get("sklearn_version")
    import sklearn
    if trained_with and trained_with != sklearn.__version__:
        warnings.warn(
            f"Artifact was created with scikit-learn {trained_with}, this environment has "
            f"{sklearn.__version__}. Predictions may differ.", RuntimeWarning, stacklevel=2)

    if use_cache:
        _CACHE[path] = artifact
    return artifact


def _to_year(value):
    """Accept '(2024)', '2024', 2024 or 2024.0; anything unparseable becomes missing."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    if not text.startswith("("):
        text = f"({text.split('.')[0]})"   # canonical form expected by the Phase 2 parser
    return dp.parse_year(pd.Series([text], dtype="object")).iloc[0]


def _to_duration(value):
    """Accept '120 min', '120', 120 or 120.0; anything unparseable becomes missing."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    if not text.endswith("min"):
        text = f"{text.split('.')[0]} min"
    return dp.parse_duration(pd.Series([text], dtype="object")).iloc[0]


def _to_text(value):
    """Normalise a text field; blank or missing becomes NA so the pipeline's missing handling applies."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return pd.NA
    text = str(value).strip()
    return text if text else pd.NA


def prepare_input(movie_data) -> pd.DataFrame:
    """Turn user input into the feature frame the fitted transformer expects.

    Accepts a dict, a list of dicts, or a DataFrame. Unknown columns are ignored, and `Rating` and
    `Votes` are dropped even when supplied. Missing optional fields stay missing and are handled by
    the pipeline exactly as during training; no values are invented.
    """
    if isinstance(movie_data, pd.DataFrame):
        frame = movie_data.copy()
    elif isinstance(movie_data, dict):
        frame = pd.DataFrame([movie_data])
    elif isinstance(movie_data, (list, tuple)):
        if not all(isinstance(m, dict) for m in movie_data):
            raise TypeError("A sequence input must contain dictionaries")
        frame = pd.DataFrame(list(movie_data))
    else:
        raise TypeError(f"Unsupported input type: {type(movie_data).__name__}. Pass a dict, list of dicts or DataFrame.")

    if frame.empty:
        raise ValueError("No movies supplied")

    missing_required = [f for f in REQUIRED_FIELDS if f not in frame.columns]
    if missing_required:
        raise ValueError(f"Missing required field(s): {missing_required}. Required: {list(REQUIRED_FIELDS)}")

    out = pd.DataFrame(index=frame.index)
    out["Year"] = [_to_year(v) for v in frame["Year"]] if "Year" in frame else pd.NA
    out["Duration_min"] = [_to_duration(v) for v in frame["Duration"]] if "Duration" in frame else pd.NA
    for col in ("Genre", "Director", "Actor 1", "Actor 2", "Actor 3"):
        out[col] = [_to_text(v) for v in frame[col]] if col in frame else pd.NA

    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Float64")
    out["Duration_min"] = pd.to_numeric(out["Duration_min"], errors="coerce").astype("Float64")

    for field in REQUIRED_FIELDS:
        column = "Duration_min" if field == "Duration" else field
        if out[column].isna().all():
            raise ValueError(f"Required field '{field}' has no usable value in any supplied row")

    for col in ("Genre", "Director", "Actor 1", "Actor 2", "Actor 3"):
        out[col] = out[col].astype("object")
    return out[list(dp.FEATURE_COLUMNS)]


def predict_movies(movie_data, path: str | Path | None = None, column: str = "Predicted Rating") -> pd.DataFrame:
    """Predict ratings for one or more movies.

    Returns a copy of the input (as a DataFrame) with the predicted rating added. `Rating` is never
    required, and `Votes` is never used. Predictions are the raw model output and are **not** clipped
    to the observed training range.
    """
    artifact = load_pipeline(path)
    features = prepare_input(movie_data)
    predictions = artifact["pipeline"].predict(features)

    if isinstance(movie_data, pd.DataFrame):
        result = movie_data.copy()
    elif isinstance(movie_data, dict):
        result = pd.DataFrame([movie_data])
    else:
        result = pd.DataFrame(list(movie_data))
    result[column] = np.asarray(predictions, dtype=float)
    return result


def predict_rating(movie_data, path: str | Path | None = None) -> float:
    """Predict the rating of a single movie and return it as a float.

    The value is the raw model output on the IMDb 1–10 scale. It is a statistical estimate from
    limited metadata, not a claim about how a film will actually be rated.
    """
    features = prepare_input(movie_data)
    if len(features) != 1:
        raise ValueError(f"predict_rating expects one movie, got {len(features)}. Use predict_movies for batches.")
    artifact = load_pipeline(path)
    return float(artifact["pipeline"].predict(features)[0])


def describe_artifact(path: str | Path | None = None) -> pd.Series:
    """Return the artifact's metadata as a Series, for quick inspection."""
    return pd.Series(load_pipeline(path)["metadata"], name="value")


if __name__ == "__main__":  # small manual check: python src/predict.py
    example = {"Name": "Pipeline smoke-test movie", "Year": "(2024)", "Duration": "120 min",
               "Genre": "Drama, Thriller", "Director": "Unknown Director",
               "Actor 1": "Unknown Actor", "Actor 2": None, "Actor 3": None}
    print(describe_artifact().to_string())
    print(f"\nPredicted rating for a smoke-test input: {predict_rating(example):.3f}")
