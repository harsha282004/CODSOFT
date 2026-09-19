"""Data cleaning and leakage-safe preprocessing for the Movie Rating Prediction project.

The module has two layers:

1. **Deterministic, row-wise cleaning** (`clean_movies`, `build_modeling_frame`).
   These functions only parse and validate each row on its own. They compute no
   statistics across rows, so they can safely be applied to the full dataset
   before any train/test split.

2. **Fitted transformers** (`NumericFeatureTransformer`, `GenreMultiHotEncoder`,
   `PeopleEncoder`, combined by `build_feature_transformer`). These learn values
   such as medians, genre vocabularies and name frequencies, so they must be
   fitted on training data only. None of them receives or uses `Rating`.

The raw CSV is only ever read; nothing in this module writes to it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted

RAW_ENCODING = "latin-1"  # the CSV is not valid UTF-8
RAW_COLUMNS = ["Name", "Year", "Duration", "Genre", "Rating", "Votes",
               "Director", "Actor 1", "Actor 2", "Actor 3"]
TARGET = "Rating"
ACTOR_COLUMNS = ["Actor 1", "Actor 2", "Actor 3"]
TEXT_COLUMNS = ["Name", "Genre", "Director", *ACTOR_COLUMNS]

# Columns passed to the feature transformer. Votes is deliberately excluded (see NON_FEATURE_COLUMNS).
FEATURE_COLUMNS = ["Year", "Duration_min", "Genre", "Director", *ACTOR_COLUMNS]
NON_FEATURE_COLUMNS = {
    "Name": "Title identifier, near-unique per film; not a descriptive attribute.",
    "Votes_eda": "Modeling assumption: accumulated audience activity may not be available when a "
                 "prediction is made. Kept for EDA and a possible later experiment only.",
    "Votes": "Raw vote text, kept unchanged for traceability; the numeric version is Votes_eda.",
    "Rating": "Target variable.",
}

YEAR_PATTERN = r"^\((\d{4})\)$"
DURATION_PATTERN = r"^(\d+) min$"
VOTES_PATTERN = r"^\d{1,3}(?:,\d{3})*$"

# Plausibility bounds used for validation only (values outside are reported, not altered)
MIN_PLAUSIBLE_YEAR = 1888  # earliest surviving motion picture
QUESTIONABLE_DURATION_MIN = 40  # below this, an entry is unlikely to be a feature-length film


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Read the raw IMDb India CSV without modifying it."""
    df = pd.read_csv(path, encoding=RAW_ENCODING)
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing expected columns: {sorted(missing)}")
    return df


# ---------------------------------------------------------------------------
# Row-wise parsing (no cross-row statistics)
# ---------------------------------------------------------------------------

def parse_year(year: pd.Series) -> pd.Series:
    """Convert '(2019)' to 2019. Values not matching the pattern become NaN."""
    return pd.to_numeric(year.str.extract(YEAR_PATTERN, expand=False), errors="coerce").astype("Float64")


def parse_duration(duration: pd.Series) -> pd.Series:
    """Convert '109 min' to 109. Values not matching the pattern become NaN."""
    return pd.to_numeric(duration.str.extract(DURATION_PATTERN, expand=False), errors="coerce").astype("Float64")


def parse_votes(votes: pd.Series) -> pd.Series:
    """Convert '25,732' to 25732.

    Only strings made of digits with optional thousands separators are accepted.
    Anything else (e.g. '$5.16M') becomes NaN instead of being guessed at.
    """
    valid = votes.str.fullmatch(VOTES_PATTERN).fillna(False).astype(bool)
    cleaned = votes.where(valid).str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce").astype("Float64")


def split_genres(genre: pd.Series) -> pd.Series:
    """Split 'Drama, Romance' into ['Drama', 'Romance'].

    Missing genres stay missing (NaN) so they can be handled explicitly downstream;
    no genre is invented for them.
    """
    def _split(value):
        if pd.isna(value):
            return np.nan
        tokens = [t.strip() for t in value.split(",") if t.strip()]
        return list(dict.fromkeys(tokens)) or np.nan  # de-duplicate, keep order

    return genre.map(_split)


def clean_movies(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic row-wise cleaning. Returns a new frame; `raw` is not modified.

    Adds parsed columns alongside the original ones:
    - Year (numeric, replaces the '(YYYY)' string), Year_raw keeps the original text
    - Duration_min (numeric minutes), duration_questionable (audit flag, not a feature)
    - Votes_eda (numeric, EDA only), votes_malformed (audit flag)
    - Genre_list (list of genre tokens)
    Text fields are stripped of surrounding whitespace; empty strings become NaN.
    """
    df = raw.copy()

    for col in TEXT_COLUMNS:
        stripped = df[col].str.strip()
        df[col] = stripped.mask(stripped == "")

    df["Year_raw"] = raw["Year"]
    df["Year"] = parse_year(raw["Year"])

    df["Duration_min"] = parse_duration(raw["Duration"])
    df["duration_questionable"] = (df["Duration_min"] < QUESTIONABLE_DURATION_MIN).fillna(False).astype(bool)

    df["Votes_eda"] = parse_votes(raw["Votes"])
    df["votes_malformed"] = (raw["Votes"].notna() & df["Votes_eda"].isna()).astype(bool)

    df["Genre_list"] = split_genres(df["Genre"])
    return df


def build_modeling_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Create the supervised modeling frame from the raw data.

    Steps (all row-wise or exact-match, no fitted statistics):
    1. clean_movies
    2. drop exact duplicate rows (identical across all raw columns), keeping the first
    3. keep only rows with a Rating (unrated rows cannot be training examples)

    Returns the frame (original row index preserved as `raw_index`) and a step-by-step row log.
    """
    log = {"raw_rows": len(raw)}
    cleaned = clean_movies(raw)

    exact_dupes = raw.duplicated(keep="first")
    log["exact_duplicates_removed"] = int(exact_dupes.sum())
    log["exact_duplicates_with_rating"] = int((exact_dupes & raw[TARGET].notna()).sum())
    cleaned = cleaned.loc[~exact_dupes]
    log["rows_after_dedup"] = len(cleaned)

    rated = cleaned[cleaned[TARGET].notna()].copy()
    log["unrated_rows_excluded"] = len(cleaned) - len(rated)

    rated["name_year_repeat"] = rated.duplicated(subset=["Name", "Year"], keep=False)
    rated.insert(0, "raw_index", rated.index)
    rated = rated.reset_index(drop=True)
    log["modeling_rows"] = len(rated)
    return rated, log


# ---------------------------------------------------------------------------
# Fitted transformers (fit on training data only)
# ---------------------------------------------------------------------------

class NumericFeatureTransformer(TransformerMixin, BaseEstimator):
    """Year and Duration features with training-fitted imputation.

    - Year: median imputation (fallback only; rated rows currently have no missing Year).
    - Duration: median of the film's release decade, learned from training rows.
      Decades with fewer than `min_group_size` known durations fall back to the
      overall training median. A `duration_missing` indicator is always added,
      because Duration missingness differs by era in this dataset.
    """

    def __init__(self, decade_width: int = 10, min_group_size: int = 30):
        self.decade_width = decade_width
        self.min_group_size = min_group_size

    def _decade(self, year: pd.Series) -> pd.Series:
        return (year // self.decade_width) * self.decade_width

    def fit(self, X: pd.DataFrame, y=None):
        year = pd.to_numeric(X["Year"], errors="coerce").astype(float)
        duration = pd.to_numeric(X["Duration_min"], errors="coerce").astype(float)

        self.year_median_ = float(year.median())
        self.duration_median_ = float(duration.median())

        known = pd.DataFrame({"decade": self._decade(year.fillna(self.year_median_)), "duration": duration}).dropna()
        stats = known.groupby("decade")["duration"].agg(["median", "size"])
        self.duration_decade_medians_ = stats.loc[stats["size"] >= self.min_group_size, "median"].to_dict()
        self.feature_names_out_ = np.array(["Year", "Duration_min", "duration_missing"], dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "duration_decade_medians_")
        year = pd.to_numeric(X["Year"], errors="coerce").astype(float).fillna(self.year_median_)
        duration = pd.to_numeric(X["Duration_min"], errors="coerce").astype(float)

        fill = self._decade(year).map(self.duration_decade_medians_).fillna(self.duration_median_)
        return pd.DataFrame({
            "Year": year,
            "Duration_min": duration.fillna(fill),
            "duration_missing": duration.isna().astype(int),
        }, index=X.index)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_


class GenreMultiHotEncoder(TransformerMixin, BaseEstimator):
    """Multi-hot encoding of the comma-separated Genre field.

    The genre vocabulary is learned from training rows. Output columns:
    - genre=<Token>: 1 if the film lists that genre
    - genre_missing: 1 if Genre is missing (all genre=<Token> columns are then 0)
    - genre_count: number of genres listed
    Genres unseen during fit are ignored in the multi-hot columns but still counted in genre_count.
    """

    def __init__(self, min_count: int = 1):
        self.min_count = min_count

    @staticmethod
    def _tokens(genre: pd.Series) -> pd.Series:
        return split_genres(genre.astype("object"))

    def fit(self, X: pd.DataFrame, y=None):
        tokens = self._tokens(X["Genre"]).dropna().explode()
        counts = tokens.value_counts()
        self.vocabulary_ = sorted(counts[counts >= self.min_count].index)
        self.feature_names_out_ = np.array(
            [f"genre={g}" for g in self.vocabulary_] + ["genre_missing", "genre_count"], dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "vocabulary_")
        tokens = self._tokens(X["Genre"])
        out = pd.DataFrame(0, index=X.index, columns=self.feature_names_out_, dtype=int)
        for idx, genres in tokens.items():
            if isinstance(genres, list):
                for g in genres:
                    if g in self.vocabulary_:
                        out.at[idx, f"genre={g}"] = 1
        out["genre_missing"] = tokens.isna().astype(int)
        out["genre_count"] = tokens.map(lambda g: len(g) if isinstance(g, list) else 0).astype(int)
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_


class PeopleEncoder(TransformerMixin, BaseEstimator):
    """Frequency + limited multi-hot encoding for person columns (Director or Actor 1–3).

    Learned from training rows only; never uses Rating.

    For each input column:
    - <col>_count: number of training films in which that person appears (in any of the
      given columns). 0 for missing or unseen names.
    - <col>_missing: 1 if the column is empty.
    Plus, pooled across the given columns:
    - <prefix>=<Name>: 1 if the person appears in the film, only for people with at least
      `min_count` training films (keeps the matrix small; rare names are represented
      only through the count features).
    - <prefix>_listed: number of non-missing names (only when more than one column is given).

    A person listed twice in the same film is counted once for that film.
    """

    def __init__(self, prefix: str, min_count: int = 10):
        self.prefix = prefix
        self.min_count = min_count

    @staticmethod
    def _names_per_film(X: pd.DataFrame) -> pd.Series:
        return X.apply(lambda row: sorted({v.strip() for v in row.dropna() if str(v).strip()}), axis=1)

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_ = list(X.columns)
        names = self._names_per_film(X).explode().dropna()
        self.name_counts_ = names.value_counts().to_dict()
        self.frequent_names_ = sorted(n for n, c in self.name_counts_.items() if c >= self.min_count)

        cols = [f"{self._slug(c)}_count" for c in self.columns_]
        cols += [f"{self._slug(c)}_missing" for c in self.columns_]
        if len(self.columns_) > 1:
            cols.append(f"{self.prefix}_listed")
        cols += [f"{self.prefix}={n}" for n in self.frequent_names_]
        self.feature_names_out_ = np.array(cols, dtype=object)
        return self

    @staticmethod
    def _slug(col: str) -> str:
        return col.lower().replace(" ", "_")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "name_counts_")
        X = X[self.columns_]
        parts = {}
        for col in self.columns_:
            parts[f"{self._slug(col)}_count"] = X[col].map(self.name_counts_).fillna(0).astype(int)
        for col in self.columns_:
            parts[f"{self._slug(col)}_missing"] = X[col].isna().astype(int)
        if len(self.columns_) > 1:
            parts[f"{self.prefix}_listed"] = X.notna().sum(axis=1).astype(int)

        frequent = set(self.frequent_names_)
        hot = pd.DataFrame(0, index=X.index, columns=[f"{self.prefix}={n}" for n in self.frequent_names_], dtype=int)
        for idx, names in self._names_per_film(X).items():
            for n in names:
                if n in frequent:
                    hot.at[idx, f"{self.prefix}={n}"] = 1
        return pd.concat([pd.DataFrame(parts, index=X.index), hot], axis=1)[list(self.feature_names_out_)]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_


FEATURE_GROUPS = ("numeric", "genre", "director", "actors")


def build_feature_transformer(director_min_count: int = 10, actor_min_count: int = 20,
                              feature_groups: tuple[str, ...] = FEATURE_GROUPS) -> ColumnTransformer:
    """Return an UNFITTED transformer mapping FEATURE_COLUMNS to a numeric feature matrix.

    Fit it on training rows only (e.g. inside an sklearn Pipeline), then transform
    validation/test rows with the fitted object.

    `feature_groups` selects a subset of the feature groups (used for ablation
    experiments); the default builds all of them.
    """
    unknown = set(feature_groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
    steps = {
        "numeric": (NumericFeatureTransformer(), ["Year", "Duration_min"]),
        "genre": (GenreMultiHotEncoder(), ["Genre"]),
        "director": (PeopleEncoder(prefix="director", min_count=director_min_count), ["Director"]),
        "actors": (PeopleEncoder(prefix="actor", min_count=actor_min_count), ACTOR_COLUMNS),
    }
    transformer = ColumnTransformer(
        transformers=[(name, *steps[name]) for name in FEATURE_GROUPS if name in feature_groups],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return transformer.set_output(transform="pandas")


def make_group_id(model_df: pd.DataFrame) -> pd.Series:
    """Group key 'normalised title|year' for group-aware splitting.

    The title is stripped, inner whitespace collapsed and case-folded; a missing year becomes 'NA'.
    Rows with a missing or blank title get a unique key from `raw_index`, so unrelated untitled
    rows are never merged into one group.
    """
    title = model_df["Name"].fillna("").str.strip().str.replace(r"\s+", " ", regex=True).str.casefold()
    year = model_df["Year"].astype("Int64").astype("string").fillna("NA")
    group = title + "|" + year
    blank = title.eq("")
    group[blank] = "__untitled__|" + model_df.loc[blank, "raw_index"].astype(str)
    return group.rename("group_id")


def get_features_and_target(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the modeling frame into the feature input X (FEATURE_COLUMNS only) and target y."""
    return model_df[FEATURE_COLUMNS].copy(), model_df[TARGET].astype(float).copy()
