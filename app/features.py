"""Shared feature engineering for training and interactive predictions."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "owners_mean",
    "review_ratio",
    "review_total",
    "log_playtime",
    "years_since_release",
]
FEATURES = NUMERIC_FEATURES + ["genres"]
TARGET = "price"


def current_reference_year() -> int:
    """Return the current calendar year for default age calculations."""
    return date.today().year


def prepare_training_data(data: pd.DataFrame, reference_year: int) -> pd.DataFrame:
    """Create the model's training columns from the raw Steam dataset."""
    df = data.copy()
    if "owners_mean" not in df.columns:
        if "owners" not in df.columns:
            raise ValueError("Dataset precisa ter a coluna 'owners' ou 'owners_mean'.")
        owner_bounds = df["owners"].astype("string").str.extract(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
        df["owners_mean"] = (
            pd.to_numeric(owner_bounds[0], errors="coerce")
            + pd.to_numeric(owner_bounds[1], errors="coerce")
        ) / 2

    if "log_playtime" in df.columns and not {
        "average_playtime", "average_playtime_forever"
    }.intersection(df.columns):
        df["log_playtime"] = pd.to_numeric(df["log_playtime"], errors="coerce")
    else:
        if "average_playtime" in df.columns:
            playtime = df["average_playtime"]
        elif "average_playtime_forever" in df.columns:
            playtime = df["average_playtime_forever"]
        else:
            playtime = pd.Series(0, index=df.index)
        df["log_playtime"] = np.log1p(pd.to_numeric(playtime, errors="coerce").clip(lower=0))

    if "review_total" in df.columns:
        df["review_total"] = pd.to_numeric(df["review_total"], errors="coerce")
    else:
        positive = pd.to_numeric(df.get("positive_ratings", 0), errors="coerce")
        negative = pd.to_numeric(df.get("negative_ratings", 0), errors="coerce")
        df["review_total"] = positive + negative
    if "review_ratio" in df.columns:
        df["review_ratio"] = pd.to_numeric(df["review_ratio"], errors="coerce")
    elif "positive_ratio" in df.columns:
        df["review_ratio"] = pd.to_numeric(df["positive_ratio"], errors="coerce")
    else:
        positive = pd.to_numeric(df.get("positive_ratings", 0), errors="coerce")
        df["review_ratio"] = positive / (df["review_total"] + 1)

    if "release_year" in df.columns:
        release_year = pd.to_numeric(df["release_year"], errors="coerce")
    elif "release_date" in df.columns:
        release_year = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    else:
        release_year = pd.Series(np.nan, index=df.index)
    df["release_year"] = release_year
    df["years_since_release"] = reference_year - release_year

    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Dataset sem colunas necessárias: " + ", ".join(missing))
    df = df[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if df.empty:
        raise ValueError("Nenhuma linha válida restou após a preparação dos dados.")
    return df


def make_prediction_row(
    owners_mean: float,
    positive_ratings: int,
    negative_ratings: int,
    average_playtime: int,
    release_year: int,
    genre: str,
    reference_year: int,
) -> pd.DataFrame:
    """Build one prediction row with exactly the columns used for training."""
    review_total = positive_ratings + negative_ratings
    return pd.DataFrame(
        [{
            "owners_mean": owners_mean,
            "review_ratio": positive_ratings / (review_total + 1),
            "review_total": review_total,
            "log_playtime": np.log1p(max(0, average_playtime)),
            "years_since_release": reference_year - release_year,
            "genres": genre,
        }],
        columns=FEATURES,
    )
