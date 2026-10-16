"""Shared data access + evaluation for all model teams.

Implements the agreements in the Model Development Meeting Summary so every
team uses identical data, folds and metrics:

    import sys; sys.path.append("src")          # from the project root
    from model_data import load_features, cv_folds, evaluate, climatology_baseline

    df = load_features("mar1")                  # or "feb1"
    for fold, train, valid in cv_folds(df):
        ...fit on train, predict valid...
        print(fold, evaluate(valid["full_bloom_doy"], predictions))

    dev, test = df[df.split == "development"], df[df.split == "test"]   # final testing only

Requires: python setup_database.py has been run once.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "etl"))
from db_connection import get_engine  # noqa: E402

TARGET = "full_bloom_doy"
ID_COLUMNS = ["station_id", "location", "prefecture", "region", "bloom_year",
              "full_bloom_date", "full_bloom_doy",
              "split", "fold1", "fold2", "fold3", "fold4", "fold5"]
FORECAST_TABLES = {"feb1": "model_features_feb1", "mar1": "model_features_mar1"}

_cache = {}


def load_features(forecast_date="mar1"):
    """Leakage-safe feature table for one forecast date ('feb1' or 'mar1').

    Rows: Somei-yoshino, bloom year >= 1964, observed full bloom, no missing
    temperature day up to the cutoff. Columns: ids, candidate features,
    target `full_bloom_doy`, and split/fold labels.
    """
    if forecast_date not in FORECAST_TABLES:
        raise ValueError(f"forecast_date must be one of {list(FORECAST_TABLES)}")
    if forecast_date not in _cache:
        engine = get_engine()
        df = pd.read_sql(f"SELECT * FROM {FORECAST_TABLES[forecast_date]} "
                         f"ORDER BY bloom_year, station_id", engine)
        engine.dispose()
        _cache[forecast_date] = df
    return _cache[forecast_date].copy()


def feature_columns(df):
    """All candidate feature columns (everything that is not an id/target/split)."""
    return [c for c in df.columns if c not in ID_COLUMNS]


def cv_folds(df):
    """Yield (fold_number, train_df, valid_df) for the 5 agreed expanding-window folds."""
    for k in range(1, 6):
        col = f"fold{k}"
        yield k, df[df[col] == "train"].copy(), df[df[col] == "valid"].copy()


def evaluate(y_true, y_pred):
    """MAE, RMSE, R2 and hit rate within +/-3 days (share of point predictions)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "n": int(len(y_true)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "hit_rate_3d": float(np.mean(np.abs(err) <= 3)),
    }


def climatology_baseline(train, valid):
    """Forecast-date-safe baseline: each station's mean full-bloom DOY in the
    training rows (overall training mean for stations without history)."""
    station_mean = train.groupby("station_id")[TARGET].mean()
    return valid["station_id"].map(station_mean).fillna(train[TARGET].mean()).to_numpy()


def cv_report(df, predict_fn):
    """Run predict_fn(train, valid) -> predictions on every fold; return a table."""
    rows = []
    for k, train, valid in cv_folds(df):
        rows.append({"fold": k, **evaluate(valid[TARGET], predict_fn(train, valid))})
    out = pd.DataFrame(rows)
    mean = out.drop(columns="fold").mean().to_dict()
    mean["n"] = int(out["n"].sum())
    return pd.concat([out, pd.DataFrame([{"fold": "mean", **mean}])], ignore_index=True)


if __name__ == "__main__":
    for date in FORECAST_TABLES:
        data = load_features(date)
        print(f"\n{date}: {len(data):,} rows, {len(feature_columns(data))} candidate features")
        print("Climatology baseline (validation folds):")
        print(cv_report(data, climatology_baseline).round(3).to_string(index=False))
