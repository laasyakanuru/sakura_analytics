"""Shared data access + evaluation for all model teams.

Implements the agreements in the Model Development Meeting Summary so every
team uses identical rows, folds and metrics:

    import sys; sys.path.append("src")          # from the project root
    from model_data import load_features, feature_columns, cv_folds, evaluate

    df = load_features("mar1")                  # or "feb1"
    X = feature_columns(df)
    for fold, train, valid in cv_folds(df):
        model.fit(train[X], train[TARGET])
        print(fold, evaluate(valid[TARGET], model.predict(valid[X])))

Where the data comes from (load_features(..., source="auto")):
  - MySQL `sakura_dw`, if your password is in .env / environment variables
    and the server answers;
  - otherwise the CSV copies in data/model/ (committed to git), so you can
    model with Python only - no MySQL needed.
Both give exactly the same table; setup_database.py refreshes the CSVs.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "etl"))
from db_connection import credentials_configured, get_engine  # noqa: E402

CSV_DIR = PROJECT_ROOT / "data" / "model"
FORECAST_TABLES = {"feb1": "model_features_feb1", "mar1": "model_features_mar1"}

TARGET = "full_bloom_doy"
FOLD_COLUMNS = ["fold1", "fold2", "fold3", "fold4", "fold5"]
TEXT_COLUMNS = ["station_id", "location", "prefecture", "region",
                "full_bloom_date", "split"] + FOLD_COLUMNS
INT_COLUMNS = ["bloom_year", "relocations_before_season", TARGET]
# Not features: identifiers, the target itself, its date, and the split labels.
ID_COLUMNS = ["station_id", "location", "prefecture", "region", "bloom_year",
              "full_bloom_date", TARGET, "split"] + FOLD_COLUMNS

_cache = {}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def _normalise(df):
    """Same dtypes whether the table came from MySQL or from CSV."""
    df = df.copy()
    for col in df.columns:
        if col in TEXT_COLUMNS:
            df[col] = df[col].map(lambda v: None if pd.isna(v) else str(v)).astype(object)
        elif col in INT_COLUMNS:
            df[col] = df[col].astype("int64")
        else:
            df[col] = pd.to_numeric(df[col]).astype("float64")
    return df.sort_values(["bloom_year", "station_id"]).reset_index(drop=True)


def _from_mysql(forecast_date):
    engine = get_engine()
    try:
        return pd.read_sql(f"SELECT * FROM {FORECAST_TABLES[forecast_date]}", engine)
    finally:
        engine.dispose()


def _from_csv(forecast_date):
    path = CSV_DIR / f"{FORECAST_TABLES[forecast_date]}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `git pull`, or build the "
                                f"database with `python3 setup_database.py`.")
    return pd.read_csv(path, dtype={"station_id": str})


def load_features(forecast_date="mar1", source="auto", verbose=True):
    """Leakage-safe feature table for one forecast date ('feb1' or 'mar1').

    source: "auto" (MySQL if configured and reachable, else CSV), "mysql" or "csv".
    Rows: Somei-yoshino, bloom year >= 1964, observed full bloom, no missing
    temperature day up to the cutoff. No NULL feature values.
    """
    if forecast_date not in FORECAST_TABLES:
        raise ValueError(f"forecast_date must be one of {list(FORECAST_TABLES)}")
    if source not in ("auto", "mysql", "csv"):
        raise ValueError("source must be 'auto', 'mysql' or 'csv'")

    key = (forecast_date, source)
    if key not in _cache:
        used = source
        if source == "mysql":
            raw = _from_mysql(forecast_date)
        elif source == "csv":
            raw = _from_csv(forecast_date)
        else:
            raw = None
            if credentials_configured():
                try:
                    raw, used = _from_mysql(forecast_date), "mysql"
                except Exception as exc:  # server down, no database yet, ...
                    if verbose:
                        print(f"(MySQL not available: {str(exc).splitlines()[0][:80]})")
            if raw is None:
                raw, used = _from_csv(forecast_date), "csv"
        _cache[key] = _normalise(raw)
        if verbose:
            where = "MySQL sakura_dw" if used == "mysql" else f"data/model/ CSV"
            print(f"Loaded {FORECAST_TABLES[forecast_date]} from {where}: "
                  f"{len(_cache[key]):,} rows")
    return _cache[key].copy()


def export_csvs():
    """Write data/model/*.csv from MySQL and verify they read back identically."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    for date, table in FORECAST_TABLES.items():
        df = _normalise(_from_mysql(date))
        path = CSV_DIR / f"{table}.csv"
        df.to_csv(path, index=False, lineterminator="\n")
        back = _normalise(_from_csv(date))
        pd.testing.assert_frame_equal(df, back, check_dtype=True)
        print(f"Wrote {path.relative_to(PROJECT_ROOT)} ({len(df):,} rows) - verified")


# --------------------------------------------------------------------------
# folds, metrics, baseline
# --------------------------------------------------------------------------
def feature_columns(df):
    """All candidate feature columns (everything that is not an id/target/split)."""
    return [c for c in df.columns if c not in ID_COLUMNS]


def cv_folds(df):
    """Yield (fold_number, train_df, valid_df) for the 5 agreed expanding-window folds."""
    for k in range(1, 6):
        col = f"fold{k}"
        yield k, df[df[col] == "train"].copy(), df[df[col] == "valid"].copy()


def development_and_test(df):
    """(development 1964-2014, test 2015-2025). Use test ONLY for the final champion."""
    return df[df["split"] == "development"].copy(), df[df["split"] == "test"].copy()


def evaluate(y_true, y_pred):
    """MAE, RMSE, R2 and hit rate within +/-3 days (share of point predictions)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "n": int(len(y_true)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        "hit_rate_3d": float(np.mean(np.abs(err) <= 3)),
    }


def climatology_baseline(train, valid):
    """Forecast-date-safe baseline: each station's mean full-bloom DOY in the
    training rows (overall training mean for stations without history)."""
    station_mean = train.groupby("station_id")[TARGET].mean()
    return valid["station_id"].map(station_mean).fillna(train[TARGET].mean()).to_numpy()


def cv_report(df, predict_fn):
    """Run predict_fn(train, valid) -> predictions on all 5 folds.
    Returns one row per fold plus a 'mean' row (n = total validation rows)."""
    rows = [{"fold": k, **evaluate(valid[TARGET], predict_fn(train, valid))}
            for k, train, valid in cv_folds(df)]
    out = pd.DataFrame(rows)
    mean = out.drop(columns="fold").mean().to_dict()
    mean["n"] = int(out["n"].sum())
    out = pd.concat([out, pd.DataFrame([{"fold": "mean", **mean}])], ignore_index=True)
    out["n"] = out["n"].astype(int)
    return out


def final_test_report(df, predict_fn):
    """Champion models only: train on all development years, score 2015-2025."""
    dev, test = development_and_test(df)
    return pd.DataFrame([{"fold": "test", **evaluate(test[TARGET], predict_fn(dev, test))}])


if __name__ == "__main__":
    for date in FORECAST_TABLES:
        data = load_features(date)
        print(f"{date}: {len(feature_columns(data))} candidate features")
        print("Climatology baseline on the 5 validation folds:")
        print(cv_report(data, climatology_baseline).round(3).to_string(index=False), "\n")
