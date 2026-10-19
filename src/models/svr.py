"""Support Vector Regression (SVR) model 

    python3 src/models/svr.py                    # tune + report both dates
    python3 src/models/svr.py --no-tune          # skip tuning, use DEFAULT_PARAMS
    python3 src/models/svr.py --target raw       # comparison: predict the raw DOY
    python3 src/models/svr.py --train-window 30  # comparison: last 30 years only
    python3 src/models/svr.py --sample small     # only stations observed in 2025
    python3 src/models/svr.py --features all     # every column in the feature table

  * data      : load_features() only (leakage-safe feb1 / mar1 tables)
  * folds     : the 5 agreed expanding-window folds from cv_folds()
                (train 1964..T, validate the next two years; T = 2004 ... 2012)
  * metrics   : evaluate() -> MAE, RMSE, R2, +-3-day hit rate
  * dates     : feb1, mar1 AND mar15 (Mar 15 added by team agreement)
  * test      : 2015-2025 stays untouched unless RUN_FINAL_TEST = True
                (champion model only, after the team picks it on validation)
  * scaling   : StandardScaler inside the model, fitted on the training fold only
  * baseline  : the agreed 30-year station climatology (printed for comparison)

Target (default "anomaly"): the SVR predicts how many days EARLIER or LATER than
usual a station blooms, not the raw bloom day:

    anomaly    = full_bloom_doy - station reference
    prediction = station reference + predicted anomaly

The station reference is that station's mean full-bloom DOY over the last
STATION_REF_YEARS (30) years of the fold's TRAINING rows - the same idea as the
agreed 30-year climatology (region mean if the station has < 10 years, then
the overall mean). It is recomputed inside every fold from training rows only,
so no validation or test information leaks in. The weather features then only
have to explain the year-to-year deviation, which is what they carry
information about; the raw-DOY SVR could not tell stations apart beyond
latitude/longitude/elevation. --target raw reproduces the earlier version.

Training window: by default the SVR trains on ALL training years of each fold.
--train-window N uses only the last N (tested with N=30 on the raw target:
slightly worse, so all years is the default).

Hyperparameter tuning: grid search over kernel, C, epsilon and gamma, scored on
the SAME 5 validation folds (GridSearchCV with refit=False, so it never trains
outside the fold training windows). The best mean-MAE setting is reported with
cv_report() like every other model, so the `mean` row is directly comparable.

Outputs (in results/svr/; prefix "svr", plus "_raw" / "_train<N>" for the
comparison options):
  <prefix>_<date>_cv.csv       fold table + mean row (commit this for the comparison)
  <prefix>_<date>_tuning.csv   every grid setting with its mean/std validation MAE
  <prefix>_<date>_test.csv     only when RUN_FINAL_TEST = True
"""
import argparse
import sys
from pathlib import Path

# Project root = the folder containing requirements.txt (works from any subfolder).
PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                    if (p / "requirements.txt").exists())
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import BaseEstimator, RegressorMixin  # noqa: E402
from sklearn.model_selection import GridSearchCV, ParameterGrid  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

from model_data import (TARGET, cv_folds, cv_report,  # noqa: E402
                        feature_columns, final_test_report, load_features)

# ============================ change these ==================================
MODEL_NAME = "svr"
RUN_FINAL_TEST = False          # champion model only!
FORECAST_DATES = ("feb1", "mar1", "mar15")
TARGET_MODE = "anomaly"         # "anomaly" (default) or "raw"
STATION_REF_YEARS = 30          # years in the station reference (anomaly target)
MIN_STATION_HISTORY = 10        # fewer years -> region mean, then overall mean
TRAIN_WINDOW_YEARS = 0          # 0 = all training years; e.g. 30 = last 30 years
SAMPLE = "full"                 # "full" or "small" (only stations observed in 2025)
FEATURE_SET = "team"            # "team" (shared candidate lists) or "all" (every column)

# Grid searched on the 5 validation folds.
#   C       : penalty for errors larger than epsilon (higher = less regularised)
#   epsilon : half-width of the "no-penalty" tube, in days
#   gamma   : RBF width (smaller = smoother, closer to linear);
#             "scale" = 1 / n_features after standardising (~0.04-0.05 here)
# A linear kernel is included as a reference: if it scores like the best RBF,
# the non-linearity adds nothing.
PARAM_GRID = [
    {
        "kernel": ["rbf"],
        "C": [0.1, 0.3, 1, 3, 10, 30, 100],
        "epsilon": [0.25, 0.5, 1.0, 2.0],
        "gamma": [0.001, 0.003, 0.01, 0.03, "scale"],
    },
    {
        "kernel": ["linear"],
        "C": [0.01, 0.03, 0.1, 0.3, 1],
        "epsilon": [0.25, 0.5, 1.0, 2.0],
    },
]

# Used with --no-tune.
DEFAULT_PARAMS = {"kernel": "rbf", "C": 10, "epsilon": 1.0, "gamma": 0.003}


# Team candidate predictors per forecast date - the same lists as
# src/models/linear_regression.py (main), so every model sees identical inputs.
# Leaves out the data-quality counts (*_missing_days), relocations_before_season
# and the February precipitation / sunshine / frost columns.
COMMON_FEATURES = [
    "latitude", "longitude", "elevation_m", "tmean_oct", "tmean_nov",
    "tmean_dec", "tmean_jan", "tmax_jan", "tmin_jan",
]
TEAM_FEATURES = {"feb1": COMMON_FEATURES}
TEAM_FEATURES["mar1"] = TEAM_FEATURES["feb1"] + [
    "tmean_feb", "tmax_feb", "tmin_feb", "chill_days_0_7c_nov_feb",
    "tmin_lowest_nov_feb", "gdd5_feb", "cum_tmax_feb", "cum_tmean_feb",
]
TEAM_FEATURES["mar15"] = TEAM_FEATURES["mar1"] + [
    "tmean_mar1_14", "tmax_mar1_14", "tmin_mar1_14", "gdd5_mar1_14",
    "cum_tmax_mar1_14", "cum_tmean_mar1_14", "chill_days_0_7c_mar1_14",
    "frost_days_mar1_14", "precip_mar1_14_mm", "sunshine_mar1_14_h",
    "cum_tmax_feb1_mar14",
]


def choose_features(df, forecast_date):
    """Team candidate list for this forecast date (default), or every column in the
    feature table with FEATURE_SET = "all". `region` is text and never used."""
    if FEATURE_SET == "all":
        return feature_columns(df)
    features = TEAM_FEATURES[forecast_date]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"{forecast_date} feature table is missing {missing}")
    return list(features)
# ============================================================================


def station_reference(train, ref_years, min_history):
    """Function mapping rows -> each station's mean full-bloom DOY over the last
    `ref_years` training years (region mean / overall mean as fallbacks)."""
    hist = train
    if ref_years:
        hist = train[train["bloom_year"] >= train["bloom_year"].max() - ref_years + 1]
    station = hist.groupby("station_id")[TARGET].agg(["mean", "size"])
    region = hist.groupby("region")[TARGET].mean()
    overall = hist[TARGET].mean()

    def lookup(rows):
        mean = rows["station_id"].map(station["mean"])
        size = rows["station_id"].map(station["size"]).fillna(0)
        fallback = rows["region"].map(region).fillna(overall)
        return np.where(size >= min_history, mean, fallback).astype(float)
    return lookup


class StationSVR(BaseEstimator, RegressorMixin):
    """StandardScaler + SVR, optionally on the station-anomaly target.

    fit/predict take the full feature-table rows (a DataFrame): `features` are
    the model inputs; station_id / region / bloom_year are only used to build
    the station reference from the training rows."""

    def __init__(self, features=None, target_mode="anomaly", ref_years=30,
                 min_history=10, kernel="rbf", C=1.0, epsilon=0.5, gamma="scale"):
        self.features = features
        self.target_mode = target_mode
        self.ref_years = ref_years
        self.min_history = min_history
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        if self.target_mode == "anomaly":
            rows = X.assign(**{TARGET: y})
            self.reference_ = station_reference(rows, self.ref_years, self.min_history)
            y = y - self.reference_(X)
        elif self.target_mode != "raw":
            raise ValueError("target_mode must be 'anomaly' or 'raw'")
        self.model_ = make_pipeline(
            StandardScaler(),
            SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon, gamma=self.gamma))
        self.model_.fit(X[self.features], y)
        return self

    def predict(self, X):
        pred = self.model_.predict(X[self.features])
        if self.target_mode == "anomaly":
            pred = pred + self.reference_(X)
        return pred


def make_model(features, params=None):
    """Return a NEW, unfitted model with the run's settings."""
    model = StationSVR(features=features, target_mode=TARGET_MODE,
                       ref_years=STATION_REF_YEARS, min_history=MIN_STATION_HISTORY)
    if params:
        model.set_params(**params)
    return model


def recent(train):
    """Keep only the last TRAIN_WINDOW_YEARS years of a fold's training rows."""
    if not TRAIN_WINDOW_YEARS:
        return train
    last = train["bloom_year"].max()
    return train[train["bloom_year"] >= last - TRAIN_WINDOW_YEARS + 1]


def fold_indices(df):
    """The 5 agreed folds as (train_positions, valid_positions) for GridSearchCV.
    Built from cv_folds(), so the validation sets are identical to every other model."""
    pos = pd.Series(np.arange(len(df)), index=df.index)
    return [(pos[recent(train).index].to_numpy(), pos[valid.index].to_numpy())
            for _, train, valid in cv_folds(df)]


def climatology_30yr(train, valid, lookback=30, min_history=10):
    """The team's agreed baseline (Yasha, Baseline Model/02_station_climatology):
    station mean full-bloom DOY over the `lookback` years before each forecast
    year (training rows only); region mean if the station has < min_history
    years, else the overall mean."""
    preds = []
    for row in valid.itertuples(index=False):
        hist = train[train["bloom_year"].between(row.bloom_year - lookback,
                                                 row.bloom_year - 1)]
        station = hist[hist["station_id"] == row.station_id]
        region = hist[hist["region"] == row.region]
        if len(station) >= min_history:
            preds.append(station[TARGET].mean())
        elif len(region):
            preds.append(region[TARGET].mean())
        else:
            preds.append(hist[TARGET].mean())
    return np.asarray(preds, dtype=float)


def tune(df, features):
    """Grid search on the 5 validation folds; returns (best_params, results table)."""
    search = GridSearchCV(
        make_model(features),
        PARAM_GRID,
        cv=fold_indices(df),
        scoring="neg_mean_absolute_error",
        refit=False,            # never fit on the full table (it contains test years)
        n_jobs=-1,
    )
    search.fit(df, df[TARGET])

    res = pd.DataFrame(search.cv_results_)
    table = pd.DataFrame({
        "kernel": res["param_kernel"],
        "C": res["param_C"],
        "epsilon": res["param_epsilon"],
        "gamma": res["param_gamma"].map(lambda g: "-" if pd.isna(g) else str(g)),
        "mean_valid_MAE": -res["mean_test_score"],
        "std_valid_MAE": res["std_test_score"],
        "mean_fit_time_s": res["mean_fit_time"],
    }).sort_values("mean_valid_MAE").reset_index(drop=True)
    return search.best_params_, table


def warn_if_on_edge(params):
    """Warn when a tuned value sits at the edge of its grid (optimum may lie outside)."""
    grid = next(g for g in PARAM_GRID if params["kernel"] in g["kernel"])
    for name, values in grid.items():
        numeric = sorted(v for v in values if not isinstance(v, str))
        value = params.get(name)
        if len(numeric) > 1 and value in (numeric[0], numeric[-1]):
            side = "lowest" if value == numeric[0] else "highest"
            print(f"  note: {name} = {value} is the {side} value in the grid - if the "
                  f"top settings differ by more than ~0.02 days, extend the grid.")


def bias_by_fold(df, features, params):
    """Mean signed error per fold (prediction - actual; + = predicted too late)."""
    rows = []
    for k, train, valid in cv_folds(df):
        train = recent(train)
        model = make_model(features, params).fit(train, train[TARGET])
        err = model.predict(valid) - valid[TARGET].to_numpy()
        rows.append(f"fold {k}: {err.mean():+.2f}")
    return "  ".join(rows)


def small_sample_stations():
    """Stations with an observed full bloom in 2025 (same rule as the LR / RF scripts:
    taken from the Mar 1 table so every forecast date uses the same stations)."""
    mar1 = load_features("mar1")
    observed = mar1["bloom_year"].eq(2025) & mar1[TARGET].notna()
    return set(mar1.loc[observed, "station_id"].astype(str))


def run_name():
    name = MODEL_NAME
    if TARGET_MODE == "raw":
        name += "_raw"
    if SAMPLE == "small":
        name += "_small"
    if FEATURE_SET == "all":
        name += "_allfeat"
    if TRAIN_WINDOW_YEARS:
        name += f"_train{TRAIN_WINDOW_YEARS}"
    return name


def run(forecast_date, do_tune, results_dir, df=None):
    """Tune, report and save one forecast date. `df` lets another script pass
    its own feature table; default: load_features(forecast_date)."""
    name = run_name()
    df = load_features(forecast_date) if df is None else df
    if SAMPLE == "small":
        df = df[df["station_id"].astype(str).isin(small_sample_stations())].copy()
    features = choose_features(df, forecast_date)
    setup = (f"target {TARGET_MODE}"
             + (f" (station ref: last {STATION_REF_YEARS} yrs)" if TARGET_MODE == "anomaly" else "")
             + f" | train {'last %d years' % TRAIN_WINDOW_YEARS if TRAIN_WINDOW_YEARS else 'all years'}"
             + f" | {SAMPLE} sample ({df['station_id'].nunique()} stations, {len(df)} rows)")

    if do_tune:
        print(f"\nTuning {name} for {forecast_date}: {len(ParameterGrid(PARAM_GRID))} "
              f"settings x 5 folds | {setup} ...")
        params, table = tune(df, features)
        table.insert(0, "forecast_date", forecast_date)
        table.to_csv(results_dir / f"{name}_{forecast_date}_tuning.csv", index=False)
        print("Top 5 settings by mean validation MAE:")
        print(table.head(5).round(3).to_string(index=False))
        lin = table[table["kernel"] == "linear"]
        if len(lin):
            print(f"Best linear-kernel setting: MAE {lin.iloc[0].mean_valid_MAE:.3f} "
                  f"(C={lin.iloc[0].C}, epsilon={lin.iloc[0].epsilon})")
        warn_if_on_edge(params)
    else:
        params = DEFAULT_PARAMS

    def predict(train, valid):
        train = recent(train)
        return make_model(features, params).fit(train, train[TARGET]).predict(valid)

    report = cv_report(df, predict)
    baseline = cv_report(df, climatology_30yr)

    print(f"\n=== {name} | forecast date {forecast_date} | {len(features)} features | "
          f"{setup} | {params} ===")
    folds = {k: f"validation {v['bloom_year'].min()}-{v['bloom_year'].max()}"
             for k, _, v in cv_folds(df)}
    printable = report.round(3)
    printable[""] = printable["fold"].map(folds).fillna("<- compare models on this row")
    print(printable.to_string(index=False))
    b = baseline.iloc[-1]
    print(f"30-yr climatology baseline (mean of folds): MAE {b.MAE:.3f}  RMSE {b.RMSE:.3f}  "
          f"R2 {b.R2:.3f}  hit_rate_3d {b.hit_rate_3d:.3f}")
    print(f"bias (mean pred - actual, days): {bias_by_fold(df, features, params)}")

    out = results_dir / f"{name}_{forecast_date}_cv.csv"
    report.insert(0, "model", name)
    report.insert(1, "forecast_date", forecast_date)
    report.insert(2, "params", str(params))
    report.to_csv(out, index=False)
    print(f"saved {out.relative_to(PROJECT_ROOT)}")

    if RUN_FINAL_TEST:
        test = final_test_report(df, predict)
        print("FINAL TEST 2015-2025:")
        print(test.round(3).to_string(index=False))
        test.insert(0, "model", name)
        test.insert(1, "forecast_date", forecast_date)
        test.insert(2, "params", str(params))
        test.to_csv(results_dir / f"{name}_{forecast_date}_test.csv", index=False)


def main():
    global TARGET_MODE, TRAIN_WINDOW_YEARS, SAMPLE, FEATURE_SET
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-tune", action="store_true",
                        help="skip the grid search and use DEFAULT_PARAMS")
    parser.add_argument("--target", choices=["anomaly", "raw"], default=TARGET_MODE,
                        help="predict the station anomaly (default) or the raw DOY")
    parser.add_argument("--train-window", type=int, default=TRAIN_WINDOW_YEARS,
                        help="train on the last N years of each fold (0 = all years)")
    parser.add_argument("--sample", choices=["full", "small"], default=SAMPLE,
                        help="full sample, or only stations observed in 2025")
    parser.add_argument("--features", choices=["team", "all"], default=FEATURE_SET,
                        help="team candidate lists (default) or every feature-table column")
    args = parser.parse_args()
    TARGET_MODE = args.target
    TRAIN_WINDOW_YEARS = args.train_window
    SAMPLE = args.sample
    FEATURE_SET = args.features

    results_dir = PROJECT_ROOT / "results" / "svr"
    results_dir.mkdir(parents=True, exist_ok=True)
    for forecast_date in FORECAST_DATES:
        run(forecast_date, not args.no_tune, results_dir)


if __name__ == "__main__":
    main()
