"""Compare all-variable OLS, all-variable Lasso and Lasso-selected OLS.

Run from the repository root: python3 src/models/linear_regression.py
Forecast dates: Feb 1, Mar 1 and Mar 15. Candidate features retain the original
lists, extended with Mar 1-14 weather for Mar 15. For each date, Lasso alpha
and selection are fixed on full-sample 1964-2004 data with internal expanding
CV. The same alpha is used for direct Lasso in all five outer folds/scopes.
Full/small sample definitions and both climatology baselines are unchanged.
Writes three CSVs in results/, each with five folds and a mean row per group.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents
                     if (p / "src" / "model_data.py").exists()), None)
if PROJECT_ROOT is None:
    raise SystemExit("Place this file in sakura_analytics/src/models/.")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import model_data  # noqa: E402
# Compatibility with older model_data.py versions; the Mar 15 CSV is required.
model_data.FORECAST_TABLES.setdefault("mar15", "model_features_mar15")
from model_data import (TARGET, climatology_baseline, cv_folds, evaluate,
                        load_features)  # noqa: E402

COMMON_FEATURES = [
    "latitude", "longitude", "elevation_m", "tmean_oct", "tmean_nov",
    "tmean_dec", "tmean_jan", "tmax_jan", "tmin_jan",
]
FEATURES = {
    "feb1": COMMON_FEATURES,
    "mar1": COMMON_FEATURES + [
        "tmean_feb", "tmax_feb", "tmin_feb",
        "chill_days_0_7c_nov_feb", "tmin_lowest_nov_feb", "gdd5_feb",
        "cum_tmax_feb", "cum_tmean_feb",
    ],
}
FEATURES["mar15"] = FEATURES["mar1"] + [
    "tmean_mar1_14", "tmax_mar1_14", "tmin_mar1_14", "gdd5_mar1_14",
    "cum_tmax_mar1_14", "cum_tmean_mar1_14", "chill_days_0_7c_mar1_14",
    "frost_days_mar1_14", "precip_mar1_14_mm", "sunshine_mar1_14_h",
    "cum_tmax_feb1_mar14",
]
LABELS = {"feb1": "February 1", "mar1": "March 1", "mar15": "March 15"}
OUTPUTS = {"ols_all": "linear_regression_ols_all_cv.csv",
           "lasso_all": "linear_regression_lasso_all_cv.csv",
           "lasso_selected_ols": "linear_regression_lasso_selected_ols_cv.csv"}
SCOPES = ("Full sample", "Small sample")
ALPHAS = np.logspace(-4, 2, 40)
METRICS = ("MAE_days", "RMSE_days", "R2", "Bias_days",
           "Within_3_days_pct")


def scores(actual, predicted):
    shared = evaluate(actual, predicted)
    error = np.asarray(predicted) - np.asarray(actual)
    return {"MAE_days": shared["MAE"], "RMSE_days": shared["RMSE"],
            "R2": shared["R2"], "Bias_days": float(error.mean()),
            "Within_3_days_pct": 100 * shared["hit_rate_3d"]}


def climatology_30yr(train, valid):
    predictions = []
    for row in valid.itertuples(index=False):
        history = train.loc[train["bloom_year"].between(
            row.bloom_year - 30, row.bloom_year - 1)]
        station = history.loc[history["station_id"].eq(row.station_id)]
        if len(station) >= 10:
            prediction = station[TARGET].mean()
        else:
            region = history.loc[history["region"].eq(row.region)]
            prediction = (region[TARGET].mean() if not region.empty
                          else history[TARGET].mean())
        predictions.append(prediction)
    return np.asarray(predictions)


def selection_folds(data):
    """Five expanding windows wholly inside the fixed 1964-2004 sample."""
    folds = []
    for start in range(1995, 2005, 2):
        train = data.loc[data["bloom_year"].lt(start)]
        valid = data.loc[data["bloom_year"].between(start, start + 1)]
        if train.empty or valid.empty:
            raise ValueError("Empty Lasso selection fold.")
        folds.append((train, valid))
    return folds


def select_features(data, features):
    selection = data.loc[data["bloom_year"].between(1964, 2004)]
    if selection.empty or int(selection["bloom_year"].max()) != 2004:
        raise ValueError("Lasso selection requires full-sample 1964-2004 data.")
    errors, converged = {}, {}
    for alpha in ALPHAS:
        fold_errors, ok = [], True
        for train, valid in selection_folds(selection):
            model = make_pipeline(
                StandardScaler(),
                Lasso(alpha=float(alpha), max_iter=100000, tol=1e-5, precompute=True),
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(train[features], train[TARGET])
            ok &= not any(issubclass(w.category, ConvergenceWarning)
                          for w in caught)
            fold_errors.append(evaluate(
                valid[TARGET], model.predict(valid[features]))["MAE"])
        errors[float(alpha)], converged[float(alpha)] = np.mean(fold_errors), ok
    eligible = [alpha for alpha in errors if converged[alpha]]
    if not eligible:
        raise ValueError("No Lasso alpha converged in every selection fold.")
    alpha = min(eligible, key=lambda value: (errors[value], value))
    selector = make_pipeline(
        StandardScaler(), Lasso(alpha=alpha, max_iter=100000, tol=1e-5, precompute=True))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        selector.fit(selection[features], selection[TARGET])
    selected = [name for name, coefficient
                in zip(features, selector[-1].coef_) if coefficient != 0]
    if not selected:
        raise ValueError("Lasso selected no variables.")
    return alpha, selected, len(selection), int(selection.station_id.nunique())


def add_metrics(row, prefix, actual, predicted):
    row.update({f"{prefix}_{metric}": value
                for metric, value in scores(actual, predicted).items()})


def make_mean(group):
    row = {"Forecast": group.iloc[0]["Forecast"],
           "Sample": group.iloc[0]["Sample"], "fold": "mean",
           "Train_years": "", "Validation_years": "",
           "Train_N": np.nan, "N": int(group["N"].sum()),
           "Selection_train_years": "1964-2004",
           "Selection_N": group.iloc[0]["Selection_N"],
           "Selection_stations": group.iloc[0]["Selection_stations"],
           "Selection_alpha": group.iloc[0]["Selection_alpha"],
           "Selected_feature_count": group.iloc[0]["Selected_feature_count"],
           "Selected_features": group.iloc[0]["Selected_features"]}
    for prefix in ("Model", "All_history", "Climatology_30yr"):
        for metric in METRICS:
            column = f"{prefix}_{metric}"
            row[column] = float(group[column].mean())
    return row


def main():
    datasets = {date: load_features(date, source="csv", verbose=False)
                for date in FEATURES}
    stations = datasets["mar1"].loc[
        datasets["mar1"]["bloom_year"].eq(2025)
        & datasets["mar1"][TARGET].notna(), "station_id"].astype(str).unique()
    if not len(stations):
        raise ValueError("No stations observed in 2025 for small sample.")

    rows = {method: [] for method in OUTPUTS}
    for date, data in datasets.items():
        features = FEATURES[date]
        development = data.loc[data["split"].eq("development")].copy()
        alpha, selected, selection_n, selection_stations = select_features(
            development, features)
        print(f"{LABELS[date]} selected: {', '.join(selected)}", flush=True)

        for fold, full_train, full_valid in cv_folds(development):
            for scope in SCOPES:
                if scope == "Full sample":
                    train, valid = full_train, full_valid
                else:
                    train = full_train.loc[
                        full_train["station_id"].astype(str).isin(stations)]
                    valid = full_valid.loc[
                        full_valid["station_id"].astype(str).isin(stations)]
                if train.empty or valid.empty:
                    raise ValueError(f"Empty fold: {date}, {scope}, {fold}")
                row = {
                    "Forecast": LABELS[date], "Sample": scope, "fold": fold,
                    "Train_years": f"{train.bloom_year.min()}-{train.bloom_year.max()}",
                    "Validation_years": f"{valid.bloom_year.min()}-{valid.bloom_year.max()}",
                    "Train_N": len(train), "N": len(valid),
                    "Selection_train_years": "1964-2004",
                    "Selection_N": selection_n,
                    "Selection_stations": selection_stations,
                    "Selection_alpha": alpha,
                    "Selected_feature_count": len(selected),
                    "Selected_features": "|".join(selected),
                }
                # Baselines use exactly the same rows for all three methods.
                add_metrics(row, "All_history", valid[TARGET],
                            climatology_baseline(train, valid))
                add_metrics(row, "Climatology_30yr", valid[TARGET],
                            climatology_30yr(train, valid))
                for method in OUTPUTS:
                    used = selected if method == "lasso_selected_ols" else features
                    estimator = (Lasso(alpha=alpha, max_iter=100000, tol=1e-5, precompute=True)
                                 if method == "lasso_all" else LinearRegression())
                    model = make_pipeline(StandardScaler(), estimator)
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", ConvergenceWarning)
                        model.fit(train[used], train[TARGET])
                    result = row.copy()
                    if method == "ols_all":
                        # No Lasso tuning or feature selection is applied to OLS-all.
                        result.update(Selection_train_years="", Selection_N=np.nan,
                                      Selection_stations=np.nan, Selection_alpha=np.nan)
                    result["Selected_feature_count"] = len(used)
                    result["Selected_features"] = "|".join(used)
                    add_metrics(result, "Model", valid[TARGET],
                                model.predict(valid[used]))
                    rows[method].append(result)

    output = PROJECT_ROOT / "results"
    output.mkdir(exist_ok=True)
    for method, filename in OUTPUTS.items():
        detail = pd.DataFrame(rows[method])
        parts = []
        for _, group in detail.groupby(["Forecast", "Sample"], sort=False):
            mean = make_mean(group)
            if method == "ols_all":
                mean["Selection_train_years"] = ""
            parts.extend([group, pd.DataFrame([mean])])
        report = pd.concat(parts, ignore_index=True)
        path = output / filename
        report.to_csv(path, index=False, float_format="%.10g")
        summary = report.loc[report["fold"].astype(str).eq("mean"), [
            "Forecast", "Sample", "N", "Model_MAE_days",
            "All_history_MAE_days", "Climatology_30yr_MAE_days"]]
        print(f"\n{method}: five-fold mean MAE")
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
