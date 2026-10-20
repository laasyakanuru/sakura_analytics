"""Log every forecasting method, its evaluation metrics and the deployed model to MLflow.

    pip install mlflow
    python3 src/models/log_models.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db      # then open http://127.0.0.1:5000

Experiment "sakura-bloom", one run per method x forecast date (x sample). Nothing is
retrained: the script reads the results the team already committed (results/,
outputs/baseline/, app/model/). Filter the runs table by the tags model_family, status,
forecast_date and sample.

  usual-date-30yr   status=deployed (Plan stage)    30-year station climatology
  svr-mar15         status=deployed (Confirm stage) SVR on weather to 14 Mar; the fitted
                                                    pipeline is logged and registered as
                                                    "sakura-svr-mar15"
  svr-feb1, svr-mar1 status=rejected                 evaluated, did not beat the usual date
                                                    on 2015-2025; metrics only, no model
  mlr-*, rf-*, gb-*  status=candidate                team models compared on the same 5
                                                    validation folds; metrics + key params

Metrics (MAE / RMSE / R2 / hit_rate_3d; bias where available):
  val_*                5 expanding-window CV folds, 2005-2014 (+ val_fold_* per fold)
  test_*               final test 2015-2025, frozen at 2014
  rolling_test_*       2015-2025, refit before every season (rolling_test.py)
  clim_test_*, clim_rolling_test_*   the usual date on the same seasons, for comparison

Re-running adds a second copy of every run; delete mlflow.db and mlruns/ first for a
clean log. By default runs go to a local SQLite store (mlflow.db, artifacts in mlruns/), which also
enables the model registry. Set MLFLOW_TRACKING_URI to log to a hosted server instead.
"""
import ast
import json
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                    if (p / "requirements.txt").exists())

import joblib  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402

EXPERIMENT = "sakura-bloom"
RESULTS_ROOT = PROJECT_ROOT / "results"
RESULTS = PROJECT_ROOT / "results" / "svr"
BASELINE = PROJECT_ROOT / "outputs" / "baseline"
MODEL_DIR = PROJECT_ROOT / "app" / "model"
METRICS = ("MAE", "RMSE", "R2", "hit_rate_3d")
SVR_DATES = {  # date -> (issue label, n features, status, note)
    "feb1": ("1 Feb", 9, "rejected", "did not beat the usual date on 2015-2025"),
    "mar1": ("1 Mar", 17, "rejected", "did not beat the usual date on 2015-2025"),
    "mar15": ("15 Mar", 28, "deployed", "Confirm stage: beats the usual date on 2015-2025"),
}


def fold_row(path, fold):
    """{metric: value} for one fold row of a results CSV ({} if missing)."""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    row = df[df["fold"].astype(str) == fold]
    return {} if row.empty else {k: float(row.iloc[0][k]) for k in METRICS}


def rolling(date, method):
    """{metric: value} from the rolling-origin test ({} if it was not run)."""
    path = RESULTS / "svr_rolling_summary.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    row = df[(df["forecast_date"] == date) & (df["method"] == method)]
    return {} if row.empty else {k: float(row.iloc[0][k]) for k in (*METRICS, "bias")}


def climatology_test():
    path = BASELINE / "baseline_test_metrics.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    row = df[df["model"] == "30yr_climatology"].iloc[0]
    return {k: float(row[k]) for k in METRICS}


def prefixed(**groups):
    return {f"{p}_{k}": v for p, values in groups.items() for k, v in values.items()}


def log_folds(path):
    """Per-fold validation metrics as steps 1-5, so MLflow can chart them."""
    if not path.exists():
        return
    df = pd.read_csv(path)
    for _, r in df[df["fold"].astype(str) != "mean"].iterrows():
        for k in METRICS:
            mlflow.log_metric(f"val_fold_{k}", float(r[k]), step=int(r["fold"]))


def log_artifacts(paths, folder):
    for path in paths:
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path=folder)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def log_usual_date(commit):
    """The Plan-stage method: 30-year station climatology (no fitted model)."""
    with mlflow.start_run(run_name="usual-date-30yr"):
        mlflow.set_tags({"method": "climatology", "model_family": "Climatology", "forecast_date": "any",
                         "sample": "full", "status": "deployed", "stage": "plan",
                         "note": "usual date for planning before 15 March", "git_commit": commit})
        mlflow.log_params({"lookback_years": 30, "min_station_history": 10,
                           "fallback": "region mean, then overall mean"})
        metrics = prefixed(
            val=fold_row(BASELINE / "climatology_30yr_cv_metrics.csv", "mean"),
            test=climatology_test(),
            rolling_test=rolling("mar15", "clim_rolling"),
        )
        mlflow.log_metrics(metrics)
        log_folds(BASELINE / "climatology_30yr_cv_metrics.csv")
        log_artifacts([BASELINE / "climatology_30yr_cv_metrics.csv",
                       BASELINE / "baseline_test_metrics.csv"], "results")
        return metrics


def log_svr(date, commit):
    label, n_features, status, note = SVR_DATES[date]
    cv_path = RESULTS / f"svr_{date}_cv.csv"
    cv = pd.read_csv(cv_path)
    params = ast.literal_eval(cv.loc[cv["fold"].astype(str) == "mean", "params"].iloc[0])
    meta_path = MODEL_DIR / f"svr_{date}_meta.json"
    model_path = MODEL_DIR / f"svr_{date}.joblib"
    deployed = status == "deployed" and model_path.exists()

    with mlflow.start_run(run_name=f"svr-{date}"):
        mlflow.set_tags({"method": "SVR", "model_family": "SVR", "sample": "full",
                         "forecast_date": date, "issue_date": label,
                         "status": status, "stage": "confirm" if deployed else "evaluated",
                         "note": note, "git_commit": commit})
        mlflow.log_params({
            **{k: params[k] for k in ("kernel", "C", "epsilon", "gamma") if k in params},
            "target": "anomaly vs 30-yr station mean",
            "sample": "full (82 stations)",
            "feature_set": "team candidate predictors",
            "n_features": n_features,
            "cv": "5 expanding-window folds, 2005-2014",
        })
        metrics = prefixed(
            val=fold_row(cv_path, "mean"),
            test=fold_row(RESULTS / f"svr_{date}_test.csv", "test"),
            rolling_test=rolling(date, "svr_C_rolling"),
            rolling_ref_only_test=rolling(date, "svr_B_ref_only"),
            clim_test=climatology_test(),
            clim_rolling_test=rolling(date, "clim_rolling"),
        )
        mlflow.log_metrics(metrics)
        log_folds(cv_path)
        log_artifacts([cv_path, RESULTS / f"svr_{date}_tuning.csv", RESULTS / f"svr_{date}_test.csv",
                       RESULTS / "svr_rolling_summary.csv", RESULTS / "svr_rolling_by_year.csv"],
                      "results")

        if deployed:
            meta = json.loads(meta_path.read_text())
            mlflow.log_params({"trained_years": f"{meta['trained_years'][0]}-{meta['trained_years'][1]}",
                               "n_training_rows": meta["n_rows"],
                               "sklearn_version": meta["sklearn_version"]})
            mlflow.log_dict({"features": meta["features"]}, "features.json")
            log_artifacts([meta_path], "model_meta")
            model = joblib.load(model_path)
            table = pd.read_csv(MODEL_DIR / f"features_{date}.csv", dtype={"station_id": str})
            example = table[meta["features"]].head(5)
            kwargs = dict(sk_model=model, signature=infer_signature(example, model.predict(example)),
                          input_example=example, registered_model_name=f"sakura-svr-{date}")
            try:
                mlflow.sklearn.log_model(name="model", **kwargs)           # MLflow 3.x
            except TypeError:
                mlflow.sklearn.log_model(artifact_path="model", **kwargs)  # MLflow 2.x
        return metrics


# ------------------------------------------------------------------ team models
DATE_KEYS = {"February 1": "feb1", "March 1": "mar1", "March 15": "mar15",
             "Feb1": "feb1", "Mar1": "mar1", "Mar15": "mar15"}


def log_candidate(run_name, tags, params, val, folds, artifacts, commit):
    """One validation-only run for a team model. folds: list of (fold, {metric: value})."""
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({**tags, "status": "candidate", "git_commit": commit,
                         "cv": "5 expanding-window folds, 2005-2014"})
        mlflow.log_params(params)
        mlflow.log_metrics({f"val_{k}": v for k, v in val.items()})
        for fold, values in folds:
            for k, v in values.items():
                mlflow.log_metric(f"val_fold_{k}", v, step=int(fold))
        log_artifacts(artifacts, "results")
    return {f"val_{k}": v for k, v in val.items()}


def log_linear_regression(commit):
    """MLR (Lasso and OLS on all candidate predictors), full and small sample, anomaly target."""
    out = {}
    for method, label in (("lasso_all", "Lasso"), ("ols_all", "OLS")):
        path = RESULTS_ROOT / f"linear_regression_{method}_cv.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for (forecast, sample), g in df.groupby(["Forecast", "Sample"], sort=False):
            date, small = DATE_KEYS[forecast], sample.startswith("Small")

            def metrics(r):
                return {"MAE": float(r["anomaly_MAE_days"]), "RMSE": float(r["anomaly_RMSE_days"]),
                        "R2": float(r["anomaly_R2"]),
                        "hit_rate_3d": float(r["anomaly_Within_3_days_pct"]) / 100,
                        "bias": float(r["anomaly_Bias_days"])}
            mean = g[g["fold"].astype(str) == "mean"].iloc[0]
            folds = [(r["fold"], metrics(r)) for _, r in g[g["fold"].astype(str) != "mean"].iterrows()]
            params = {"estimator": label, "target": "anomaly vs 30-yr station mean",
                      "n_features": int(mean["anomaly_Selected_feature_count"])}
            if method == "lasso_all":
                params["alpha"] = float(mean["anomaly_Selection_alpha"])
            name = f"mlr-{label.lower()}-{date}-{'small' if small else 'full'}"
            out[name + " (candidate)"] = log_candidate(
                name, {"method": f"MLR ({label})", "model_family": "MLR", "forecast_date": date,
                       "sample": "small" if small else "full"},
                params, metrics(mean), folds, [path], commit)
    return out


def log_random_forest(commit):
    """RF: best model per forecast date for the large and small samples."""
    best = RESULTS_ROOT / "rf" / "★best_models_anomaly_ref30_trainall_comparison.csv"
    if not best.exists():
        return {}
    out = {}
    for _, r in pd.read_csv(best, encoding="utf-8-sig").iterrows():
        key, size = r["dataset"].split("-")
        date, small = DATE_KEYS[key], size == "small"
        fold_path = (RESULTS_ROOT / "rf" / f"{date}_large_anomaly_ref30_trainall_best_candidate_fold_results"
                     f"{'_small' if small else ''}.csv")
        folds = []
        if fold_path.exists():
            for _, f in pd.read_csv(fold_path, encoding="utf-8-sig").iterrows():
                folds.append((f["fold"], {"MAE": float(f["valid_MAE"]), "RMSE": float(f["valid_RMSE"]),
                                          "R2": float(f["valid_R2"]),
                                          "hit_rate_3d": float(f["valid_hit_rate_3d"])}))
        params = {"target": "anomaly vs 30-yr station mean", "training_window": str(r["training_window"]),
                  "n_features": int(r["features"]), "max_depth": str(r["max_depth"]),
                  "max_features": str(r["max_features"]), "min_samples_leaf": int(r["min_samples_leaf"]),
                  "n_estimators": int(r["n_estimators"])}
        val = {"MAE": float(r["valid_MAE"]), "RMSE": float(r["valid_RMSE"]), "R2": float(r["valid_R2"]),
               "hit_rate_3d": float(r["valid_hit_rate_3d"]), "train_MAE": float(r["train_MAE"])}
        name = f"rf-{date}-{'small' if small else 'large'}"
        out[name + " (candidate)"] = log_candidate(
            name, {"method": "Random forest", "model_family": "RF", "forecast_date": date,
                   "sample": "small" if small else "full"},
            params, val, folds, [best, fold_path], commit)
    return out


def log_gradient_boosting(commit):
    """GB: tuned anomaly model per forecast date (full sample, all training years)."""
    folder = RESULTS_ROOT / "gradient_boosting" / "03_anomaly"
    out = {}
    for date in ("feb1", "mar1", "mar15"):
        cv_path, tune_path = folder / f"gb_anomaly_{date}_cv.csv", folder / f"gb_anomaly_{date}_tuning.csv"
        if not cv_path.exists():
            continue
        params = {"target": "anomaly vs 30-yr station mean", "training_window": "all"}
        if tune_path.exists():
            tune = pd.read_csv(tune_path)
            best = tune.loc[tune["MAE"].idxmin()]
            params.update({"learning_rate": float(best["learning_rate"]),
                           "max_depth": int(best["max_depth"]), "n_estimators": int(best["n_estimators"])})
        cv = pd.read_csv(cv_path)
        folds = [(r["fold"], {k: float(r[k]) for k in METRICS})
                 for _, r in cv[cv["fold"].astype(str) != "mean"].iterrows()]
        name = f"gb-{date}-full"
        out[name + " (candidate)"] = log_candidate(
            name, {"method": "Gradient boosting", "model_family": "GB", "forecast_date": date,
                   "sample": "full"},
            params, fold_row(cv_path, "mean"), folds,
            [cv_path, tune_path, RESULTS_ROOT / "gradient_boosting" / "04_feature_importance"
             / f"gb_{date}_feature_importance.png"], commit)
    return out


def main():
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENT)
    commit = git_commit()

    summary = {"usual-date-30yr (deployed: plan)": log_usual_date(commit)}
    for date, (_, _, status, _) in SVR_DATES.items():
        if (RESULTS / f"svr_{date}_cv.csv").exists():
            summary[f"svr-{date} ({status})"] = log_svr(date, commit)

    for logger in (log_linear_regression, log_random_forest, log_gradient_boosting):
        summary.update(logger(commit))

    cols = ["val_MAE", "test_MAE", "rolling_test_MAE"]
    table = pd.DataFrame(summary).T.reindex(columns=cols)
    table["date"] = [next((d for d in ("mar15", "mar1", "feb1") if d in name), "any") for name in table.index]
    table = table.sort_values(["date", "val_MAE"], key=lambda c: c.map(
        {"any": 0, "feb1": 1, "mar1": 2, "mar15": 3}) if c.name == "date" else c)
    print(table.drop(columns="date").round(3).to_string())
    print(f"\nLogged {len(summary)} runs to {mlflow.get_tracking_uri()} (experiment '{EXPERIMENT}').")
    if mlflow.get_tracking_uri().startswith("sqlite"):
        print("View: mlflow ui --backend-store-uri sqlite:///mlflow.db  ->  http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
