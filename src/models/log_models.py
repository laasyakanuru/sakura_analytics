"""Log the forecasting methods, their evaluation metrics and the deployed model to MLflow.

    pip install mlflow
    python3 src/models/log_models.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db      # then open http://127.0.0.1:5000

Experiment "sakura-svr", one run per method. Nothing is retrained: the script reads
what the pipeline already produced (results/svr/, outputs/baseline/, app/model/).

  usual-date-30yr   status=deployed (Plan stage)    30-year station climatology
  svr-mar15         status=deployed (Confirm stage) SVR on weather to 14 Mar; the fitted
                                                    pipeline is logged and registered as
                                                    "sakura-svr-mar15"
  svr-feb1, svr-mar1 status=rejected                 evaluated, did not beat the usual date
                                                    on 2015-2025; metrics only, no model

Metrics (MAE / RMSE / R2 / hit_rate_3d; bias where available):
  val_*                5 expanding-window CV folds, 2005-2014 (+ val_fold_* per fold)
  test_*               final test 2015-2025, frozen at 2014
  rolling_test_*       2015-2025, refit before every season (rolling_test.py)
  clim_test_*, clim_rolling_test_*   the usual date on the same seasons, for comparison

By default runs go to a local SQLite store (mlflow.db, artifacts in mlruns/), which also
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

EXPERIMENT = "sakura-svr"
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
        mlflow.set_tags({"method": "climatology", "status": "deployed", "stage": "plan",
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
        mlflow.set_tags({"method": "SVR", "forecast_date": date, "issue_date": label,
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


def main():
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENT)
    commit = git_commit()

    summary = {"usual-date-30yr (deployed: plan)": log_usual_date(commit)}
    for date, (_, _, status, _) in SVR_DATES.items():
        if (RESULTS / f"svr_{date}_cv.csv").exists():
            summary[f"svr-{date} ({status})"] = log_svr(date, commit)

    cols = ["val_MAE", "test_MAE", "rolling_test_MAE"]
    print(pd.DataFrame(summary).T.reindex(columns=cols).round(3).to_string())
    print(f"\nLogged {len(summary)} runs to {mlflow.get_tracking_uri()} (experiment '{EXPERIMENT}').")
    if mlflow.get_tracking_uri().startswith("sqlite"):
        print("View: mlflow ui --backend-store-uri sqlite:///mlflow.db  ->  http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
