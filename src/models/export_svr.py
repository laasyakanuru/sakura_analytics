"""Train the deployed SVR on ALL years and save it for the app.

    python3 src/models/export_svr.py                 # the deployed model (15 Mar)
    python3 src/models/export_svr.py --date feb1     # another date, e.g. for analysis

Only the 15 March model is deployed: on the 2015-2025 test the 1 Feb and 1 Mar
models did not beat the 30-year climatology, so the app uses the usual date for
planning before 15 March (see src/models/rolling_test.py).

For each date the settings and validation metrics are read from the committed
results/svr/svr_<date>_cv.csv (full sample, team candidate predictors), and the
test metrics from results/svr/svr_<date>_test.csv when the final test was run.
Run the final test BEFORE exporting: the deployed models are refit on every year,
including the 2015-2025 test years.

Writes to app/model/ for each date:
  svr_<date>.joblib       StandardScaler + SVR pipeline (predicts the anomaly)
  svr_<date>_meta.json    features, params, metrics, station references and names
  features_<date>.csv     the feature table the app and batch_forecast.py read
"""
import argparse
import ast
import json
import sys
from pathlib import Path

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                    if (p / "requirements.txt").exists())
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

from model_data import TARGET, load_features  # noqa: E402
from svr import MIN_STATION_HISTORY, STATION_REF_YEARS, TEAM_FEATURES  # noqa: E402

DATES = ("feb1", "mar1", "mar15")
DEPLOYED = ("mar15",)
ISSUE = {"feb1": ("1 Feb", 32), "mar1": ("1 Mar", 60), "mar15": ("15 Mar", 74)}  # label, day of year
RESULTS = PROJECT_ROOT / "results" / "svr"
BASELINE_TEST = PROJECT_ROOT / "outputs" / "baseline" / "baseline_test_metrics.csv"
ROLLING_TEST = RESULTS / "svr_rolling_summary.csv"
OUT_DIR = PROJECT_ROOT / "app" / "model"


def metrics(path, fold):
    """(params, metric dict) from a results CSV row, or (None, None) if absent."""
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    row = df[df["fold"].astype(str) == fold].iloc[0]
    keys = ("MAE", "RMSE", "R2", "hit_rate_3d", "n")
    return ast.literal_eval(row["params"]), {k: float(row[k]) for k in keys}


def baseline_test():
    """30-year climatology on the same 2015-2025 test (identical for every date)."""
    if not BASELINE_TEST.exists():
        return None
    df = pd.read_csv(BASELINE_TEST)
    row = df[df["model"] == "30yr_climatology"].iloc[0]
    return {k: float(row[k]) for k in ("MAE", "RMSE", "R2", "hit_rate_3d", "n")}


def rolling_test(date):
    """Rolling-origin test 2015-2025 (refit every year) for this date, by method, if run."""
    if not ROLLING_TEST.exists():
        return None
    df = pd.read_csv(ROLLING_TEST)
    df = df[df["forecast_date"] == date]
    keys = ("MAE", "RMSE", "R2", "hit_rate_3d", "bias", "n")
    return {r["method"]: {k: float(r[k]) for k in keys} for _, r in df.iterrows()} or None


def station_references(df):
    """Mean full-bloom DOY per station over the last STATION_REF_YEARS years,
    falling back to the region mean, then the overall mean (same rule as svr.py)."""
    hist = df[df["bloom_year"] >= df["bloom_year"].max() - STATION_REF_YEARS + 1]
    station = hist.groupby("station_id")[TARGET].agg(["mean", "size"])
    region = hist.groupby("region")[TARGET].mean()
    overall = float(hist[TARGET].mean())
    info = df.sort_values("bloom_year").drop_duplicates("station_id", keep="last")
    refs = {}
    for row in info.itertuples(index=False):
        sid = str(row.station_id)
        if sid in station.index and station.loc[sid, "size"] >= MIN_STATION_HISTORY:
            refs[sid] = float(station.loc[sid, "mean"])
        else:
            refs[sid] = float(region.get(row.region, overall))
    return refs, info


def export(date):
    params, validation = metrics(RESULTS / f"svr_{date}_cv.csv", "mean")
    if params is None:
        raise SystemExit(f"results/svr/svr_{date}_cv.csv not found - run svr.py first.")
    _, test = metrics(RESULTS / f"svr_{date}_test.csv", "test")

    df = load_features(date)
    df["station_id"] = df["station_id"].astype(str)
    features = list(TEAM_FEATURES[date])
    refs, info = station_references(df)

    model = make_pipeline(StandardScaler(), SVR(**params))
    model.fit(df[features], df[TARGET] - df["station_id"].map(refs))

    label, issue_doy = ISSUE[date]
    lead = pd.Series(refs) - issue_doy          # usual days of notice per station
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_DIR / f"svr_{date}.joblib")
    meta = {
        "forecast_date": date,
        "issue_label": label,
        "issue_doy": issue_doy,
        "typical_lead_days": float(lead.median()),
        "params": params,
        "features": features,
        "trained_years": [int(df["bloom_year"].min()), int(df["bloom_year"].max())],
        "n_rows": int(len(df)),
        "sklearn_version": sklearn.__version__,
        "validation": validation,
        "test": test,
        "baseline_test": baseline_test(),
        "rolling_test": rolling_test(date),
        "station_reference_years": STATION_REF_YEARS,
        "station_reference": refs,
        "stations": {str(r.station_id): {"location": str(r.location),
                                         "prefecture": str(r.prefecture),
                                         "region": str(r.region)}
                     for r in info.itertuples(index=False)},
    }
    (OUT_DIR / f"svr_{date}_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    cols = ["station_id", "location", "prefecture", "region", "bloom_year",
            "full_bloom_date", TARGET] + features
    df[cols].to_csv(OUT_DIR / f"features_{date}.csv", index=False)
    print(f"{date}: {params} | validation MAE {validation['MAE']:.3f}"
          + (f" | test MAE {test['MAE']:.3f}" if test else " | no test run")
          + (f" (climatology {meta['baseline_test']['MAE']:.3f})" if test and meta["baseline_test"] else "")
          + f" | median notice {lead.median():.0f} days | sklearn {sklearn.__version__}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", choices=DATES,
                        help=f"export this date instead of the deployed {', '.join(DEPLOYED)}")
    args = parser.parse_args()
    for date in ([args.date] if args.date else DEPLOYED):
        export(date)


if __name__ == "__main__":
    main()
