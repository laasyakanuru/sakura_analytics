"""Train the champion SVR on ALL years and save it for the Streamlit app.

    python3 src/models/export_svr.py

Champion (validation MAE 2.950 days): Mar 15 forecast, full sample, team
candidate predictors, anomaly target vs 30-year station reference.
Run the final test (RUN_FINAL_TEST in svr.py) BEFORE using this: the deployed
model is refit on every year, including the 2015-2025 test years.

Writes to app/model/:
  svr_mar15.joblib      StandardScaler + SVR pipeline (predicts the anomaly)
  svr_mar15_meta.json   features, params, station references, station names
  features_mar15.csv    the feature table the app looks rows up in
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                    if (p / "requirements.txt").exists())
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

import joblib  # noqa: E402
import sklearn  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

from model_data import TARGET, load_features  # noqa: E402
from svr import MIN_STATION_HISTORY, STATION_REF_YEARS, TEAM_FEATURES  # noqa: E402

FORECAST_DATE = "mar15"
PARAMS = {"kernel": "rbf", "C": 30, "epsilon": 2.0, "gamma": 0.003}
OUT_DIR = PROJECT_ROOT / "app" / "model"


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


def main():
    df = load_features(FORECAST_DATE)
    df["station_id"] = df["station_id"].astype(str)
    features = list(TEAM_FEATURES[FORECAST_DATE])
    refs, info = station_references(df)

    reference = df["station_id"].map(refs)
    model = make_pipeline(StandardScaler(), SVR(**PARAMS))
    model.fit(df[features], df[TARGET] - reference)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_DIR / "svr_mar15.joblib")
    meta = {
        "forecast_date": FORECAST_DATE,
        "params": PARAMS,
        "features": features,
        "trained_years": [int(df["bloom_year"].min()), int(df["bloom_year"].max())],
        "n_rows": int(len(df)),
        "sklearn_version": sklearn.__version__,
        "validation_mae_days": 2.950,
        "station_reference_years": STATION_REF_YEARS,
        "station_reference": refs,
        "stations": {str(r.station_id): {"location": str(r.location),
                                         "prefecture": str(r.prefecture),
                                         "region": str(r.region)}
                     for r in info.itertuples(index=False)},
    }
    (OUT_DIR / "svr_mar15_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    cols = ["station_id", "location", "prefecture", "region", "bloom_year",
            "full_bloom_date", TARGET] + features
    df[cols].to_csv(OUT_DIR / "features_mar15.csv", index=False)
    print(f"Trained on {len(df):,} rows ({meta['trained_years'][0]}-{meta['trained_years'][1]}), "
          f"{len(refs)} stations, sklearn {sklearn.__version__}. Saved to {OUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
