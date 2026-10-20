"""Blind forward test of the champion model (15 March SVR) on the 2026 season.

    python3 src/models/forward_2026.py forecast     # step 1: issue the forecast, blind
    git add app/forecasts/forecast_2026_mar15.csv results/svr/forward_2026_forecast.sha256
    git commit -m "Issue blind 2026 forecast"       # freeze it before looking at results
    python3 src/models/forward_2026.py score        # step 2: compare with observed 2026 blooms

Step 1 (forecast)
  * Uses the DEPLOYED model app/model/svr_mar15.joblib, trained on 1964-2025 only, and its
    station references (each station's mean full bloom over 1996-2025). Nothing is refit.
  * Reads data/forward_2026/model/model_features_mar15_2026.csv and DROPS the observed
    columns (full_bloom_doy, full_bloom_date) before the model or any other code sees the rows.
  * Saves app/forecasts/forecast_2026_mar15.csv (no observed values in it) and writes its
    SHA-256 to results/svr/forward_2026_forecast.sha256, so any later edit is detectable.
  * Refuses to overwrite an existing forecast (use --force only if you have NOT scored yet).

Step 2 (score)
  * Checks the forecast file still matches the saved SHA-256, then joins the observed 2026
    full-bloom dates and scores the SVR and the usual date (the same 1996-2025 station means).
  * Writes results/svr/forward_2026_scores.csv (summary) and
    results/svr/forward_2026_by_station.csv (per station).
"""
import argparse
import datetime as dt
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                    if (p / "requirements.txt").exists())
sys.path.insert(0, str(PROJECT_ROOT / "app"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FEATURES_2026 = PROJECT_ROOT / "data" / "forward_2026" / "model" / "model_features_mar15_2026.csv"
FORECAST = PROJECT_ROOT / "app" / "forecasts" / "forecast_2026_mar15.csv"
HASH_FILE = PROJECT_ROOT / "results" / "svr" / "forward_2026_forecast.sha256"
SCORES = PROJECT_ROOT / "results" / "svr" / "forward_2026_scores.csv"
BY_STATION = PROJECT_ROOT / "results" / "svr" / "forward_2026_by_station.csv"
OBSERVED_COLUMNS = ("full_bloom_doy", "full_bloom_date")
YEAR = 2026


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forecast(force=False):
    if FORECAST.exists() and not force:
        sys.exit(f"{FORECAST.relative_to(PROJECT_ROOT)} already exists. The 2026 forecast is "
                 "issued once; use --force only if you have not scored it yet.")
    from predictor import BloomPredictor
    bp = BloomPredictor("mar15")
    if bp.meta["trained_years"][1] >= YEAR:
        sys.exit(f"The deployed model was trained up to {bp.meta['trained_years'][1]}; "
                 "it must not include 2026. Re-export it on 1964-2025 first.")

    rows = pd.read_csv(FEATURES_2026, dtype={"station_id": str})
    rows = rows.drop(columns=[c for c in OBSERVED_COLUMNS if c in rows.columns])   # blind
    assert not any(c in rows.columns for c in OBSERVED_COLUMNS)
    rows = rows[rows["bloom_year"] == YEAR]

    fc = bp.predict_batch(rows, YEAR)
    fc["model"] = (f"SVR {bp.meta['params']} trained "
                   f"{bp.meta['trained_years'][0]}-{bp.meta['trained_years'][1]}")
    fc["issued_at"] = dt.datetime.now().isoformat(timespec="seconds")
    FORECAST.parent.mkdir(parents=True, exist_ok=True)
    fc.to_csv(FORECAST, index=False)

    HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    HASH_FILE.write_text(f"{sha256(FORECAST)}  {FORECAST.relative_to(PROJECT_ROOT)}\n")
    first, last = fc.iloc[0], fc.iloc[-1]
    print(f"Blind 2026 forecast issued for {len(fc)} stations "
          f"(earliest {first.location} {first.predicted_date}, latest {last.location} {last.predicted_date}).")
    print(f"Saved {FORECAST.relative_to(PROJECT_ROOT)} and its SHA-256 in "
          f"{HASH_FILE.relative_to(PROJECT_ROOT)}. Commit both before scoring.")


def metrics(pred, obs):
    err = np.asarray(pred, float) - np.asarray(obs, float)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return {"n": len(err), "MAE": np.abs(err).mean(), "RMSE": np.sqrt(np.mean(err ** 2)),
            "R2": 1 - np.sum(err ** 2) / ss_tot if ss_tot > 0 else np.nan,
            "hit_rate_3d": np.mean(np.abs(err) <= 3), "bias": err.mean()}


def score():
    if not FORECAST.exists() or not HASH_FILE.exists():
        sys.exit("No issued forecast found. Run `forward_2026.py forecast` first.")
    saved = HASH_FILE.read_text().split()[0]
    if sha256(FORECAST) != saved:
        sys.exit("The forecast file has changed since it was issued (SHA-256 mismatch). "
                 "Scoring refused.")

    fc = pd.read_csv(FORECAST, dtype={"station_id": str})
    obs = pd.read_csv(FEATURES_2026, dtype={"station_id": str})[["station_id", "full_bloom_doy"]]
    df = fc.merge(obs, on="station_id", how="inner").dropna(subset=["full_bloom_doy"])
    df["svr_error"] = (df["predicted_doy"] - df["full_bloom_doy"]).round(1)
    df["usual_date_error"] = (df["reference_doy"] - df["full_bloom_doy"]).round(1)

    summary = pd.DataFrame([
        {"method": "SVR 15 March (deployed)", **metrics(df["predicted_doy"], df["full_bloom_doy"])},
        {"method": "Usual date (1996-2025 mean)", **metrics(df["reference_doy"], df["full_bloom_doy"])},
    ])
    SCORES.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SCORES, index=False)
    cols = ["station_id", "location", "prefecture", "region", "predicted_doy", "reference_doy",
            "full_bloom_doy", "svr_error", "usual_date_error"]
    df[cols].sort_values("svr_error", key=abs, ascending=False).to_csv(BY_STATION, index=False)

    print(f"2026 forward test: {len(df)} stations, forecast issued {fc['issued_at'].iloc[0]} "
          "(file unchanged since issue)\n")
    print(summary.set_index("method").round(3).to_string())
    print("\nLargest SVR misses (days, + = forecast too late):")
    print(df[cols].sort_values("svr_error", key=abs, ascending=False).head(5).to_string(index=False))
    print(f"\nSaved {SCORES.relative_to(PROJECT_ROOT)} and {BY_STATION.relative_to(PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)
    f = sub.add_parser("forecast", help="issue the blind 2026 forecast")
    f.add_argument("--force", action="store_true", help="overwrite an unscored forecast")
    sub.add_parser("score", help="score the issued forecast against observed 2026 blooms")
    args = parser.parse_args()
    forecast(args.force) if args.step == "forecast" else score()


if __name__ == "__main__":
    main()
