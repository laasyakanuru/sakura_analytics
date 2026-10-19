"""Batch forecast: score every station for one season and save the result.

    python3 src/models/batch_forecast.py                    # latest season in the feature table
    python3 src/models/batch_forecast.py --year 2025
    python3 src/models/batch_forecast.py --year 2026 --input data/new_season_mar15.csv

Run once per forecast issue date (15 March). The Streamlit app's "Official forecast"
tab reads the saved file, so every viewer sees the same frozen forecast.

--input: a CSV with station_id plus the model's feature columns, one row per station
(for a season that is not in the feature table yet). Without it, the season's rows
are taken from app/model/features_mar15.csv.

Writes app/forecasts/forecast_<year>_mar15.csv. When the season's observed bloom
dates are known they are added (observed_doy, error_days) so the forecast can be scored.
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                    if (p / "requirements.txt").exists())
sys.path.insert(0, str(PROJECT_ROOT / "app"))
from predictor import BloomPredictor  # noqa: E402

OUT_DIR = PROJECT_ROOT / "app" / "forecasts"


def main():
    bp = BloomPredictor()
    table = bp.table
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, default=int(table["bloom_year"].max()))
    parser.add_argument("--input", type=Path, help="CSV of station_id + features for a new season")
    args = parser.parse_args()

    if args.input:
        rows = pd.read_csv(args.input, dtype={"station_id": str})
    else:
        rows = table[table["bloom_year"] == args.year]
        if rows.empty:
            raise SystemExit(f"No rows for {args.year} in features_mar15.csv; pass --input.")

    forecast = bp.predict_batch(rows, args.year)

    observed = table[table["bloom_year"] == args.year].set_index("station_id")["full_bloom_doy"]
    if not observed.empty:
        forecast["observed_doy"] = forecast["station_id"].map(observed)
        forecast["error_days"] = (forecast["predicted_doy"] - forecast["observed_doy"]).round(1)

    forecast["model"] = f"SVR {bp.meta['params']} trained {bp.meta['trained_years'][0]}-{bp.meta['trained_years'][1]}"
    forecast["generated_at"] = dt.datetime.now().isoformat(timespec="seconds")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"forecast_{args.year}_mar15.csv"
    forecast.to_csv(out, index=False)
    msg = f"{len(forecast)} stations forecast for {args.year}"
    if "error_days" in forecast:
        msg += f" (in-sample MAE vs observed: {forecast['error_days'].abs().mean():.2f} days)"
    print(f"{msg}. Saved {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
