"""Batch forecast: score every station for one season and save a frozen copy.

    python3 src/models/batch_forecast.py                     # latest season, deployed model(s)
    python3 src/models/batch_forecast.py --year 2024
    python3 src/models/batch_forecast.py --year 2026 --date mar15 --input data/new_mar15.csv

The Streamlit app computes forecasts itself; this script keeps an archived copy of
what was issued (e.g. run on 15 March each season).

--input: a CSV with station_id plus that date's model features, one row per
station (for a season that is not in the feature table yet). Needs --date.
Without it, the season's rows come from app/model/features_<date>.csv.

Writes app/forecasts/forecast_<year>_<date>.csv. When the season's observed bloom
dates are known they are added (observed_doy, error_days) so it can be scored.
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                    if (p / "requirements.txt").exists())
sys.path.insert(0, str(PROJECT_ROOT / "app"))
from predictor import DATES, BloomPredictor  # noqa: E402

OUT_DIR = PROJECT_ROOT / "app" / "forecasts"


def run(date, year, input_csv=None):
    bp = BloomPredictor(date)
    table = bp.table
    year = int(table["bloom_year"].max()) if year is None else year
    if input_csv:
        rows = pd.read_csv(input_csv, dtype={"station_id": str})
    else:
        rows = table[table["bloom_year"] == year]
        if rows.empty:
            raise SystemExit(f"No {year} rows in features_{date}.csv; pass --input.")

    forecast = bp.predict_batch(rows, year)
    observed = table[table["bloom_year"] == year].set_index("station_id")["full_bloom_doy"]
    if not observed.empty:
        forecast["observed_doy"] = forecast["station_id"].map(observed)
        forecast["error_days"] = (forecast["predicted_doy"] - forecast["observed_doy"]).round(1)
    forecast["model"] = (f"SVR {bp.meta['params']} trained "
                         f"{bp.meta['trained_years'][0]}-{bp.meta['trained_years'][1]}")
    forecast["generated_at"] = dt.datetime.now().isoformat(timespec="seconds")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"forecast_{year}_{date}.csv"
    forecast.to_csv(out, index=False)
    msg = f"{date} {year}: {len(forecast)} stations"
    if "error_days" in forecast:
        msg += f", in-sample MAE vs observed {forecast['error_days'].abs().mean():.2f} days"
    print(f"{msg}. Saved {out.relative_to(PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, help="bloom year (default: latest in the feature table)")
    parser.add_argument("--date", choices=DATES, help="one forecast date (default: every exported model)")
    parser.add_argument("--input", type=Path, help="CSV of station_id + features for a new season")
    args = parser.parse_args()
    if args.input and not args.date:
        parser.error("--input needs --date (each date uses different features)")
    exported = [d for d in DATES if (PROJECT_ROOT / "app" / "model" / f"svr_{d}.joblib").exists()]
    for date in ([args.date] if args.date else exported):
        run(date, args.year, args.input)


if __name__ == "__main__":
    main()
