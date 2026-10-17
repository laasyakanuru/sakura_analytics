"""PROOF OF CONCEPT - Mar 15 forecast date for the SVR (not an agreed team date).

    python3 src/models/svr/mar15_poc.py            # build features, tune, compare
    python3 src/models/svr/mar15_poc.py --no-tune  # quick run with DEFAULT_PARAMS

Question: the feb1 / mar1 models are late in years with a warm March (2008,
2013). Does seeing the first half of March fix that, and how much accuracy does
a forecast 2 weeks later buy?

What it does (nothing shared is changed; everything stays in this folder and
outputs/svr/mar15_poc/):
  1. Builds a Mar 15 feature table in Python from the cleaned daily weather
     (data/processed/02_cleaned_source_data/weather/02_cleaned_weather_daily_somei_1952_2025.csv.gz),
     using weather up to 14 March only:
       - all Mar 1 features, unchanged (taken from load_features("mar1"));
       - first half of March: tmean / tmax / tmin, gdd5, cum_tmax, cum_tmean,
         precipitation and sunshine for Mar 1-14 (same formulas as the SQL in
         sql/05_model_feature_tables.sql), plus chill and frost days in Mar 1-14;
       - cum_tmax_feb1_mar14: accumulated Tmax since Feb 1 (the 600 degC-rule quantity).
     Rows = the Mar 1 table's rows that also have no missing temperature day in
     Mar 1-14 (decided with data available on Mar 15). It first recomputes some
     Mar 1 features from the same daily file and checks they match the official
     table, to show the daily data and formulas agree with the warehouse.
     No station has ever reached full bloom before day 78; Mar 15 is day 74
     (75 in leap years), so every row is still a genuine forecast.
  2. Runs svr_model.py's SVR (same folds, metrics, anomaly target, tuning) on
     Mar 15, and on Feb 1 / Mar 1 restricted to the same rows, so all three
     dates are compared on identical station-years.
  3. Prints the comparison and the mean error per validation year.

The test years 2015-2025 are not used (RUN_FINAL_TEST stays False).
"""
import argparse
import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = next(p for p in HERE.parents if (p / "requirements.txt").exists())
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import svr_model  # noqa: E402
from model_data import TARGET, cv_folds, load_features  # noqa: E402

DAILY = (PROJECT_ROOT / "data" / "processed" / "02_cleaned_source_data" / "weather"
         / "02_cleaned_weather_daily_somei_1952_2025.csv.gz")
OUT_DIR = PROJECT_ROOT / "outputs" / "svr" / "mar15_poc"


def load_daily():
    w = pd.read_csv(DAILY, dtype={"station_id": str}, parse_dates=["date"],
                    usecols=["date", "station_id", "mean_temperature_c",
                             "maximum_temperature_c", "minimum_temperature_c",
                             "precipitation_mm", "sunshine_duration_hours"])
    w = w.rename(columns={"mean_temperature_c": "tm", "maximum_temperature_c": "tx",
                          "minimum_temperature_c": "tn", "precipitation_mm": "pr",
                          "sunshine_duration_hours": "su"})
    w["month"] = w["date"].dt.month
    w["day"] = w["date"].dt.day
    w["bloom_year"] = w["date"].dt.year + (w["month"] >= 10)
    return w


def window_features(w, mask, suffix):
    """Same formulas as sql/05_model_feature_tables.sql, for the days in `mask`."""
    d = w[mask].copy()
    d["gdd5"] = (d["tm"] - 5).clip(lower=0)
    d["temp_missing"] = d[["tm", "tx", "tn"]].isna().any(axis=1)
    g = d.groupby(["station_id", "bloom_year"])
    n_days = g.size()
    out = pd.DataFrame({
        f"tmean_{suffix}": g["tm"].mean().round(2),
        f"tmax_{suffix}": g["tx"].mean().round(2),
        f"tmin_{suffix}": g["tn"].mean().round(2),
        f"gdd5_{suffix}": g["gdd5"].sum().round(2),
        f"cum_tmax_{suffix}": g["tx"].sum().round(2),
        f"cum_tmean_{suffix}": g["tm"].sum().round(2),
        f"chill_days_0_7c_{suffix}": g["tm"].apply(lambda s: s.between(0, 7.2).sum()),
        f"frost_days_{suffix}": g["tn"].apply(lambda s: (s < 0).sum()),
        f"precip_{suffix}_mm": (g["pr"].mean() * n_days).round(2),
        f"precip_{suffix}_missing_days": g["pr"].apply(lambda s: s.isna().sum()),
        f"sunshine_{suffix}_h": (g["su"].mean() * n_days).round(2),
        f"sunshine_{suffix}_missing_days": g["su"].apply(lambda s: s.isna().sum()),
        f"temp_missing_days_{suffix}": g["temp_missing"].sum(),
    })
    return out


def check_against_official(w, mar1):
    """Recompute February features and compare with the official Mar 1 table."""
    feb = window_features(w, w["month"] == 2, "feb")
    m = mar1.set_index(["station_id", "bloom_year"]).join(feb, rsuffix="_poc", how="left")
    for col in ["tmean_feb", "gdd5_feb", "cum_tmax_feb", "precip_feb_mm"]:
        diff = (m[col] - m[f"{col}_poc"]).abs().max()
        print(f"  check {col:16s} max |official - recomputed| = {diff:.3f}")
        if diff > 0.02:
            raise SystemExit(f"{col} does not match the official table - stop.")


def build_mar15():
    print("Building Mar 15 features from the cleaned daily weather ...")
    mar1 = load_features("mar1")
    w = load_daily()
    check_against_official(w, mar1)

    first_half = (w["month"] == 3) & (w["day"] <= 14)
    mar = window_features(w, first_half, "mar1_14")
    feb1_mar14 = (w["month"] == 2) | first_half
    acc = w[feb1_mar14].groupby(["station_id", "bloom_year"])["tx"].sum().round(2)
    mar["cum_tmax_feb1_mar14"] = acc

    df = mar1.set_index(["station_id", "bloom_year"]).join(mar, how="inner")
    complete = (df["temp_missing_days_mar1_14"] == 0) & df["sunshine_mar1_14_h"].notna() \
        & df["precip_mar1_14_mm"].notna()
    dropped = len(df) - int(complete.sum())
    df = df[complete].drop(columns="temp_missing_days_mar1_14").reset_index()
    df = df.sort_values(["bloom_year", "station_id"]).reset_index(drop=True)
    # same column order as the official tables: features first, ids/target/splits last
    tail = ["full_bloom_date", TARGET, "split", "fold1", "fold2", "fold3", "fold4", "fold5"]
    df = df[[c for c in df.columns if c not in tail] + tail]
    print(f"  Mar 15 table: {len(df):,} rows ({dropped} Mar 1 rows dropped: missing "
          f"temperature / precipitation / sunshine for all of Mar 1-14)")
    return df


def per_year_bias(df, features, params):
    rows = []
    for _, train, valid in cv_folds(df):
        model = svr_model.make_model(features, params).fit(train, train[TARGET])
        err = model.predict(valid) - valid[TARGET].to_numpy()
        rows.append(pd.Series(err, index=valid["bloom_year"].to_numpy()))
    return pd.concat(rows).groupby(level=0).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-tune", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mar15 = build_mar15()
    mar15.to_csv(OUT_DIR / "model_features_mar15.csv", index=False)
    keys = mar15.set_index(["station_id", "bloom_year"]).index

    tables = {}
    for date in ("feb1", "mar1"):
        d = load_features(date, verbose=False)
        tables[date] = d[d.set_index(["station_id", "bloom_year"]).index.isin(keys)] \
            .reset_index(drop=True)
    tables["mar15"] = mar15

    summary, biases = [], {}
    for date, df in tables.items():
        svr_model.run(date, not args.no_tune, OUT_DIR, df=df)
        cv = pd.read_csv(OUT_DIR / f"{svr_model.run_name()}_{date}_cv.csv")
        mean = cv[cv["fold"] == "mean"].iloc[0]
        params = ast.literal_eval(mean["params"])  # written by svr_model.run
        biases[date] = per_year_bias(df, svr_model.choose_features(df), params)
        summary.append({"forecast_date": date, "features": len(svr_model.choose_features(df)),
                        "MAE": mean["MAE"], "RMSE": mean["RMSE"], "R2": mean["R2"],
                        "hit_rate_3d": mean["hit_rate_3d"]})

    summary = pd.DataFrame(summary)
    bias = pd.DataFrame(biases).T
    bias.columns = [int(c) for c in bias.columns]
    summary.to_csv(OUT_DIR / "comparison.csv", index=False)
    bias.round(2).to_csv(OUT_DIR / "bias_by_year.csv")

    n = len(mar15)
    print(f"\n=== Lead time vs accuracy (SVR, same {n:,} station-years, mean of 5 folds) ===")
    print(summary.round(3).to_string(index=False))
    print("\nMean error by validation year (days, + = predicted too late):")
    print(bias.round(2).to_string())
    print(f"\nsaved {OUT_DIR.relative_to(PROJECT_ROOT)}/ (features, cv tables, comparison)")


if __name__ == "__main__":
    main()
