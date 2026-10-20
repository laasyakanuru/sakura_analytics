"""Rolling-origin (walk-forward) test of the SVR, alongside the frozen final test.

    python3 src/models/rolling_test.py                  # all three forecast dates
    python3 src/models/rolling_test.py --dates mar15

Question: is the drop from validation to the 2015-2025 test caused by the model,
or by the model going stale between its 2014 training cut-off and the forecast year?

For every test year Y (2015-2025) and forecast date, three SVR variants:
  A  frozen     model and 30-yr station reference fitted on 1964-2014 (= the official test)
  B  ref-only   same frozen model, but the station reference recomputed from all seasons < Y
  C  rolling    model and reference refitted on all seasons < Y (retrain every year)
plus the 30-year climatology baseline, frozen (training rows <= 2014) and rolling (< Y).

Nothing is tuned here: kernel, C, epsilon and gamma are read from the validated
results/svr/svr_<date>_cv.csv. Season Y is never part of its own training data.
The frozen result (A) stays the official test number; run this ONCE and report it
as a second evaluation design.

Writes:
  results/svr/svr_rolling_summary.csv   MAE / RMSE / R2 / hit rate / bias per date and method
  results/svr/svr_rolling_by_year.csv   the same per test year (with years since 2014)
"""
import argparse
import ast
import sys
from pathlib import Path

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                    if (p / "requirements.txt").exists())
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import svr  # noqa: E402
from model_data import TARGET, evaluate, load_features  # noqa: E402

DATES = ("feb1", "mar1", "mar15")
TRAIN_END, TEST_YEARS = 2014, range(2015, 2026)
RESULTS = PROJECT_ROOT / "results" / "svr"
METHODS = {
    "clim_frozen": "Climatology, frozen at 2014",
    "clim_rolling": "Climatology, rolling",
    "svr_A_frozen": "SVR A: frozen (official test)",
    "svr_B_ref_only": "SVR B: frozen model, rolling reference",
    "svr_C_rolling": "SVR C: retrained every year",
}


def validated_params(date):
    cv = pd.read_csv(RESULTS / f"svr_{date}_cv.csv")
    return ast.literal_eval(cv.loc[cv["fold"].astype(str) == "mean", "params"].iloc[0])


def predictions(date):
    """One row per test station-season with every method's prediction."""
    df = load_features(date)
    features = svr.choose_features(df, date)
    params = validated_params(date)
    dev = df[df["bloom_year"] <= TRAIN_END]
    frozen = svr.make_model(features, params).fit(dev, dev[TARGET])
    print(f"{date}: {len(features)} features, {params}", flush=True)

    out = []
    for year in TEST_YEARS:
        test = df[df["bloom_year"] == year]
        if test.empty:
            continue
        past = df[df["bloom_year"] < year]               # everything known before season Y
        rolling = svr.make_model(features, params).fit(past, past[TARGET])
        ref_now = svr.station_reference(past, svr.STATION_REF_YEARS,
                                        svr.MIN_STATION_HISTORY)
        out.append(pd.DataFrame({
            "forecast_date": date,
            "bloom_year": year,
            "years_since_training": year - TRAIN_END,
            "station_id": test["station_id"].to_numpy(),
            "observed": test[TARGET].to_numpy(dtype=float),
            "clim_frozen": svr.climatology_30yr(dev, test),
            "clim_rolling": svr.climatology_30yr(past, test),
            "svr_A_frozen": frozen.predict(test),
            "svr_B_ref_only": frozen.model_.predict(test[features]) + ref_now(test),
            "svr_C_rolling": rolling.predict(test),
        }))
        print(f"  {year} done ({len(test)} stations)", flush=True)
    return pd.concat(out, ignore_index=True)


def score(group):
    rows = []
    for method in METHODS:
        m = evaluate(group["observed"], group[method])
        m["bias"] = float((group[method] - group["observed"]).mean())
        rows.append({"method": method, **m})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dates", nargs="+", choices=DATES, default=list(DATES))
    args = parser.parse_args()

    preds = pd.concat([predictions(d) for d in args.dates], ignore_index=True)

    summary = pd.concat(
        [score(g).assign(forecast_date=d) for d, g in preds.groupby("forecast_date", sort=False)],
        ignore_index=True)
    by_year = pd.concat(
        [score(g).assign(forecast_date=d, bloom_year=y, years_since_training=y - TRAIN_END)
         for (d, y), g in preds.groupby(["forecast_date", "bloom_year"], sort=False)],
        ignore_index=True)
    front = ["forecast_date", "bloom_year", "years_since_training", "method"]
    summary = summary[["forecast_date"] + [c for c in summary if c != "forecast_date"]]
    by_year = by_year[front + [c for c in by_year if c not in front]]
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS / "svr_rolling_summary.csv", index=False)
    by_year.to_csv(RESULTS / "svr_rolling_by_year.csv", index=False)

    pd.set_option("display.width", 140)
    for date, s in summary.groupby("forecast_date", sort=False):
        print(f"\n=== {date}: 2015-2025 test, {int(s['n'].iloc[0])} station-seasons ===")
        t = s.set_index("method")[["MAE", "RMSE", "R2", "hit_rate_3d", "bias"]]
        t.index = [METHODS[m] for m in t.index]
        print(t.round(3).to_string())

    gaps = preds.assign(gap=pd.cut(preds["years_since_training"], [0, 3, 7, 11],
                                   labels=["2015-17", "2018-21", "2022-25"]))
    mae = pd.DataFrame([
        {"forecast_date": d, "years": str(gap),
         **{m: (g[m] - g["observed"]).abs().mean() for m in METHODS}}
        for (d, gap), g in gaps.groupby(["forecast_date", "gap"], observed=True, sort=False)
    ]).set_index(["forecast_date", "years"])
    print("\n=== MAE by years since the 2014 training cut-off ===")
    print(mae.round(2).to_string())
    print(f"\nsaved {(RESULTS / 'svr_rolling_summary.csv').relative_to(PROJECT_ROOT)} and "
          f"{(RESULTS / 'svr_rolling_by_year.csv').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
