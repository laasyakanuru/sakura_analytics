"""Pandas re-implementation of sql/05_model_feature_tables.sql.

Reproduces MySQL DECIMAL arithmetic exactly: daily values are held as integer
hundredths, AVG() is rounded half away from zero to 6 decimals (DECIMAL(6,2)
average scale) and then ROUND(..., 2) half away from zero.

Usage: python build_model_features.py <project_dir> <daily_csv_gz> <last_bloom_year> <out_dir>
"""
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(sys.argv[1])
DAILY = Path(sys.argv[2])
LAST_YEAR = int(sys.argv[3])
OUT = Path(sys.argv[4])
OUT.mkdir(parents=True, exist_ok=True)

import types
_stub = types.ModuleType("db_connection")
_stub.credentials_configured = lambda: False
_stub.get_engine = lambda: None
sys.modules["db_connection"] = _stub   # no MySQL here; only _normalise is used
sys.path.insert(0, str(PROJECT / "src"))
from model_data import _normalise  # noqa: E402  (same dtypes as the team's CSV export)


def rha(x, nd):
    """ROUND half away from zero of a Fraction to nd decimals -> Fraction."""
    if x is None:
        return None
    q = Fraction(10) ** nd
    v = abs(x) * q
    n = int(v)
    if v - n >= Fraction(1, 2):
        n += 1
    return Fraction(n if x >= 0 else -n) / q


def avg2(total, count):
    """MySQL ROUND(AVG(decimal(.,2)), 2): exact mean -> 6 dp -> 2 dp."""
    if count == 0:
        return None
    return rha(rha(Fraction(int(total), 100 * int(count)), 6), 2)


def fr2(x):
    return None if x is None else float(x)


# ---------------------------------------------------------------- dimensions
meta_dir = PROJECT / "data/processed/01_source_data/01_station_metadata"
names = pd.read_csv(meta_dir / "01_01_station_name.csv", dtype={"station_id": str}, encoding="utf-8-sig")
elev = pd.read_csv(meta_dir / "01_02_station_elevations.csv", dtype={"station_id": str}, encoding="utf-8-sig")
spec = pd.read_csv(meta_dir / "01_03_cherry_species.csv", dtype={"station_id": str}, encoding="utf-8-sig")
dim_station = names.merge(elev, on="station_id").merge(spec, on="station_id")
dim_station = dim_station[dim_station.cherry_species == "Somei-yoshino"]

season = pd.read_csv(PROJECT / "data/processed/02_cleaned_source_data/weather/03_weather_features_somei_station_year.csv",
                     dtype={"station_id": str})
season = season[["station_id", "year", "relocations_before_season", "full_bloom_date", "full_bloom_doy"]]
season = season.rename(columns={"year": "bloom_year"})

# ---------------------------------------------------------------- daily facts
d = pd.read_csv(DAILY, dtype={"station_id": str},
                usecols=["date", "station_id", "mean_temperature_c", "maximum_temperature_c",
                         "minimum_temperature_c", "precipitation_mm", "sunshine_duration_hours"])
d["date"] = pd.to_datetime(d.date)
d["month"] = d.date.dt.month
d["day"] = d.date.dt.day
d["bloom_year"] = d.date.dt.year + (d.month >= 10).astype(int)
for src, dst in [("mean_temperature_c", "tm"), ("maximum_temperature_c", "tx"), ("minimum_temperature_c", "tn"),
                 ("precipitation_mm", "pr"), ("sunshine_duration_hours", "su")]:
    d[dst] = np.round(d[src] * 100).astype("Int64")          # exact hundredths
d = d[d.bloom_year.between(1964, LAST_YEAR)]


def split_row(y):
    if y <= 2014:
        split = "development"
    elif y <= 2025:
        split = "test"
    else:
        split = f"forward_{y}"
    folds = []
    for k in range(1, 6):
        tr_end, va_end = 2002 + 2 * k, 2004 + 2 * k
        folds.append("train" if y <= tr_end else "valid" if y <= va_end else None)
    if y > 2025:
        folds = [None] * 5
    return [split] + folds


def build(cutoff):
    """cutoff: 'jan31' | 'feb_end' | 'mar14'"""
    months = {"jan31": [10, 11, 12, 1], "feb_end": [10, 11, 12, 1, 2], "mar14": [10, 11, 12, 1, 2, 3]}[cutoff]
    w = d[d.month.isin(months)]
    if cutoff == "mar14":
        w = w[(w.month != 3) | (w.day <= 14)]
    rows = []
    for (sid, y), g in w.groupby(["station_id", "bloom_year"], sort=True):
        r = {"station_id": sid, "bloom_year": y}
        tm, tx, tn, pr, su, mo = (g[c] for c in ["tm", "tx", "tn", "pr", "su", "month"])
        r["temp_missing_days_to_cutoff"] = int((tm.isna() | tx.isna() | tn.isna()).sum())

        def avg(col, m):
            s = col[mo.isin(m)].dropna()
            return fr2(avg2(s.sum(), len(s)))

        def ssum(col, m):
            s = col[mo.isin(m)].dropna()
            return None if len(s) == 0 else float(Fraction(int(s.sum()), 100))

        def gdd(m):
            s = tm[mo.isin(m)].dropna()
            return None if len(s) == 0 else float(Fraction(int(np.maximum(s.to_numpy(dtype=np.int64) - 500, 0).sum()), 100))

        def scaled(col, m):          # ROUND(AVG(col) * COUNT(days), 2)
            s = col[mo.isin(m)].dropna()
            n_days = int(mo.isin(m).sum())
            if len(s) == 0:
                return None
            return fr2(rha(rha(Fraction(int(s.sum()), 100 * len(s)), 6) * n_days, 2))

        def cnt(mask):
            return int(mask.fillna(False).sum())

        def chill(m):
            return cnt(mo.isin(m) & (tm >= 0) & (tm <= 720))

        def frost(m):
            return cnt(mo.isin(m) & (tn < 0))

        def miss(col, m):
            return int((mo.isin(m) & col.isna()).sum())

        def tmin_lowest(m):
            s = tn[mo.isin(m)].dropna()
            return None if len(s) == 0 else float(Fraction(int(s.min()), 100))

        r["tmean_oct"], r["tmean_nov"], r["tmean_dec"], r["tmean_jan"] = (avg(tm, [k]) for k in (10, 11, 12, 1))
        if cutoff == "jan31":
            r["tmax_jan"], r["tmin_jan"] = avg(tx, [1]), avg(tn, [1])
            r["chill_days_0_7c_nov_jan"] = chill([11, 12, 1])
            r["tmin_lowest_nov_jan"] = tmin_lowest([11, 12, 1])
            r["frost_days_jan"] = frost([1])
            r["gdd5_jan"], r["cum_tmax_jan"], r["cum_tmean_jan"] = gdd([1]), ssum(tx, [1]), ssum(tm, [1])
            r["precip_jan_mm"], r["precip_jan_missing_days"] = scaled(pr, [1]), miss(pr, [1])
            r["sunshine_jan_h"], r["sunshine_jan_missing_days"] = scaled(su, [1]), miss(su, [1])
        else:
            r["tmean_feb"] = avg(tm, [2])
            r["tmax_jan"], r["tmin_jan"], r["tmax_feb"], r["tmin_feb"] = avg(tx, [1]), avg(tn, [1]), avg(tx, [2]), avg(tn, [2])
            r["chill_days_0_7c_nov_feb"] = chill([11, 12, 1, 2])
            r["tmin_lowest_nov_feb"] = tmin_lowest([11, 12, 1, 2])
            r["frost_days_jan_feb"] = frost([1, 2])
            r["gdd5_feb"], r["cum_tmax_feb"], r["cum_tmean_feb"] = gdd([2]), ssum(tx, [2]), ssum(tm, [2])
            r["precip_feb_mm"], r["precip_feb_missing_days"] = scaled(pr, [2]), miss(pr, [2])
            r["sunshine_feb_h"], r["sunshine_feb_missing_days"] = scaled(su, [2]), miss(su, [2])
            if cutoff == "mar14":
                r["tmean_mar1_14"], r["tmax_mar1_14"], r["tmin_mar1_14"] = avg(tm, [3]), avg(tx, [3]), avg(tn, [3])
                r["gdd5_mar1_14"], r["cum_tmax_mar1_14"], r["cum_tmean_mar1_14"] = gdd([3]), ssum(tx, [3]), ssum(tm, [3])
                r["chill_days_0_7c_mar1_14"] = chill([3])
                r["frost_days_mar1_14"] = frost([3])
                r["precip_mar1_14_mm"], r["precip_mar1_14_missing_days"] = scaled(pr, [3]), miss(pr, [3])
                r["sunshine_mar1_14_h"], r["sunshine_mar1_14_missing_days"] = scaled(su, [3]), miss(su, [3])
                r["cum_tmax_feb1_mar14"] = ssum(tx, [2, 3])
        rows.append(r)
    a = pd.DataFrame(rows)
    t = (a.merge(season, on=["station_id", "bloom_year"])
          .merge(dim_station, on="station_id"))
    t = t[t.full_bloom_doy.notna() & (t.temp_missing_days_to_cutoff == 0)].copy()
    sp = pd.DataFrame([[y] + split_row(y) for y in sorted(t.bloom_year.unique())],
                      columns=["bloom_year", "split", "fold1", "fold2", "fold3", "fold4", "fold5"])
    t = t.merge(sp, on="bloom_year")

    # whole period missing -> same-year regional mean of the rows in this table
    fill_cols = {"jan31": ["precip_jan_mm", "sunshine_jan_h"],
                 "feb_end": ["precip_feb_mm", "sunshine_feb_h"],
                 "mar14": ["precip_feb_mm", "sunshine_feb_h", "precip_mar1_14_mm", "sunshine_mar1_14_h"]}[cutoff]
    for c in fill_cols:
        for (reg, y), g in t.groupby(["region", "bloom_year"]):
            nn = g[c].dropna()
            if g[c].isna().any() and len(nn):
                tot = sum(Fraction(str(v)) for v in nn)
                val = rha(rha(tot / len(nn), 6), 2)
                t.loc[g.index[g[c].isna()], c] = float(val)
    return t


ORDER = {
    "feb1": ["station_id", "location", "prefecture", "region", "latitude", "longitude", "elevation_m",
             "bloom_year", "relocations_before_season", "tmean_oct", "tmean_nov", "tmean_dec", "tmean_jan",
             "tmax_jan", "tmin_jan", "chill_days_0_7c_nov_jan", "tmin_lowest_nov_jan", "frost_days_jan",
             "gdd5_jan", "cum_tmax_jan", "cum_tmean_jan", "precip_jan_mm", "precip_jan_missing_days",
             "sunshine_jan_h", "sunshine_jan_missing_days"],
    "mar1": ["station_id", "location", "prefecture", "region", "latitude", "longitude", "elevation_m",
             "bloom_year", "relocations_before_season", "tmean_oct", "tmean_nov", "tmean_dec", "tmean_jan",
             "tmean_feb", "tmax_jan", "tmin_jan", "tmax_feb", "tmin_feb", "chill_days_0_7c_nov_feb",
             "tmin_lowest_nov_feb", "frost_days_jan_feb", "gdd5_feb", "cum_tmax_feb", "cum_tmean_feb",
             "precip_feb_mm", "precip_feb_missing_days", "sunshine_feb_h", "sunshine_feb_missing_days"],
}
ORDER["mar15"] = ORDER["mar1"] + ["tmean_mar1_14", "tmax_mar1_14", "tmin_mar1_14", "gdd5_mar1_14",
                                  "cum_tmax_mar1_14", "cum_tmean_mar1_14", "chill_days_0_7c_mar1_14",
                                  "frost_days_mar1_14", "precip_mar1_14_mm", "precip_mar1_14_missing_days",
                                  "sunshine_mar1_14_h", "sunshine_mar1_14_missing_days", "cum_tmax_feb1_mar14"]
TAIL = ["full_bloom_date", "full_bloom_doy", "split", "fold1", "fold2", "fold3", "fold4", "fold5"]

for name, cutoff in [("feb1", "jan31"), ("mar1", "feb_end"), ("mar15", "mar14")]:
    t = build(cutoff)
    t = t[ORDER[name] + TAIL].sort_values(["bloom_year", "station_id"], kind="stable").reset_index(drop=True)
    t = _normalise(t)
    t.to_csv(OUT / f"model_features_{name}.csv", index=False, lineterminator="\n")
    print(name, t.shape, t.groupby("split").size().to_dict())
