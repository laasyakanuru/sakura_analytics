#%%
"""
Weather cleaning + feature engineering for the sakura bloom-date project.

Inputs
  data/raw/weather_by_station/weather_*_daily_quality.csv      (JMA values + quality codes)
  data/processed/01_source_data/01_station_metadata/*.csv
  data/processed/01_source_data/02_sakura_dates/*.csv

Outputs (data/processed/02_cleaned_source_data/weather/), Somei-yoshino stations only (82 of 102)
  02_cleaned_weather_daily_somei_1952_2025.csv.gz  one row per station x day, cleaned + imputed, with flags
  03_weather_features_somei_station_year.csv       one row per station x bloom year, model-ready features + targets
  04_weather_gap_log_somei.csv                     every missing-value run and how it was handled
  05_weather_station_relocations_somei.csv         JMA homogeneity breaks (station moves / instrument changes)
  06_weather_imputation_validation_somei.csv       masked-gap test of the imputation methods

All 102 stations are cleaned and used as imputation neighbours; only the outputs are filtered.

Cleaning rules (see the EDA report in reference/ for the evidence behind each one)
  1. Trust JMA quality codes 8 (normal) and 5 (quasi-normal). Code 4 ("insufficient data")
     is set to missing; codes 1 / 0 are already missing.
  2. Leading blocks before a station began observing a variable are NOT imputed
     (e.g. Tateyama temperature starts 1968, sunshine starts 1961 everywhere).
  3. Temperature gaps (mean / max / min) of up to MAX_IMPUTE_GAP days are filled with the
     neighbour-difference method: nearby stations' values on the missing day plus the
     average station-minus-neighbour difference in the 60 days around the gap.
     If no neighbour has data, gaps of up to 3 days fall back to linear interpolation.
     Longer gaps stay missing; affected seasons are flagged in the feature table.
  4. Precipitation and sunshine are never imputed (too local / too patchy); features using
     them carry their own missing-day counts.
  5. Values are not otherwise altered. Two known artefacts are FLAGGED, not fixed:
       - 1953-1963: daily mean can sit below daily min (different JMA day boundary for extremes)
       - station relocations (homogeneity breaks) can cause step changes
  6. After imputation Tmin <= Tmean <= Tmax is enforced on imputed rows only.

Run from anywhere:  python src/02_preprocess/02_clean_weather_and_features.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

#%% ---------------------------------------------------------------- settings
if "__file__" in globals():
    PROJECT_DIR = Path(__file__).resolve().parents[2]
else:  # notebook / interactive window: run from src/02_preprocess
    PROJECT_DIR = Path("../..").resolve()

RAW_DIR = PROJECT_DIR / "data/raw/weather_by_station"
META_DIR = PROJECT_DIR / "data/processed/01_source_data/01_station_metadata"
DATES_DIR = PROJECT_DIR / "data/processed/01_source_data/02_sakura_dates"
OUT_DIR = PROJECT_DIR / "data/processed/02_cleaned_source_data/weather"

START, END = "1952-01-01", "2025-12-31"
FIRST_SEASON, LAST_SEASON = 1953, 2025

TEMP_VARS = ["mean_temperature_c", "maximum_temperature_c", "minimum_temperature_c"]
OTHER_VARS = ["precipitation_mm", "sunshine_duration_hours"]
ALL_VARS = TEMP_VARS + OTHER_VARS
GOOD_QUALITY = {8, 5}

MAX_IMPUTE_GAP = 60      # days; longer temperature gaps are left missing
LINEAR_FALLBACK_GAP = 3  # days; used only when no neighbour has data
N_NEIGHBOURS = 3
MAX_NEIGHBOUR_KM = 300
OFFSET_WINDOW = 60       # days either side of a gap used to estimate the offset

SEED = 42
SPECIES = "Somei-yoshino"   # outputs are restricted to this species

#%% ---------------------------------------------------------------- load
def load_metadata():
    names = pd.read_csv(META_DIR / "01_01_station_name.csv", dtype={"station_id": str}, encoding="utf-8-sig")
    elev = pd.read_csv(META_DIR / "01_02_station_elevations.csv", dtype={"station_id": str}, encoding="utf-8-sig")
    species = pd.read_csv(META_DIR / "01_03_cherry_species.csv", dtype={"station_id": str}, encoding="utf-8-sig")
    meta = names.merge(elev, on="station_id").merge(species, on="station_id")
    assert len(meta) == 102 and meta.station_id.is_unique
    return meta.sort_values("station_id").reset_index(drop=True)


def load_raw_weather():
    files = sorted(RAW_DIR.glob("weather_*_daily_quality.csv"))
    assert files, f"no quality files in {RAW_DIR}"
    frames = [pd.read_csv(f, dtype={"station_id": str}, encoding="utf-8-sig") for f in files]
    raw = pd.concat(frames, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"], format="%Y-%m-%d")
    raw = raw[(raw.date >= START) & (raw.date <= END)]
    assert not raw.duplicated(["station_id", "date"]).any()
    return raw.sort_values(["station_id", "date"]).reset_index(drop=True)


def load_bloom_dates():
    def read(name, col):
        df = pd.read_csv(DATES_DIR / name, dtype={"station_id": str}, encoding="utf-8-sig")
        df[col] = pd.to_datetime(df[col], format="%d/%m/%Y")
        return df
    fl = read("02_01_sakura_flowering_1953_2025.csv", "bloom_date").rename(columns={"bloom_date": "flowering_date"})
    fb = read("02_02_sakura_full_bloom_1953_2025.csv", "full_bloom_date")
    d = fl.merge(fb, on=["station_id", "year"], how="outer")
    d["flowering_doy"] = d.flowering_date.dt.dayofyear
    d["full_bloom_doy"] = d.full_bloom_date.dt.dayofyear
    d["bloom_interval_days"] = (d.full_bloom_date - d.flowering_date).dt.days
    return d


#%% ---------------------------------------------------------------- helpers
def to_wide(df, col, stations, dates):
    w = df.pivot(index="date", columns="station_id", values=col)
    return w.reindex(index=dates, columns=stations).to_numpy(dtype=float)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def neighbour_table(meta):
    lat, lon = meta.latitude.to_numpy(), meta.longitude.to_numpy()
    dist = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(dist, np.inf)
    order = np.argsort(dist, axis=1)
    return dist, order


def missing_runs(mask):
    """Return list of (start_idx, end_idx_inclusive) for True runs in a 1-D bool array."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    return list(zip(starts, ends))


def neighbour_estimate(X, j, s, e, dist, order):
    """Neighbour-difference estimate for station j over rows s..e (inclusive). None if impossible."""
    n = X.shape[0]
    lo, hi = max(0, s - OFFSET_WINDOW), min(n, e + 1 + OFFSET_WINDOW)
    ctx = np.r_[lo:s, e + 1:hi]
    target_ctx = X[ctx, j]
    preds, weights = [], []
    for k in order[j]:
        if dist[j, k] > MAX_NEIGHBOUR_KM or len(preds) == N_NEIGHBOURS:
            break
        gap_vals = X[s:e + 1, k]
        if np.isnan(gap_vals).any():
            continue
        diff = target_ctx - X[ctx, k]
        diff = diff[~np.isnan(diff)]
        if len(diff) < 20:
            continue
        resid_sd = max(diff.std(), 0.1)
        preds.append(gap_vals + diff.mean())
        weights.append(1 / resid_sd ** 2)   # neighbours that track the station closely count more
    if not preds:
        return None
    w = np.array(weights)
    return (np.array(preds) * w[:, None]).sum(0) / w.sum()


def linear_fill(x, s, e):
    if s == 0 or e == len(x) - 1 or np.isnan(x[s - 1]) or np.isnan(x[e + 1]):
        return None
    return np.linspace(x[s - 1], x[e + 1], e - s + 3)[1:-1]


#%% ---------------------------------------------------------------- cleaning
def apply_quality_codes(raw):
    df = raw[["station_id", "location", "date"]].copy()
    for v in ALL_VARS:
        q = raw[f"{v}_quality_code"]
        df[v] = raw[v].where(q.isin(GOOD_QUALITY))
        df[f"{v}_dropped_q4"] = raw[v].notna() & (q == 4)
    return df


def relocation_breaks(raw):
    """Homogeneity number changes inside a downloaded year = real break. Changes on 1 Jan are
    an artefact of the per-year download (numbers restart in every request)."""
    rows = []
    for sid, g in raw.groupby("station_id"):
        h = g["mean_temperature_c_homogeneity_number"]
        changed = g.loc[h.diff().fillna(0).ne(0), "date"]
        for d in changed:
            if not (d.month == 1 and d.day == 1):
                rows.append({"station_id": sid, "location": g.location.iloc[0], "break_date": d.date()})
    return pd.DataFrame(rows)


def impute_temperatures(clean, meta, dates):
    stations = meta.station_id.tolist()
    dist, order = neighbour_table(meta)
    log, filled, methods = [], {}, {}
    for v in TEMP_VARS:
        X = to_wide(clean, v, stations, dates)
        Xf = X.copy()
        M = np.full(X.shape, "", dtype=object)
        M[~np.isnan(X)] = "observed"
        for j, sid in enumerate(stations):
            col = X[:, j]
            first = np.argmax(~np.isnan(col)) if (~np.isnan(col)).any() else len(col)
            for s, e in missing_runs(np.isnan(col)):
                length = e - s + 1
                rec = {"station_id": sid, "variable": v, "start": dates[s].date(), "end": dates[e].date(),
                       "length_days": length}
                if e < first:
                    M[s:e + 1, j] = "not_yet_observed"
                    rec["handling"] = "left missing: before first observation"
                elif length > MAX_IMPUTE_GAP:
                    M[s:e + 1, j] = "missing_long_gap"
                    rec["handling"] = f"left missing: gap > {MAX_IMPUTE_GAP} days"
                else:
                    est = neighbour_estimate(X, j, s, e, dist, order)
                    if est is not None:
                        Xf[s:e + 1, j] = np.round(est, 1)
                        M[s:e + 1, j] = "imputed_neighbour"
                        rec["handling"] = "neighbour-difference"
                    elif length <= LINEAR_FALLBACK_GAP and (lin := linear_fill(col, s, e)) is not None:
                        Xf[s:e + 1, j] = np.round(lin, 1)
                        M[s:e + 1, j] = "imputed_linear"
                        rec["handling"] = "linear interpolation"
                    else:
                        M[s:e + 1, j] = "missing_no_donor"
                        rec["handling"] = "left missing: no neighbour data"
                log.append(rec)
        filled[v], methods[v] = Xf, M
    return filled, methods, pd.DataFrame(log)


def validate_imputation(clean, meta, dates, n_trials=600):
    """Hide blocks of real observations and compare methods. Returns one row per trial."""
    rng = np.random.default_rng(SEED)
    stations = meta.station_id.tolist()
    dist, order = neighbour_table(meta)
    rows = []
    lengths = [1, 2, 3, 5, 7, 10, 15, 30, 60]
    for v in TEMP_VARS:
        X = to_wide(clean, v, stations, dates)
        for _ in range(n_trials):
            L = int(rng.choice(lengths))
            j = int(rng.integers(len(stations)))
            s = int(rng.integers(OFFSET_WINDOW + 1, len(dates) - L - OFFSET_WINDOW - 1))
            e = s + L - 1
            truth = X[s:e + 1, j].copy()
            if np.isnan(X[s - 1:e + 2, j]).any():
                continue
            Xm = X.copy()
            Xm[s:e + 1, j] = np.nan
            nb = neighbour_estimate(Xm, j, s, e, dist, order)
            lin = linear_fill(Xm[:, j], s, e)
            rows.append({"variable": v, "station_id": stations[j], "gap_length": L,
                         "mae_neighbour": np.nan if nb is None else np.abs(nb - truth).mean(),
                         "mae_linear": np.abs(lin - truth).mean()})
    return pd.DataFrame(rows)


#%% ---------------------------------------------------------------- features
def season_features(daily, meta, bloom, breaks):
    """One row per station x season year Y. A season runs 1 Oct (Y-1) .. 30 Jun (Y)."""
    d = daily.copy()
    d["month"] = d.date.dt.month
    d["season"] = d.date.dt.year + (d.month >= 10).astype(int)
    d = d[(d.month >= 10) | (d.month <= 6)]
    d = d[d.season.between(FIRST_SEASON, LAST_SEASON)]
    Tm, Tx, Tn = TEMP_VARS
    d["gdd5"] = (d[Tm] - 5).clip(lower=0)
    d["chill_0_7"] = d[Tm].between(0, 7.2).astype(float).where(d[Tm].notna())
    d["frost"] = (d[Tn] < 0).astype(float).where(d[Tn].notna())
    keys = ["station_id", "season"]

    def window(m_from, m_to, day_to=None):
        # months in season order: Oct=10, Nov=11, Dec=12, Jan=1 ...
        so = d.month.where(d.month >= 10, d.month + 12)
        a = m_from if m_from >= 10 else m_from + 12
        b = m_to if m_to >= 10 else m_to + 12
        w = so.between(a, b)
        if day_to is not None:
            w &= ~((so == b) & (d.date.dt.day > day_to))
        return d[w]

    out = []
    # monthly means
    month_names = {10: "oct", 11: "nov", 12: "dec", 1: "jan", 2: "feb", 3: "mar", 4: "apr"}
    for m, name in month_names.items():
        g = d[d.month == m].groupby(keys)
        out.append(g[Tm].mean().rename(f"tmean_{name}"))
        if m in (1, 2, 3):
            out.append(g[Tx].mean().rename(f"tmax_{name}"))
            out.append(g[Tn].mean().rename(f"tmin_{name}"))

    w = window(11, 2).groupby(keys)
    out += [w["chill_0_7"].sum(min_count=1).rename("chill_days_0_7c_nov_feb"),
            w[Tn].min().rename("tmin_lowest_nov_feb")]
    w = window(1, 3).groupby(keys)
    out.append(w["frost"].sum(min_count=1).rename("frost_days_jan_mar"))
    for to_m, tag in [(2, "feb"), (3, "mar")]:
        w = window(2, to_m).groupby(keys)
        out += [w["gdd5"].sum(min_count=1).rename(f"gdd5_feb1_to_{tag}_end"),
                w[Tx].sum(min_count=1).rename(f"cum_tmax_feb1_to_{tag}_end"),
                w[Tm].sum(min_count=1).rename(f"cum_tmean_feb1_to_{tag}_end")]
    w = window(2, 3).groupby(keys)
    out += [w["precipitation_mm"].sum(min_count=1).rename("precip_feb_mar_mm"),
            w["precipitation_mm"].apply(lambda s: s.isna().sum()).rename("precip_feb_mar_missing_days"),
            w["sunshine_duration_hours"].sum(min_count=1).rename("sunshine_feb_mar_h"),
            w["sunshine_duration_hours"].apply(lambda s: s.isna().sum()).rename("sunshine_feb_mar_missing_days")]

    # temperature-sum rules: first day of year when the running sum from 1 Feb reaches a threshold
    sp = d[(d.month >= 2) & (d.month <= 6)].sort_values(keys + ["date"]).copy()
    sp["cum_tmax"] = sp.groupby(keys)[Tx].cumsum()
    sp["cum_tmean"] = sp.groupby(keys)[Tm].cumsum()
    for col, thr, name in [("cum_tmax", 600, "rule600_tmax_pred_doy"), ("cum_tmean", 400, "rule400_tmean_pred_doy")]:
        hit = sp[sp[col] >= thr].groupby(keys).date.min().dt.dayofyear.rename(name)
        out.append(hit)

    # completeness of the core temperature window (Oct..May) after cleaning
    core = d[(d.month >= 10) | (d.month <= 5)]
    out.append(core.groupby(keys)[TEMP_VARS].apply(lambda x: int(x.isna().any(axis=1).sum()))
               .rename("temp_missing_days_oct_may"))
    out.append(core.groupby(keys)["temp_imputed"].sum().rename("temp_imputed_days_oct_may"))

    feat = pd.concat(out, axis=1).reset_index().rename(columns={"season": "year"})
    # a cumulative-sum rule is only valid if no day in Feb..its hit date is missing; simplest
    # safe rule: drop rule predictions for seasons with any missing temperature day Feb-May
    spring_missing = (d[(d.month >= 2) & (d.month <= 5)].groupby(keys)[TEMP_VARS]
                      .apply(lambda x: x.isna().any(axis=None)).rename("bad").reset_index()
                      .rename(columns={"season": "year"}))
    feat = feat.merge(spring_missing, on=["station_id", "year"], how="left")
    for c in ["rule600_tmax_pred_doy", "rule400_tmean_pred_doy"] + [c for c in feat if c.startswith(("cum_", "gdd5_"))]:
        feat.loc[feat.bad.fillna(True), c] = np.nan
    feat = feat.drop(columns="bad")
    feat["season_temp_complete"] = (feat.temp_missing_days_oct_may == 0).astype(int)

    # station-level relocation index: how many JMA breaks happened before this season
    feat["relocations_before_season"] = [
        int(((breaks.station_id == s) & (pd.to_datetime(breaks.break_date) < pd.Timestamp(y - 1, 10, 1))).sum())
        for s, y in zip(feat.station_id, feat.year)]
    feat["pre_1964_definition"] = (feat.year < 1964).astype(int)

    full = (meta[["station_id", "location", "prefecture", "region", "cherry_species",
                  "latitude", "longitude", "elevation_m"]]
            .merge(feat, on="station_id", how="right")
            .merge(bloom[["station_id", "year", "flowering_date", "flowering_doy", "full_bloom_date",
                          "full_bloom_doy", "bloom_interval_days"]], on=["station_id", "year"], how="left"))
    for c in ["flowering_date", "full_bloom_date"]:
        full[c] = full[c].dt.strftime("%Y-%m-%d")
    num = full.select_dtypes("float").columns
    full[num] = full[num].round(2)
    lead = ["station_id", "location", "prefecture", "region", "cherry_species", "latitude", "longitude",
            "elevation_m", "year", "flowering_date", "flowering_doy", "full_bloom_date", "full_bloom_doy",
            "bloom_interval_days"]
    return full[lead + [c for c in full.columns if c not in lead]].sort_values(["station_id", "year"])


#%% ---------------------------------------------------------------- run
if __name__ == "__main__" or "__file__" not in globals():
    meta = load_metadata()
    raw = load_raw_weather()
    bloom = load_bloom_dates()
    dates = pd.date_range(START, END, freq="D")
    stations = meta.station_id.tolist()
    print(f"raw rows {len(raw):,}  stations {raw.station_id.nunique()}  days {len(dates):,}")

    clean = apply_quality_codes(raw)
    print("quality-4 values dropped:", {v: int(clean[f'{v}_dropped_q4'].sum()) for v in ALL_VARS})

    relocation_breaks_cache = relocation_breaks(raw)
    print("relocation breaks:", len(relocation_breaks_cache))

    filled, methods, gap_log = impute_temperatures(clean, meta, dates)
    print(gap_log.groupby(["variable", "handling"]).size().to_string())

    # back to long format
    idx = pd.MultiIndex.from_product([dates, stations], names=["date", "station_id"])
    daily = pd.DataFrame(index=idx).reset_index()
    for v in TEMP_VARS:
        daily[v] = filled[v].reshape(-1)
        daily[v.replace("_c", "_flag")] = methods[v].reshape(-1)
    for v in OTHER_VARS:
        daily[v] = to_wide(clean, v, stations, dates).reshape(-1)
    daily = daily.sort_values(["station_id", "date"]).reset_index(drop=True)

    # rule 6: keep imputed values physically consistent
    Tm, Tx, Tn = TEMP_VARS
    imp = daily.filter(like="temperature_flag").apply(lambda c: c.str.startswith("imputed")).any(axis=1)
    daily.loc[imp, Tx] = daily.loc[imp, [Tx, Tm, Tn]].max(axis=1)
    daily.loc[imp, Tn] = daily.loc[imp, [Tx, Tm, Tn]].min(axis=1)
    daily["temp_imputed"] = imp.astype(int)
    daily["pre_1964_mean_below_min"] = ((daily.date.dt.year < 1964) & (daily[Tm] < daily[Tn] - 0.05)).astype(int)

    # validation of imputation methods
    validation = validate_imputation(clean, meta, dates)
    print(validation.groupby("gap_length")[["mae_neighbour", "mae_linear"]].mean().round(2))

    features = season_features(daily, meta, bloom, relocation_breaks_cache)
    print("feature table:", features.shape)

    # ---- keep Somei-yoshino stations only in the outputs (the team models this species only).
    # All 102 stations were still used above as imputation neighbours, which improves gap filling.
    keep = set(meta.loc[meta.cherry_species == SPECIES, "station_id"])
    print(f"writing {SPECIES} stations only: {len(keep)} of {len(meta)}")
    out = lambda df: df[df.station_id.isin(keep)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("0[2-6]_weather_*"):          # files from the earlier all-species run
        old.unlink()
    for old in OUT_DIR.glob("02_cleaned_weather_daily_*"):
        old.unlink()
    out_daily = out(daily).copy()
    out_daily["date"] = out_daily.date.dt.strftime("%Y-%m-%d")
    out_daily.to_csv(OUT_DIR / "02_cleaned_weather_daily_somei_1952_2025.csv.gz", index=False, na_rep="",
                     compression={"method": "gzip", "mtime": 0})
    out(features).to_csv(OUT_DIR / "03_weather_features_somei_station_year.csv", index=False, na_rep="")
    out(gap_log).to_csv(OUT_DIR / "04_weather_gap_log_somei.csv", index=False)
    out(relocation_breaks_cache).to_csv(OUT_DIR / "05_weather_station_relocations_somei.csv", index=False)
    out(validation).round(3).to_csv(OUT_DIR / "06_weather_imputation_validation_somei.csv", index=False)
    print("saved to", OUT_DIR)
