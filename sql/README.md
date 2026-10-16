# Sakura Yosoku — MySQL Star Schema

**Most people do not need to run these files by hand.** `python3 setup_database.py`
in the project root runs them in the right order and checks the results
(see the root `README.md`). This page explains what each file does.

## Files

| File | What it does | Changes data? |
|---|---|---|
| `01_create_star_schema.sql` | Creates `sakura_dw` and the five tables; fills `dim_date` (1952–2025) and `dim_season` (1953–2025). Run **once on an empty server** — it fails if the tables already exist. | Yes |
| `02_validation.sql` | Row counts, coverage, missing targets, duplicate keys and foreign-key checks. | No |
| `03_olap_queries.sql` | Descriptive OLAP queries (region, decade, earliest stations, Jan temperature, imputation) and the `vw_sakura_march1_features` view. | Creates a view |
| `04_season_integration_validation.sql` | Date-to-season mapping, grain checks and recomputation of stored features from daily weather. See `README_04_SEASON_INTEGRATION_VALIDATION.md`. | No |
| `05_model_feature_tables.sql` | Builds `ref_cv_split`, `model_features_feb1` and `model_features_mar1` — the leakage-safe modelling inputs agreed in the Model Development Meeting. Safe to re-run. | Rebuilds those 3 tables |
| `schema_design/` | ERD image and MySQL Workbench model (`.mwb`). | — |

## Build order

1. `01_create_star_schema.sql`
2. `src/etl/load_stations.py` → `dim_station` (102 rows)
3. `src/etl/load_daily_weather.py` → `fact_daily_weather` (2,216,378 rows)
4. `src/etl/load_bloom_season.py` → `fact_bloom_season` (5,986 rows)
5. `03_olap_queries.sql`, then `05_model_feature_tables.sql`
6. `02_validation.sql` and `04_season_integration_validation.sql` to check

`setup_database.py` does exactly this. To run a single file by hand in MySQL
Workbench: File → Open SQL Script → ⚡ Execute. For `04`, raise the result-row
limit first (B1 lists ~26,000 rows).

## Expected results (fresh build, 2026-10-16)

- Row counts: 102 / 27,029 / 73 / 2,216,378 / 5,986.
- `fact_bloom_season`: 82 Somei-yoshino stations × 73 seasons; 825 missing full-bloom dates.
- `dim_station`: 102 stations (82 Somei-yoshino, 10 Ezo-yamazakura, 9 Hikan-zakura, 1 Chishima-zakura).
- Duplicate-key and foreign-key checks: 0.
- `04` A1/A2: 0 mismatches (Jan–Mar 44 missing, Apr 43; sunshine 610 missing).
- `model_features_feb1` / `model_features_mar1`: 4,365 rows each (3,837 development, 528 test).

## Why `05` recomputes features instead of using `fact_bloom_season`

Several stored columns are not safe for the agreed forecast dates:

- `chill_days_0_7c_nov_feb`, `tmin_lowest_nov_feb` include February → not allowed for Feb 1.
- `season_temp_complete` looks at Oct–May, and `gdd5_*` / `cum_*` were set to NULL
  when any Feb–May day was missing → uses information after the forecast date.
- `rule600_tmax_pred_doy`, `rule400_tmean_pred_doy` need temperatures until the
  threshold is reached (always after Mar 1).

`05` recomputes every feature from `fact_daily_weather` with an explicit date cutoff;
where definitions overlap, the values equal the stored ones exactly.

## Notes

- `vw_sakura_march1_features` (from `03`) is kept for the descriptive analysis; for
  modelling use `model_features_mar1`, which avoids the issues above (it also keeps
  one station-year the view drops because of a missing day in March–May).
- The original interactive `sakura.sql` contains temporary deletes, restorations and
  ALTER statements; do not rerun it as a deployment script.
