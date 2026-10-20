# 2026 season data for Sakura Yosoku

Everything in this folder is additive: no existing file in the repository was changed, so the current pipeline, MySQL warehouse and `data/model/*.csv` keep working as before.

Built on 20 Oct 2026 from repo commit 9cd154b (main; its data inputs and pipeline code are unchanged in e8ff3d6) with the team's own pipeline, extended to the 2026 bloom season.

## What to use

| File | What it is |
|---|---|
| `model/model_features_{feb1,mar1,mar15}_2026.csv` | **2026 rows only** (48 Somei-yoshino stations), same columns as `data/model/model_features_*.csv`. `split = forward_2026`, fold columns empty. Use as an unseen forward test of the champion model. |
| `model/full_with_2026/model_features_*.csv` | Full tables 1964–2026 (4,413 rows). Rows 1964–2025 are **byte-identical** to the committed `data/model/*.csv`; only the 48 rows for 2026 are new. |
| `sakura_dates/*_1953_2026.csv` | Flowering and full-bloom source files with a 2026 row for all 102 stations (blank where JMA did not observe). |
| `raw_weather_2026/weather_*_daily(.csv,_quality.csv)` | JMA daily weather 2026-01-01 to 2026-06-30, 102 stations, same format as `data/raw/weather_by_station/`. `_coverage_2026.csv`: no failed requests, every day retrieved. |
| `cleaned/` | Cleaned daily weather 1952-01-01 to 2026-06-30 (`02_cleaned_weather_daily_somei_1952_2026.csv.gz`) and the station-season feature table including season 2026 (`03_weather_features_somei_station_year_1953_2026.csv`). |
| `tools/` | `02_clean_weather_and_features_2026.patch` (the only code change: dates and file names), `build_model_features.py` (pandas version of `sql/05_model_feature_tables.sql`), the parsed JMA 2026 bloom table. |

## How it was built

1. **Weather:** `src/01_prepare_source_data/01_prepare_station_weather.py prepare` (all 102 stations resolved), then `download --start 2026-01-01 --end 2026-06-30`. The new rows were appended to each station's raw files (which ended on 2025-12-31).
2. **Bloom dates:** JMA "2026年のさくらの開花状況 / 満開状況" pages (data.jma.go.jp/sakura/data/sakura_kaika.html, sakura_mankai.html), mapped to station IDs by JMA station name. 57 flowering and 56 full-bloom dates; 48 are Somei-yoshino.
3. **Cleaning and features:** `src/02_preprocess/02_clean_weather_and_features.py` with END = 2026-06-30 and LAST_SEASON = 2026 (see patch). No other rule changed.
4. **Model tables:** MySQL is not available in the build environment, so `tools/build_model_features.py` reimplements `sql/05_model_feature_tables.sql` in pandas with exact DECIMAL rounding.

## Checks

- Re-running the original cleaning script reproduces all committed outputs byte for byte.
- With 2026 added, every cleaned daily row up to 2025-12-31 and every station-season row up to 2025 is identical to the committed files; gap log and relocation files are unchanged.
- `build_model_features.py` reproduces the committed `model_features_{feb1,mar1,mar15}.csv` byte for byte for 1964–2025.
- Tokyo February 2026 daily mean, maximum and minimum temperatures match JMA's daily table (etrn) exactly.
- 2026: no missing or imputed temperature days at the 82 Somei-yoshino stations; no missing precipitation or sunshine in the feature windows.

## Notes

- `split = forward_2026` is a new label; existing code that selects `split == "development"` or `"test"` ignores these rows.
- JMA observed full bloom at 48 of the 82 Somei-yoshino stations in 2026, exactly the same stations as in 2025.
- The imputation-validation file (06_…) changes slightly because its random masked gaps are drawn from a longer period; it is not used by the models.
