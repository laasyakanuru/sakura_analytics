# Sakura Yosoku — MySQL Star Schema

Cleaned and reorganized from the user's `sakura.sql` (716 lines of interactive work).

## Files

1. `01_create_star_schema.sql` — Creates **five final tables**, including `dim_date` (1952–2025) and `dim_season` (1953–2025). **Run on a NEW/EMPTY `sakura_dw` only**; it does not populate station metadata or fact data.
2. `02_validation.sql` — Read-only checks for row counts, coverage, missing targets, duplicate keys, and FK consistency. Run after loading.
3. `03_olap_queries.sql` — Historical OLAP queries and a March 1 feature view.

## Rebuild sequence for another group member

1. Install MySQL 8+, Python, pandas, SQLAlchemy, and PyMySQL; clone `sakura_analytics`.
2. On a **new/empty** MySQL instance, run `01_create_star_schema.sql`.
3. Populate `dim_station` from the three metadata CSVs in `data/processed/01_source_data/01_station_metadata/` (join on `station_id`). This import step was done via MySQL Workbench staging tables in the original workflow and is **not automated in these three SQL scripts**. Confirm 102 station records before proceeding.
4. Run `python3 src/etl/load_daily_weather.py`, then `python3 src/etl/load_bloom_season.py`. These scripts must exist in the repository and connect to `sakura_dw`; do not upload credentials.
5. Run `02_validation.sql`, then `03_olap_queries.sql`.

## Scope and caveats

- `dim_station`: 102 source stations (82 Somei-yoshino + 20 other-species stations).
- `fact_daily_weather`: 82 Somei-yoshino stations, 1952–2025, expected 2,216,378 rows.
- `fact_bloom_season`: 82 stations × 73 seasons, expected 5,986 rows; 825 missing full-bloom dates.
- Recommended initial modelling filter: bloom year >= 1964, season temperature complete, and observed full-bloom target.
- Weather features after a forecast issue date must never be used for that issue date. The March 1 view is only a **candidate** training extract.
- These scripts do **not** include an export of the existing MySQL database or actual ETL Python files.
- The original interactive `sakura.sql` includes temporary deletes, restorations, repeated queries, and ALTER statements; do not rerun it as a deployment script.
