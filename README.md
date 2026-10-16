# Sakura Yosoku — Cherry Blossom Full-Bloom Forecasting

Predict the Somei-yoshino full-bloom date (day of year) for 82 JMA stations at
two forecast dates, **Feb 1** and **Mar 1**, from daily weather 1952–2025.

Contents
1. [Which path is yours?](#1-which-path-is-yours)
2. [Modelling with Python only (most teammates)](#2-modelling-with-python-only-most-teammates)
3. [Rules every model must follow](#3-rules-every-model-must-follow)
4. [Building the MySQL database](#4-building-the-mysql-database)
5. [Database design and SQL files](#5-database-design-and-sql-files)
6. [Expected results](#6-expected-results)
7. [Repository layout](#7-repository-layout)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Which path is yours?

| You want to… | You need | Go to |
|---|---|---|
| Build / tune a model (MLR, decision tree, gradient boosting, SVR, baseline) | Python only | [Section 2](#2-modelling-with-python-only-most-teammates) |
| Build or check the MySQL warehouse, run SQL in Workbench | MySQL Server 8+ and Python | [Section 4](#4-building-the-mysql-database) |

The modelling data is stored twice with identical content: in MySQL
(`sakura_dw`) and as CSV files in `data/model/`. Python code reads MySQL when
it is set up on your computer and the CSVs otherwise, so **you do not need
MySQL to build models.**

> Windows: type `python` instead of `python3` in every command below.

---

## 2. Modelling with Python only (most teammates)

### 2.1 One-time setup

```bash
git clone https://github.com/natsumisugimoto817/sakura_analytics.git
cd sakura_analytics
python3 -m pip install -r requirements.txt
```

### 2.2 Every time you start working

```bash
cd sakura_analytics
git pull
```

`git pull` also brings any updated `data/model/*.csv`.

### 2.3 Run the example model

```bash
python3 src/example_model.py
```

It loads the data, trains a linear regression on the 5 agreed folds for both
forecast dates, prints the scores next to the climatology baseline, and saves
them under `results/`. If this works, your setup is complete.

### 2.4 Start your own model

```bash
cp src/example_model.py src/models/gradient_boosting.py     # macOS / Linux
copy src\example_model.py src\models\gradient_boosting.py  # Windows
```

(use your model's name instead of `gradient_boosting`)

Open the copy and change only the block marked `change these`:

| Setting | What to put there |
|---|---|
| `MODEL_NAME` | e.g. `"gradient_boosting"` — used in the results file names |
| `make_model()` | return a **new, unfitted** model (examples are in the docstring) |
| `choose_features(df)` | the features you picked from the candidate list (default: all) |
| `RUN_FINAL_TEST` | keep `False`; only for the champion model (see [Section 3](#3-rules-every-model-must-follow)) |

Then run it:

```bash
python3 src/models/gradient_boosting.py
```

Output, for `feb1` and `mar1`:

```
=== gradient_boosting | forecast date mar1 | 23 features ===
fold   n   MAE  RMSE    R2  hit_rate_3d
   1 148  ...                               validation 2005-2006
   ...
   5  96  ...                               validation 2013-2014
mean 575  ...                               <- compare models on this row
climatology baseline (mean of folds): MAE 3.956  RMSE 4.687  R2 0.804  hit_rate_3d 0.440
saved results/gradient_boosting_mar1_cv.csv
```

To tune hyperparameters, change `make_model()` and run again: the `mean` row is
the validation score. Commit your model file and `results/*_cv.csv` so the team
can compare all models in one place.

### 2.5 Notes per model

| Task | Owner | Notes |
|---|---|---|
| Baseline | Yasha | `climatology_baseline` (each station's mean bloom day in the training years) is already printed by every model. Do **not** use `rule600_tmax_pred_doy` / `rule400_tmean_pred_doy` as a baseline — see [Section 3](#3-rules-every-model-must-follow). |
| TF-IDF / text analytics | Yasha | Not part of this database; there is no text data here. |
| Multiple linear regression | Hairong | Keep `StandardScaler` (as in the example) to compare coefficients. Watch for correlated features (e.g. `tmean_feb` vs `cum_tmean_feb`). |
| Decision tree | Natsumi | `DecisionTreeRegressor(max_depth=..., random_state=42)`; no scaling needed. |
| Gradient boosting | Leyi | `GradientBoostingRegressor(random_state=42)`; no scaling needed. |
| Support vector regression | Laasya | Must scale: `make_pipeline(StandardScaler(), SVR(...))`. |

### 2.6 Using the data in your own code or notebook

```python
import sys; sys.path.append("src")      # path to the src folder from where you run
from model_data import (load_features, feature_columns, cv_folds, evaluate,
                        cv_report, climatology_baseline, TARGET)

df = load_features("mar1")              # or "feb1"
X = feature_columns(df)                 # all candidate features

for fold, train, valid in cv_folds(df):
    model.fit(train[X], train[TARGET])
    print(fold, evaluate(valid[TARGET], model.predict(valid[X])))
```

| Function | Returns |
|---|---|
| `load_features(date)` | the feature table for `"feb1"` or `"mar1"` (add `source="csv"` or `"mysql"` to force one) |
| `feature_columns(df)` | all candidate feature names |
| `cv_folds(df)` | `(fold, train_df, valid_df)` for the 5 agreed folds |
| `evaluate(y_true, y_pred)` | `{n, MAE, RMSE, R2, hit_rate_3d}` |
| `cv_report(df, predict_fn)` | all 5 folds + mean row; `predict_fn(train, valid)` returns predictions |
| `climatology_baseline(train, valid)` | baseline predictions |
| `final_test_report(df, predict_fn)` | train on 1964–2014, score 2015–2025 (champion only) |

### 2.7 What is in the feature tables

One row per station and bloom year: Somei-yoshino only, bloom years 1964–2025,
observed full-bloom date, no missing temperature day up to the cutoff
(4,365 rows in each table).

| Column group | `feb1` (weather to 31 Jan) | `mar1` (weather to 28/29 Feb) |
|---|---|---|
| Station | `latitude`, `longitude`, `elevation_m`, `relocations_before_season` | same |
| Monthly mean temperature | `tmean_oct` … `tmean_jan` | `tmean_oct` … `tmean_feb` |
| Monthly max / min | `tmax_jan`, `tmin_jan` | `tmax_jan`, `tmin_jan`, `tmax_feb`, `tmin_feb` |
| Chilling | `chill_days_0_7c_nov_jan`, `tmin_lowest_nov_jan` | `chill_days_0_7c_nov_feb`, `tmin_lowest_nov_feb` |
| Frost days (Tmin < 0) | `frost_days_jan` | `frost_days_jan_feb` |
| Warmth in the last month | `gdd5_jan`, `cum_tmax_jan`, `cum_tmean_jan` | `gdd5_feb`, `cum_tmax_feb`, `cum_tmean_feb` |
| Precipitation / sunshine | `precip_jan_mm`, `sunshine_jan_h` (+ `_missing_days`) | `precip_feb_mm`, `sunshine_feb_h` (+ `_missing_days`) |
| Not features | `station_id`, `location`, `prefecture`, `region`, `bloom_year`, `full_bloom_date`, **`full_bloom_doy` (target)**, `split`, `fold1`–`fold5` | same |

Definitions: chill day = daily mean 0–7.2 °C; `gdd5` = sum of max(Tmean − 5, 0);
`cum_*` = sum of daily values over the month. Precipitation and sunshine
totals are (mean of observed days) × (days in month); `*_missing_days` says
how many days were estimated. One month with no sunshine data at all
(station 604, Feb 2008) uses the same-year mean of its region.
No column contains empty values.

`feature_columns(df)` returns everything except the "Not features" row.
`region` is text; one-hot encode it yourself if you want to use it.

---

## 3. Rules every model must follow

These come from the Model Development Meeting and keep the models comparable.

1. **Same data**: use only `load_features()` (or the `model_features_*`
   tables). Do not add columns from `fact_bloom_season`: many contain weather
   after the forecast date (e.g. `tmean_mar`, `precip_feb_mar_mm`, and for
   Feb 1 also `chill_days_0_7c_nov_feb`, `tmin_lowest_nov_feb`).
2. **Same folds**: use `cv_folds()` — five expanding windows, training from
   1964 up to year *T* and validating on the next two years:

   | Fold | Training | Validation |
   |---|---|---|
   | 1 | 1964–2004 | 2005–2006 |
   | 2 | 1964–2006 | 2007–2008 |
   | 3 | 1964–2008 | 2009–2010 |
   | 4 | 1964–2010 | 2011–2012 |
   | 5 | 1964–2012 | 2013–2014 |

3. **Same metrics**: `evaluate()` — MAE, RMSE, R², and the ±3-day hit rate
   (share of point predictions within 3 days of the actual full bloom; not a
   prediction interval).
4. **Both forecast dates**: report `feb1` and `mar1` (the example does both).
5. **Test years 2015–2025 stay untouched** until the team has chosen the
   champion per forecast date on validation. Then set `RUN_FINAL_TEST = True`
   for the champions only.
6. **Fair baseline**: `rule600_tmax_pred_doy` and `rule400_tmean_pred_doy`
   need real temperatures until the threshold is reached, which is always
   after Mar 1. They are not available on the forecast date; use
   `climatology_baseline` instead.
7. **Fit preprocessing on the training fold only** (scalers, imputers, feature
   selection). Using a `Pipeline` inside `make_model()` does this
   automatically.

Good to know: all 82 stations appear in training years, but only 48 stations
have bloom observations from 2013 onward (JMA reduced observations), so the
test set covers those 48 stations.

---

## 4. Building the MySQL database

### 4.1 First build

You need MySQL Server 8+ running on your computer (MySQL Workbench is only the
client — the server must also be installed and started) and the Python setup
from 2.1.

```bash
python3 setup_database.py
```

It asks for your MySQL username and password once, then:

1. runs `sql/01_create_star_schema.sql` (database + 5 tables);
2. loads `dim_station`, `fact_daily_weather`, `fact_bloom_season` with
   `src/etl/load_*.py`;
3. runs `sql/03_olap_queries.sql` (analysis view);
4. runs `sql/05_model_feature_tables.sql` (model tables);
5. exports `data/model/*.csv` and verifies they match MySQL;
6. prints a PASS/FAIL table of 26 health checks (the checks from `02` and
   `04` plus split / leakage / NULL checks).

Expect about 5–15 minutes; most of it is loading 2.2 million weather rows.
Success ends with `✓ All checks passed`. In MySQL Workbench, refresh the
schema list to see `sakura_dw`.

You never need to open the five SQL files one by one; the script runs them
in the right order. They are still there to read or to run single queries.

### 4.2 Other commands

| Command | When |
|---|---|
| `python3 setup_database.py --checks` | Check an existing database (read-only) |
| `python3 setup_database.py --model-tables` | You already have `sakura_dw` and only need the new or updated model tables and CSVs (e.g. after pulling a change to `05`) — about 1 minute |
| `python3 setup_database.py --reset` | Delete `sakura_dw` and rebuild everything |

If you changed data or SQL and the CSVs in `data/model/` changed, commit them
so Python-only teammates get the update.

### 4.3 Saving your password (optional)

Copy `.env.example` to `.env` and fill in your password; scripts then stop
asking, and `load_features()` reads from MySQL instead of the CSVs. `.env` is
git-ignored — never commit it.

---

## 5. Database design and SQL files

### 5.1 Star schema (`sakura_dw`)

ERD: `sql/schema_design/Sakura_Star_Schema.png` (Workbench model:
`Sakura_Warehouse.mwb`).

| Table | Grain | Rows |
|---|---|---|
| `dim_station` | station (102: 82 Somei-yoshino + 20 other species) | 102 |
| `dim_date` | calendar day 1952-01-01 … 2025-12-31 | 27,029 |
| `dim_season` | bloom season *Y* = 1 Oct (*Y*−1) … 30 Jun *Y*, 1953–2025 | 73 |
| `fact_daily_weather` | station × day (82 Somei-yoshino stations) | 2,216,378 |
| `fact_bloom_season` | station × bloom season: flowering / full-bloom dates and seasonal weather features | 5,986 |

Model layer (derived from the star schema by `05`, rebuilt at any time):

| Table | Content |
|---|---|
| `ref_cv_split` | one row per bloom year 1964–2025: `split`, `fold1`–`fold5` |
| `model_features_feb1` | Feb 1 feature table (Section 2.7) |
| `model_features_mar1` | Mar 1 feature table (Section 2.7) |

### 5.2 SQL files

| File | What it does | Changes data? |
|---|---|---|
| `01_create_star_schema.sql` | Creates `sakura_dw` and the five tables; fills `dim_date` and `dim_season`. Only on a server without `sakura_dw` — fails if the tables exist. | Yes |
| `02_validation.sql` | Row counts, coverage, missing targets, duplicate keys, foreign keys. | No |
| `03_olap_queries.sql` | Descriptive OLAP queries (bloom timing by region, decade, earliest stations, January temperature, imputation by region) and the `vw_sakura_march1_features` view. | Creates a view |
| `04_season_integration_validation.sql` | Season mapping, grain and feature-consistency checks (5.3). | No |
| `05_model_feature_tables.sql` | Builds the model layer with explicit forecast-date cutoffs. Safe to re-run. | Rebuilds 3 model tables |

To run one file by hand in Workbench: File → Open SQL Script → ⚡ Execute.
For `04`, raise the result-row limit first (B1 lists ~26,000 rows).

### 5.3 What `04` checks

Bloom outcomes are one row per station-season while weather is daily, so
wrong joins or aggregations would silently distort features. `04` is
read-only and checks:

| Section | Check | Why it matters |
|---|---|---|
| B1 | Maps each date to an Oct–Jun bloom season | Oct–Dec weather must count toward the *following* bloom year |
| B2 | Dates in more than one season (expect 0 rows) | One weather day must not feed two seasons |
| B3 | Null or reversed season boundaries (expect 0 rows) | Invalid windows distort features |
| C1 | Aggregates Dec–Feb weather to station-season *before* joining bloom data | Shows the correct grain; joining daily rows to bloom rows multiplies targets. Descriptive only — Dec–Feb is not available on Feb 1 |
| C2 | Duplicate keys in both fact tables (expect 0 rows) | Each fact table keeps its grain |
| A1 | Recomputes Jan–Apr mean temperature from daily data and compares with the stored values | Catches processing / rounding errors |
| A2 | Same for Feb–Mar precipitation and sunshine totals | Same, for the other weather features |
| A3 | Stations with missing Jan/Feb mean temperature | Shows where gaps concentrate |

Results: `MATCH` = within 0.011; `MISMATCH` = investigate (do not overwrite
automatically); `MISSING` = one side is empty.

### 5.4 Why `05` recomputes the features

Several stored columns in `fact_bloom_season` are not safe for the agreed
forecast dates:

- `chill_days_0_7c_nov_feb` and `tmin_lowest_nov_feb` include February
  (not allowed on Feb 1);
- `season_temp_complete` uses Oct–May completeness, and `gdd5_*` / `cum_*`
  were emptied when any Feb–May day was missing — information from after the
  forecast date;
- `rule600_*` / `rule400_*` need temperatures after Mar 1.

`05` recomputes every feature from `fact_daily_weather` with an explicit date
cutoff and decides row completeness only from days up to the cutoff. Where the
definitions are the same, the values equal the stored ones exactly.
`vw_sakura_march1_features` (from `03`) is kept for descriptive analysis only.

### 5.5 How the CSVs behind the database were made

`src/01_prepare_source_data` downloads and combines JMA station weather;
`src/02_preprocess/02_clean_weather_and_features.py` applies quality codes,
imputes short temperature gaps from neighbouring stations, and writes the
cleaned files in `data/processed/02_cleaned_source_data/weather/` (details in
the README there and the EDA report in `reference/`).

---

## 6. Expected results

From a fresh build on MySQL 8.0 (2026-10-16):

- Row counts: 102 / 27,029 / 73 / 2,216,378 / 5,986; 825 missing full-bloom
  dates; 0 duplicate keys; 0 foreign-key orphans.
- `04`: A1 — Jan, Feb, Mar 5,942 MATCH / 44 MISSING, Apr 5,943 / 43,
  0 MISMATCH; A2 — precipitation 5,944 MATCH / 42 MISSING, sunshine 5,376 MATCH
  / 610 MISSING, 0 MISMATCH; B2, B3, C2 return no rows.
- Model tables: 4,365 rows each (3,837 development, 528 test); no empty
  values; no full bloom on or before Mar 1 (earliest is day 78).
- Climatology baseline, mean of 5 validation folds: MAE 3.96, RMSE 4.69,
  R² 0.80, ±3-day hit rate 0.44.

---

## 7. Repository layout

```
README.md                  this file
requirements.txt           Python packages
setup_database.py          build / check the MySQL database, export the CSVs
.env.example               template for saving your MySQL login (copy to .env)
data/model/                model_features_feb1.csv, model_features_mar1.csv
data/processed/, data/raw/ source and cleaned data used by the loaders
sql/                       01-05 SQL files, schema_design/ (ERD)
src/model_data.py          shared loading, folds, metrics, baseline
src/example_model.py       template - copy to src/models/<your_model>.py
src/models/                your model scripts
results/                   cross-validation results (*_cv.csv)
src/etl/                   loaders + db_connection.py (shared MySQL login)
src/01_prepare_source_data/, src/02_preprocess/   how the source CSVs were made
reference/                 EDA reports
```

---

## 8. Troubleshooting

| Message | Fix |
|---|---|
| `No module named ...` | `python3 -m pip install -r requirements.txt`, using the same `python3` you run the script with |
| `model_features_....csv not found` | Run `git pull` from the project folder |
| `Can't connect to MySQL server on 'localhost'` | The MySQL server is not running. macOS: System Settings → MySQL → Start. Windows: Services → MySQL80 → Start. (Modelling still works from the CSVs.) |
| `Access denied for user ...` | Wrong MySQL username or password — use the ones you use in MySQL Workbench |
| `'cryptography' package is required` | `python3 -m pip install -r requirements.txt` |
| `Database 'sakura_dw' already exists` | Use `--checks`, `--model-tables` or `--reset` (4.2) |
| A health check shows FAIL | Run `python3 setup_database.py --reset` once; if it still fails, share the PASS/FAIL table with the team |
| `ValueError: Input X contains NaN` | You added a column that is not from `load_features()`; the provided features have no empty values |
