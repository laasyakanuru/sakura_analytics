# Sakura Yosoku — Cherry Blossom Full-Bloom Forecasting

Predict the Somei-yoshino full-bloom date (day of year) for 82 JMA stations at
three forecast dates, **Feb 1**, **Mar 1** and **Mar 15**, from daily weather 1952–2025.

- **Building models?** Read this page (Python only, no MySQL needed).
- **MySQL warehouse, star schema, SQL files:** see [`sql/README.md`](sql/README.md)
  (star schema diagram: [`sql/star_schema/`](sql/star_schema/)).

Contents
1. [Which path is yours?](#1-which-path-is-yours)
2. [Modelling with Python only (most teammates)](#2-modelling-with-python-only-most-teammates)
3. [Rules every model must follow](#3-rules-every-model-must-follow)
4. [Repository layout](#4-repository-layout)
5. [Troubleshooting](#5-troubleshooting)

---

## 1. Which path is yours?

| You want to… | You need | Go to |
|---|---|---|
| Build / tune a model (MLR, decision tree, gradient boosting, SVR, baseline) | Python only | [Section 2](#2-modelling-with-python-only-most-teammates) |
| Build or check the MySQL warehouse, run SQL in Workbench | MySQL Server 8+ and Python | [`sql/README.md`](sql/README.md) |

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

It loads the data, trains a linear regression on the 5 agreed folds for all three
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

Output, for `feb1`, `mar1` and `mar15`:

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

df = load_features("mar1")              # or "feb1" / "mar15"
X = feature_columns(df)                 # all candidate features

for fold, train, valid in cv_folds(df):
    model.fit(train[X], train[TARGET])
    print(fold, evaluate(valid[TARGET], model.predict(valid[X])))
```

| Function | Returns |
|---|---|
| `load_features(date)` | the feature table for `"feb1"`, `"mar1"` or `"mar15"` (add `source="csv"` or `"mysql"` to force one) |
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

| Column group | `feb1` (weather to 31 Jan) | `mar1` (weather to 28/29 Feb) | `mar15` (weather to 14 Mar): all `mar1` columns, plus |
|---|---|---|---|
| Station | `latitude`, `longitude`, `elevation_m`, `relocations_before_season` | same | – |
| Monthly mean temperature | `tmean_oct` … `tmean_jan` | `tmean_oct` … `tmean_feb` | `tmean_mar1_14` |
| Monthly max / min | `tmax_jan`, `tmin_jan` | `tmax_jan`, `tmin_jan`, `tmax_feb`, `tmin_feb` | `tmax_mar1_14`, `tmin_mar1_14` |
| Chilling | `chill_days_0_7c_nov_jan`, `tmin_lowest_nov_jan` | `chill_days_0_7c_nov_feb`, `tmin_lowest_nov_feb` | `chill_days_0_7c_mar1_14` |
| Frost days (Tmin < 0) | `frost_days_jan` | `frost_days_jan_feb` | `frost_days_mar1_14` |
| Warmth in the last month | `gdd5_jan`, `cum_tmax_jan`, `cum_tmean_jan` | `gdd5_feb`, `cum_tmax_feb`, `cum_tmean_feb` | `gdd5_mar1_14`, `cum_tmax_mar1_14`, `cum_tmean_mar1_14`, `cum_tmax_feb1_mar14` |
| Precipitation / sunshine | `precip_jan_mm`, `sunshine_jan_h` (+ `_missing_days`) | `precip_feb_mm`, `sunshine_feb_h` (+ `_missing_days`) | `precip_mar1_14_mm`, `sunshine_mar1_14_h` (+ `_missing_days`) |
| Not features | `station_id`, `location`, `prefecture`, `region`, `bloom_year`, `full_bloom_date`, **`full_bloom_doy` (target)**, `split`, `fold1`–`fold5` | same | – |

Definitions: chill day = daily mean 0–7.2 °C; `gdd5` = sum of max(Tmean − 5, 0);
`cum_*` = sum of daily values over the month. Precipitation and sunshine
totals are (mean of observed days) × (days in month); `*_missing_days` says
how many days were estimated. One month with no sunshine data at all
(station 604, Feb 2008; in `mar15` also 1–14 Mar 2008) uses the same-year mean
of its region.

`mar15` contains every `mar1` column with identical values (same rows: all
4,365 also have complete temperatures to 14 Mar), plus the same quantities for
1–14 March (`*_mar1_14`; totals = mean of observed days × 14) and
`cum_tmax_feb1_mar14` = daily maximum temperature summed from 1 Feb to 14 Mar.
The earliest full bloom in the data is day 78, so every Mar 15 row is still a
forecast.
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
4. **All three forecast dates**: report `feb1`, `mar1` and `mar15` (the example
   does all three). Mar 15 was added after the meeting by team agreement.
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

## 4. Repository layout

```
README.md                  this file (modelling)
requirements.txt           Python packages
config.json                file paths used by the data-preparation scripts
data/model/                model_features_feb1.csv, _mar1.csv, _mar15.csv
data/processed/, data/raw/ source and cleaned data
sql/                       MySQL warehouse: 01-05 SQL files, setup_database.py,
                           star_schema/ (ERD) - see sql/README.md
src/model_data.py          shared loading, folds, metrics, baseline
src/example_model.py       template - copy to src/models/<your_model>.py
src/models/                your model scripts
results/                   cross-validation results (*_cv.csv)
src/etl/                   loaders that fill MySQL + db_connection.py
src/01_prepare_source_data/, src/02_preprocess/   how the source CSVs were made
reference/                 EDA reports
```

---

## 5. Troubleshooting

| Message | Fix |
|---|---|
| `No module named ...` | `python3 -m pip install -r requirements.txt`, using the same `python3` you run the script with |
| `model_features_....csv not found` | Run `git pull` from the project folder |
| `(MySQL not available: ...)` before `Loaded ... from data/model/ CSV` | Fine — MySQL is not running, so the identical CSV copy is used |
| `ValueError: Input X contains NaN` | You added a column that is not from `load_features()`; the provided features have no empty values |
| Anything about building MySQL | See [Troubleshooting in `sql/README.md`](sql/README.md#8-troubleshooting) |
