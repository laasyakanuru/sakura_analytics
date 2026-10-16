# Sakura Yosoku — Cherry Blossom Full-Bloom Forecasting

Predict the Somei-yoshino full-bloom date for 82 JMA stations, at two forecast
dates (**Feb 1** and **Mar 1**), from daily weather 1952–2025.
All cleaned data lives in a MySQL star schema called `sakura_dw`.

## Quick start (build the database on your own computer)

You need: **MySQL Server 8+** running locally (MySQL Workbench is only the
client — the server must also be installed and started) and **Python 3.10+**.

```bash
git clone https://github.com/natsumisugimoto817/sakura_analytics.git
cd sakura_analytics
python3 -m pip install -r requirements.txt
python3 setup_database.py
```

`setup_database.py` asks for your MySQL username and password once, then
creates the database, loads all data, builds the model tables and prints a
PASS/FAIL table of 24 health checks. Expect roughly 5–15 minutes; almost all of
it is loading 2.2 million weather rows. When it ends with
`✓ All checks passed`, you are done — open MySQL Workbench and refresh the
schema list to see `sakura_dw`.

| Command | When |
|---|---|
| `python3 setup_database.py` | First time, on a MySQL without `sakura_dw` |
| `python3 setup_database.py --checks` | Re-check an existing database (read-only) |
| `python3 setup_database.py --reset` | Delete `sakura_dw` and rebuild it from scratch |

Tired of typing the password? Copy `.env.example` to `.env` and fill it in.
`.env` is git-ignored; never commit passwords.

## Using the data for modelling

Two ready-made, **leakage-safe** tables follow the team agreements in the
Model Development Meeting Summary:

| Table | Forecast date | Weather used | Rows |
|---|---|---|---|
| `model_features_feb1` | Feb 1 | 1 Oct (Y-1) – 31 Jan (Y) | 4,365 |
| `model_features_mar1` | Mar 1 | 1 Oct (Y-1) – 28/29 Feb (Y) | 4,365 |

Both already apply the agreed filters (bloom year ≥ 1964, Somei-yoshino only,
observed full-bloom date, no missing temperature day up to the cutoff) and carry
the agreed split columns:

| Column | Values |
|---|---|
| `split` | `development` (1964–2014) / `test` (2015–2025, final testing only) |
| `fold1` … `fold5` | `train` / `valid` / NULL — expanding window: fold *k* trains on 1964…(2002+2k) and validates on the next two years |

The target is `full_bloom_doy` (day of year). Everything else that is not an
id, target or split column is a candidate feature.

### In Python (recommended — same folds and metrics for every team)

```python
import sys; sys.path.append("src")      # run from the project root
from model_data import load_features, feature_columns, cv_folds, evaluate, cv_report, climatology_baseline

df = load_features("mar1")               # or "feb1"
X_cols = feature_columns(df)

for fold, train, valid in cv_folds(df):
    model.fit(train[X_cols], train["full_bloom_doy"])
    print(fold, evaluate(valid["full_bloom_doy"], model.predict(valid[X_cols])))
```

`evaluate` returns MAE, RMSE, R² and the ±3-day hit rate.
`cv_report(df, predict_fn)` runs all five folds and adds a mean row.
`python3 src/model_data.py` prints the climatology baseline for both forecast dates.

### In MySQL Workbench

```sql
SELECT * FROM sakura_dw.model_features_mar1 WHERE fold1 = 'train';
```

### Leakage rules (please read)

- Only use columns from `model_features_feb1` / `model_features_mar1` as
  features. Do **not** pull extra columns from `fact_bloom_season`: many of
  them include weather after the forecast date (e.g. `tmean_mar`,
  `chill_days_0_7c_nov_feb` for Feb 1, `precip_feb_mar_mm`).
- `rule600_tmax_pred_doy` and `rule400_tmean_pred_doy` need temperatures until
  the threshold is reached, which is always after Mar 1. They are neither
  features nor a fair forecast-date baseline. Use `climatology_baseline`
  (each station's training-period mean) as the forecast-date baseline.
- Do not tune anything on the `test` rows; they are for the final champion
  evaluation only.
- Station coverage changes: all 82 stations appear in training years, but only
  48 stations have targets from 2013 onwards (JMA reduced observations).

## Repository layout

```
setup_database.py          one-command database build + health checks
requirements.txt           Python packages
src/etl/                   loaders for dim_station, fact_daily_weather, fact_bloom_season
src/etl/db_connection.py   shared MySQL login (env vars / .env / prompt)
src/model_data.py          shared data loading, CV folds, metrics, baseline
src/01_prepare_source_data, src/02_preprocess   how the CSVs were produced
sql/                       schema, validation, analysis and model-table SQL (see sql/README.md)
data/                      raw and cleaned CSVs used by the loaders
reference/                 EDA reports
```

## Troubleshooting

| Message | Fix |
|---|---|
| `Can't connect to MySQL server on 'localhost'` | The MySQL server is not running. macOS: System Settings → MySQL → Start. Windows: Services → MySQL80 → Start. |
| `Access denied for user 'root'@'localhost'` | Wrong password — use the one you use to log in with MySQL Workbench. |
| `'cryptography' package is required` | `python3 -m pip install -r requirements.txt` |
| `Database 'sakura_dw' already exists` | Use `--checks` to verify it, or `--reset` to rebuild. |
| `No module named pandas` (or similar) | `python3 -m pip install -r requirements.txt` with the same `python3` you run the script with. |
| A check shows FAIL | Run `python3 setup_database.py --reset` once; if it still fails, share the PASS/FAIL table with the team. |
