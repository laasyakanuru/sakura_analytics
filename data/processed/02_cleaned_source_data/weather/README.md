# Cleaned weather data (Somei-yoshino)

Cleaned daily JMA weather and season-level weather features for the 82 Somei-yoshino stations, 1952–2025.

## How to run

```bash
python src/02_preprocess/02_clean_weather_and_features.py   # cleaning + features (~2 min)
python src/02_preprocess/04_eda_weather_report.py          # EDA report -> reference/weather-eda.html
```

EDA comes in two forms:
- `src/02_preprocess/03_eda_weather_rough.ipynb`: rough working notebook (checks, tables, quick plots) with outputs saved
- `src/02_preprocess/04_eda_weather_report.py`: builds the polished report `reference/weather-eda.html`

Input: `data/raw/weather_by_station/*_daily_quality.csv`, station metadata and sakura dates in `data/processed/01_source_data/`.

## Output files

| File | One row per | What it is |
|---|---|---|
| `03_weather_features_somei_station_year.csv` | station × bloom year | **Main modelling table**: station info, flowering & full-bloom dates, weather features |
| `02_cleaned_weather_daily_somei_1952_2025.csv.gz` | station × day | Cleaned daily weather with imputation flags (pandas reads `.csv.gz` directly) |
| `04_weather_gap_log_somei.csv` | missing-value run | Each temperature gap and how it was handled |
| `05_weather_station_relocations_somei.csv` | relocation | JMA station moves / instrument changes |
| `06_weather_imputation_validation_somei.csv` | test gap | Accuracy test of the gap-filling methods |

## Cleaning rules (short)

1. Keep JMA quality codes 8 and 5; code 4 ("insufficient data") is treated as missing.
2. Temperature gaps up to 60 days are filled from up to 3 nearby stations (neighbour-difference method, ~0.8 °C error). All 102 stations serve as neighbours.
3. Longer gaps, and periods before a station started observing, stay missing. Precipitation and sunshine are never filled.
4. Not fixed, only flagged: the pre-1964 definition change (daily mean can be below the minimum) and station relocations.

## Joining with the sakura data

Join on `station_id` + `year`, reading `station_id` as text:

```python
wx = pd.read_csv("data/processed/02_cleaned_source_data/weather/03_weather_features_somei_station_year.csv",
                 dtype={"station_id": str})
```

`year` is the bloom year: weather features for year Y cover 1 Oct (Y−1) to 30 Jun (Y).
The table already includes flowering and full-bloom dates and matches `01_cleaned_full_bloom_data` exactly.

## Suggested modelling filter

```python
df = wx[(wx.year >= 1964) & (wx.season_temp_complete == 1)]
```

If predicting before March, use only features up to end of February (`*_feb_end` and earlier months);
March/April features overlap the bloom date at southern stations.

Full details, charts and the column dictionary: `reference/weather-eda.html`.
