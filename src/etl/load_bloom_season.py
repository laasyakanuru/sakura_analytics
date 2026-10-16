
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db_connection import get_engine

# ------------------------------------------
# STEP 1: EXTRACT
# ------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data/processed/02_cleaned_source_data/weather"
    / "03_weather_features_somei_station_year.csv"
)

df = pd.read_csv(
    DATA_PATH,
    dtype={"station_id": str}
)

print("Source rows:", len(df))
print("Source columns:", len(df.columns))


# ------------------------------------------
# STEP 2: TRANSFORM
# ------------------------------------------

# Remove attributes already stored in DIM_STATION
dimension_columns = [
    "location",
    "prefecture",
    "region",
    "cherry_species",
    "latitude",
    "longitude",
    "elevation_m"
]

df = df.drop(columns=dimension_columns)

# Match the MySQL column name
df = df.rename(columns={"year": "season_key"})

# Convert date columns
for col in ["flowering_date", "full_bloom_date"]:
    df[col] = pd.to_datetime(
        df[col],
        errors="raise"
    ).dt.date

# Check missing keys
assert df[["station_id", "season_key"]].notna().all().all(), \
    "Missing primary key values!"

# Check duplicate station-years
assert not df.duplicated(
    ["station_id", "season_key"]
).any(), "Duplicate station-season records!"

# Convert missing values to SQL NULL
df = df.astype(object).where(pd.notna(df), None)

print("Transformed columns:", len(df.columns))
print("Ready to load:", len(df), "rows")


# ------------------------------------------
# STEP 3: CONNECT TO MYSQL
# ------------------------------------------

engine = get_engine()


# ------------------------------------------
# STEP 4: VALIDATE AND LOAD
# ------------------------------------------

with engine.begin() as conn:

    existing = conn.execute(
        text("SELECT COUNT(*) FROM fact_bloom_season")
    ).scalar_one()

    if existing != 0:
        raise RuntimeError(
            f"Table already contains {existing} rows. "
            "Import stopped to prevent duplicates."
        )

    # Verify columns match MySQL table
    result = conn.execute(
        text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'sakura_dw'
              AND table_name = 'fact_bloom_season'
            ORDER BY ordinal_position
        """)
    )

    mysql_columns = [row[0] for row in result]

    missing_in_mysql = set(df.columns) - set(mysql_columns)
    missing_in_csv = set(mysql_columns) - set(df.columns)

    if missing_in_mysql or missing_in_csv:
        raise ValueError(
            f"Columns missing in MySQL: {missing_in_mysql}\n"
            f"Columns missing in CSV: {missing_in_csv}"
        )

    # Verify station foreign keys
    station_ids = {
        str(row[0]) for row in conn.execute(
            text("SELECT station_id FROM dim_station")
        )
    }

    unknown_stations = set(df["station_id"]) - station_ids

    if unknown_stations:
        raise ValueError(
            f"Unknown station IDs: {unknown_stations}"
        )

    # Verify season foreign keys
    season_keys = {
        row[0] for row in conn.execute(
            text("SELECT season_key FROM dim_season")
        )
    }

    unknown_seasons = set(df["season_key"]) - season_keys

    if unknown_seasons:
        raise ValueError(
            f"Unknown season keys: {unknown_seasons}"
        )

    # Load records into the existing fact table
    df.to_sql(
        "fact_bloom_season",
        con=conn,
        if_exists="append",
        index=False,
        chunksize=1000,
        method=None
    )

    print("Rows inserted:", len(df))

print("ETL completed successfully!")

engine.dispose()
