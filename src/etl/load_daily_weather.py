from pathlib import Path
from getpass import getpass

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# 1. Locate project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data/processed/02_cleaned_source_data/weather"
    / "02_cleaned_weather_daily_somei_1952_2025.csv.gz"
)

# 2. Connect to local MySQL
username = input("MySQL username [root]: ").strip() or "root"
password = getpass("MySQL password: ")

url = URL.create(
    "mysql+pymysql",
    username=username,
    password=password,
    host="localhost",
    port=3306,
    database="sakura_dw"
)

engine = create_engine(url, pool_pre_ping=True)

# 3. Check existing records
with engine.connect() as conn:
    existing = conn.execute(
        text("SELECT COUNT(*) FROM fact_daily_weather")
    ).scalar_one()

if existing != 0:
    raise RuntimeError(
        f"Fact table already contains {existing} rows. "
        "Stop to avoid duplicate loading."
    )

# 4. Read and load in batches
total_loaded = 0

for chunk in pd.read_csv(
    DATA_PATH,
    compression="gzip",
    dtype={"station_id": str},
    chunksize=20000
):
    # Convert date to date_key
    dates = pd.to_datetime(chunk["date"], errors="raise")

    chunk["date_key"] = dates.dt.strftime("%Y%m%d").astype(int)

    # Remove original date column
    chunk = chunk.drop(columns=["date"])

    # Convert NaN to SQL NULL
    chunk = chunk.astype(object).where(pd.notna(chunk), None)

    # Insert into MySQL
    chunk.to_sql(
        "fact_daily_weather",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=2000,
        method=None
    )

    total_loaded += len(chunk)

    print(f"Loaded {total_loaded:,} rows")

print("Import completed!")
print(f"Total loaded: {total_loaded:,}")



