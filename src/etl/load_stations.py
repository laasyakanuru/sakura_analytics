from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db_connection import get_engine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
META_DIR = (
    PROJECT_ROOT
    / "data/processed/01_source_data/01_station_metadata"
)


def read_metadata(filename):
    return pd.read_csv(
        META_DIR / filename,
        dtype={"station_id": str},
        encoding="utf-8-sig",
    )


def main():
    # Read and merge station metadata.
    names = read_metadata("01_01_station_name.csv")
    elevations = read_metadata("01_02_station_elevations.csv")
    species = read_metadata("01_03_cherry_species.csv")

    stations = (
        names.merge(
            elevations,
            on="station_id",
            how="outer",
            validate="one_to_one",
        )
        .merge(
            species,
            on="station_id",
            how="outer",
            validate="one_to_one",
        )
    )

    # Validate station records before loading.
    if len(stations) != 102:
        raise ValueError(
            f"Expected 102 stations, found {len(stations)}."
        )

    if stations.isna().any().any():
        missing_columns = stations.columns[
            stations.isna().any()
        ].tolist()
        raise ValueError(
            f"Missing metadata in columns: {missing_columns}"
        )

    if stations["station_id"].duplicated().any():
        raise ValueError("Duplicate station IDs found.")

    print(f"Source stations: {len(stations)}")

    # Connect to the local MySQL database.
    engine = get_engine()

    try:
        # Load all station records in a single transaction.
        with engine.begin() as conn:
            existing = conn.execute(
                text("SELECT COUNT(*) FROM dim_station")
            ).scalar_one()

            if existing != 0:
                raise RuntimeError(
                    f"dim_station already contains {existing} rows. "
                    "Import stopped to prevent duplicate loading."
                )

            stations.to_sql(
                "dim_station",
                con=conn,
                if_exists="append",
                index=False,
                chunksize=1000,
            )

            loaded = conn.execute(
                text("SELECT COUNT(*) FROM dim_station")
            ).scalar_one()

            if loaded != len(stations):
                raise RuntimeError(
                    f"Expected {len(stations)} rows, found {loaded}. "
                    "The transaction will be rolled back."
                )

        print(f"Successfully loaded {loaded} stations.")

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()