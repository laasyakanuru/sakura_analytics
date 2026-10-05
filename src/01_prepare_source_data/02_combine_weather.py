#%%
import json
from pathlib import Path
import pandas as pd

# Combine station-level weather CSVs for the selected years.
# Preserve station IDs, location names, and missing measurements.
def combine_weather_csvs(
    input_dir,
    output_file,
    start_year,
    end_year,
    expected_stations=102,
):
    input_dir = Path(input_dir)
    output_file = Path(output_file)

    if start_year > end_year:
        raise ValueError("start_year must not exceed end_year.")

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    # Find station analysis files and exclude quality-detail files.
    files = sorted(
        path
        for path in input_dir.glob("weather_*.csv")
        if not path.name.endswith("_quality.csv")
        and path.resolve() != output_file.resolve()
    )
    files = [
        path for path in files
        if path.resolve() != output_file.resolve()
    ]

    if not files:
        raise FileNotFoundError("No analysis CSV files found.")

    frames = []
    expected_columns = None
    first_columns = ["station_id", "location", "date"]

    for path in files:
        # Read IDs as text to preserve leading zeros.
        df = pd.read_csv(
            path,
            encoding="utf-8-sig",
            dtype={"station_id": "string", "location": "string"},
        )

        missing_columns = set(first_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"{path.name}: missing columns {sorted(missing_columns)}"
            )

        # Prevent combining different measurement selections.
        if expected_columns is None:
            expected_columns = list(df.columns)
        elif set(df.columns) != set(expected_columns):
            raise ValueError(
                f"{path.name}: inconsistent measurement columns."
            )

        for column in ["station_id", "location"]:
            df[column] = df[column].str.strip()
            if (df[column].isna() | df[column].eq("")).any():
                raise ValueError(f"{path.name}: missing {column}.")

        # Require explicit ISO dates to avoid month/day ambiguity.
        date_text = df["date"].astype("string")
        valid_format = date_text.str.fullmatch(r"\d{4}-\d{2}-\d{2}")

        if not valid_format.fillna(False).all():
            raise ValueError(f"{path.name}: dates must use YYYY-MM-DD.")

        df["date"] = pd.to_datetime(
            date_text,
            format="%Y-%m-%d",
            errors="raise",
        )

        # Keep both boundary years.
        selected = df["date"].dt.year.between(start_year, end_year)
        frames.append(df.loc[selected, expected_columns].copy())

    combined = pd.concat(frames, ignore_index=True)

    if combined.empty:
        raise ValueError("No records found for the selected years.")

    # Require one consistent name for each station.
    name_counts = combined.groupby("station_id")["location"].nunique()
    if (name_counts > 1).any():
        raise ValueError("Some station IDs have multiple location names.")

    # Each station-date pair must be unique.
    if combined.duplicated(["station_id", "date"]).any():
        raise ValueError("Duplicate station-date records found.")

    station_count = combined["station_id"].nunique()
    if station_count != expected_stations:
        raise ValueError(
            f"Expected {expected_stations} stations; found {station_count}."
        )

    # Verify that every station has every day in the selected period.
    expected_days = len(
        pd.date_range(
            start=f"{start_year}-01-01",
            end=f"{end_year}-12-31",
            freq="D",
        )
    )

    counts = combined.groupby("station_id").size()
    incomplete = counts[counts != expected_days]

    if not incomplete.empty:
        raise ValueError(
            "Incomplete date coverage for these stations:\n"
            + incomplete.to_string()
        )

    # Order identifier columns before measurement columns.
    measurement_columns = [
        column for column in expected_columns
        if column not in first_columns
    ]

    combined = (
        combined[first_columns + measurement_columns]
        .sort_values(["station_id", "date"])
        .reset_index(drop=True)
    )

    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    
    # Remove location names; retain station IDs for joining metadata.
    combined = combined.drop(columns=["location"])

    # Save missing measurements as blank cells, not zeros.
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
        na_rep="",
    )

    print(f"Stations: {station_count:,}")
    print(f"Records: {len(combined):,}")
    print(f"Period: {start_year}-01-01 to {end_year}-12-31")
    print(f"File size: {output_file.stat().st_size / 1024**2:,.1f} MiB")
    print(f"Saved to: {output_file}")

    return combined


# Set the analysis period here.
START_YEAR = 1975
END_YEAR = 2025

# Locate the repository.
# For a .py file saved under src/01_prepare_source_data/.
if "__file__" in globals():
    PROJECT_DIR = Path(__file__).resolve().parents[2]
else:
    # For direct execution in a notebook.
    PROJECT_DIR = Path(
        r"C:\Users\natsu\OneDrive\ドキュメント\GitHub"
        r"\sakura_analytics"
    )

# Load input and output paths from config.json.
CONFIG_FILE = PROJECT_DIR / "config.json"

with CONFIG_FILE.open(encoding="utf-8-sig") as file:
    config = json.load(file)
# %%
