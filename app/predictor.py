"""Load the exported SVR bundle and turn feature values into a bloom-date forecast."""
import datetime as dt
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "model"


class BloomPredictor:
    def __init__(self, model_dir=MODEL_DIR):
        self.model = joblib.load(model_dir / "svr_mar15.joblib")
        self.meta = json.loads((model_dir / "svr_mar15_meta.json").read_text())
        self.features = self.meta["features"]
        self.table = pd.read_csv(model_dir / "features_mar15.csv", dtype={"station_id": str})

    def stations(self):
        """DataFrame of station_id, location, prefecture, region (sorted by name)."""
        rows = [{"station_id": sid, **info} for sid, info in self.meta["stations"].items()]
        return pd.DataFrame(rows).sort_values(["location", "prefecture"]).reset_index(drop=True)

    def history(self, station_id):
        return self.table[self.table["station_id"] == str(station_id)].sort_values("bloom_year")

    def predict(self, station_id, year, values):
        """values: dict feature -> number. Returns a dict with the forecast."""
        missing = [f for f in self.features if f not in values or pd.isna(values[f])]
        if missing:
            raise ValueError(f"missing feature values: {', '.join(missing)}")
        reference = self.meta["station_reference"][str(station_id)]
        x = pd.DataFrame([[float(values[f]) for f in self.features]], columns=self.features)
        anomaly = float(self.model.predict(x)[0])
        doy = reference + anomaly
        date = dt.date(int(year), 1, 1) + dt.timedelta(days=int(np.rint(doy)) - 1)
        return {"doy": doy, "date": date, "anomaly": anomaly, "reference": reference}

    def predict_batch(self, rows, year):
        """Score many stations at once. rows: DataFrame with station_id + the model
        features (one row per station). Returns one forecast row per station."""
        rows = rows.copy()
        rows["station_id"] = rows["station_id"].astype(str)
        missing = [f for f in self.features if f not in rows.columns]
        if missing:
            raise ValueError(f"missing feature columns: {', '.join(missing)}")
        unknown = sorted(set(rows["station_id"]) - set(self.meta["station_reference"]))
        if unknown:
            raise ValueError(f"stations not in the model: {', '.join(unknown)}")
        bad = rows[self.features].isna().any(axis=1)
        if bad.any():
            raise ValueError("missing feature values for stations: "
                             + ", ".join(rows.loc[bad, "station_id"]))
        reference = rows["station_id"].map(self.meta["station_reference"]).astype(float)
        anomaly = self.model.predict(rows[self.features].astype(float))
        doy = reference.to_numpy() + anomaly
        start = pd.Timestamp(int(year), 1, 1)
        info = pd.DataFrame.from_dict(self.meta["stations"], orient="index")
        out = pd.DataFrame({
            "station_id": rows["station_id"].to_numpy(),
            "location": rows["station_id"].map(info["location"]).to_numpy(),
            "prefecture": rows["station_id"].map(info["prefecture"]).to_numpy(),
            "region": rows["station_id"].map(info["region"]).to_numpy(),
            "latitude": rows["latitude"].to_numpy(),
            "longitude": rows["longitude"].to_numpy(),
            "year": int(year),
            "reference_doy": reference.round(1).to_numpy(),
            "anomaly_days": np.round(anomaly, 1),
            "predicted_doy": np.round(doy, 1),
            "predicted_date": [(start + pd.Timedelta(days=int(np.rint(d)) - 1)).date().isoformat()
                               for d in doy],
        })
        return out.sort_values("predicted_doy").reset_index(drop=True)
