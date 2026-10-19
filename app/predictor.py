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
