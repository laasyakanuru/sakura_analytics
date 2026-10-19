"""Sakura Yosoku - full-bloom forecast (Somei-yoshino) from the champion SVR.

    streamlit run app/streamlit_app.py

Official forecast : batch forecasts saved by src/models/batch_forecast.py (frozen,
                    the same for every viewer).
What-if           : live single-station prediction from editable feature values.
"""
import datetime as dt
from pathlib import Path

import pandas as pd
import sklearn
import streamlit as st

from predictor import BloomPredictor

FORECAST_DIR = Path(__file__).resolve().parent / "forecasts"

st.set_page_config(page_title="Sakura Yosoku", page_icon="🌸", layout="wide")


@st.cache_resource
def load():
    return BloomPredictor()


@st.cache_data
def load_forecast(path, mtime):
    return pd.read_csv(path, dtype={"station_id": str}, parse_dates=["predicted_date"])


bp = load()
meta = bp.meta
if sklearn.__version__ != meta["sklearn_version"]:
    st.warning(f"Model was saved with scikit-learn {meta['sklearn_version']} but this app runs "
               f"{sklearn.__version__}; pin scikit-learn=={meta['sklearn_version']} in requirements.txt.")

st.title("🌸 Sakura Yosoku — full-bloom forecast")
st.caption(
    f"Somei-yoshino full-bloom dates forecast on **15 March** by an RBF Support Vector Regression "
    f"on {len(bp.features)} weather and location features. Cross-validated MAE "
    f"{meta['validation_mae_days']:.2f} days; deployed model refit on "
    f"{meta['trained_years'][0]}–{meta['trained_years'][1]}."
)

official, whatif = st.tabs(["📋 Official forecast", "🔧 What-if"])

# ============================================================ official (batch)
with official:
    files = sorted(FORECAST_DIR.glob("forecast_*_mar15.csv"), reverse=True)
    if not files:
        st.info("No batch forecast yet. Run `python3 src/models/batch_forecast.py` and commit "
                "`app/forecasts/`.")
    else:
        years = {int(f.stem.split("_")[1]): f for f in files}
        year = st.selectbox("Season", list(years), key="official_year")
        fc = load_forecast(years[year], years[year].stat().st_mtime)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stations", len(fc))
        first, last = fc.iloc[0], fc.iloc[-1]
        c2.metric("Earliest bloom", first.predicted_date.strftime("%d %b"), first.location, delta_color="off")
        c3.metric("Latest bloom", last.predicted_date.strftime("%d %b"), last.location, delta_color="off")
        if "error_days" in fc and fc["error_days"].notna().any():
            c4.metric("MAE vs observed", f"{fc['error_days'].abs().mean():.2f} days")
            st.caption("This season was part of the training data, so the error vs observed is "
                       "optimistic. Use the cross-validated MAE above as the honest accuracy.")
        else:
            c4.metric("Typical error", f"± {meta['validation_mae_days']:.1f} days")

        st.map(fc, latitude="latitude", longitude="longitude", size=8000)

        cols = ["location", "prefecture", "region", "predicted_date", "anomaly_days", "reference_doy"]
        cols += [c for c in ("observed_doy", "error_days") if c in fc]
        show = fc[cols].copy()
        show["predicted_date"] = show["predicted_date"].dt.strftime("%d %b")
        st.dataframe(
            show.rename(columns={"predicted_date": "forecast", "anomaly_days": "vs usual (days)",
                                 "reference_doy": "usual DOY", "observed_doy": "observed DOY",
                                 "error_days": "error (days)"}),
            hide_index=True, use_container_width=True,
        )
        st.download_button("Download CSV", years[year].read_bytes(), years[year].name, "text/csv")
        st.caption(f"Generated {fc['generated_at'].iloc[0]} · {fc['model'].iloc[0]}")

# ============================================================ what-if (real time)
with whatif:
    stations = bp.stations()
    labels = stations["location"] + " (" + stations["prefecture"] + ")"
    tokyo = labels.str.startswith("Tokyo")
    left, right = st.columns([1, 2])
    with left:
        pick = st.selectbox("Station", range(len(stations)),
                            index=int(tokyo.idxmax()) if tokyo.any() else 0,
                            format_func=lambda i: labels[i])
        station = stations.iloc[pick]
        hist = bp.history(station.station_id)
        mode = st.radio("Season", ["Past season (features on file)", "New season (enter features)"])
        if mode.startswith("Past") and not hist.empty:
            year = st.selectbox("Bloom year", hist["bloom_year"].tolist()[::-1])
            base = hist[hist["bloom_year"] == year].iloc[0]
            values = {f: float(base[f]) for f in bp.features}
            actual_doy = float(base["full_bloom_doy"])
        else:
            year = st.number_input("Bloom year", 1964, 2100, dt.date.today().year)
            last = hist.iloc[-1] if not hist.empty else None
            values = {f: (float(last[f]) if last is not None else 0.0) for f in bp.features}
            actual_doy = None
            st.info("Starting values are copied from this station's latest season. "
                    "Edit them with the weather observed up to 14 March.")
    with right:
        st.subheader(f"{station.location}, {station.prefecture} · {year}")
        with st.expander("Feature values (editable)", expanded=actual_doy is None):
            edited = st.data_editor(
                pd.DataFrame({"feature": bp.features, "value": [values[f] for f in bp.features]}),
                disabled=["feature"], hide_index=True, use_container_width=True,
                key=f"{station.station_id}-{year}-{mode}",
            )
            values = dict(zip(edited["feature"], edited["value"]))

        try:
            result = bp.predict(station.station_id, year, values)
        except ValueError as err:
            st.error(str(err))
            st.stop()

        c1, c2, c3 = st.columns(3)
        c1.metric("Forecast full bloom", result["date"].strftime("%d %b %Y"))
        c2.metric("Usual date (30-yr mean)",
                  (dt.date(int(year), 1, 1) + dt.timedelta(days=round(result["reference"]) - 1)).strftime("%d %b"))
        c3.metric("Earlier / later than usual", f"{result['anomaly']:+.1f} days")
        if actual_doy is not None:
            actual = dt.date(int(year), 1, 1) + dt.timedelta(days=int(actual_doy) - 1)
            st.metric("Observed full bloom", actual.strftime("%d %b %Y"),
                      f"error {result['doy'] - actual_doy:+.1f} days", delta_color="off")

    if not hist.empty:
        st.subheader("Observed full-bloom day of year at this station")
        chart = hist.set_index("bloom_year")[["full_bloom_doy"]].rename(columns={"full_bloom_doy": "observed"})
        chart["30-yr station mean"] = result["reference"]
        st.line_chart(chart)

with st.expander("About the model"):
    st.markdown(
        f"""
- **Target:** days earlier or later than the station's mean full-bloom date over the last
  {meta['station_reference_years']} years; the forecast adds that back to the station mean.
- **Model:** StandardScaler + SVR ({', '.join(f'{k}={v}' for k, v in meta['params'].items())}),
  tuned on five expanding-window validation folds (2005–2014); test MAE 2015–2025: 3.72 days
  (30-year climatology: 5.04).
- **Official forecast** = batch scores saved by `src/models/batch_forecast.py`;
  **What-if** = live prediction from the values you enter.
- **Inputs:** {', '.join(f'`{f}`' for f in bp.features)}.
- Built with scikit-learn {meta['sklearn_version']}.
"""
    )
