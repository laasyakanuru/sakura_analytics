"""Sakura Yosoku - full-bloom forecast (Somei-yoshino) from the champion SVR.

    streamlit run app/streamlit_app.py
"""
import datetime as dt

import pandas as pd
import streamlit as st

from predictor import BloomPredictor

st.set_page_config(page_title="Sakura Yosoku", page_icon="🌸", layout="wide")


@st.cache_resource
def load():
    return BloomPredictor()


bp = load()
meta = bp.meta
import sklearn  # noqa: E402
if sklearn.__version__ != meta["sklearn_version"]:
    st.warning(f"Model was saved with scikit-learn {meta['sklearn_version']} but this app runs "
               f"{sklearn.__version__}; pin scikit-learn=={meta['sklearn_version']} in requirements.txt.")
stations = bp.stations()

st.title("🌸 Sakura Yosoku — full-bloom forecast")
st.caption(
    f"Somei-yoshino full-bloom date forecast as of **15 March**, from an RBF Support Vector "
    f"Regression on {len(bp.features)} weather and location features. Cross-validated MAE "
    f"{meta['validation_mae_days']:.2f} days; deployed model refit on "
    f"{meta['trained_years'][0]}–{meta['trained_years'][1]}."
)

# ----------------------------------------------------------------- inputs
with st.sidebar:
    st.header("Station and season")
    labels = stations["location"] + " (" + stations["prefecture"] + ")"
    default = int(labels.str.startswith("Tokyo").idxmax()) if labels.str.startswith("Tokyo").any() else 0
    pick = st.selectbox("Station", range(len(stations)), index=default, format_func=lambda i: labels[i])
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
                "Edit them below with the weather observed up to 14 March.")

st.subheader(f"{station.location}, {station.prefecture} · {year}")

with st.expander("Feature values (editable)", expanded=actual_doy is None):
    edited = st.data_editor(
        pd.DataFrame({"feature": bp.features, "value": [values[f] for f in bp.features]}),
        disabled=["feature"], hide_index=True, use_container_width=True, key=f"{station.station_id}-{year}-{mode}",
    )
    values = dict(zip(edited["feature"], edited["value"]))

# ----------------------------------------------------------------- forecast
try:
    result = bp.predict(station.station_id, year, values)
except ValueError as err:
    st.error(str(err))
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Forecast full bloom", result["date"].strftime("%d %b %Y"))
c2.metric("Usual date (30-yr station mean)",
          (dt.date(int(year), 1, 1) + dt.timedelta(days=round(result["reference"]) - 1)).strftime("%d %b"))
c3.metric("Earlier / later than usual", f"{result['anomaly']:+.1f} days")
if actual_doy is not None:
    actual = dt.date(int(year), 1, 1) + dt.timedelta(days=int(actual_doy) - 1)
    c4.metric("Observed full bloom", actual.strftime("%d %b %Y"),
              f"error {result['doy'] - actual_doy:+.1f} days", delta_color="off")
    st.caption("Past seasons were part of the training data, so their errors look better "
               "than a true forecast would. Use the cross-validated MAE above as the honest accuracy.")
else:
    c4.metric("Typical error", f"± {meta['validation_mae_days']:.1f} days")

# ----------------------------------------------------------------- history
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
  tuned on five expanding-window validation folds (2005–2014).
- **Inputs:** {', '.join(f'`{f}`' for f in bp.features)}.
- Built with scikit-learn {meta['sklearn_version']}.
"""
    )
