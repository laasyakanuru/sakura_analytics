"""Sakura Yosoku - Somei-yoshino full-bloom forecasts for tour operators.

    streamlit run app/streamlit_app.py

Shows the batch forecasts saved by src/models/batch_forecast.py for the three
issue dates (1 Feb, 1 Mar, 15 Mar): earlier forecasts give more notice, later
ones are more accurate. The app never runs the model; it only reads the files.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR, FORECAST_DIR = APP_DIR / "model", APP_DIR / "forecasts"
DATES = ("feb1", "mar1", "mar15")

st.set_page_config(page_title="Sakura Yosoku", page_icon="🌸", layout="wide")


@st.cache_data
def load_meta():
    out = {}
    for d in DATES:
        path = MODEL_DIR / f"svr_{d}_meta.json"
        if path.exists():
            m = json.loads(path.read_text())
            m.pop("station_reference", None)
            m.pop("stations", None)
            out[d] = m
    return out


@st.cache_data
def load_forecast(path, mtime):
    return pd.read_csv(path, dtype={"station_id": str}, parse_dates=["predicted_date"])


def forecast_files():
    """{year: {date: path}} for every saved batch forecast."""
    files = {}
    for f in FORECAST_DIR.glob("forecast_*_*.csv"):
        _, year, date = f.stem.split("_")
        if date in DATES:
            files.setdefault(int(year), {})[date] = f
    return files


meta = load_meta()
files = forecast_files()

st.title("🌸 Sakura Yosoku — full-bloom forecasts")
st.caption("Somei-yoshino full-bloom dates for Japanese weather stations, forecast by a Support "
           "Vector Regression on weather up to each issue date. **Earlier forecasts give more notice; "
           "later ones are more accurate.**")

if not meta or not files:
    st.info("No forecasts yet. Run `python3 src/models/export_svr.py` and "
            "`python3 src/models/batch_forecast.py`, then commit `app/model/` and `app/forecasts/`.")
    st.stop()

year = st.selectbox("Season", sorted(files, reverse=True))
season = {d: load_forecast(p, p.stat().st_mtime) for d, p in files[year].items()}
available = [d for d in DATES if d in season and d in meta]

tabs = st.tabs(["⚖️ Choose your forecast"] + [f"📅 {meta[d]['issue_label']}" for d in available])


def pct(x):
    return f"{100 * x:.0f}%"


# ------------------------------------------------------------- trade-off tab
with tabs[0]:
    st.subheader("Notice vs accuracy")
    rows = []
    for d in available:
        m, v, t = meta[d], meta[d]["validation"], meta[d].get("test")
        rows.append({
            "Forecast issued": m["issue_label"],
            "Typical notice before bloom": f"~{m['typical_lead_days']:.0f} days",
            "Average error": f"± {v['MAE']:.1f} days",
            "Within ±3 days": pct(v["hit_rate_3d"]),
            "Tested on 2015–2025": f"± {t['MAE']:.1f} days" if t else "—",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Average error and ±3-day share are cross-validated on 2005–2014 seasons the model "
               "had not seen. Notice = median days between the issue date and the usual bloom date.")

    c1, c2, c3 = st.columns(3)
    c1.markdown("**Book early → 1 Feb**  \nTwo months' notice for flights, hotels and guides. "
                "Plan with a buffer of about ±4 days.")
    c2.markdown("**Fine-tune → 1 Mar**  \nAbout a month out. Accuracy is close to the 1 Feb "
                "forecast, so use it mainly to confirm or shift plans.")
    c3.markdown("**Lock in → 15 Mar**  \nAbout three weeks out and the most accurate: "
                "six in ten forecasts land within ±3 days.")

    st.subheader(f"Compare the forecasts for one station · {year}")
    names = pd.concat(season.values())[["station_id", "location", "prefecture"]].drop_duplicates("station_id")
    names = names.sort_values("location").reset_index(drop=True)
    labels = names["location"] + " (" + names["prefecture"] + ")"
    default = labels.str.startswith("Tokyo")
    i = st.selectbox("Station", range(len(names)), index=int(default.idxmax()) if default.any() else 0,
                     format_func=lambda k: labels[k])
    sid = names.loc[i, "station_id"]

    cols = st.columns(len(available) + 1)
    observed = None
    for col, d in zip(cols, available):
        row = season[d][season[d]["station_id"] == sid]
        if row.empty:
            col.metric(f"Issued {meta[d]['issue_label']}", "—")
            continue
        r = row.iloc[0]
        col.metric(f"Issued {meta[d]['issue_label']}", r.predicted_date.strftime("%d %b"),
                   f"{int(r.notice_days)} days' notice", delta_color="off")
        if "observed_doy" in r and pd.notna(r.observed_doy):
            observed = r.observed_doy
    if observed is not None:
        obs = pd.Timestamp(year, 1, 1) + pd.Timedelta(days=int(observed) - 1)
        cols[-1].metric("Observed", obs.strftime("%d %b"))
        st.caption("Past seasons were part of the training data, so these look better than a true "
                   "forecast would. Use the cross-validated errors above as the honest accuracy.")

# ------------------------------------------------------------- one tab per date
for tab, d in zip(tabs[1:], available):
    with tab:
        m, fc = meta[d], season[d]
        v, t = m["validation"], m.get("test")
        st.markdown(f"**Issued {m['issue_label']} {year}** using weather up to the day before · "
                    f"typical error ± {v['MAE']:.1f} days · {pct(v['hit_rate_3d'])} within ±3 days"
                    + (f" · tested on 2015–2025: ± {t['MAE']:.1f} days" if t else ""))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stations", len(fc))
        first, last = fc.iloc[0], fc.iloc[-1]
        c2.metric("Earliest bloom", first.predicted_date.strftime("%d %b"), first.location, delta_color="off")
        c3.metric("Latest bloom", last.predicted_date.strftime("%d %b"), last.location, delta_color="off")
        c4.metric("Median notice", f"{fc['notice_days'].median():.0f} days")

        st.map(fc, latitude="latitude", longitude="longitude", size=8000)

        cols = ["location", "prefecture", "region", "predicted_date", "notice_days",
                "anomaly_days", "reference_doy"]
        cols += [c for c in ("observed_doy", "error_days") if c in fc]
        show = fc[cols].copy()
        show["predicted_date"] = show["predicted_date"].dt.strftime("%d %b")
        st.dataframe(
            show.rename(columns={"predicted_date": "forecast", "notice_days": "days' notice",
                                 "anomaly_days": "vs usual (days)", "reference_doy": "usual DOY",
                                 "observed_doy": "observed DOY", "error_days": "error (days)"}),
            hide_index=True, use_container_width=True,
        )
        st.download_button("Download CSV", files[year][d].read_bytes(), files[year][d].name,
                           "text/csv", key=f"dl-{d}")
        st.caption(f"Generated {fc['generated_at'].iloc[0]} · {fc['model'].iloc[0]}")

with st.expander("About the model"):
    st.markdown(
        """
- **Target:** days earlier or later than each station's mean full-bloom date over the last
  30 years; the forecast adds that back to the station mean.
- **Model:** StandardScaler + RBF Support Vector Regression, one per issue date, tuned on five
  expanding-window validation folds (2005–2014) and refit on all seasons for deployment.
- **Inputs:** location plus temperature, chill and heat-sum features up to the issue date
  (9 for 1 Feb, 17 for 1 Mar, 28 for 15 Mar).
- **Batch forecasts:** `src/models/batch_forecast.py` scores every station once per issue date;
  this app only displays the saved results.
"""
    )
