"""Sakura Yosoku - Somei-yoshino full-bloom forecasts for tour operators.

    streamlit run app/streamlit_app.py

Shows the batch forecasts saved by src/models/batch_forecast.py for the three
issue dates (1 Feb, 1 Mar, 15 Mar): earlier forecasts give more notice, later
ones are more accurate. The app never runs the model; it only reads the files.
"""
import base64
import json
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR, FORECAST_DIR = APP_DIR / "model", APP_DIR / "forecasts"
DATES = ("feb1", "mar1", "mar15")

st.set_page_config(page_title="Sakura Yosoku", page_icon="🌸", layout="wide")

# Sakura icon (app/sakura.png, 128x128 transparent PNG) as a data URL for the map.
SAKURA_ICON = {
    "url": "data:image/png;base64," + base64.b64encode((APP_DIR / "sakura.png").read_bytes()).decode(),
    "width": 128, "height": 128, "anchorY": 64,
}
JAPAN_VIEW = pdk.ViewState(latitude=36.8, longitude=137.5, zoom=4.2)


def sakura_map(fc, selected=None):
    """Pydeck map with a sakura flower per station; the selected station is drawn larger."""
    pts = fc[["station_id", "location", "prefecture", "latitude", "longitude",
              "predicted_date", "notice_days"]].copy()
    pts["forecast"] = pts["predicted_date"].dt.strftime("%d %b")
    pts["icon"] = [SAKURA_ICON] * len(pts)
    pts["size"] = (pts["station_id"] == selected).map({True: 46, False: 26})
    pts = pts.drop(columns="predicted_date").sort_values("size")   # selected on top
    dots = pdk.Layer(   # small pink dot under each flower, so stations always show
        "ScatterplotLayer", pts, get_position=["longitude", "latitude"],
        get_fill_color=[217, 79, 134, 200], get_radius=4, radius_units="pixels",
    )
    flowers = pdk.Layer(
        "IconLayer", pts, get_icon="icon", get_position=["longitude", "latitude"],
        get_size="size", size_units="pixels", pickable=True,
    )
    view = JAPAN_VIEW
    if selected is not None and (pts["station_id"] == selected).any():
        r = pts[pts["station_id"] == selected].iloc[0]
        view = pdk.ViewState(latitude=float(r.latitude), longitude=float(r.longitude), zoom=6)
    return pdk.Deck(
        layers=[dots, flowers], initial_view_state=view,
        map_provider="carto", map_style=pdk.map_styles.CARTO_LIGHT,
        tooltip={"html": "<b>{location}</b>, {prefecture}<br/>Full bloom {forecast}"
                         "<br/>{notice_days} days' notice"},
    )


def station_picker(df, key, allow_all=False):
    """Station dropdown sorted by name; returns station_id (or None for 'All stations')."""
    names = df[["station_id", "location", "prefecture"]].drop_duplicates("station_id")
    names = names.sort_values("location").reset_index(drop=True)
    labels = (names["location"] + " (" + names["prefecture"] + ")").tolist()
    ids = names["station_id"].tolist()
    if allow_all:
        labels, ids = ["All stations"] + labels, [None] + ids
        default = 0
    else:
        default = next((i for i, l in enumerate(labels) if l.startswith("Tokyo")), 0)
    i = st.selectbox("Station", range(len(labels)), index=default,
                     format_func=lambda k: labels[k], key=key)
    return ids[i]


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
           "only the 15 March forecast has proven more reliable than the usual date.**")

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


def beats_baseline(m):
    """True / False if the test was run and the baseline is known, else None."""
    t, b = m.get("test"), m.get("baseline_test")
    return None if not (t and b) else t["MAE"] < b["MAE"]


def honest_error(m):
    """Test MAE when available (unseen 2015-2025 seasons), else the validation MAE."""
    return (m.get("test") or m["validation"])["MAE"]


# ------------------------------------------------------------- trade-off tab
with tabs[0]:
    st.subheader("Notice vs accuracy")
    base = next((meta[d]["baseline_test"] for d in available if meta[d].get("baseline_test")), None)
    rows = []
    if base:
        rows.append({
            "Forecast": "Usual date (30-yr station mean)",
            "Typical notice before bloom": "any time",
            "Error on 2015–2025": f"± {base['MAE']:.1f} days",
            "Within ±3 days (2015–2025)": pct(base["hit_rate_3d"]),
            "Error on 2005–2014": "± 3.8 days",
            "Better than usual date?": "—",
        })
    for d in available:
        m, v, t = meta[d], meta[d]["validation"], meta[d].get("test")
        verdict = beats_baseline(m)
        rows.append({
            "Forecast": f"SVR issued {m['issue_label']}",
            "Typical notice before bloom": f"~{m['typical_lead_days']:.0f} days",
            "Error on 2015–2025": f"± {t['MAE']:.1f} days" if t else "not tested",
            "Within ±3 days (2015–2025)": pct(t["hit_rate_3d"]) if t else "—",
            "Error on 2005–2014": f"± {v['MAE']:.1f} days",
            "Better than usual date?": {True: "✅ yes", False: "❌ no", None: "—"}[verdict],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Errors are mean absolute errors on seasons the model had not seen: 2005–2014 was used to "
               "choose the model, 2015–2025 was held back as a final test. Notice = median days between "
               "the issue date and the usual bloom date.")

    st.warning("**Recent springs have come early.** On 2015–2025 even the usual date ran about "
               "3–4 days late, and the 1 Feb and 1 Mar forecasts pushed it later still, so they did "
               "worse than the usual date. Only the 15 March forecast, which sees early-March "
               "warmth, beat it.")

    c1, c2, c3 = st.columns(3)
    c1.markdown("**Plan early → usual date**  \nFor bookings months ahead, start from the station's "
                "usual date, lean a few days earlier, and keep a buffer of about ±5 days.")
    c2.markdown("**1 Feb / 1 Mar → indicative only**  \nUseful to see which way the model leans, "
                "but on recent seasons these forecasts were not more reliable than the usual date.")
    c3.markdown("**Lock in → 15 Mar**  \nAbout three weeks out and the only forecast that beat "
                "the usual date on unseen seasons (± 3.7 days).")

    st.subheader(f"Compare the forecasts for one station · {year}")
    sid = station_picker(pd.concat(season.values()), key="compare_station")

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
                    f"error ± {v['MAE']:.1f} days on 2005–2014"
                    + (f" · ± {t['MAE']:.1f} days on 2015–2025" if t else ""))
        if beats_baseline(m) is False:
            st.warning(f"On the 2015–2025 test this forecast (± {t['MAE']:.1f} days) did worse than "
                       f"simply using each station's usual date (± {m['baseline_test']['MAE']:.1f} days). "
                       "Treat it as indicative only.")

        sid = station_picker(fc, key=f"station-{d}", allow_all=True)
        c1, c2, c3, c4 = st.columns(4)
        if sid is None:
            c1.metric("Stations", len(fc))
            first, last = fc.iloc[0], fc.iloc[-1]
            c2.metric("Earliest bloom", first.predicted_date.strftime("%d %b"), first.location, delta_color="off")
            c3.metric("Latest bloom", last.predicted_date.strftime("%d %b"), last.location, delta_color="off")
            c4.metric("Median notice", f"{fc['notice_days'].median():.0f} days")
        else:
            r = fc[fc["station_id"] == sid].iloc[0]
            c1.metric("Forecast full bloom", r.predicted_date.strftime("%d %b %Y"))
            c2.metric("Notice", f"{int(r.notice_days)} days")
            c3.metric("Earlier / later than usual", f"{r.anomaly_days:+.1f} days")
            if "observed_doy" in r and pd.notna(r.observed_doy):
                obs = pd.Timestamp(year, 1, 1) + pd.Timedelta(days=int(r.observed_doy) - 1)
                c4.metric("Observed", obs.strftime("%d %b"), f"error {r.error_days:+.1f} days",
                          delta_color="off")
            else:
                c4.metric("Typical error", f"± {honest_error(m):.1f} days")

        st.pydeck_chart(sakura_map(fc, sid), use_container_width=True)

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
- **Final test 2015–2025:** the 15 March model beat the 30-year climatology (± 3.7 vs ± 5.0 days);
  the 1 Feb and 1 Mar models did not (± 6.0 and ± 5.9). Recent warm winters lie outside the
  1964–2014 training range, and the early models read them as signs of a later bloom.
- **Batch forecasts:** `src/models/batch_forecast.py` scores every station once per issue date;
  this app only displays the saved results.
"""
    )
