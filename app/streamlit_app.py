"""Sakura Yosoku - Somei-yoshino full-bloom dates for tour operators, in two stages.

    streamlit run app/streamlit_app.py

1. Plan  (any time)      each station's usual date: mean full bloom over the 30 seasons
                         before the one being planned (works for the upcoming season too)
2. Confirm (from 15 Mar) the deployed SVR (app/model/svr_mar15.joblib, exported by
                         src/models/export_svr.py) run in the cloud on weather up to 14 March

Forecasts issued on 1 Feb and 1 Mar were tested and dropped: on 2015-2025 they did not
beat the usual date (src/models/rolling_test.py).
"""
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import sklearn
import streamlit as st

from predictor import BloomPredictor

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
DATE = "mar15"            # the deployed forecast
LOOKBACK, MIN_HISTORY = 30, 10

st.set_page_config(page_title="Sakura Yosoku", page_icon="🌸", layout="wide")

# ------------------------------------------------------------------ map helpers
SAKURA_ICON = {
    "url": "data:image/png;base64," + base64.b64encode((APP_DIR / "sakura.png").read_bytes()).decode(),
    "width": 128, "height": 128, "anchorY": 64,
}
JAPAN_VIEW = pdk.ViewState(latitude=36.8, longitude=137.5, zoom=4.2)


def sakura_map(df, when_col, detail_col, selected=None):
    """Pydeck map with a small sakura flower per station; the selected one is larger.
    Tooltip shows `when_col` and `detail_col` (both text columns of df)."""
    pts = df[["station_id", "location", "prefecture", "latitude", "longitude"]].copy()
    pts["when"], pts["detail"] = df[when_col].astype(str), df[detail_col].astype(str)
    pts["icon"] = [SAKURA_ICON] * len(pts)

    def flowers(rows, px):
        return pdk.Layer(
            "IconLayer", rows, get_icon="icon", get_position=["longitude", "latitude"],
            get_size=px, size_scale=1, size_units="pixels",
            size_min_pixels=px, size_max_pixels=px, pickable=True,
        )

    layers = [
        pdk.Layer("ScatterplotLayer", pts, get_position=["longitude", "latitude"],
                  get_fill_color=[217, 79, 134, 200], get_radius=3, radius_units="pixels",
                  radius_min_pixels=3, radius_max_pixels=3),
        flowers(pts[pts["station_id"] != selected], 22),
    ]
    view = JAPAN_VIEW
    if selected is not None and (pts["station_id"] == selected).any():
        sel = pts[pts["station_id"] == selected]
        layers.append(flowers(sel, 40))
        view = pdk.ViewState(latitude=float(sel.latitude.iloc[0]),
                             longitude=float(sel.longitude.iloc[0]), zoom=6)
    return pdk.Deck(layers=layers, initial_view_state=view,
                    map_provider="carto", map_style=pdk.map_styles.CARTO_LIGHT,
                    tooltip={"html": "<b>{location}</b>, {prefecture}<br/>{when}<br/>{detail}"})


def station_picker(df, key, allow_all=False):
    """Station dropdown sorted by name; returns station_id (or None for 'All stations')."""
    names = df[["station_id", "location", "prefecture"]].drop_duplicates("station_id")
    names = names.sort_values("location").reset_index(drop=True)
    labels = (names["location"] + " (" + names["prefecture"] + ")").tolist()
    ids = names["station_id"].tolist()
    if allow_all:
        labels, ids, default = ["All stations"] + labels, [None] + ids, 0
    else:
        default = next((i for i, l in enumerate(labels) if l.startswith("Tokyo")), 0)
    i = st.selectbox("Station", range(len(labels)), index=default,
                     format_func=lambda k: labels[k], key=key)
    return ids[i]


def doy_to_date(year, doy):
    return pd.Timestamp(int(year), 1, 1) + pd.Timedelta(days=int(np.rint(doy)) - 1)


# ------------------------------------------------------------------ data
@st.cache_resource
def load_predictor():
    """The deployed 15 March model, loaded once per app instance."""
    return BloomPredictor(DATE) if (MODEL_DIR / f"svr_{DATE}.joblib").exists() else None


@st.cache_data(show_spinner="Working out the usual dates…")
def usual_dates(year):
    """Stage 1: each station's usual date for `year` = mean full-bloom DOY over the 30 seasons
    before it (region mean if a station has < 10 of them, else the overall mean).
    Stations: those observed in `year`, or for an upcoming season those observed last season."""
    table = load_predictor().table
    hist = table[table["bloom_year"].between(year - LOOKBACK, year - 1)]
    stations = table[table["bloom_year"] == year]
    if stations.empty:
        stations = table[table["bloom_year"] == table["bloom_year"].max()]
    stats = hist.groupby("station_id")["full_bloom_doy"].agg(["mean", "size"])
    region = hist.groupby("region")["full_bloom_doy"].mean()
    out = stations[["station_id", "location", "prefecture", "region", "latitude", "longitude"]].copy()
    mean = out["station_id"].map(stats["mean"])
    size = out["station_id"].map(stats["size"]).fillna(0)
    fallback = out["region"].map(region).fillna(hist["full_bloom_doy"].mean())
    out["usual_doy"] = np.where(size >= MIN_HISTORY, mean, fallback).round(1)
    out["usual_date"] = [doy_to_date(year, d) for d in out["usual_doy"]]
    observed = table[table["bloom_year"] == year].set_index("station_id")["full_bloom_doy"]
    out["observed_doy"] = out["station_id"].map(observed)
    out["error_days"] = (out["usual_doy"] - out["observed_doy"]).round(1)
    return out.sort_values("usual_doy").reset_index(drop=True)


@st.cache_data(show_spinner="Running the SVR model…")
def forecast(year):
    """Stage 2: cloud inference with the deployed SVR for every station with data this season."""
    bp = load_predictor()
    rows = bp.table[bp.table["bloom_year"] == year]
    if rows.empty:
        return None
    fc = bp.predict_batch(rows, year)
    observed = rows.set_index("station_id")["full_bloom_doy"]
    fc["observed_doy"] = fc["station_id"].map(observed)
    fc["error_days"] = (fc["predicted_doy"] - fc["observed_doy"]).round(1)
    fc["predicted_date"] = pd.to_datetime(fc["predicted_date"])
    return fc


bp = load_predictor()
st.title("🌸 Sakura Yosoku — when will the cherry blossoms peak?")
st.caption("Somei-yoshino full-bloom dates for Japanese weather stations, for tour operators: "
           "**plan with the usual date, then confirm with our 15 March forecast.**")
if bp is None:
    st.info("No model yet. Run `python3 src/models/export_svr.py` and commit `app/model/`.")
    st.stop()

meta = bp.meta
if meta["sklearn_version"] != sklearn.__version__:
    st.warning(f"The model was saved with scikit-learn {meta['sklearn_version']} but this app runs "
               f"{sklearn.__version__}; pin the same version in app/requirements.txt.")

# accuracy on 2015-2025, retrained each season (rolling test); fall back to the frozen test
rolling = meta.get("rolling_test") or {}
plan_acc = rolling.get("clim_rolling") or meta.get("baseline_test")
svr_acc = rolling.get("svr_C_rolling") or meta.get("test") or meta["validation"]
window = int(np.ceil(plan_acc["MAE"])) if plan_acc else 5

last = int(bp.table["bloom_year"].max())
seasons = [last + 1] + sorted(bp.table["bloom_year"].unique().tolist(), reverse=True)
year = st.selectbox("Season", seasons, index=1,
                    format_func=lambda y: f"{y} (upcoming)" if y > last else str(y))
plan = usual_dates(int(year))
fc = forecast(int(year))

how, plan_tab, confirm_tab = st.tabs(["⚖️ How to use", "🗓️ Plan · usual date", "🌸 Confirm · 15 Mar forecast"])

# ------------------------------------------------------------------ how to use
with how:
    st.subheader("Two stages")
    rows = [
        {"Stage": "1 · Plan", "When": "any time, even a year ahead",
         "What you get": "each station's usual full-bloom date (last 30 seasons)",
         "Typical error*": f"± {plan_acc['MAE']:.1f} days" if plan_acc else "—",
         "Within ±3 days*": f"{100 * plan_acc['hit_rate_3d']:.0f}%" if plan_acc else "—"},
        {"Stage": "2 · Confirm", "When": f"from 15 March (~{meta['typical_lead_days']:.0f} days before bloom)",
         "What you get": "SVR forecast from weather up to 14 March",
         "Typical error*": f"± {svr_acc['MAE']:.1f} days",
         "Within ±3 days*": f"{100 * svr_acc['hit_rate_3d']:.0f}%"},
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("*Measured on the 2015–2025 seasons, which were held back from model building, "
               "with both methods updated each season.")

    c1, c2 = st.columns(2)
    c1.markdown(f"**Plan → book with a buffer.** Use the usual date to pick your travel window "
                f"and keep about ±{window} days of flexibility in bookings.")
    c2.markdown("**Confirm → fine-tune on 15 March.** About three weeks out, our forecast narrows "
                "the window: half of its forecasts land within ±3 days.")
    if plan_acc and plan_acc.get("bias", 0) > 1:
        st.warning(f"**Recent springs have come early.** Over 2015–2025, full bloom arrived on average "
                   f"about {plan_acc['bias']:.0f} days *before* the usual date, so lean your plans "
                   "a little earlier.")
    with st.expander("Why no forecast before 15 March?"):
        st.markdown("We also built forecasts issued on **1 February** and **1 March**. On the 2015–2025 "
                    "seasons they were *less* accurate than the usual date: winter weather alone does "
                    "not tell us enough, while early-March warmth does. So until 15 March, the usual "
                    "date is the most reliable guide.")

    st.subheader(f"One station · {year}")
    sid = station_picker(plan, key="how_station")
    p = plan[plan["station_id"] == sid].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Plan · usual date", p.usual_date.strftime("%d %b"))
    if fc is not None and (fc["station_id"] == sid).any():
        r = fc[fc["station_id"] == sid].iloc[0]
        c2.metric(f"Confirm · 15 Mar forecast ({int(r.notice_days)} days' notice)",
                  r.predicted_date.strftime("%d %b"))
    else:
        c2.metric("Confirm · 15 Mar forecast", "from 15 Mar")
    if pd.notna(p.observed_doy):
        c3.metric("Observed full bloom", doy_to_date(year, p.observed_doy).strftime("%d %b"))
        st.caption("For past seasons the 15 March model had seen these years in training, so its "
                   "match looks better than a true forecast would. Use the errors above as the honest "
                   "accuracy.")

# ------------------------------------------------------------------ stage 1: plan
with plan_tab:
    st.markdown(f"**Usual full-bloom date for {year}**: each station's average over the "
                f"{LOOKBACK} seasons before it ({year - LOOKBACK}–{year - 1}). "
                f"Plan with about ±{window} days of flexibility.")
    sid = station_picker(plan, key="plan_station", allow_all=True)
    c1, c2, c3, c4 = st.columns(4)
    if sid is None:
        c1.metric("Stations", len(plan))
        c2.metric(f"Earliest · {plan.iloc[0].location}", plan.iloc[0].usual_date.strftime("%d %b"))
        c3.metric(f"Latest · {plan.iloc[-1].location}", plan.iloc[-1].usual_date.strftime("%d %b"))
        c4.metric("Plan with a buffer of", f"± {window} days")
    else:
        p = plan[plan["station_id"] == sid].iloc[0]
        c1.metric("Usual date", p.usual_date.strftime("%d %b %Y"))
        c2.metric("Likely window",
                  f"{(p.usual_date - pd.Timedelta(days=window)).strftime('%d %b')} – "
                  f"{(p.usual_date + pd.Timedelta(days=window)).strftime('%d %b')}")
        if pd.notna(p.observed_doy):
            c3.metric(f"Observed · usual date was off by {p.error_days:+.1f} days",
                      doy_to_date(year, p.observed_doy).strftime("%d %b"))

    shown = plan.assign(when=plan["usual_date"].dt.strftime("usual date %d %b"),
                        detail=f"plan ± {window} days")
    st.pydeck_chart(sakura_map(shown, "when", "detail", sid), use_container_width=True)

    cols = ["location", "prefecture", "region", "usual_date"]
    cols += [c for c in ("observed_doy", "error_days") if plan[c].notna().any()]
    table = plan[cols].copy()
    table["window"] = [f"{(d - pd.Timedelta(days=window)).strftime('%d %b')} – "
                       f"{(d + pd.Timedelta(days=window)).strftime('%d %b')}" for d in plan["usual_date"]]
    table["usual_date"] = table["usual_date"].dt.strftime("%d %b")
    st.dataframe(table.rename(columns={"usual_date": "usual date", "window": "likely window",
                                       "observed_doy": "observed DOY",
                                       "error_days": "usual date off by (days)"}),
                 hide_index=True, use_container_width=True)
    st.download_button("Download CSV", plan.to_csv(index=False).encode(),
                       f"usual_dates_{year}.csv", "text/csv", key="dl-plan")

# ------------------------------------------------------------------ stage 2: confirm
with confirm_tab:
    v, t = meta["validation"], meta.get("test")
    st.markdown(f"**15 March forecast for {year}** from weather up to 14 March · typical error "
                f"± {svr_acc['MAE']:.1f} days on 2015–2025 (usual date: ± {plan_acc['MAE']:.1f})"
                if plan_acc else f"**15 March forecast for {year}**")
    if fc is None:
        st.info(f"The {year} forecast is issued on 15 March {year}, once weather up to 14 March is "
                "available. Until then, plan with the usual date.")
    else:
        sid = station_picker(fc, key="confirm_station", allow_all=True)
        c1, c2, c3, c4 = st.columns(4)
        if sid is None:
            c1.metric("Stations", len(fc))
            c2.metric(f"Earliest · {fc.iloc[0].location}", fc.iloc[0].predicted_date.strftime("%d %b"))
            c3.metric(f"Latest · {fc.iloc[-1].location}", fc.iloc[-1].predicted_date.strftime("%d %b"))
            c4.metric("Median notice", f"{fc['notice_days'].median():.0f} days")
        else:
            r = fc[fc["station_id"] == sid].iloc[0]
            c1.metric("Forecast full bloom", r.predicted_date.strftime("%d %b %Y"))
            c2.metric("Notice", f"{int(r.notice_days)} days")
            c3.metric("Earlier / later than usual", f"{r.anomaly_days:+.1f} days")
            if pd.notna(r.observed_doy):
                c4.metric(f"Observed · forecast error {r.error_days:+.1f} days",
                          doy_to_date(year, r.observed_doy).strftime("%d %b"))
            else:
                c4.metric("Typical error", f"± {svr_acc['MAE']:.1f} days")

        shown = fc.assign(when=fc["predicted_date"].dt.strftime("full bloom %d %b"),
                          detail=fc["notice_days"].astype(int).astype(str) + " days' notice")
        st.pydeck_chart(sakura_map(shown, "when", "detail", sid), use_container_width=True)

        cols = ["location", "prefecture", "region", "predicted_date", "notice_days",
                "anomaly_days", "reference_doy"]
        cols += [c for c in ("observed_doy", "error_days") if fc[c].notna().any()]
        table = fc[cols].copy()
        table["predicted_date"] = table["predicted_date"].dt.strftime("%d %b")
        st.dataframe(table.rename(columns={"predicted_date": "forecast", "notice_days": "days' notice",
                                           "anomaly_days": "vs usual (days)", "reference_doy": "usual DOY",
                                           "observed_doy": "observed DOY", "error_days": "error (days)"}),
                     hide_index=True, use_container_width=True)
        st.download_button("Download CSV", fc.to_csv(index=False).encode(),
                           f"forecast_{year}_mar15.csv", "text/csv", key="dl-confirm")
        p = meta["params"]
        st.caption(f"Computed in the cloud by the deployed SVR ({p['kernel']}, C={p['C']}, "
                   f"epsilon={p['epsilon']}, gamma={p['gamma']}), trained on "
                   f"{meta['trained_years'][0]}–{meta['trained_years'][1]} · scikit-learn {sklearn.__version__}")

with st.expander("About the model"):
    rt = f"; retrained each season: ± {svr_acc['MAE']:.2f}" if rolling.get("svr_C_rolling") else ""
    st.markdown(
        f"""
- **Stage 1 · usual date:** each station's mean full-bloom date over the 30 seasons before the one
  being planned (region mean if a station has fewer than 10 of them).
- **Stage 2 · 15 March forecast:** StandardScaler + RBF Support Vector Regression on
  {len(bp.features)} location, temperature, chill and heat-sum features up to 14 March. It predicts
  how many days earlier or later than the usual date a station will bloom.
- **Accuracy (mean absolute error, days):** validation 2005–2014 ± {meta['validation']['MAE']:.2f};
  final test 2015–2025 ± {meta['test']['MAE']:.2f} with the model frozen at 2014{rt}.
- **Why not earlier?** Forecasts issued on 1 February and 1 March were tested and did not beat the
  usual date on 2015–2025; early-March warmth is what makes the forecast work.
- **Deployment:** the fitted model (joblib) is deployed with this app on Streamlit Community Cloud
  and runs server-side; results are cached per season.
""" if meta.get("test") else ""
    )
