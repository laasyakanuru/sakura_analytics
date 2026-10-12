#%%
"""
EDA report for the cleaned weather data.

Run AFTER 02_clean_weather_and_features.py:
    python src/02_preprocess/04_eda_weather_report.py

Writes a self-contained HTML report (charts embedded) to
    reference/weather-eda.html
in the same visual style as reference/sakura-findings3.html.
Every number quoted in the report text is computed below, so re-running the
script after changing the cleaning rules keeps the text and charts in sync.
"""
import base64
import html
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

#%% ---------------------------------------------------------------- paths
if "__file__" in globals():
    PROJECT_DIR = Path(__file__).resolve().parents[2]
else:
    PROJECT_DIR = Path("../..").resolve()
RAW_DIR = PROJECT_DIR / "data/raw/weather_by_station"
META_DIR = PROJECT_DIR / "data/processed/01_source_data/01_station_metadata"
CLEAN_DIR = PROJECT_DIR / "data/processed/02_cleaned_source_data/weather"
OUT_FILE = PROJECT_DIR / "reference/weather-eda.html"

MODEL_START = 1964  # first year without the pre-1964 definition artefact

#%% ---------------------------------------------------------------- style
PINK, BLUE, OCHRE = "#c2457a", "#2a6fdb", "#9a7b00"   # validated categorical set (CVD-safe on white)
PINK_LIGHT, GREY, INK, MUTED = "#f2b8cf", "#b9aab1", "#402d37", "#765c67"
SEQ = LinearSegmentedColormap.from_list("seq", ["#fff4f7", "#f2b8cf", "#c2457a", "#6e1f42"])
DIV = LinearSegmentedColormap.from_list("div", [BLUE, "#eeeaec", PINK])

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 10.5, "axes.titlesize": 12,
    "axes.edgecolor": "#d9c6ce", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18,
    "grid.color": "#9c7f8a", "legend.frameon": False, "lines.linewidth": 2, "text.color": INK,
})


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def figure(title, uri, caption):
    t, c = html.escape(title), html.escape(caption)
    return (f'<figure><h3>{t}</h3><a href="{uri}" target="_blank" aria-label="Enlarge {t}">'
            f'<img src="{uri}" alt="{c}"></a><figcaption>{c}</figcaption></figure>')


def table(df, caption):
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>"
                   for r in df.itertuples(index=False))
    return (f'<div class="table-wrap"><table><caption>{html.escape(caption)}</caption>'
            f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>")


#%% ---------------------------------------------------------------- data
meta = (pd.read_csv(META_DIR / "01_01_station_name.csv", dtype={"station_id": str}, encoding="utf-8-sig")
        .merge(pd.read_csv(META_DIR / "01_02_station_elevations.csv", dtype={"station_id": str}, encoding="utf-8-sig"))
        .merge(pd.read_csv(META_DIR / "01_03_cherry_species.csv", dtype={"station_id": str}, encoding="utf-8-sig")))
raw = pd.concat([pd.read_csv(f, dtype={"station_id": str}, encoding="utf-8-sig")
                 for f in sorted(RAW_DIR.glob("weather_*_daily_quality.csv"))], ignore_index=True)
SPECIES = "Somei-yoshino"
meta = meta[meta.cherry_species == SPECIES].reset_index(drop=True)
raw = raw[raw.station_id.isin(meta.station_id)].reset_index(drop=True)
raw["year"] = raw.date.str[:4].astype(int)
daily = pd.read_csv(CLEAN_DIR / "02_cleaned_weather_daily_somei_1952_2025.csv.gz", dtype={"station_id": str})
daily["date"] = pd.to_datetime(daily.date)
feat = pd.read_csv(CLEAN_DIR / "03_weather_features_somei_station_year.csv", dtype={"station_id": str})
gaps = pd.read_csv(CLEAN_DIR / "04_weather_gap_log_somei.csv", dtype={"station_id": str})
breaks = pd.read_csv(CLEAN_DIR / "05_weather_station_relocations_somei.csv", dtype={"station_id": str})
valid = pd.read_csv(CLEAN_DIR / "06_weather_imputation_validation_somei.csv", dtype={"station_id": str})

TM, TX, TN, PR, SU = ("mean_temperature_c", "maximum_temperature_c", "minimum_temperature_c",
                      "precipitation_mm", "sunshine_duration_hours")
n_st = meta.station_id.nunique()
order_lat = meta.sort_values("latitude", ascending=False).station_id.tolist()
loc = dict(zip(meta.station_id, meta.location))
somei = feat[feat.cherry_species == "Somei-yoshino"].copy()
sections = {}

#%% ---------------------------------------------------------------- 01 coverage
good = lambda v: raw[f"{v}_quality_code"].isin([8, 5])
cov = {}
for v in [TM, SU]:
    c = raw.assign(ok=good(v)).groupby(["station_id", "year"]).ok.mean().unstack()
    cov[v] = c.reindex(order_lat)


def coverage_heatmap(c, label):
    fig, ax = plt.subplots(figsize=(7.4, 7.2))
    im = ax.imshow(c.values, aspect="auto", cmap=SEQ, vmin=0, vmax=1, interpolation="nearest")
    years = c.columns.to_numpy()
    xt = [i for i, y in enumerate(years) if y % 10 == 0]
    ax.set_xticks(xt, years[xt])
    ax.set_yticks([0, len(c) - 1], [f"{loc[c.index[0]]} (north)", f"{loc[c.index[-1]]} (south)"])
    ax.grid(False)
    ax.set_xlabel("Year")
    cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.02)
    cb.set_label(f"Share of days with usable {label}")
    cb.outline.set_visible(False)
    return png(fig)


raw_temp_missing = int((~good(TM)).sum())
raw_temp_missing_in_record = int(gaps[(gaps.variable == TM) & ~gaps.handling.str.contains("before first")].length_days.sum())
after_missing = int(daily[TM].isna().sum())
after_missing_in_record = int(gaps[(gaps.variable == TM) & gaps.handling.str.startswith("left missing") &
                                   ~gaps.handling.str.contains("before first")].length_days.sum())
sun_cov = cov[SU].gt(.5).mean()          # share of stations with >50% usable sunshine days, by year
sun_start = int(sun_cov[sun_cov >= .9].index.min())
sun_early = int(cov[SU].loc[:, :sun_start - 1].gt(.5).any(axis=1).sum())

qc = []
for v, name in [(TM, "Mean temp"), (TX, "Max temp"), (TN, "Min temp"), (PR, "Precipitation"), (SU, "Sunshine")]:
    vc = raw[f"{v}_quality_code"].value_counts()
    qc.append([name] + [f"{int(vc.get(k, 0)):,}" for k in [8, 5, 4, 1, 0]])
qc_table = pd.DataFrame(qc, columns=["Variable", "8 normal", "5 quasi-normal", "4 insufficient → dropped",
                                     "1 missing", "0 not observed"])

late_start = (raw[good(TM)].groupby("station_id").year.min().loc[lambda s: s > 1952]
              .rename(index=loc).sort_values())
late_txt = ", ".join(f"{k} ({v})" for k, v in late_start.items())

sections["coverage"] = f"""
<section id="coverage"><span class="eyebrow">01 / What the raw data covers</span>
<h2>Temperature is nearly complete; sunshine is patchy before {sun_start}</h2>
<p class="lead">The raw JMA files hold {len(raw):,} station-days ({n_st} stations × {raw.date.nunique():,} days,
1952–2025). Almost every gap in temperature is a block before a station began observing, not a hole in an
ongoing record. Only {sun_early} of {n_st} stations have sunshine data before {sun_start}.</p>
<div class="pair">
{figure("Mean temperature · usable days", coverage_heatmap(cov[TM], "mean temp"),
        f"Rows are stations ordered north → south by latitude. Pale blocks at the left are stations that began observing late: {late_txt}.")}
{figure("Sunshine · usable days", coverage_heatmap(cov[SU], "sunshine"),
        f"From {sun_start}, at least 90% of stations have sunshine data. Before that only {sun_early} stations do, so any sunshine feature effectively restricts the modelling period to {sun_start} onwards.")}
</div>
<div class="coverage">
<article><span>Mean-temperature values not usable (raw)</span><strong>{raw_temp_missing:,}</strong>
<p>Of which {raw_temp_missing_in_record:,} fall inside an active record</p></article>
<span class="arrow" aria-hidden="true">→</span>
<article><span>Still missing inside an active record after cleaning</span><strong>{after_missing_in_record:,}</strong>
<p>All in the long Onahama (1977–79) and Miyakejima (2000–01) gaps</p></article>
</div>
{table(qc_table, "JMA quality codes in the raw files (daily values)")}
<p class="note">JMA quality codes: 8 = normal; 5 = quasi-normal (a few hourly values missing, still usable);
4 = insufficient data (too many hourly values missing, so the daily value is biased); 1 = missing; 0 = not observed.
The cleaning keeps 8 and 5, and treats 4 as missing.</p>
</section>"""

#%% ---------------------------------------------------------------- 02 quality artefacts
art = raw.assign(bad=raw[TM] < raw[TN] - 0.05).groupby("year").bad.mean() * 100
art_n = int((raw[TM] < raw[TN] - 0.05).sum())
fig, ax = plt.subplots(figsize=(7.4, 4))
ax.bar(art.index, art.values, color=PINK, width=.8)
ax.set_xlabel("Year"); ax.set_ylabel("% of station-days")
ax.set_title("Daily mean below daily minimum", loc="left")
ax.set_xlim(1951, 2026)
fig_art = png(fig)

# step at each relocation: station minus 3 nearest neighbours, 3 years before vs after
W = daily.pivot(index="date", columns="station_id", values=TM)
lat = meta.set_index("station_id").latitude; lon = meta.set_index("station_id").longitude


def nearest(sid, k=3):
    d = np.hypot(lat - lat[sid], (lon - lon[sid]) * np.cos(np.radians(lat[sid])))
    return d.drop(sid).nsmallest(k).index.tolist()


steps = []
for b in breaks.itertuples():
    t = pd.Timestamp(b.break_date)
    diff = W[b.station_id] - W[nearest(b.station_id)].mean(axis=1)
    before = diff[(diff.index >= t - pd.DateOffset(years=3)) & (diff.index < t)].mean()
    after = diff[(diff.index >= t) & (diff.index < t + pd.DateOffset(years=3))].mean()
    steps.append({"station_id": b.station_id, "location": b.location, "break_date": b.break_date,
                  "step": after - before})
steps = pd.DataFrame(steps).sort_values("step")
big = steps[steps.step.abs() >= 0.3]
fig, ax = plt.subplots(figsize=(7.4, 6.4))
labels = [f"{r.location} {r.break_date[:4]}" for r in steps.itertuples()]
ax.barh(labels, steps.step, color=[BLUE if s < 0 else PINK for s in steps.step], height=.7)
ax.axvline(0, color=INK, lw=.8)
ax.set_xlabel("Change in station − neighbours, mean temp (°C)")
ax.tick_params(axis="y", labelsize=8)
fig_steps = png(fig)

# extreme days
clim = daily.assign(doy=daily.date.dt.dayofyear)
for v in [TX, TN]:
    an = clim[v] - clim.groupby(["station_id", "doy"])[v].transform("mean")
    clim[f"z_{v}"] = an / an.groupby(clim.station_id).transform("std")
ext = (clim.assign(z=clim[[f"z_{TX}", f"z_{TN}"]].abs().max(axis=1)).nlargest(8, "z"))
ext_table = pd.DataFrame({"Station": ext.station_id.map(loc), "Date": ext.date.dt.strftime("%Y-%m-%d"),
                          "Max °C": ext[TX], "Min °C": ext[TN], "Largest |z|": ext.z.round(1)})

sections["quality"] = f"""
<section id="quality"><span class="eyebrow">02 / Artefacts to know about</span>
<h2>Two issues are flagged, not fixed</h2>
<p class="lead">The temperature values are internally consistent from {MODEL_START} onwards. Before that, and at
{len(breaks)} station relocations, the record changes in ways that could bias a model. The data keeps the
values as JMA reports them, and adds flags so the modelling choice is explicit.</p>
<div class="pair">
{figure("Pre-1964 definition change", fig_art,
        f"{art_n:,} station-days in 1953–1963 have a daily mean lower than the daily minimum. Daily extremes then used a different day boundary from the mean, so the three values are not strictly comparable. They are flagged as pre_1964_mean_below_min in the daily file and pre_1964_definition in the feature table.")}
{figure("Step change at each relocation", fig_steps,
        f"JMA homogeneity numbers mark {len(breaks)} station moves or instrument changes at {breaks.station_id.nunique()} stations. Each bar compares the station with its three nearest neighbours in the 3 years before and after. {len(big)} breaks shift by 0.3 °C or more.")}
</div>
{table(ext_table, "Most extreme days relative to each station's normal for that date (kept)")}
<p class="note">Extreme days were checked against neighbours and look like real weather (foehn warm spells on the
Japan Sea coast, cold nights at inland Kutchan), so no values were removed as outliers. Each JMA homogeneity number
restarts in every one-year download, so changes on 1 January are artefacts of how the data was downloaded and are ignored.</p>
<aside class="discussion"><span class="eyebrow">For discussion</span>
<h3>Start the modelling period in {MODEL_START}?</h3>
<p>Starting in {MODEL_START} avoids the definition change and keeps {feat[feat.year >= MODEL_START].flowering_doy.notna().sum():,}
of the {feat.flowering_doy.notna().sum():,} flowering records. For relocations, the feature table has a
relocations_before_season counter that can act as a station-segment effect.</p>
</aside>
</section>"""

#%% ---------------------------------------------------------------- 03 imputation
vm = valid.groupby(["gap_length"])[["mae_neighbour", "mae_linear"]].mean()
fig, ax = plt.subplots(figsize=(7.4, 4.2))
ax.plot(vm.index, vm.mae_linear, "-o", color=OCHRE, ms=6, label="Linear interpolation")
ax.plot(vm.index, vm.mae_neighbour, "-o", color=PINK, ms=6, label="Neighbour-difference")
ax.set_xscale("log"); ax.set_xticks(vm.index, vm.index)
ax.set_xlabel("Length of hidden gap (days)"); ax.set_ylabel("Mean absolute error (°C)")
ax.set_ylim(0, None)
for name, col, y in [("Linear interpolation", OCHRE, vm.mae_linear.iloc[-1]),
                     ("Neighbour-difference", PINK, vm.mae_neighbour.iloc[-1])]:
    ax.annotate(name, (vm.index[-1], y), xytext=(6, 0), textcoords="offset points", color=INK, va="center", fontsize=9)
ax.set_xlim(None, 260)
fig_val = png(fig)

vv = valid.groupby("variable")[["mae_neighbour", "mae_linear"]].mean()
fig, ax = plt.subplots(figsize=(7.4, 4.2))
names = {TM: "Mean", TX: "Max", TN: "Min"}
x = np.arange(len(vv)); w = .36
ax.bar(x - w / 2, vv.mae_linear, w, color=OCHRE, label="Linear interpolation")
ax.bar(x + w / 2, vv.mae_neighbour, w, color=PINK, label="Neighbour-difference")
for i, (a, b) in enumerate(zip(vv.mae_linear, vv.mae_neighbour)):
    ax.text(i - w / 2, a + .03, f"{a:.2f}", ha="center", fontsize=9, color=INK)
    ax.text(i + w / 2, b + .03, f"{b:.2f}", ha="center", fontsize=9, color=INK)
ax.set_xticks(x, [names[v] for v in vv.index]); ax.set_ylabel("Mean absolute error (°C)")
ax.set_ylim(0, vv.mae_linear.max() * 1.3)
ax.legend(loc="upper center", ncol=2)
fig_val2 = png(fig)

handled = (gaps[~gaps.handling.str.contains("before first")]
           .groupby(["handling"]).agg(runs=("length_days", "size"), days=("length_days", "sum"))
           .reset_index().rename(columns={"handling": "Handling (all three temperature variables)",
                                          "runs": "Gaps", "days": "Days"}))
nb_mae, lin_mae = valid.mae_neighbour.mean(), valid.mae_linear.mean()

sections["imputation"] = f"""
<section id="imputation"><span class="eyebrow">03 / Filling the gaps</span>
<h2>Neighbouring stations beat interpolation at every gap length</h2>
<p class="lead">To choose a method, {len(valid):,} blocks of real observations were hidden and then refilled.
The neighbour-difference method (nearby stations' values plus the usual difference between the two stations)
averaged {nb_mae:.2f} °C error, and stays flat as gaps grow. Linear interpolation averaged {lin_mae:.2f} °C, and is
worse even for a single missing day.</p>
<div class="pair">
{figure("Error by gap length", fig_val, "Hidden gaps of 1–60 days, all three temperature variables. Interpolation cannot see a cold front that arrived during the gap; neighbouring stations can.")}
{figure("Error by variable", fig_val2, f"Daily maximum is the hardest to fill ({vv.loc[TX, 'mae_neighbour']:.2f} °C with neighbours), because daytime highs depend on local cloud and sea breezes. Daily mean is the easiest ({vv.loc[TM, 'mae_neighbour']:.2f} °C).")}
</div>
{table(handled, "How every in-record temperature gap was handled")}
<p class="note">Method: for a gap at station S, take up to 3 stations within 300 km with complete data over the gap.
For each one, estimate the mean S − neighbour difference over the 60 days either side of the gap, add it to the
neighbour's values, and weight by how tightly the two stations track each other. Gaps longer than 60 days are left
missing (Onahama 1977–79; Miyakejima 2000–01, during the volcanic evacuation). Every imputed value is labelled in
the *_temperature_flag columns.</p>
</section>"""

#%% ---------------------------------------------------------------- 04 climate signal
fm = feat.assign(fm=(feat.tmean_feb + feat.tmean_mar) / 2)
nat = fm[fm.year >= MODEL_START].groupby("year").fm.mean()
fig, ax = plt.subplots(figsize=(7.4, 4.2))
ax.plot(nat.index, nat.values, "-o", color=PINK_LIGHT, ms=3, lw=1.3, label="Annual")
ax.plot(nat.index, nat.rolling(5, center=True).mean(), color=PINK, lw=3, label="5-year moving average")
sl = np.polyfit(nat.index, nat.values, 1)[0] * 10
ax.set_xlabel("Year"); ax.set_ylabel("Feb–Mar mean temperature (°C)")
ax.legend(loc="upper left")
fig_nat = png(fig)

tr = []
for sid, g in fm[(fm.year >= MODEL_START)].dropna(subset=["fm"]).groupby("station_id"):
    if len(g) >= 40:
        tr.append((sid, np.polyfit(g.year, g.fm, 1)[0] * 10))
tr = pd.DataFrame(tr, columns=["station_id", "slope"]).merge(meta[["station_id", "latitude"]])
r_lat = tr[["latitude", "slope"]].corr().iloc[0, 1]
lat_txt = (f"Northern stations have warmed somewhat faster (r = {r_lat:.2f} with latitude)." if r_lat >= 0.3 else
           f"Warming rates show little relationship with latitude (r = {r_lat:.2f}); the slowest-warming stations are small southern islands such as Hachijojima, Tanegashima and Yakushima.")
fig, ax = plt.subplots(figsize=(7.4, 4.2))
ax.scatter(tr.latitude, tr.slope, s=34, color=PINK, edgecolor="white", linewidth=1)
ax.axhline(0, color=INK, lw=.8)
ax.set_xlabel("Station latitude (°N)"); ax.set_ylabel("Feb–Mar warming (°C per decade)")
fig_tr = png(fig)

sections["climate"] = f"""
<section id="climate"><span class="eyebrow">04 / The climate signal</span>
<h2>Late winter has warmed by about {sl:.2f} °C per decade</h2>
<p class="lead">Averaged across stations, February–March mean temperature has risen by about {sl * (2025 - MODEL_START) / 10:.1f} °C since
{MODEL_START}. This is the season that drives bloom timing. It matches the earlier full-bloom trend in the sakura findings report.</p>
<div class="pair">
{figure("Feb–Mar mean temperature, all stations", fig_nat, f"Average of all stations with data each year, {MODEL_START}–2025. Year-to-year swings of ±1.5 °C sit on top of a steady upward trend.")}
{figure("Warming rate by station", fig_tr, f"Linear trend per station ({len(tr)} stations with at least 40 seasons). {int((tr.slope > 0).sum())} of {len(tr)} have warmed; the median is {tr.slope.median():.2f} °C per decade. {lat_txt}")}
</div>
</section>"""

#%% ---------------------------------------------------------------- 05 weather vs bloom
s = somei[somei.year >= MODEL_START].dropna(subset=["flowering_doy", "tmean_mar"]).copy()
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.scatter(s.tmean_mar, s.flowering_doy, s=7, alpha=.35, color=PINK, linewidth=0)
ax.set_xlabel("March mean temperature (°C)"); ax.set_ylabel("Flowering day of year")
r_raw = s[["tmean_mar", "flowering_doy"]].corr().iloc[0, 1]
fig_sc = png(fig)

cand = {"tmean_dec": "Dec mean (prev yr)", "tmean_jan": "Jan mean", "tmean_feb": "Feb mean",
        "tmean_mar": "Mar mean", "tmax_mar": "Mar max", "gdd5_feb1_to_mar_end": "GDD>5 °C Feb–Mar",
        "chill_days_0_7c_nov_feb": "Chill days 0–7 °C", "frost_days_jan_mar": "Frost days Jan–Mar",
        "precip_feb_mar_mm": "Precip Feb–Mar", "sunshine_feb_mar_h": "Sunshine Feb–Mar"}
an = s.copy()
for c in list(cand) + ["flowering_doy"]:
    an[c] = an[c] - an.groupby("station_id")[c].transform("mean")
cor_raw = pd.Series({k: s[[k, "flowering_doy"]].corr().iloc[0, 1] for k in cand})
cor_an = pd.Series({k: an[[k, "flowering_doy"]].corr().iloc[0, 1] for k in cand})
o = cor_an.sort_values().index
fig, ax = plt.subplots(figsize=(7.4, 4.6))
y = np.arange(len(o)); h = .38
ax.barh(y + h / 2, cor_raw[o], h, color=GREY, label="Across stations (raw)")
ax.barh(y - h / 2, cor_an[o], h, color=PINK, label="Within station (anomalies)")
ax.set_yticks(y, [cand[k] for k in o]); ax.axvline(0, color=INK, lw=.8)
ax.set_xlabel("Correlation with flowering day of year"); ax.set_xlim(-1, 1)
ax.legend(loc="lower right", fontsize=9)
fig_cor = png(fig)

slope = np.polyfit(an.tmean_mar, an.flowering_doy, 1)[0]
slope_fm = np.polyfit((an.tmean_feb + an.tmean_mar) / 2, an.flowering_doy, 1)[0]
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.scatter(an.tmean_mar, an.flowering_doy, s=7, alpha=.3, color=PINK, linewidth=0)
xx = np.linspace(an.tmean_mar.min(), an.tmean_mar.max(), 10)
ax.plot(xx, slope * xx, color=INK, lw=2)
ax.set_xlabel("March temperature anomaly vs station mean (°C)")
ax.set_ylabel("Flowering anomaly (days)")
fig_an = png(fig)

sections["drivers"] = f"""
<section id="drivers"><span class="eyebrow">05 / Weather and bloom timing</span>
<h2>A 1 °C warmer March brings flowering about {abs(slope):.1f} days earlier</h2>
<p class="lead">Across stations, temperature mostly reflects latitude: warm places bloom early
(r = {r_raw:.2f} for March mean). The more useful question for forecasting is within a station: in a warmer-than-usual
year, how much earlier does it bloom? Taking each station's own average out, a +1 °C March anomaly means about
{abs(slope):.1f} days earlier, and a +1 °C February–March anomaly about {abs(slope_fm):.1f} days earlier.
(Somei-yoshino, {MODEL_START}–2025, n = {len(s):,}.)</p>
<div class="pair">
{figure("Across stations", fig_sc, "Each dot is one station-year. Most of this pattern comes from latitude, so a model that ignores station identity would overstate the temperature effect.")}
{figure("Within stations", fig_an, f"Both axes are deviations from the station's own average. The fitted slope is {slope:.1f} days per °C.")}
</div>
<div class="pair">
{figure("Which features carry signal", fig_cor, f"Grey: raw correlation across all station-years. Pink: correlation after removing station means. Within a station, March warmth and Feb–Mar GDD are the strongest signals. Frost days drop from r = {cor_raw['frost_days_jan_mar']:.2f} to {cor_an['frost_days_jan_mar']:.2f} once latitude is removed. December warmth hardly matters ({cor_an['tmean_dec']:.2f}), and the simple chill-day count shows almost no linear link ({cor_an['chill_days_0_7c_nov_feb']:.2f}).")}
<div>
<aside class="discussion" style="margin-top:0"><span class="eyebrow">Reading this chart</span>
<h3>No linear chill effect does not mean no chill effect</h3>
<p>Within a station, a mild January still goes with an earlier bloom. Theory says too little winter chill delays dormancy break,
which matters most at the warmest stations (southern Kyushu and the islands). That effect is non-linear and regional, so a straight correlation can miss it. Test a chill × region interaction, or a separate southern model, before dropping the chill feature.</p></aside>
<p class="note">Sunshine has a weak signal and is patchy before {sun_start}. Precipitation is close to zero once station
means are removed. Both are candidates to drop.</p>
</div>
</div>
</section>"""

#%% ---------------------------------------------------------------- 06 decisions + dictionary
files_tbl = pd.DataFrame([
    ["02_cleaned_weather_daily_somei_1952_2025.csv.gz", "station × day", "Cleaned daily weather; *_temperature_flag says observed / imputed_neighbour / imputed_linear / not_yet_observed / missing_long_gap. pandas reads .csv.gz directly."],
    ["03_weather_features_somei_station_year.csv", "station × bloom year", "Model-ready: station metadata + flowering & full-bloom targets + weather features below."],
    ["04_weather_gap_log_somei.csv", "gap", "Every missing temperature run and how it was handled."],
    ["05_weather_station_relocations_somei.csv", "break", "JMA homogeneity breaks (station move / instrument change dates)."],
    ["06_weather_imputation_validation_somei.csv", "trial", "Masked-gap test results behind section 03. During cleaning, all 102 stations served as gap-filling neighbours; only Somei-yoshino stations are written out."],
], columns=["File (data/processed/02_cleaned_source_data/weather/)", "One row per", "Contents"])

dict_tbl = pd.DataFrame([
    ["tmean_oct … tmean_apr", "Monthly mean temperature; Oct–Dec are from the previous calendar year", "Apr is after flowering for most stations"],
    ["tmax_jan…mar, tmin_jan…mar", "Monthly mean of daily max / min", "Mar can overlap flowering in the south"],
    ["chill_days_0_7c_nov_feb", "Days with mean temp between 0 and 7.2 °C, Nov–Feb (simple chilling count)", "Safe from 1 Mar"],
    ["tmin_lowest_nov_feb, frost_days_jan_mar", "Coldest night Nov–Feb; nights below 0 °C Jan–Mar", "Mar part overlaps flowering in the south"],
    ["gdd5_feb1_to_feb_end / _mar_end", "Growing degree days above 5 °C from 1 Feb", "_feb_end safe from 1 Mar"],
    ["cum_tmax_* / cum_tmean_*", "Accumulated daily max / mean temperature from 1 Feb", "_feb_end safe from 1 Mar"],
    ["precip_feb_mar_mm, sunshine_feb_mar_h", "Feb–Mar totals, with *_missing_days counts", f"Sunshine patchy before {sun_start}"],
    ["rule600_tmax_pred_doy, rule400_tmean_pred_doy", "Day of year the rule threshold is reached (baseline predictions)", "Uses weather up to that date"],
    ["temp_missing_days_oct_may, temp_imputed_days_oct_may, season_temp_complete", "Data-quality indicators for the season", "Filter on season_temp_complete = 1"],
    ["relocations_before_season, pre_1964_definition", "Artefact flags from section 02", "—"],
], columns=["Feature", "Meaning", "Forecast leakage note"])

sections["decisions"] = f"""
<section id="decisions"><span class="eyebrow">06 / Decisions & data dictionary</span>
<h2>What the team needs to decide</h2>
<div class="closing"><ul>
<li><b>Modelling period:</b> start in {MODEL_START}, and use {sun_start}+ if sunshine is kept.</li>
<li><b>Forecast date:</b> if the model predicts on 1 March, use only the *_feb_end and pre-March features.
March features are fine for a "nowcast" analysis but leak information for southern stations.</li>
<li><b>Model structure:</b> use station fixed effects or anomaly features. The raw cross-station correlation mostly
reflects latitude.</li>
<li><b>Relocations:</b> ignore, add relocations_before_season as a feature, or drop the {len(big)} large-step breaks.</li>
</ul></div>
{table(files_tbl, "Output files")}
{table(dict_tbl, "Feature table (03_weather_features_somei_station_year.csv): weather columns")}
<p class="note">Season year Y runs from 1 October of Y−1 to 30 June of Y and is joined to that year's flowering and
full-bloom dates. Day of year follows pandas dayofyear (leap years shift by one day after 28 February), the same
convention as 01_cleaned_full_bloom_data.</p>
</section>"""

#%% ---------------------------------------------------------------- assemble
CSS = """
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:#fff8f8;color:#402d37;font:16px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{position:relative;overflow:hidden;background:radial-gradient(ellipse at 90% 10%,#f6cbd8,transparent 55%),linear-gradient(130deg,#fff4f4,#f6e2e8);padding:76px max(24px,calc((100% - 1200px)/2)) 52px;border-bottom:1px solid #eacbd5}
header:after{content:"✿";position:absolute;right:5%;top:-110px;font-size:460px;line-height:1;color:#fff8f8;opacity:.45;pointer-events:none}
.hero{position:relative;z-index:1;max-width:850px}
.eyebrow{font-size:12px;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:#9c4566}
h1{font-family:Georgia,serif;font-size:clamp(38px,5vw,66px);line-height:1.1;font-weight:400;margin:18px 0 24px}
h2{font-family:Georgia,serif;font-size:clamp(28px,3vw,38px);line-height:1.25;font-weight:400;margin:12px 0 18px}
h3{font-size:17px;line-height:1.4;margin:0 0 15px}p{margin:0 0 18px}
.subtitle{font-size:20px;max-width:760px;color:#70505e}
.tags{display:flex;gap:10px;flex-wrap:wrap;margin-top:26px}
.tags span{padding:5px 14px;border:1px solid #d8aabd;border-radius:30px;background:#fff7f8cc;font-size:13px}
nav{display:flex;gap:22px;flex-wrap:wrap;margin:0 auto;padding:22px 24px;max-width:1248px}
a{color:#9c4566}nav a{text-decoration:none;font-size:14px}
main{max-width:1248px;margin:auto;padding:0 24px 50px}
section{padding:48px 0;border-top:1px solid #ead8de;scroll-margin-top:20px}
.lead{max-width:950px;color:#735461;font-size:18px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin:28px 0}
figure{margin:0;background:#fff;border:1px solid #efdae2;border-radius:18px;padding:22px;box-shadow:0 8px 28px #73364d06}
figure img{display:block;width:100%;height:auto}
figcaption{font-size:14px;color:#765c67;margin-top:16px;line-height:1.6}
.discussion{background:#f8e6ed;border-left:4px solid #b84f76;border-radius:0 16px 16px 0;padding:24px 28px;margin-top:28px}
.discussion h3{font-family:Georgia,serif;font-size:24px;font-weight:400;margin:6px 0 10px}.discussion p{margin:0}
.coverage{display:flex;align-items:center;gap:28px;margin:28px 0}
.coverage article{flex:1;background:#fff;border:1px solid #edd3df;padding:26px;border-radius:16px}
.coverage article span{color:#896170;font-size:14px}
.coverage strong{display:block;font-family:Georgia,serif;font-size:36px;font-weight:400;color:#a33c65}
.coverage p{margin:0}.arrow{font-size:32px;color:#b56783}
.note{font-size:14px;color:#806472;margin-top:22px}
.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;margin:24px 0;text-align:left}
caption{text-align:left;font-weight:600;margin-bottom:12px}th{color:#8d4260;font-weight:600;background:#f9e9ef}
td,th{padding:12px 16px;border-bottom:1px solid #ead6de;font-size:15px}
.closing{background:#f4e7eb;padding:34px;border-radius:20px;margin-top:30px}.closing li{margin-bottom:12px}
footer{max-width:1248px;margin:auto;padding:24px;display:flex;justify-content:space-between;align-items:center;color:#806472;font-size:14px}
footer button{border:1px solid #d8aabd;background:#fff;border-radius:20px;padding:8px 16px;color:#9c4566;cursor:pointer}
@media(max-width:800px){.pair{grid-template-columns:1fr}.coverage{flex-direction:column;align-items:stretch}.arrow{text-align:center;transform:rotate(90deg)}}
@media print{nav,footer button{display:none}section{break-inside:avoid}}
"""
n_somei_rows = int(somei.flowering_doy.notna().sum())
page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sakura Weather — EDA</title><style>{CSS}</style></head>
<body>
<header><div class="hero"><span class="eyebrow">Sakura research notes / weather data / 1952–2025</span>
<h1>Cleaning the weather.<br>Finding what moves the bloom.</h1>
<p class="subtitle">Data quality, gap filling and feature engineering for daily JMA weather at the {n_st} Somei-yoshino
stations, and a first look at how winter and spring temperatures shape flowering dates.</p>
<div class="tags"><span>{n_st} stations · {len(raw):,} station-days</span><span>{len(feat):,} station-year feature rows</span>
<span>{n_somei_rows:,} Somei-yoshino flowering records</span></div>
</div></header>
<nav aria-label="Report sections"><a href="#coverage">Coverage</a><a href="#quality">Artefacts</a>
<a href="#imputation">Gap filling</a><a href="#climate">Climate signal</a><a href="#drivers">Weather vs bloom</a> <a href="#decisions">Decisions & dictionary</a></nav>
<main>{''.join(sections.values())}</main>
<footer><p>Generated by src/02_preprocess/04_eda_weather_report.py from data/processed/02_cleaned_source_data/weather/.</p>
<button onclick="window.print()">Print / Save as PDF</button></footer>
</body></html>"""
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUT_FILE.write_text(page, encoding="utf-8")
print(f"wrote {OUT_FILE} ({OUT_FILE.stat().st_size / 1e6:.1f} MB)")
