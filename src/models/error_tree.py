"""Error tree for the deployed 15 March SVR: where, and in which years, does it miss?

    python3 src/models/error_tree.py               # trees + images + MLflow
    python3 src/models/error_tree.py --no-mlflow

A shallow decision tree is fitted to the SVR's *errors* (not to the bloom date), using a
few readable station descriptors (latitude, elevation, how variable the station's bloom
dates are, relocations). Each leaf is a kind of station with its own typical error.

Season-weather descriptors (Feb / early-March temperature vs normal, the model's predicted
shift) were also tried: their splits did not hold on unseen years (leave-one-year-out R²
around 0 or below), so they are reported for comparison but left out of the trees.

Errors explained (all out-of-sample):
  test   2015-2025, SVR frozen at 2014 (the official final test, 528 station-seasons)
  2026   the blind forward forecast (app/forecasts/forecast_2026_mar15.csv), used only
         to check whether the rules found on 2015-2025 still hold

Two trees:
  abs    |forecast - observed|   -> where the model is badly off
  bias   forecast - observed     -> where it is consistently late (+) or early (-)

The tree is kept small (depth 3, >= 30 station-seasons per leaf) and checked with
leave-one-year-out CV, so a split must help on years it did not see.

Writes results/svr/:
  error_tree_abs.png, error_tree_bias.png   cherry-blossom themed tree diagrams
  error_tree_leaves.csv                      one row per leaf: rule, size, errors (test, 2026)
  error_tree_rows.csv                        every explained station-season with its features
and logs them to the svr-mar15 MLflow run (or a new run if that one is missing).
"""
import argparse
import ast
import os
import sys
from pathlib import Path

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                    if (p / "requirements.txt").exists())
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

import logging  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Ellipse, FancyBboxPatch  # noqa: E402
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

import svr  # noqa: E402
from model_data import TARGET, load_features  # noqa: E402

DATE = "mar15"
TRAIN_END, REF_YEARS = 2014, 30
RESULTS = PROJECT_ROOT / "results" / "svr"
FEATURES_2026 = PROJECT_ROOT / "data" / "forward_2026" / "model" / "model_features_mar15_2026.csv"
FORECAST_2026 = PROJECT_ROOT / "app" / "forecasts" / "forecast_2026_mar15.csv"
TREE = {"max_depth": 3, "min_samples_leaf": 30, "random_state": 0}

# descriptor -> (label shown on the tree, number format incl. unit)
DESCRIPTORS = {
    "latitude": ("Latitude", "{:.1f} °N"),
    "elevation_m": ("Elevation", "{:.0f} m"),
    "station_sd_doy": ("Bloom-date spread (SD)", "{:.1f} days"),
    "relocations_before_season": ("Station relocations", "{:.1f}"),
    "march_anomaly_c": ("1–14 Mar temp vs normal", "{:+.1f} °C"),
    "feb_anomaly_c": ("Feb temp vs normal", "{:+.1f} °C"),
    "predicted_shift_days": ("Predicted shift from usual", "{:+.1f} days"),
}
TREE_FEATURES = ["latitude", "elevation_m", "station_sd_doy", "relocations_before_season"]
WEATHER_FEATURES = ["march_anomaly_c", "feb_anomaly_c", "predicted_shift_days"]

# cherry-blossom palette (matches the deck)
BLUSH, PETAL, ROSE, DEEP, PLUM, BRANCH = "#FFF6F9", "#FBD9E6", "#E888AE", "#C2477A", "#4A2B3A", "#E7A9C2"
PINKS = LinearSegmentedColormap.from_list("sakura", ["#FFF0F5", "#F7C4D8", "#E57BA6", "#B8336A"])


# ------------------------------------------------------------------ data
def describe(rows, history):
    """Readable descriptors for each row, using only `history` (seasons before the forecast)."""
    hist = history[history["bloom_year"] > history["bloom_year"].max() - REF_YEARS]
    by_station = hist.groupby("station_id")
    out = pd.DataFrame(index=rows.index)
    out["latitude"] = rows["latitude"].astype(float)
    out["elevation_m"] = rows["elevation_m"].astype(float)
    sd = by_station[TARGET].std()
    out["station_sd_doy"] = rows["station_id"].map(sd).fillna(sd.median())
    for col, name in (("tmean_mar1_14", "march_anomaly_c"), ("tmean_feb", "feb_anomaly_c")):
        normal = rows["station_id"].map(by_station[col].mean())
        out[name] = rows[col].astype(float) - normal.fillna(hist[col].mean())
    out["relocations_before_season"] = rows["relocations_before_season"].astype(float)
    return out


def test_errors(df):
    """2015-2025 with the SVR frozen at 2014 (identical to the official final test)."""
    features = svr.choose_features(df, DATE)
    cv = pd.read_csv(RESULTS / f"svr_{DATE}_cv.csv")
    params = ast.literal_eval(cv.loc[cv["fold"].astype(str) == "mean", "params"].iloc[0])
    dev = df[df["bloom_year"] <= TRAIN_END]
    test = df[df["bloom_year"] > TRAIN_END].copy()
    model = svr.make_model(features, params).fit(dev, dev[TARGET])
    pred = model.predict(test)
    reference = model.reference_(test)
    usual = svr.climatology_30yr(dev, test)
    rows = test[["station_id", "location", "region", "bloom_year"]].copy()
    rows["set"] = "test"
    rows["observed_doy"] = test[TARGET].astype(float)
    rows["forecast_doy"] = pred
    rows["usual_doy"] = usual
    rows = rows.join(describe(test, dev))
    rows["predicted_shift_days"] = pred - reference
    print(f"test: {len(rows)} station-seasons, SVR {params}")
    return rows


def errors_2026(df):
    """The frozen blind 2026 forecast, scored against the observed bloom."""
    if not (FEATURES_2026.exists() and FORECAST_2026.exists()):
        print("2026 files not found: skipping the 2026 check")
        return None
    feats = pd.read_csv(FEATURES_2026, dtype={"station_id": str})
    fc = pd.read_csv(FORECAST_2026, dtype={"station_id": str}).set_index("station_id")
    feats = feats[feats["station_id"].isin(fc.index)]
    rows = feats[["station_id", "location", "region", "bloom_year"]].copy()
    rows["set"] = "2026"
    rows["observed_doy"] = feats[TARGET].astype(float)
    rows["forecast_doy"] = feats["station_id"].map(fc["predicted_doy"]).astype(float)
    rows["usual_doy"] = feats["station_id"].map(fc["reference_doy"]).astype(float)
    rows = rows.join(describe(feats, df[df["bloom_year"] <= 2025]))
    rows["predicted_shift_days"] = feats["station_id"].map(fc["anomaly_days"]).astype(float)
    print(f"2026: {len(rows)} station-seasons")
    return rows


# ------------------------------------------------------------------ trees
def fit_tree(rows, target, features=TREE_FEATURES):
    X = rows[features]
    tree = DecisionTreeRegressor(**TREE).fit(X, rows[target])
    # leave-one-year-out: does each tree beat "every row gets the overall mean" on unseen years?
    sse_tree = sse_mean = 0.0
    for tr, va in LeaveOneGroupOut().split(X, groups=rows["bloom_year"]):
        t = DecisionTreeRegressor(**TREE).fit(X.iloc[tr], rows[target].iloc[tr])
        y = rows[target].iloc[va].to_numpy()
        sse_tree += np.sum((y - t.predict(X.iloc[va])) ** 2)
        sse_mean += np.sum((y - rows[target].iloc[tr].mean()) ** 2)
    return tree, 1 - sse_tree / sse_mean


def rules(tree):
    """node id -> readable rule for the path from the root; repeated splits on one
    descriptor are merged into a single range (e.g. 36.6 °N < Latitude ≤ 40.3 °N)."""
    t, bounds = tree.tree_, {0: {}}
    for node in range(t.node_count):
        if t.children_left[node] == -1:
            continue
        name, thr = TREE_FEATURES[t.feature[node]], t.threshold[node]
        lo, hi = bounds[node].get(name, (None, None))
        bounds[t.children_left[node]] = {**bounds[node], name: (lo, thr)}
        bounds[t.children_right[node]] = {**bounds[node], name: (thr, hi)}
    out = {}
    for node, b in bounds.items():
        parts = []
        for name, (lo, hi) in b.items():
            label, fmt = DESCRIPTORS[name]
            if lo is not None and hi is not None:
                parts.append(f"{fmt.format(lo)} < {label} ≤ {fmt.format(hi)}")
            elif hi is not None:
                parts.append(f"{label} ≤ {fmt.format(hi)}")
            else:
                parts.append(f"{label} > {fmt.format(lo)}")
        out[node] = parts
    return out


def leaf_table(tree, kind, test, new):
    paths, out = rules(tree), []
    X = TREE_FEATURES
    leaf_test = tree.apply(test[X])
    leaf_new = tree.apply(new[X]) if new is not None else np.array([])
    for leaf in np.unique(leaf_test):
        a = test[leaf_test == leaf]
        b = new[leaf_new == leaf] if new is not None else test.iloc[:0]
        out.append({
            "tree": kind, "leaf": int(leaf), "rule": " AND ".join(paths[leaf]) or "all",
            "n_test": len(a), "share_test": len(a) / len(test),
            "svr_MAE_test": a["abs_error"].mean(), "usual_MAE_test": a["usual_abs_error"].mean(),
            "svr_bias_test": a["error"].mean(),
            "n_2026": len(b),
            "svr_MAE_2026": b["abs_error"].mean() if len(b) else np.nan,
            "usual_MAE_2026": b["usual_abs_error"].mean() if len(b) else np.nan,
            "svr_bias_2026": b["error"].mean() if len(b) else np.nan,
            "stations_2026": ", ".join(sorted(b["location"])) if len(b) else "",
        })
    return pd.DataFrame(out).sort_values("svr_MAE_test" if kind == "abs" else "svr_bias_test",
                                         ascending=False)


# ------------------------------------------------------------------ drawing
def sakura(ax, x, y, r, alpha=0.35, color=ROSE, angle=0):
    """A small five-petal cherry blossom at (x, y) in axes coordinates."""
    for k in range(5):
        a = np.deg2rad(angle + 72 * k)
        ax.add_patch(Ellipse((x + 0.55 * r * np.cos(a), y + 0.55 * r * np.sin(a) * 2),
                             r, r * 1.6, angle=np.rad2deg(a) - 90, transform=ax.transAxes,
                             color=color, alpha=alpha, lw=0, zorder=0))
    ax.add_patch(Ellipse((x, y), r * 0.45, r * 0.9, transform=ax.transAxes,
                         color="#FFF3B0", alpha=min(1, alpha + 0.3), lw=0, zorder=0))


def layout(tree):
    """x, y (axes coords) per node: leaves evenly spread, parents centred over children."""
    t, pos, leaves = tree.tree_, {}, []

    def walk(node, depth):
        if t.children_left[node] == -1:
            leaves.append(node)
            pos[node] = [None, depth]
            return
        walk(t.children_left[node], depth + 1)
        walk(t.children_right[node], depth + 1)
        pos[node] = [None, depth]
    walk(0, 0)
    for i, leaf in enumerate(leaves):
        pos[leaf][0] = (i + 0.5) / len(leaves)

    def centre(node):
        if pos[node][0] is None:
            pos[node][0] = (centre(t.children_left[node]) + centre(t.children_right[node])) / 2
        return pos[node][0]
    centre(0)
    depth = max(d for _, d in pos.values())
    return {n: (0.04 + 0.92 * x, 0.76 - 0.6 * d / max(depth, 1)) for n, (x, d) in pos.items()}, len(leaves)


def draw(tree, rows, kind, cv_r2, new, path):
    t = tree.tree_
    pos, n_leaves = layout(tree)
    X = TREE_FEATURES
    leaf_new = tree.apply(new[X]) if new is not None else None
    values = t.value[:, 0, 0]
    scale = np.abs(values).max() if kind == "bias" else values.max()
    lo = 0 if kind == "bias" else values.min()

    plt.rcParams["font.family"] = ["Open Sans", "Helvetica", "Arial", "DejaVu Sans"]
    fig = plt.figure(figsize=(16, 9), facecolor=BLUSH)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    for x, y, r, a, ang in ((0.965, 0.93, 0.05, 0.30, 10), (0.915, 0.975, 0.03, 0.22, 40),
                            (0.985, 0.82, 0.018, 0.18, 0), (0.975, 0.045, 0.028, 0.22, 25),
                            (0.94, 0.02, 0.016, 0.16, 60)):
        sakura(ax, x, y, r, a, angle=ang)

    if kind == "abs":
        title = "Where the 15 March forecast misses most"
        sub = ("Error tree on |forecast − observed|, 2015–2025 test seasons "
               f"({len(rows)} station-seasons) · splits checked on unseen years: "
               f"leave-one-year-out R² {cv_r2:.2f}")
    else:
        title = "Where the 15 March forecast leans early or late"
        sub = ("Error tree on forecast − observed, 2015–2025 test seasons "
               f"({len(rows)} station-seasons) · splits checked on unseen years: "
               f"leave-one-year-out R² {cv_r2:.2f}")
    fig.text(0.035, 0.96, title, fontsize=26, color=PLUM, family=["Georgia", "DejaVu Serif"], va="top")
    fig.text(0.035, 0.89, sub, fontsize=12, color="#8A5A70", va="top")

    box_w = min(0.125, 0.92 / n_leaves - 0.022)
    for node, (x, y) in pos.items():          # branches first, under the boxes
        if t.children_left[node] == -1:
            continue
        for child, word in ((t.children_left[node], "yes"), (t.children_right[node], "no")):
            cx, cy = pos[child]
            ax.plot([x, x, cx, cx], [y - 0.05, (y + cy) / 2, (y + cy) / 2, cy + 0.06],
                    color=BRANCH, lw=2.2, solid_capstyle="round", zorder=1)
            ax.text(cx + (-0.012 if word == "yes" else 0.012), (y + cy) / 2 + 0.014, word, ha="center", fontsize=10,
                    color=DEEP, style="italic", zorder=2)

    for node, (x, y) in pos.items():
        v, n = values[node], int(t.n_node_samples[node])
        leaf = t.children_left[node] == -1
        if kind == "abs":
            headline = f"avg miss {v:.1f} days"
            shade = (v - lo) / (scale - lo + 1e-9)
        else:
            headline = (f"forecast {abs(v):.1f} d too {'late' if v > 0 else 'early'}"
                        if abs(v) >= 0.05 else "on target")
            shade = abs(v) / (scale + 1e-9)
        face = PINKS(0.08 + 0.85 * shade)
        ink = "white" if shade > 0.55 else PLUM
        if leaf:
            lines = [headline, f"{n} station-seasons", f"({n / len(rows):.0%} of test)"]
            if leaf_new is not None:
                b = new[leaf_new == node]
                if len(b):
                    stat = (b["abs_error"].mean() if kind == "abs" else b["error"].mean())
                    lines.append(f"2026: {stat:+.1f} d ({len(b)} stn)" if kind == "bias"
                                 else f"2026: {stat:.1f} d ({len(b)} stn)")
                else:
                    lines.append("2026: no stations")
            w, h = box_w, 0.15
        else:
            name = X[t.feature[node]]
            label, fmt = DESCRIPTORS[name]
            lines = [f"{label} ≤ {fmt.format(t.threshold[node])} ?", headline, f"n = {n}"]
            w, h = 0.19, 0.11
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.006,rounding_size=0.018",
                                    fc=face, ec=DEEP if leaf else ROSE, lw=1.6 if leaf else 1.2,
                                    zorder=3))
        step = (h - 0.03) / len(lines)
        for i, line in enumerate(lines):
            ax.text(x, y + h / 2 - 0.015 - step * (i + 0.5), line,
                    ha="center", va="center", fontsize=(10.5 if leaf else 11) if i == 0 else 9.5,
                    weight="bold" if i == 0 else "normal", color=ink, zorder=4)

    note = ("Darker = bigger average miss." if kind == "abs"
            else "Darker = stronger lean · ‘too late’ = full bloom came earlier than forecast.")
    fig.text(0.035, 0.035, note + " 2026 lines: the blind 2026 season sorted into the same leaves.",
             fontsize=10.5, color="#8A5A70")
    fig.savefig(path, dpi=200, facecolor=BLUSH)
    plt.close(fig)
    print(f"saved {path.relative_to(PROJECT_ROOT)}")


# ------------------------------------------------------------------ MLflow
def log_to_mlflow(paths, metrics):
    import mlflow
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment("sakura-bloom")
    found = mlflow.search_runs(experiment_names=["sakura-bloom"],
                               filter_string="tags.mlflow.runName = 'svr-mar15'",
                               order_by=["start_time DESC"], max_results=1)
    if len(found):
        run = mlflow.start_run(run_id=found.iloc[0]["run_id"])
        where = "svr-mar15 run"
    else:
        run = mlflow.start_run(run_name="svr-mar15-error-tree")
        mlflow.set_tags({"model_family": "SVR", "forecast_date": DATE, "status": "analysis"})
        where = "new run svr-mar15-error-tree"
    with run:
        mlflow.set_tag("error_tree", "depth 3, min leaf 30, on 2015-2025 test errors")
        mlflow.log_params({"error_tree_" + k: v for k, v in TREE.items() if k != "random_state"})
        mlflow.log_metrics(metrics)
        for p in paths:
            mlflow.log_artifact(str(p), artifact_path="error_tree")
    print(f"logged to MLflow ({where}) at {mlflow.get_tracking_uri()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    args = parser.parse_args()

    df = load_features(DATE)
    test = test_errors(df)
    new = errors_2026(df)
    for r in (test, new):
        if r is not None:
            r["error"] = r["forecast_doy"] - r["observed_doy"]
            r["abs_error"] = r["error"].abs()
            r["usual_abs_error"] = (r["usual_doy"] - r["observed_doy"]).abs()

    RESULTS.mkdir(parents=True, exist_ok=True)
    tables, metrics, paths = [], {}, []
    for kind, target in (("abs", "abs_error"), ("bias", "error")):
        tree, cv_r2 = fit_tree(test, target)
        _, cv_weather = fit_tree(test, target, TREE_FEATURES + WEATHER_FEATURES)
        metrics[f"error_tree_{kind}_loyo_r2_with_weather"] = float(cv_weather)
        path = RESULTS / f"error_tree_{kind}.png"
        draw(tree, test, kind, cv_r2, new, path)
        tables.append(leaf_table(tree, kind, test, new))
        metrics[f"error_tree_{kind}_loyo_r2"] = float(cv_r2)
        metrics[f"error_tree_{kind}_leaves"] = int(tree.get_n_leaves())
        paths.append(path)
        print(f"\n{kind} tree · leave-one-year-out R² {cv_r2:.3f} "
              f"({'splits generalise' if cv_r2 > 0 else 'splits do NOT beat the overall mean'}) · "
              f"adding season-weather descriptors: {cv_weather:.3f}")
        print(tables[-1].drop(columns=["tree", "leaf", "stations_2026"]).round(2).to_string(index=False))

    leaves = pd.concat(tables, ignore_index=True)
    leaves.to_csv(RESULTS / "error_tree_leaves.csv", index=False)
    pd.concat([test, new]).to_csv(RESULTS / "error_tree_rows.csv", index=False)
    paths += [RESULTS / "error_tree_leaves.csv", RESULTS / "error_tree_rows.csv"]
    print(f"\nsaved results/svr/error_tree_leaves.csv, error_tree_rows.csv")
    metrics["error_tree_test_MAE"] = float(test["abs_error"].mean())

    if not args.no_mlflow:
        log_to_mlflow(paths, metrics)


if __name__ == "__main__":
    main()
