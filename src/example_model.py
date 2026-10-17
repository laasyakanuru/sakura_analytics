"""Example model - copy this file to start your own model.

    cp src/example_model.py src/models/gradient_boosting.py
    # edit MODEL_NAME, make_model() and (optionally) choose_features()
    python3 src/models/gradient_boosting.py

What it does, for ALL THREE forecast dates (Feb 1, Mar 1 and Mar 15):
  1. loads the agreed leakage-safe feature table (MySQL or data/model/ CSV);
  2. trains on each of the 5 agreed CV folds and scores the validation years
     with MAE / RMSE / R2 / +-3-day hit rate;
  3. prints the climatology baseline next to it for comparison;
  4. saves the fold table to results/<MODEL_NAME>_<forecast_date>_cv.csv.

The 2015-2025 test years are NOT used unless RUN_FINAL_TEST = True. Only switch
that on for the champion model, after the team has picked it on validation.
"""
import sys
from pathlib import Path

# Find the project root (the folder containing requirements.txt), so this file
# works from src/, src/models/ or anywhere else inside the project.
PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                    if (p / "requirements.txt").exists())
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from model_data import (TARGET, climatology_baseline, cv_report,  # noqa: E402
                        feature_columns, final_test_report, load_features)

# ============================ change these ==================================
MODEL_NAME = "example_linear_regression"
RUN_FINAL_TEST = False          # champion model only!


def make_model():
    """Return a NEW, unfitted model. Examples:
        from sklearn.tree import DecisionTreeRegressor
        return DecisionTreeRegressor(max_depth=5, random_state=42)

        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(random_state=42)

        from sklearn.svm import SVR                       # SVR needs scaling
        return make_pipeline(StandardScaler(), SVR(C=10))
    """
    return make_pipeline(StandardScaler(), LinearRegression())


def choose_features(df):
    """Pick features from the agreed candidate list (default: all of them).
    Example: return ["latitude", "elevation_m", "tmean_jan", "tmean_feb"]"""
    return feature_columns(df)
# ============================================================================


def main():
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    for forecast_date in ("feb1", "mar1", "mar15"):
        df = load_features(forecast_date)
        features = choose_features(df)

        def predict(train, valid):
            model = make_model()
            model.fit(train[features], train[TARGET])
            return model.predict(valid[features])

        report = cv_report(df, predict)
        baseline = cv_report(df, climatology_baseline)

        print(f"\n=== {MODEL_NAME} | forecast date {forecast_date} | "
              f"{len(features)} features ===")
        print(report.round(3).to_string(index=False))
        b = baseline.iloc[-1]
        print(f"climatology baseline (mean of folds): MAE {b.MAE:.3f}  RMSE {b.RMSE:.3f}  "
              f"R2 {b.R2:.3f}  hit_rate_3d {b.hit_rate_3d:.3f}")

        out = results_dir / f"{MODEL_NAME}_{forecast_date}_cv.csv"
        report.insert(0, "model", MODEL_NAME)
        report.insert(1, "forecast_date", forecast_date)
        report.to_csv(out, index=False)
        print(f"saved {out.relative_to(PROJECT_ROOT)}")

        if RUN_FINAL_TEST:
            test = final_test_report(df, predict)
            print("FINAL TEST 2015-2025:")
            print(test.round(3).to_string(index=False))
            test.insert(0, "model", MODEL_NAME)
            test.insert(1, "forecast_date", forecast_date)
            test.to_csv(results_dir / f"{MODEL_NAME}_{forecast_date}_test.csv", index=False)


if __name__ == "__main__":
    main()
