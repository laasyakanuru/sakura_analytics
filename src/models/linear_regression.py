"""Compare OLS, Ridge, and Lasso for February 1 and March 1 forecasts.

Run from the project root:
    python3 src/models/linear_regression.py
    python3 src/models/linear_regression.py --cv-only

Data: repository CSVs loaded through model_data.load_features().
Small sample: stations with observed full bloom in 2025.
All models use the repository's five CV folds; Ridge and Lasso alpha values
minimize mean validation MAE. StandardScaler is fitted within each fold.
Tuned CV scores describe tuning performance, not independent validation.

Default output: core historical test metrics for 2015-2025, grouped by
forecast date and evaluated on the same small-sample test cohort.
Final model: select predictors using March 1 full-sample Lasso, then refit
sklearn OLS on March 1 small-sample development data (1964-2014).

Saves two files under results/:
    linear_regression_test_comparison.csv
    linear_regression_mar1_small_selected_test.csv

--cv-only prints CV metrics and final model coefficients without evaluating
test outcomes or saving files. The small-sample definition still uses 2025
station availability.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents
     if (p / 'src' / 'model_data.py').exists()), None
)
if PROJECT_ROOT is None:
    raise SystemExit('Place this file in sakura_analytics/src/models/.')
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from model_data import TARGET, cv_folds, evaluate, load_features  # noqa: E402

MODEL_NAME = 'linear_regression'
SCOPES = ('Full sample', 'Small sample')
# Preserve the notebook's candidates; three March columns have new names.
COMMON_FEATURES = [
    'latitude', 'longitude', 'elevation_m', 'tmean_oct', 'tmean_nov',
    'tmean_dec', 'tmean_jan', 'tmax_jan', 'tmin_jan',
]
FEATURES = {
    'feb1': COMMON_FEATURES,
    'mar1': COMMON_FEATURES + [
        'tmean_feb', 'tmax_feb', 'tmin_feb', 'chill_days_0_7c_nov_feb',
        'tmin_lowest_nov_feb', 'gdd5_feb', 'cum_tmax_feb', 'cum_tmean_feb',
    ],
}
LABELS = {'feb1': 'February 1', 'mar1': 'March 1'}
METRICS = ['N', 'MAE_days', 'RMSE_days', 'R2', 'Bias_days', 'Within_3_days_pct']


def make_model(method='OLS', alpha=None):
    if method == 'OLS':
        estimator = LinearRegression()
    elif method == 'Ridge':
        estimator = Ridge(alpha=alpha)
    elif method == 'Lasso':
        estimator = Lasso(alpha=alpha, max_iter=100000, tol=1e-5)
    else:
        raise ValueError(method)
    return make_pipeline(StandardScaler(), estimator)


def scores(actual, predicted):
    shared = evaluate(actual, predicted)
    error = np.asarray(predicted) - np.asarray(actual)
    return dict(N=shared['n'], MAE_days=shared['MAE'],
                RMSE_days=shared['RMSE'], R2=shared['R2'],
                Bias_days=float(error.mean()),
                Within_3_days_pct=100 * shared['hit_rate_3d'])


def save_csv(frame, directory, name):
    frame.to_csv(directory / f'{MODEL_NAME}_{name}.csv', index=False,
                 float_format='%.10g')


def fold_scores(training, validation, features, method, alpha):
    model = make_model(method, alpha)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(training[features], training[TARGET])
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return {**scores(validation[TARGET], model.predict(validation[features])),
            'Converged': converged}


def fit_candidate(data, features, method):
    dev = data.loc[data['split'].eq('development')]
    folds = list(cv_folds(dev))
    if len(folds) != 5 or any(t.empty or v.empty for _, t, v in folds):
        raise ValueError('Expected five nonempty repository CV folds.')
    grid = ([None] if method == 'OLS' else
            np.logspace(-4, 5 if method == 'Ridge' else 2, 40))
    trials = []
    for alpha in grid:
        for fold, training, validation in folds:
            trials.append({'Alpha': alpha, 'fold': fold,
                           **fold_scores(training, validation, features, method, alpha)})
    trials = pd.DataFrame(trials)
    if method == 'OLS':
        alpha, best = None, trials.copy()
    else:
        eligible = trials.groupby('Alpha')['Converged'].all()
        converged_trials = trials.loc[trials['Alpha'].isin(eligible[eligible].index)]
        if converged_trials.empty:
            raise ValueError(f'{method}: no alpha converged in all five folds.')
        ranking = converged_trials.groupby('Alpha', as_index=False)['MAE_days'].mean()
        ranking = ranking.sort_values(['MAE_days', 'Alpha'])
        alpha = float(ranking.iloc[0]['Alpha'])
        best = trials.loc[trials['Alpha'].eq(alpha)].copy()
    mean = {metric: best[metric].mean() for metric in METRICS}
    mean['N'] = int(best['N'].sum())
    best = pd.concat([best, pd.DataFrame([
        {'Alpha': alpha, 'fold': 'mean', **mean}
    ])], ignore_index=True)
    fitted = make_model(method, alpha)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        fitted.fit(dev[features], dev[TARGET])
    return fitted, alpha, best, trials


def coefficient_table(model, features):
    scaler, estimator = model.steps[0][1], model.steps[-1][1]
    raw = estimator.coef_ / scaler.scale_
    intercept = float(estimator.intercept_ - np.dot(raw, scaler.mean_))
    table = pd.DataFrame({'Variable': features,
                          'Standardized_coefficient': estimator.coef_,
                          'Original_unit_coefficient': raw})
    return intercept, table


def show(frame):
    print(frame.to_string(index=False, na_rep='-',
                          float_format=lambda x: f'{x:.4g}'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cv-only', action='store_true',
                        help='Skip historical holdout metrics and predictions.')
    args = parser.parse_args()
    out = PROJECT_ROOT / 'results'
    out.mkdir(exist_ok=True)
    datasets = {date: load_features(date, source='csv', verbose=False)
                for date in FEATURES}
    mar = datasets['mar1']
    small_stations = mar.loc[mar['bloom_year'].eq(2025)
                             & mar[TARGET].notna(), 'station_id'].unique()
    if not len(small_stations):
        raise ValueError('No observed stations in 2025 for small sample.')
    models, cohorts, comparisons = {}, {}, []
    unconverged_fits = 0
    for date, data in datasets.items():
        features = FEATURES[date]
        if data[features + [TARGET]].isna().any().any():
            raise ValueError(f'{date}: missing values in provided model data.')
        small = data.loc[data['station_id'].isin(small_stations)].copy()
        common_test = small.loc[small['split'].eq('test')]
        if common_test.empty:
            raise ValueError(f'{date}: empty holdout cohort.')
        date_rows = []
        for scope in SCOPES:
            cohort = data if scope == 'Full sample' else small
            cohorts[(date, scope)] = cohort
            for method in ('OLS', 'Ridge', 'Lasso'):
                fitted, alpha, cv, trials = fit_candidate(cohort, features, method)
                unconverged_fits += int((~trials['Converged']).sum())
                models[(date, scope, method)] = fitted
                if args.cv_only:
                    metrics = cv.loc[cv['fold'].eq('mean'), METRICS].iloc[0].to_dict()
                else:
                    metrics = scores(common_test[TARGET], fitted.predict(common_test[features]))
                date_rows.append({'Forecast': LABELS[date], 'Training sample': scope,
                                  'Model': method, 'Alpha': alpha, **metrics})
        table = pd.DataFrame(date_rows)
        table['N'] = table['N'].astype(int)
        comparisons.append(table)
        phase = 'CV tuning metrics (mean of 5 folds)' if args.cv_only else 'historical test 2015-2025, common small-sample cohort'
        print(f'\n=== {LABELS[date]} | {phase} ===')
        show(table.drop(columns='Forecast'))

    # Prescribed selection: full-sample Lasso -> small-sample OLS.
    selector = models[('mar1', 'Full sample', 'Lasso')]
    _, selection = coefficient_table(selector, FEATURES['mar1'])
    selection['Selected'] = selector.steps[-1][1].coef_ != 0
    selected = selection.loc[selection['Selected'], 'Variable'].tolist()
    if not selected:
        raise ValueError('Full-sample March Lasso selected no predictors.')
    cohort = cohorts[('mar1', 'Small sample')]
    train = cohort.loc[cohort['split'].eq('development')]
    test = cohort.loc[cohort['split'].eq('test')]
    final = make_model()
    final.fit(train[selected], train[TARGET])
    intercept, coefficients = coefficient_table(final, selected)
    info = {
        'Forecast': 'March 1', 'Training sample': 'Small sample',
        'Model': 'OLS after full-sample Lasso selection',
        'Selection_alpha': float(selector.steps[-1][1].alpha),
        'Train_N': len(train), 'Train_stations': train['station_id'].nunique(),
        'Train_year_start': int(train['bloom_year'].min()),
        'Train_year_end': int(train['bloom_year'].max()),
        'Feature_count': len(selected), 'Original_unit_intercept': intercept,
        'Selected_features': '|'.join(selected),
    }
    print('\n=== Final: March 1 | Small sample | Lasso-selected OLS ===')
    print(f'Selection: full-sample Lasso, alpha={info["Selection_alpha"]:.6g}')
    print(f'Training: {len(train):,} rows, {info["Train_stations"]} stations, '
          f'{info["Train_year_start"]}-{info["Train_year_end"]}')
    print(f'Original-unit intercept: {intercept:.6g}')
    show(coefficients)
    if not args.cv_only:
        prediction = final.predict(test[selected])
        metrics = scores(test[TARGET], prediction)
        show(pd.DataFrame([metrics]))
        save_csv(pd.DataFrame([{**info, **metrics}]), out, 'mar1_small_selected_test')
        save_csv(pd.concat(comparisons, ignore_index=True), out, 'test_comparison')
    print('\nSmall sample = stations observed in 2025 (retrospective cohort).')
    print('Alpha CV metrics are tuning scores; final specification follows the notebook.')
    if unconverged_fits:
        print(f'{unconverged_fits} unconverged CV fits: affected alphas excluded from selection.')
    if not args.cv_only:
        print(f'Two CSV files saved to {out}')


if __name__ == '__main__':
    main()
