# Sakura Yosoku — Season Integration & Feature Validation

**Companion to:** `04_sakura_quality_and_integration.sql`  
**Database:** `sakura_dw` | **Compatibility:** MySQL 8+

## Why this SQL add-on matters for modelling

Cherry blossom prediction uses **one outcome per station and flowering season**, while the source weather observations are recorded **daily**. A model trained on incorrectly joined or inconsistently calculated weather features can produce misleading relationships, unreliable evaluation results, and an inflated number of training observations.

This SQL add-on provides **read-only checks** that help establish whether seasonal predictors are mapped, aggregated, and stored consistently before they are used for modelling. It is a validation and analytical preparation layer—not a model-training pipeline and not a guarantee of improved predictive accuracy.

## Modelling value of the checks

| SQL section | What it checks | Why it matters for the prediction model |
|---|---|---|
| **B1 — Date-to-season mapping** | Assigns dates to October–June bloom seasons using season boundaries. | Ensures October–December weather is associated with the *following* bloom year rather than the calendar year in which it occurred. |
| **B2 — Season overlap** | Finds dates assigned to more than one season. | Detects ambiguous time mappings that could cause the same weather day to contribute to multiple seasonal feature sets. |
| **B3 — Season boundaries** | Flags null or reversed start/end dates. | Helps prevent incomplete or invalid time windows from distorting derived predictors. |
| **C1 — Winter weather aggregation** | Aggregates December–February daily temperatures to one station-season record **before** joining bloom outcomes. | Avoids multiplying one bloom target across many daily rows and demonstrates the correct grain for modelling. |
| **C2 — Grain uniqueness** | Checks duplicate `(station_id, date_key)` and `(station_id, season_key)` combinations. | Verifies that each fact table respects its intended observation unit; primary keys should already prevent duplicates. |
| **A1 — Monthly temperature consistency** | Recomputes January–April mean temperatures from daily observations and compares them with stored seasonal features. | Flags potential differences in key temperature predictors caused by processing, missing-value treatment, or rounding. |
| **A2 — Rainfall and sunshine consistency** | Recomputes February–March precipitation and sunshine totals and compares them with stored seasonal totals. | Identifies possible inconsistencies in additional weather predictors before they enter retrospective analysis or appropriately timed forecasts. |
| **A3 — Missing winter features** | Lists stations with missing January or February mean-temperature features and their affected years. | Shows whether feature gaps are concentrated in particular stations or historical periods, informing exclusion or imputation decisions. |

## How to use the results

**Temperature, precipitation, and sunshine comparisons:** `MATCH` means the stored and recalculated values agree within the script's **0.011-unit tolerance**; `MISMATCH` calls for investigation, not automatic replacement. `MISSING` means a comparison cannot be completed because at least one value is absent. The calculations assume averages of available daily observations and SQL `SUM` behavior, so differences may reflect legitimate ETL rules, including imputation or missing-data handling.

**Season and key checks:** B2, B3, and C2 should normally return **zero rows**. A nonzero result warrants investigation before using affected records in training.

**Aggregated winter-weather extract:** C1 returns weather summaries at station-season grain, joined to observed full-bloom outcomes. It is useful for examining predictor–target relationships, but is **not automatically a validated training set**. It uses an inner join and excludes rows with missing full-bloom targets; its output therefore need not include every station-season.

**Missing-feature report:** A3 identifies where January/February temperature gaps occur. It does not itself determine whether a row should be dropped or imputed; that decision belongs in the modelling pipeline and should be applied consistently within cross-validation.
