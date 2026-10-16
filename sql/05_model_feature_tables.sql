/* SAKURA YOSOKU | 05 Model feature tables (Feb 1 and Mar 1 forecasts)
   MySQL 8+ | Run after ETL | Safe to re-run (drops and rebuilds only the
   three tables below; never touches dim_* or fact_* tables).

   Implements the team agreements in the Model Development Meeting Summary:
   - Data filters: bloom year >= 1964, Somei-yoshino stations only.
   - Forecast dates: Feb 1 model uses weather through Jan 31 only;
     Mar 1 model uses weather through Feb 28/29 only.
   - Same CV folds for all teams; final test 2015-2025.

   Every weather feature here is recomputed from fact_daily_weather with an
   explicit date cutoff, instead of being copied from fact_bloom_season.
   Reason: several stored columns are not leakage-safe for these dates:
   - chill_days_0_7c_nov_feb / tmin_lowest_nov_feb include February
     (not allowed for the Feb 1 model);
   - season_temp_complete, gdd5_* and cum_* were nulled using Oct-May / Feb-May
     missingness, i.e. information from after the forecast date;
   - rule600_tmax_pred_doy / rule400_tmean_pred_doy need temperatures until the
     threshold is reached (always after Mar 1), so they are NOT features and
     NOT a fair forecast-date baseline.

   Row filter "temp_missing_days_to_cutoff = 0" only looks at days up to the
   cutoff, so it is decided with information available on the forecast date.
*/
USE sakura_dw;

/* ------------------------------------------------------------------
   1. Shared train / validation / test split (one row per bloom year)
   fold_k = 'train' | 'valid' | NULL (year not used in that fold)
   ------------------------------------------------------------------ */
DROP TABLE IF EXISTS ref_cv_split;
CREATE TABLE ref_cv_split (
    bloom_year INT NOT NULL PRIMARY KEY,
    split VARCHAR(12) NOT NULL,          -- 'development' (1964-2014) or 'test' (2015-2025)
    fold1 VARCHAR(5), fold2 VARCHAR(5), fold3 VARCHAR(5),
    fold4 VARCHAR(5), fold5 VARCHAR(5),
    CONSTRAINT fk_split_season FOREIGN KEY (bloom_year) REFERENCES dim_season(season_key)
) ENGINE=InnoDB;

-- Fold k: train 1964..(2002+2k), validate the next two years.
INSERT INTO ref_cv_split
SELECT season_key,
       CASE WHEN season_key <= 2014 THEN 'development' ELSE 'test' END,
       CASE WHEN season_key <= 2004 THEN 'train' WHEN season_key <= 2006 THEN 'valid' END,
       CASE WHEN season_key <= 2006 THEN 'train' WHEN season_key <= 2008 THEN 'valid' END,
       CASE WHEN season_key <= 2008 THEN 'train' WHEN season_key <= 2010 THEN 'valid' END,
       CASE WHEN season_key <= 2010 THEN 'train' WHEN season_key <= 2012 THEN 'valid' END,
       CASE WHEN season_key <= 2012 THEN 'train' WHEN season_key <= 2014 THEN 'valid' END
FROM dim_season
WHERE season_key BETWEEN 1964 AND 2025;

/* ------------------------------------------------------------------
   2. Feb 1 forecast: weather 1 Oct (Y-1) .. 31 Jan (Y)
   ------------------------------------------------------------------ */
DROP TABLE IF EXISTS model_features_feb1;
CREATE TABLE model_features_feb1 AS
SELECT s.station_id, s.location, s.prefecture, s.region,
       s.latitude, s.longitude, s.elevation_m,
       a.bloom_year, b.relocations_before_season,
       a.tmean_oct, a.tmean_nov, a.tmean_dec, a.tmean_jan,
       a.tmax_jan, a.tmin_jan,
       a.chill_days_0_7c_nov_jan, a.tmin_lowest_nov_jan,
       a.frost_days_jan,
       a.gdd5_jan, a.cum_tmax_jan, a.cum_tmean_jan,
       a.precip_jan_mm, a.precip_jan_missing_days,
       a.sunshine_jan_h, a.sunshine_jan_missing_days,
       b.full_bloom_date, b.full_bloom_doy,
       c.split, c.fold1, c.fold2, c.fold3, c.fold4, c.fold5
FROM (
    SELECT station_id, bloom_year,
           SUM(tm IS NULL OR tx IS NULL OR tn IS NULL) AS temp_missing_days_to_cutoff,
           ROUND(AVG(CASE WHEN month = 10 THEN tm END), 2) AS tmean_oct,
           ROUND(AVG(CASE WHEN month = 11 THEN tm END), 2) AS tmean_nov,
           ROUND(AVG(CASE WHEN month = 12 THEN tm END), 2) AS tmean_dec,
           ROUND(AVG(CASE WHEN month = 1  THEN tm END), 2) AS tmean_jan,
           ROUND(AVG(CASE WHEN month = 1  THEN tx END), 2) AS tmax_jan,
           ROUND(AVG(CASE WHEN month = 1  THEN tn END), 2) AS tmin_jan,
           SUM(month IN (11,12,1) AND tm BETWEEN 0 AND 7.2) AS chill_days_0_7c_nov_jan,
           MIN(CASE WHEN month IN (11,12,1) THEN tn END) AS tmin_lowest_nov_jan,
           SUM(month = 1 AND tn < 0) AS frost_days_jan,
           ROUND(SUM(CASE WHEN month = 1 THEN GREATEST(tm - 5, 0) END), 2) AS gdd5_jan,
           ROUND(SUM(CASE WHEN month = 1 THEN tx END), 2) AS cum_tmax_jan,
           ROUND(SUM(CASE WHEN month = 1 THEN tm END), 2) AS cum_tmean_jan,
           ROUND(SUM(CASE WHEN month = 1 THEN pr END), 2) AS precip_jan_mm,
           SUM(month = 1 AND pr IS NULL) AS precip_jan_missing_days,
           ROUND(SUM(CASE WHEN month = 1 THEN su END), 2) AS sunshine_jan_h,
           SUM(month = 1 AND su IS NULL) AS sunshine_jan_missing_days
    FROM (
        SELECT w.station_id, d.year + (d.month >= 10) AS bloom_year, d.month,
               w.mean_temperature_c AS tm, w.maximum_temperature_c AS tx,
               w.minimum_temperature_c AS tn, w.precipitation_mm AS pr,
               w.sunshine_duration_hours AS su
        FROM fact_daily_weather w
        JOIN dim_date d ON w.date_key = d.date_key
        WHERE d.month IN (10, 11, 12, 1)
          AND d.full_date BETWEEN '1963-10-01' AND '2025-01-31'
    ) daily
    GROUP BY station_id, bloom_year
) a
JOIN fact_bloom_season b ON b.station_id = a.station_id AND b.season_key = a.bloom_year
JOIN dim_station s ON s.station_id = a.station_id
JOIN ref_cv_split c ON c.bloom_year = a.bloom_year
WHERE s.cherry_species = 'Somei-yoshino'
  AND b.full_bloom_doy IS NOT NULL
  AND a.temp_missing_days_to_cutoff = 0;

ALTER TABLE model_features_feb1 ADD PRIMARY KEY (station_id, bloom_year);

/* ------------------------------------------------------------------
   3. Mar 1 forecast: weather 1 Oct (Y-1) .. 28/29 Feb (Y)
   ------------------------------------------------------------------ */
DROP TABLE IF EXISTS model_features_mar1;
CREATE TABLE model_features_mar1 AS
SELECT s.station_id, s.location, s.prefecture, s.region,
       s.latitude, s.longitude, s.elevation_m,
       a.bloom_year, b.relocations_before_season,
       a.tmean_oct, a.tmean_nov, a.tmean_dec, a.tmean_jan, a.tmean_feb,
       a.tmax_jan, a.tmin_jan, a.tmax_feb, a.tmin_feb,
       a.chill_days_0_7c_nov_feb, a.tmin_lowest_nov_feb,
       a.frost_days_jan_feb,
       a.gdd5_feb, a.cum_tmax_feb, a.cum_tmean_feb,
       a.precip_feb_mm, a.precip_feb_missing_days,
       a.sunshine_feb_h, a.sunshine_feb_missing_days,
       b.full_bloom_date, b.full_bloom_doy,
       c.split, c.fold1, c.fold2, c.fold3, c.fold4, c.fold5
FROM (
    SELECT station_id, bloom_year,
           SUM(tm IS NULL OR tx IS NULL OR tn IS NULL) AS temp_missing_days_to_cutoff,
           ROUND(AVG(CASE WHEN month = 10 THEN tm END), 2) AS tmean_oct,
           ROUND(AVG(CASE WHEN month = 11 THEN tm END), 2) AS tmean_nov,
           ROUND(AVG(CASE WHEN month = 12 THEN tm END), 2) AS tmean_dec,
           ROUND(AVG(CASE WHEN month = 1  THEN tm END), 2) AS tmean_jan,
           ROUND(AVG(CASE WHEN month = 2  THEN tm END), 2) AS tmean_feb,
           ROUND(AVG(CASE WHEN month = 1  THEN tx END), 2) AS tmax_jan,
           ROUND(AVG(CASE WHEN month = 1  THEN tn END), 2) AS tmin_jan,
           ROUND(AVG(CASE WHEN month = 2  THEN tx END), 2) AS tmax_feb,
           ROUND(AVG(CASE WHEN month = 2  THEN tn END), 2) AS tmin_feb,
           SUM(month IN (11,12,1,2) AND tm BETWEEN 0 AND 7.2) AS chill_days_0_7c_nov_feb,
           MIN(CASE WHEN month IN (11,12,1,2) THEN tn END) AS tmin_lowest_nov_feb,
           SUM(month IN (1,2) AND tn < 0) AS frost_days_jan_feb,
           ROUND(SUM(CASE WHEN month = 2 THEN GREATEST(tm - 5, 0) END), 2) AS gdd5_feb,
           ROUND(SUM(CASE WHEN month = 2 THEN tx END), 2) AS cum_tmax_feb,
           ROUND(SUM(CASE WHEN month = 2 THEN tm END), 2) AS cum_tmean_feb,
           ROUND(SUM(CASE WHEN month = 2 THEN pr END), 2) AS precip_feb_mm,
           SUM(month = 2 AND pr IS NULL) AS precip_feb_missing_days,
           ROUND(SUM(CASE WHEN month = 2 THEN su END), 2) AS sunshine_feb_h,
           SUM(month = 2 AND su IS NULL) AS sunshine_feb_missing_days
    FROM (
        SELECT w.station_id, d.year + (d.month >= 10) AS bloom_year, d.month,
               w.mean_temperature_c AS tm, w.maximum_temperature_c AS tx,
               w.minimum_temperature_c AS tn, w.precipitation_mm AS pr,
               w.sunshine_duration_hours AS su
        FROM fact_daily_weather w
        JOIN dim_date d ON w.date_key = d.date_key
        WHERE d.month IN (10, 11, 12, 1, 2)
          AND d.full_date BETWEEN '1963-10-01' AND '2025-02-28'
    ) daily
    GROUP BY station_id, bloom_year
) a
JOIN fact_bloom_season b ON b.station_id = a.station_id AND b.season_key = a.bloom_year
JOIN dim_station s ON s.station_id = a.station_id
JOIN ref_cv_split c ON c.bloom_year = a.bloom_year
WHERE s.cherry_species = 'Somei-yoshino'
  AND b.full_bloom_doy IS NOT NULL
  AND a.temp_missing_days_to_cutoff = 0;

ALTER TABLE model_features_mar1 ADD PRIMARY KEY (station_id, bloom_year);

/* ------------------------------------------------------------------
   4. Quick summary (rows per split)
   ------------------------------------------------------------------ */
SELECT 'feb1' AS forecast_date, split, COUNT(*) AS station_years,
       COUNT(DISTINCT station_id) AS stations
FROM model_features_feb1 GROUP BY split
UNION ALL
SELECT 'mar1', split, COUNT(*), COUNT(DISTINCT station_id)
FROM model_features_mar1 GROUP BY split
ORDER BY forecast_date, split;
