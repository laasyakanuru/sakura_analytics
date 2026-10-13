-- Sakura Yosoku | OLAP analysis and March 1 feature view
-- Read-only SELECT queries, except CREATE OR REPLACE VIEW.
USE sakura_dw;

-- Q1. Regional historical full-bloom timing (1964+).
SELECT s.region, COUNT(f.full_bloom_doy) AS observations,
       ROUND(AVG(f.full_bloom_doy),1) AS avg_bloom_doy
FROM fact_bloom_season f
JOIN dim_station s ON f.station_id = s.station_id
WHERE f.season_key >= 1964 AND f.full_bloom_doy IS NOT NULL
GROUP BY s.region ORDER BY avg_bloom_doy ASC;

-- Q2. Full-bloom timing by decade; descriptive, sensitive to station coverage.
SELECT FLOOR(f.season_key / 10) * 10 AS decade,
       COUNT(*) AS observations,
       ROUND(AVG(f.full_bloom_doy),2) AS avg_bloom_doy
FROM fact_bloom_season f
WHERE f.season_key >= 1964 AND f.full_bloom_doy IS NOT NULL
GROUP BY FLOOR(f.season_key / 10) * 10 ORDER BY decade;

-- Q3. Earliest-blooming stations with >=20 observed seasons.
SELECT s.location, s.region, COUNT(*) AS observations,
       ROUND(AVG(f.full_bloom_doy),1) AS avg_bloom_doy
FROM fact_bloom_season f
JOIN dim_station s ON f.station_id = s.station_id
WHERE f.season_key >= 1964 AND f.full_bloom_doy IS NOT NULL
GROUP BY f.station_id, s.location, s.region
HAVING COUNT(*) >= 20
ORDER BY avg_bloom_doy ASC LIMIT 10;

-- Q4. January temperature vs bloom timing (regional descriptive comparison).
SELECT s.region, COUNT(*) AS observations,
       ROUND(AVG(f.tmean_jan),2) AS avg_jan_temp,
       ROUND(AVG(f.full_bloom_doy),1) AS avg_bloom_doy
FROM fact_bloom_season f
JOIN dim_station s ON f.station_id = s.station_id
WHERE f.season_key >= 1964 AND f.season_temp_complete = 1
  AND f.full_bloom_doy IS NOT NULL AND f.tmean_jan IS NOT NULL
GROUP BY s.region ORDER BY avg_jan_temp DESC;

-- Q5. Weather imputation by region.
SELECT s.region, COUNT(*) AS total_records,
       SUM(f.temp_imputed = 1) AS imputed_records,
       ROUND(100.0 * SUM(f.temp_imputed = 1) / NULLIF(COUNT(*),0),2) AS imputation_pct
FROM fact_daily_weather f
JOIN dim_station s ON f.station_id = s.station_id
GROUP BY s.region ORDER BY imputation_pct DESC;

-- March 1 baseline feature view: contains ONLY columns defined from
-- observations through end of February, but additional availability checks
-- are still needed for southern stations and historical forecast timing.
-- This view is NOT a final leakage-validated training dataset.
CREATE OR REPLACE VIEW vw_sakura_march1_features AS
SELECT f.station_id, s.location, s.region, s.latitude, s.longitude,
       s.elevation_m, f.season_key AS bloom_year,
       f.tmean_oct, f.tmean_nov, f.tmean_dec,
       f.tmean_jan, f.tmax_jan, f.tmin_jan,
       f.tmean_feb, f.tmax_feb, f.tmin_feb,
       f.chill_days_0_7c_nov_feb, f.tmin_lowest_nov_feb,
       f.gdd5_feb1_to_feb_end, f.cum_tmax_feb1_to_feb_end,
       f.cum_tmean_feb1_to_feb_end, f.full_bloom_doy
FROM fact_bloom_season f
JOIN dim_station s ON f.station_id = s.station_id
WHERE f.season_key >= 1964
  AND f.season_temp_complete = 1
  AND f.full_bloom_doy IS NOT NULL;

SELECT COUNT(*) AS march1_candidate_rows FROM vw_sakura_march1_features;
