/* SAKURA YOSOKU | 04 Season Integration & Validation
   MySQL 8+ | Read-only | Run after ETL
   A: derived feature consistency; B: date/season mapping;
   C: safe integration of different fact grains.
   Does not change schema, ERD, data, or views.
*/
USE sakura_dw;

/* B1: Date to season mapping */
SELECT d.date_key, d.full_date, d.year AS calendar_year,
       s.season_key AS bloom_year, s.season_start_date, s.season_end_date
FROM dim_date d JOIN dim_season s
 ON d.full_date BETWEEN s.season_start_date AND s.season_end_date
ORDER BY d.full_date;

/* B2: Season overlap; expect zero rows */
SELECT d.full_date, COUNT(*) AS matching_seasons
FROM dim_date d JOIN dim_season s
 ON d.full_date BETWEEN s.season_start_date AND s.season_end_date
GROUP BY d.full_date HAVING COUNT(*) > 1;

/* B3: Invalid season boundaries; expect zero rows */
SELECT season_key, season_start_date, season_end_date
FROM dim_season
WHERE season_start_date IS NULL OR season_end_date IS NULL
   OR season_start_date > season_end_date;

/* C1: Aggregate winter daily weather to station-season before joining bloom.
   Descriptive only: Dec-Feb is NOT available for Feb 1 forecasts. */
WITH winter_weather AS (
 SELECT w.station_id, s.season_key,
        ROUND(AVG(w.mean_temperature_c),2) AS avg_winter_temp,
        COUNT(w.mean_temperature_c) AS valid_temp_days
 FROM fact_daily_weather w
 JOIN dim_date d ON w.date_key = d.date_key
 JOIN dim_season s ON d.full_date BETWEEN s.season_start_date AND s.season_end_date
 WHERE d.month IN (12,1,2)
 GROUP BY w.station_id,s.season_key
)
SELECT b.station_id, b.season_key AS bloom_year,
       ww.avg_winter_temp, ww.valid_temp_days, b.full_bloom_doy
FROM fact_bloom_season b JOIN winter_weather ww
 ON b.station_id=ww.station_id AND b.season_key=ww.season_key
WHERE b.full_bloom_doy IS NOT NULL
ORDER BY b.season_key,b.station_id;

/* C2: Grain uniqueness; each should return zero rows */
SELECT station_id,date_key,COUNT(*) AS n FROM fact_daily_weather
GROUP BY station_id,date_key HAVING COUNT(*)>1;
SELECT station_id,season_key,COUNT(*) AS n FROM fact_bloom_season
GROUP BY station_id,season_key HAVING COUNT(*)>1;

/* A1: January-April monthly mean temperature validation summary.
   Assumes arithmetic average of available daily means and 2-decimal rounding. */
WITH monthly AS (
 SELECT w.station_id,d.year AS bloom_year,d.month,
        ROUND(AVG(w.mean_temperature_c),2) AS calculated_mean,
        COUNT(w.mean_temperature_c) AS valid_days
 FROM fact_daily_weather w JOIN dim_date d ON w.date_key=d.date_key
 WHERE d.month BETWEEN 1 AND 4
 GROUP BY w.station_id,d.year,d.month
), stored AS (
 SELECT station_id,season_key,1 AS month_no,tmean_jan AS stored_mean FROM fact_bloom_season
 UNION ALL SELECT station_id,season_key,2,tmean_feb FROM fact_bloom_season
 UNION ALL SELECT station_id,season_key,3,tmean_mar FROM fact_bloom_season
 UNION ALL SELECT station_id,season_key,4,tmean_apr FROM fact_bloom_season
), comparison AS (
 SELECT b.station_id,b.season_key,b.month_no,
        CASE WHEN b.stored_mean IS NULL OR m.calculated_mean IS NULL THEN 'MISSING'
             WHEN ABS(b.stored_mean-m.calculated_mean)<=0.011 THEN 'MATCH'
             ELSE 'MISMATCH' END AS status
 FROM stored b LEFT JOIN monthly m
 ON b.station_id=m.station_id AND b.season_key=m.bloom_year AND b.month_no=m.month
)
SELECT month_no,status,COUNT(*) AS station_seasons
FROM comparison GROUP BY month_no,status ORDER BY month_no,status;

/* A2: February-March precipitation and sunshine validation.
   SUM ignores NULLs; mismatches require ETL missingness-rule review. */
WITH totals AS (
 SELECT w.station_id,d.year AS bloom_year,
        ROUND(SUM(w.precipitation_mm),2) AS calc_precip,
        ROUND(SUM(w.sunshine_duration_hours),2) AS calc_sunshine,
        COUNT(*) AS recorded_days,
        COUNT(w.precipitation_mm) AS valid_precip_days,
        COUNT(w.sunshine_duration_hours) AS valid_sunshine_days
 FROM fact_daily_weather w JOIN dim_date d ON w.date_key=d.date_key
 WHERE d.month IN (2,3)
 GROUP BY w.station_id,d.year
), comparison AS (
 SELECT b.station_id,b.season_key,
        CASE WHEN b.precip_feb_mar_mm IS NULL OR t.calc_precip IS NULL THEN 'MISSING'
             WHEN ABS(b.precip_feb_mar_mm-t.calc_precip)<=0.011 THEN 'MATCH'
             ELSE 'MISMATCH' END AS precip_status,
        CASE WHEN b.sunshine_feb_mar_h IS NULL OR t.calc_sunshine IS NULL THEN 'MISSING'
             WHEN ABS(b.sunshine_feb_mar_h-t.calc_sunshine)<=0.011 THEN 'MATCH'
             ELSE 'MISMATCH' END AS sunshine_status
 FROM fact_bloom_season b LEFT JOIN totals t
 ON b.station_id=t.station_id AND b.season_key=t.bloom_year
)
SELECT 'precip_feb_mar_mm' AS feature,precip_status AS status,COUNT(*) AS station_seasons
FROM comparison GROUP BY precip_status
UNION ALL
SELECT 'sunshine_feb_mar_h',sunshine_status,COUNT(*)
FROM comparison GROUP BY sunshine_status
ORDER BY feature,status;

/* A3: Investigate missing Jan/Feb stored features by station */
SELECT b.station_id,s.location,s.region,COUNT(*) AS missing_seasons,
       MIN(b.season_key) AS first_missing_year,MAX(b.season_key) AS last_missing_year
FROM fact_bloom_season b LEFT JOIN dim_station s ON b.station_id=s.station_id
WHERE b.tmean_jan IS NULL OR b.tmean_feb IS NULL
GROUP BY b.station_id,s.location,s.region
ORDER BY missing_seasons DESC,b.station_id;

/* NOTES:
   Prior local result: 5942 matching Jan/Feb records, 44 missing, 0 mismatches.
   This script has NOT been executed against your local MySQL instance.
   Recalculate derived features if daily weather or ETL rules change.
   Review issue-date leakage and cardinality independently.
*/
