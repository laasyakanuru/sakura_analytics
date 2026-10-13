-- Sakura Yosoku | Read-only validation | Run AFTER ETL.
USE sakura_dw;

-- 1. Row counts: 102 / 27029 / 73 / 2216378 / 5986 expected.
SELECT 'dim_station' AS table_name, COUNT(*) AS row_count FROM dim_station
UNION ALL SELECT 'dim_date', COUNT(*) FROM dim_date
UNION ALL SELECT 'dim_season', COUNT(*) FROM dim_season
UNION ALL SELECT 'fact_daily_weather', COUNT(*) FROM fact_daily_weather
UNION ALL SELECT 'fact_bloom_season', COUNT(*) FROM fact_bloom_season;

-- 2. Bloom coverage and missing targets.
SELECT COUNT(*) AS total_rows,
       COUNT(DISTINCT station_id) AS station_count,
       MIN(season_key) AS first_year, MAX(season_key) AS last_year,
       SUM(full_bloom_date IS NULL) AS missing_full_bloom,
       SUM(season_temp_complete = 1) AS complete_temp_seasons
FROM fact_bloom_season;

-- 3. Daily weather coverage and imputation.
SELECT COUNT(*) AS total_weather_rows,
       COUNT(DISTINCT station_id) AS weather_stations,
       MIN(date_key) AS earliest_date_key, MAX(date_key) AS latest_date_key,
       SUM(temp_imputed = 1) AS imputed_rows,
       ROUND(100.0 * SUM(temp_imputed = 1) / NULLIF(COUNT(*),0), 2) AS imputation_pct
FROM fact_daily_weather;

-- 4. Dimension species distribution (82 Somei-yoshino expected).
SELECT cherry_species, COUNT(*) AS station_count
FROM dim_station GROUP BY cherry_species ORDER BY station_count DESC;

-- 5. Duplicate natural keys (PK constraints should prevent these).
SELECT station_id, date_key, COUNT(*) AS n
FROM fact_daily_weather GROUP BY station_id, date_key HAVING COUNT(*) > 1;
SELECT station_id, season_key, COUNT(*) AS n
FROM fact_bloom_season GROUP BY station_id, season_key HAVING COUNT(*) > 1;

-- 6. Missing dimension references (should be zero due to enforced FKs).
SELECT COUNT(*) AS unmatched_weather_stations
FROM fact_daily_weather f LEFT JOIN dim_station d ON f.station_id = d.station_id
WHERE d.station_id IS NULL;
SELECT COUNT(*) AS unmatched_weather_dates
FROM fact_daily_weather f LEFT JOIN dim_date d ON f.date_key = d.date_key
WHERE d.date_key IS NULL;
SELECT COUNT(*) AS unmatched_bloom_stations
FROM fact_bloom_season f LEFT JOIN dim_station d ON f.station_id = d.station_id
WHERE d.station_id IS NULL;
SELECT COUNT(*) AS unmatched_bloom_seasons
FROM fact_bloom_season f LEFT JOIN dim_season d ON f.season_key = d.season_key
WHERE d.season_key IS NULL;

-- 7. Approximate modelling eligibility; not a final leakage-safe training set.
SELECT COUNT(*) AS potential_modelling_rows
FROM fact_bloom_season
WHERE season_key >= 1964 AND season_temp_complete = 1 AND full_bloom_date IS NOT NULL;
