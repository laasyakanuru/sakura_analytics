-- Sakura Yosoku | MySQL 8+ | Fresh database setup
-- Run ONLY on a new database. Does not import CSV data.
-- Load station metadata before fact ETLs (see README).
CREATE DATABASE IF NOT EXISTS sakura_dw CHARACTER SET utf8mb4;
USE sakura_dw;

CREATE TABLE dim_station (
    station_id VARCHAR(20) NOT NULL PRIMARY KEY,
    location VARCHAR(100),
    jma_station_id VARCHAR(30),
    prefecture VARCHAR(100),
    region VARCHAR(100),
    latitude DECIMAL(10,6),
    longitude DECIMAL(10,6),
    elevation_m DECIMAL(10,2),
    cherry_species VARCHAR(100)
) ENGINE=InnoDB;

CREATE TABLE dim_date (
    date_key INT NOT NULL PRIMARY KEY,
    full_date DATE NOT NULL UNIQUE,
    year INT,
    month INT,
    day INT,
    quarter INT,
    day_of_year INT
) ENGINE=InnoDB;

-- Generate every date from 1952-01-01 through 2025-12-31.
-- Five digits generate enough offsets; filter limits to the actual range.
INSERT INTO dim_date (date_key, full_date, year, month, day, quarter, day_of_year)
SELECT CAST(DATE_FORMAT(d.full_date, '%Y%m%d') AS UNSIGNED),
       d.full_date, YEAR(d.full_date), MONTH(d.full_date),
       DAY(d.full_date), QUARTER(d.full_date), DAYOFYEAR(d.full_date)
FROM (
    SELECT DATE_ADD('1952-01-01', INTERVAL n.num DAY) AS full_date
    FROM (
        SELECT a.n + b.n*10 + c.n*100 + e.n*1000 + f.n*10000 AS num
        FROM (SELECT 0 n UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) a
        CROSS JOIN (SELECT 0 n UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) b
        CROSS JOIN (SELECT 0 n UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) c
        CROSS JOIN (SELECT 0 n UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) e
        CROSS JOIN (SELECT 0 n UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) f
    ) n
    WHERE n.num <= DATEDIFF('2025-12-31','1952-01-01')
) d;

CREATE TABLE dim_season (
    season_key INT NOT NULL PRIMARY KEY,
    season_start_date DATE NOT NULL,
    season_end_date DATE NOT NULL,
    decade INT
) ENGINE=InnoDB;

-- Season Y covers 1 Oct (Y-1) to 30 Jun Y.
INSERT INTO dim_season (season_key, season_start_date, season_end_date, decade)
SELECT 1953 + n.num,
       DATE_ADD(MAKEDATE(1953 + n.num - 1, 1), INTERVAL 9 MONTH),
       DATE_SUB(DATE_ADD(MAKEDATE(1953 + n.num, 1), INTERVAL 6 MONTH), INTERVAL 1 DAY),
       FLOOR((1953 + n.num) / 10) * 10
FROM (
    SELECT a.n + b.n*10 AS num
    FROM (SELECT 0 n UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) a
    CROSS JOIN (SELECT 0 n UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) b
) n WHERE n.num BETWEEN 0 AND 72;

CREATE TABLE fact_daily_weather (
    station_id VARCHAR(20) NOT NULL,
    date_key INT NOT NULL,
    mean_temperature_c DECIMAL(6,2),
    maximum_temperature_c DECIMAL(6,2),
    minimum_temperature_c DECIMAL(6,2),
    precipitation_mm DECIMAL(8,2),
    sunshine_duration_hours DECIMAL(6,2),
    mean_temperature_flag VARCHAR(30),
    maximum_temperature_flag VARCHAR(30),
    minimum_temperature_flag VARCHAR(30),
    temp_imputed TINYINT,
    pre_1964_mean_below_min TINYINT,
    PRIMARY KEY (station_id, date_key),
    CONSTRAINT fk_weather_station FOREIGN KEY (station_id) REFERENCES dim_station(station_id),
    CONSTRAINT fk_weather_date FOREIGN KEY (date_key) REFERENCES dim_date(date_key)
) ENGINE=InnoDB;

CREATE TABLE fact_bloom_season (
    station_id VARCHAR(20) NOT NULL,
    season_key INT NOT NULL,
    flowering_date DATE,
    flowering_doy SMALLINT,
    full_bloom_date DATE,
    full_bloom_doy SMALLINT,
    bloom_interval_days SMALLINT,
    tmean_oct DECIMAL(6,2),
    tmean_nov DECIMAL(6,2),
    tmean_dec DECIMAL(6,2),
    tmean_jan DECIMAL(6,2),
    tmean_feb DECIMAL(6,2),
    chill_days_0_7c_nov_feb SMALLINT,
    gdd5_feb1_to_feb_end DECIMAL(10,2),
    cum_tmax_feb1_to_feb_end DECIMAL(10,2),
    temp_missing_days_oct_may SMALLINT,
    temp_imputed_days_oct_may SMALLINT,
    season_temp_complete TINYINT,
    relocations_before_season SMALLINT,
    pre_1964_definition TINYINT,
    tmax_jan DECIMAL(6,2),
    tmin_jan DECIMAL(6,2),
    tmax_feb DECIMAL(6,2),
    tmin_feb DECIMAL(6,2),
    tmean_mar DECIMAL(6,2),
    tmax_mar DECIMAL(6,2),
    tmin_mar DECIMAL(6,2),
    tmean_apr DECIMAL(6,2),
    tmin_lowest_nov_feb DECIMAL(6,2),
    frost_days_jan_mar SMALLINT,
    cum_tmean_feb1_to_feb_end DECIMAL(10,2),
    gdd5_feb1_to_mar_end DECIMAL(10,2),
    cum_tmax_feb1_to_mar_end DECIMAL(10,2),
    cum_tmean_feb1_to_mar_end DECIMAL(10,2),
    precip_feb_mar_mm DECIMAL(10,2),
    precip_feb_mar_missing_days SMALLINT,
    sunshine_feb_mar_h DECIMAL(10,2),
    sunshine_feb_mar_missing_days SMALLINT,
    rule600_tmax_pred_doy SMALLINT,
    rule400_tmean_pred_doy SMALLINT,
    PRIMARY KEY (station_id, season_key),
    CONSTRAINT fk_bloom_station FOREIGN KEY (station_id) REFERENCES dim_station(station_id),
    CONSTRAINT fk_bloom_season FOREIGN KEY (season_key) REFERENCES dim_season(season_key)
) ENGINE=InnoDB;
