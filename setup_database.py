"""One-command setup for the Sakura Yosoku MySQL warehouse (sakura_dw).

    python setup_database.py              # build everything on a fresh MySQL
    python setup_database.py --reset      # drop sakura_dw and rebuild from scratch
    python setup_database.py --checks     # only re-run the health checks

Steps (about 5-15 minutes, mostly loading 2.2 million weather rows):
  1. sql/01_create_star_schema.sql        create sakura_dw + 5 tables
  2. src/etl/load_stations.py             dim_station        (102 rows)
  3. src/etl/load_daily_weather.py        fact_daily_weather (2,216,378 rows)
  4. src/etl/load_bloom_season.py         fact_bloom_season  (5,986 rows)
  5. sql/03_olap_queries.sql              vw_sakura_march1_features view
  6. sql/05_model_feature_tables.sql      ref_cv_split + model_features_feb1/mar1
  7. health checks (02 + 04 + leakage/split checks) with a PASS/FAIL table

MySQL credentials are asked once (or read from .env / environment variables,
see src/etl/db_connection.py) and passed on to every step.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "etl"))

from sqlalchemy import text  # noqa: E402

from db_connection import DATABASE, get_engine  # noqa: E402

SQL_DIR = PROJECT_ROOT / "sql"
ETL_DIR = PROJECT_ROOT / "src" / "etl"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def split_sql(script):
    """Split a .sql file into statements (handles comments and quoted text)."""
    statements, buf = [], []
    i, n = 0, len(script)
    quote = None
    while i < n:
        ch = script[i]
        nxt = script[i + 1] if i + 1 < n else ""
        if quote:
            buf.append(ch)
            if ch == "\\" and nxt:
                buf.append(nxt)
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in ("'", '"', "`"):
            quote = ch
            buf.append(ch)
        elif ch == "-" and nxt == "-":
            while i < n and script[i] != "\n":
                i += 1
            continue
        elif ch == "#":
            while i < n and script[i] != "\n":
                i += 1
            continue
        elif ch == "/" and nxt == "*":
            end = script.find("*/", i + 2)
            i = n if end == -1 else end + 2
            buf.append(" ")
            continue
        elif ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
        else:
            buf.append(ch)
        i += 1
    stmt = "".join(buf).strip()
    if stmt:
        statements.append(stmt)
    return statements


def run_sql_file(path, database=None):
    """Execute every statement in a .sql file; result sets are discarded."""
    engine = get_engine(database=database)
    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        for stmt in split_sql(path.read_text(encoding="utf-8")):
            cur.execute(stmt)
            if cur.description:
                cur.fetchall()
        raw.commit()
    finally:
        raw.close()
        engine.dispose()


def run_etl(script):
    """Run an ETL script as its own process; credentials pass via environment."""
    result = subprocess.run([sys.executable, str(ETL_DIR / script)], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        sys.exit(f"\n✗ {script} failed (see the error above). Nothing else was run.\n"
                 f"  Fix the problem, then rebuild with:  python setup_database.py --reset")


def database_state():
    """Return (exists, number_of_tables)."""
    engine = get_engine(database=None)
    with engine.connect() as conn:
        exists = conn.execute(text("SELECT COUNT(*) FROM information_schema.schemata "
                                   "WHERE schema_name = :db"), {"db": DATABASE}).scalar_one()
        tables = conn.execute(text("SELECT COUNT(*) FROM information_schema.tables "
                                   "WHERE table_schema = :db"), {"db": DATABASE}).scalar_one()
    engine.dispose()
    return bool(exists), tables


def step(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


# --------------------------------------------------------------------------
# health checks: (description, SQL returning one number, expected value)
# --------------------------------------------------------------------------
CHECKS = [
    # --- 02_validation.sql: row counts and coverage
    ("dim_station rows", "SELECT COUNT(*) FROM dim_station", 102),
    ("dim_station Somei-yoshino stations",
     "SELECT COUNT(*) FROM dim_station WHERE cherry_species = 'Somei-yoshino'", 82),
    ("dim_date rows (1952-2025)", "SELECT COUNT(*) FROM dim_date", 27029),
    ("dim_season rows (1953-2025)", "SELECT COUNT(*) FROM dim_season", 73),
    ("fact_daily_weather rows", "SELECT COUNT(*) FROM fact_daily_weather", 2216378),
    ("fact_bloom_season rows", "SELECT COUNT(*) FROM fact_bloom_season", 5986),
    ("fact_bloom_season stations", "SELECT COUNT(DISTINCT station_id) FROM fact_bloom_season", 82),
    ("missing full-bloom targets", "SELECT SUM(full_bloom_date IS NULL) FROM fact_bloom_season", 825),
    # --- 02 / 04 C2: grain uniqueness
    ("duplicate (station, date) in weather",
     "SELECT COUNT(*) FROM (SELECT 1 FROM fact_daily_weather GROUP BY station_id, date_key "
     "HAVING COUNT(*) > 1) x", 0),
    ("duplicate (station, season) in bloom",
     "SELECT COUNT(*) FROM (SELECT 1 FROM fact_bloom_season GROUP BY station_id, season_key "
     "HAVING COUNT(*) > 1) x", 0),
    # --- 02: foreign keys
    ("weather rows with unknown station",
     "SELECT COUNT(*) FROM fact_daily_weather f LEFT JOIN dim_station d "
     "ON f.station_id = d.station_id WHERE d.station_id IS NULL", 0),
    ("bloom rows with unknown season",
     "SELECT COUNT(*) FROM fact_bloom_season f LEFT JOIN dim_season d "
     "ON f.season_key = d.season_key WHERE d.season_key IS NULL", 0),
    # --- 04 B2 / B3: season mapping
    ("dates mapped to >1 season (B2)",
     "SELECT COUNT(*) FROM (SELECT d.full_date FROM dim_date d JOIN dim_season s "
     "ON d.full_date BETWEEN s.season_start_date AND s.season_end_date "
     "GROUP BY d.full_date HAVING COUNT(*) > 1) x", 0),
    ("invalid season boundaries (B3)",
     "SELECT COUNT(*) FROM dim_season WHERE season_start_date IS NULL "
     "OR season_end_date IS NULL OR season_start_date > season_end_date", 0),
    # --- 04 A1: stored Jan-Apr means vs recomputed from daily data
    ("Jan-Apr monthly mean mismatches (A1)",
     """SELECT COUNT(*) FROM fact_bloom_season b
        JOIN (SELECT w.station_id, d.year, d.month, ROUND(AVG(w.mean_temperature_c), 2) AS m
              FROM fact_daily_weather w JOIN dim_date d ON w.date_key = d.date_key
              WHERE d.month BETWEEN 1 AND 4 GROUP BY w.station_id, d.year, d.month) c
          ON c.station_id = b.station_id AND c.year = b.season_key
        WHERE ABS(CASE c.month WHEN 1 THEN b.tmean_jan WHEN 2 THEN b.tmean_feb
                               WHEN 3 THEN b.tmean_mar ELSE b.tmean_apr END - c.m) > 0.011""", 0),
    # --- 05: model tables agreed in the meeting
    ("CV split years (1964-2025)", "SELECT COUNT(*) FROM ref_cv_split", 62),
    ("test years = 2015-2025",
     "SELECT COUNT(*) FROM ref_cv_split WHERE split = 'test' AND bloom_year BETWEEN 2015 AND 2025", 11),
    ("fold 1 = train 1964-2004 / valid 2005-2006",
     "SELECT SUM(fold1 = 'train' AND bloom_year <= 2004) + SUM(fold1 = 'valid' "
     "AND bloom_year IN (2005, 2006)) FROM ref_cv_split", 43),
    ("fold 5 = train 1964-2012 / valid 2013-2014",
     "SELECT SUM(fold5 = 'train' AND bloom_year <= 2012) + SUM(fold5 = 'valid' "
     "AND bloom_year IN (2013, 2014)) FROM ref_cv_split", 51),
    ("model_features_feb1 rows", "SELECT COUNT(*) FROM model_features_feb1", 4365),
    ("model_features_mar1 rows", "SELECT COUNT(*) FROM model_features_mar1", 4365),
    ("feature rows before 1964 or non-Somei",
     "SELECT COUNT(*) FROM model_features_mar1 m JOIN dim_station s ON s.station_id = m.station_id "
     "WHERE m.bloom_year < 1964 OR s.cherry_species <> 'Somei-yoshino'", 0),
    ("full bloom on/before Mar 1 (target already known)",
     "SELECT COUNT(*) FROM model_features_mar1 WHERE full_bloom_doy <= 59", 0),
    ("Mar 1 Feb-mean equals stored tmean_feb",
     "SELECT COUNT(*) FROM model_features_mar1 m JOIN fact_bloom_season b "
     "ON b.station_id = m.station_id AND b.season_key = m.bloom_year "
     "WHERE ABS(m.tmean_feb - b.tmean_feb) > 0.011", 0),
]


def run_checks():
    step("Health checks")
    engine = get_engine()
    failures = 0
    with engine.connect() as conn:
        for desc, sql, expected in CHECKS:
            try:
                value = conn.execute(text(sql)).scalar()
                value = int(value) if value is not None else None
                ok = value == expected
                shown = f"{value:,}" if value is not None else "NULL"
            except Exception as exc:  # e.g. table missing
                ok, shown = False, f"error: {str(exc).splitlines()[0][:60]}"
            failures += not ok
            print(f"  {'PASS' if ok else 'FAIL'}  {desc:<48} {shown:>12}  (expected {expected:,})")
    engine.dispose()
    print()
    if failures:
        print(f"✗ {failures} check(s) failed. See README.md > Troubleshooting.")
    else:
        print("✓ All checks passed. sakura_dw is ready.\n"
              "  Model data:  model_features_feb1, model_features_mar1 (see README.md)")
    return failures


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reset", action="store_true",
                        help="DROP the existing sakura_dw database and rebuild it")
    parser.add_argument("--checks", action="store_true",
                        help="only run the health checks on the existing database")
    parser.add_argument("--yes", action="store_true", help="do not ask before --reset")
    args = parser.parse_args()

    started = time.time()
    try:
        exists, n_tables = database_state()
    except Exception as exc:
        message = str(getattr(exc, "orig", exc))
        if "1045" in message:
            hint = "Wrong MySQL username or password (use the one you use in MySQL Workbench)."
        elif "2003" in message or "Can't connect" in message:
            hint = ("MySQL server is not running. macOS: System Settings -> MySQL -> Start. "
                    "Windows: Services -> MySQL80 -> Start.")
        else:
            hint = message
        sys.exit(f"\n✗ Could not connect to MySQL: {hint}")

    if args.checks:
        sys.exit(1 if run_checks() else 0)

    if exists and n_tables > 0 and not args.reset:
        print(f"\nDatabase '{DATABASE}' already exists ({n_tables} tables). Nothing was changed.\n"
              f"  - Re-run the checks only:   python setup_database.py --checks\n"
              f"  - Rebuild from scratch:     python setup_database.py --reset")
        sys.exit(0)

    if exists and args.reset:
        if not args.yes:
            answer = input(f"\nThis will DELETE the database '{DATABASE}' and rebuild it. "
                           f"Type yes to continue: ").strip().lower()
            if answer != "yes":
                sys.exit("Cancelled.")
        engine = get_engine(database=None)
        with engine.begin() as conn:
            conn.execute(text(f"DROP DATABASE {DATABASE}"))
        engine.dispose()
        print(f"Dropped {DATABASE}.")

    step("1/6  Create schema (sql/01_create_star_schema.sql)")
    run_sql_file(SQL_DIR / "01_create_star_schema.sql")
    print("Created sakura_dw with dim_date and dim_season filled.")

    step("2/6  Load stations (src/etl/load_stations.py)")
    run_etl("load_stations.py")

    step("3/6  Load daily weather (src/etl/load_daily_weather.py) - the slow step")
    run_etl("load_daily_weather.py")

    step("4/6  Load bloom seasons (src/etl/load_bloom_season.py)")
    run_etl("load_bloom_season.py")

    step("5/6  Create analysis view (sql/03_olap_queries.sql)")
    run_sql_file(SQL_DIR / "03_olap_queries.sql")
    print("Created vw_sakura_march1_features.")

    step("6/6  Build model tables (sql/05_model_feature_tables.sql)")
    run_sql_file(SQL_DIR / "05_model_feature_tables.sql")
    print("Created ref_cv_split, model_features_feb1, model_features_mar1.")

    failures = run_checks()
    print(f"Total time: {(time.time() - started) / 60:.1f} min")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
