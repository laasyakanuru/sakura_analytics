"""Shared MySQL connection for every ETL / modelling script.

Credentials are looked up in this order, so you type them at most once:
  1. Environment variables  SAKURA_DB_USER, SAKURA_DB_PASSWORD,
                            SAKURA_DB_HOST (default localhost),
                            SAKURA_DB_PORT (default 3306)
  2. A `.env` file in the project root with the same keys (git-ignored;
     copy `.env.example` to `.env` if you want to skip the prompt).
  3. An interactive prompt.
"""
import os
from getpass import getpass
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import URL

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE = "sakura_dw"


def _read_env_file():
    env_file = PROJECT_ROOT / ".env"
    values = {}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_credentials():
    """Return (user, password, host, port); prompts only for what is missing."""
    file_values = _read_env_file()

    def lookup(key, default=None):
        return os.environ.get(key) or file_values.get(key) or default

    user = lookup("SAKURA_DB_USER")
    password = lookup("SAKURA_DB_PASSWORD")
    host = lookup("SAKURA_DB_HOST", "localhost")
    port = int(lookup("SAKURA_DB_PORT", "3306"))

    if user is None:
        user = input("MySQL username [root]: ").strip() or "root"
    if password is None:
        password = getpass(f"MySQL password for {user}: ")

    # Remember for the rest of this process and any child process.
    os.environ.update(SAKURA_DB_USER=user, SAKURA_DB_PASSWORD=password,
                      SAKURA_DB_HOST=host, SAKURA_DB_PORT=str(port))
    return user, password, host, port


def get_engine(database=DATABASE):
    """SQLAlchemy engine for sakura_dw (or the bare server when database=None)."""
    user, password, host, port = get_credentials()
    url = URL.create(
        "mysql+pymysql",
        username=user,
        password=password,
        host=host,
        port=port,
        database=database,
        query={"charset": "utf8mb4"},
    )
    return create_engine(url, pool_pre_ping=True)
