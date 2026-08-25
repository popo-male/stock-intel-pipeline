"""
Database connection, configuration, and schema initialization module.
Provides get_db context manager and init_db runner.
"""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from src.core.logger import logger
from src.core.settings import settings
from src.crud.crud_news_articles import sync_tickers

_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    """Returns or lazily initializes the database connection pool."""
    global _pool
    if _pool is None:
        if not settings.DATABASE_URL:
            raise ValueError(
                "DATABASE_URL is not set in environment or .env file. "
                "Please configure your PostgreSQL connection string."
            )
        try:
            logger.info("Initializing database connection pool...")
            _pool = ConnectionPool(
                conninfo=settings.DATABASE_URL,
                min_size=1,
                max_size=10,
                open=True,
                kwargs={"row_factory": dict_row},
            )
        except Exception as e:
            logger.error(f"Failed to create database connection pool: {e}")
            raise
    return _pool


@contextmanager
def get_db() -> Generator:
    """
    Context manager providing a database connection with auto-commit / rollback.
    Usage:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
    """
    pool = get_pool()
    with pool.connection() as conn:
        yield conn


def init_db(source_dir: Path | str | None = None) -> None:
    """
    Initializes database schema by executing all .sql files in the source/ directory,
    and synchronizes tickers from config.yaml into the tickers table.
    """
    if source_dir is None:
        source_dir = Path(__file__).resolve().parents[2] / "source"
    else:
        source_dir = Path(source_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source SQL directory not found at: {source_dir}")

    sql_files = sorted(source_dir.glob("*.sql"))
    if not sql_files:
        raise FileNotFoundError(f"No .sql schema files found inside {source_dir}")

    logger.info(
        f"Executing database schema initialization from {len(sql_files)} SQL file(s)..."
    )

    with get_db() as conn, conn.cursor() as cursor:
        for sql_file in sql_files:
            logger.info(f"Executing SQL file: {sql_file.name}")
            with sql_file.open("r", encoding="utf-8") as f:
                sql_content = f.read()
            cursor.execute(sql_content)

    logger.info("Database schema initialized successfully from source/ .sql files!")

    # Synchronize tickers from config.yaml into tickers table
    synced_count = sync_tickers()
    logger.info(
        f"Synchronized {synced_count} tickers from config.yaml into 'tickers' table."
    )


def close_db() -> None:
    """Closes the active database connection pool."""
    global _pool
    if _pool is not None:
        try:
            _pool.close(timeout=1.0)
            logger.info("Database connection pool closed successfully.")
        except Exception as e:
            logger.error(f"Error closing database connection pool: {e}")
        finally:
            _pool = None
