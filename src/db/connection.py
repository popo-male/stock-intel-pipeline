from contextlib import contextmanager

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from src.core.logger import logger
from src.core.settings import settings

try:
    pool = ConnectionPool(
        conninfo=settings.DATABASE_URL,
        min_size=2,
        max_size=10,
        open=True,
        kwargs={"row_factory": dict_row},
    )
except Exception as e:
    logger.error(f"Error creating connection pool: {e}")
    raise


def get_db_connection():
    """
    Returns the pool's built-in connection context manager directly.
    Usage remains identical: with get_db_connection() as conn:
    """
    return pool.connection()
