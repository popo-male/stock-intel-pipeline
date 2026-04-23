import psycopg2
from psycopg2.extras import RealDictCursor

from core.settings import settings


def get_db_connection():
    if settings.DATABASE_URL:
        return psycopg2.connect(
            settings.DATABASE_URL,
            cursor_factory=RealDictCursor,
        )

    if not all(
        [
            settings.DB_HOST,
            settings.DB_PORT,
            settings.DB_NAME,
            settings.DB_USER,
            settings.DB_PASSWORD,
        ]
    ):
        raise ValueError(
            "Database configuration is incomplete. Set DATABASE_URL or the DB_* variables in .env."
        )

    return psycopg2.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        dbname=settings.DB_NAME,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        cursor_factory=RealDictCursor,
    )
