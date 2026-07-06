import uuid
from typing import Any

from src.db.connection import get_db_connection


def setup_database() -> None:
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tickers (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    symbol TEXT NOT NULL UNIQUE,
                    name TEXT,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS articles_v2 (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    title TEXT NOT NULL,
                    url TEXT NOT NULL UNIQUE,
                    summary TEXT,
                    published_at TIMESTAMPTZ,
                    source TEXT,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    sentiment_score REAL,
                    sentiment_label TEXT,
                    bullets JSONB,
                    keywords JSONB
                );
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS article_tickers (
                    article_id UUID REFERENCES articles_v2(id) ON DELETE CASCADE,
                    ticker_id UUID REFERENCES tickers(id) ON DELETE CASCADE,
                    PRIMARY KEY (article_id, ticker_id)
                );
                """
            )

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles_v2 (published_at DESC);"
            )
            cursor.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_article_tickers_reverse ON article_tickers (ticker_id, article_id);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_articles_unsc_jsonb ON articles_v2 (id) WHERE sentiment_score IS NULL;"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_articles_unsum_jsonb ON articles_v2 (id) WHERE bullets IS NULL;"
            )


def insert_ticker(cursor: Any, symbol: str, name: str | None = None) -> uuid.UUID:
    """Insert a ticker (with optional display name) and return its ID."""
    cursor.execute(
        """
        INSERT INTO tickers (symbol, name)
        VALUES (%s, %s)
        ON CONFLICT (symbol) DO UPDATE SET name = COALESCE(EXCLUDED.name, tickers.name)
        RETURNING id;
        """,
        (symbol, name),
    )
    result = cursor.fetchone()
    if not result:
        raise ValueError(f"Failed to retrieve or generate ID for ticker: {symbol}")
    return result["id"]


def insert_article(cursor: Any, article: dict[str, Any]) -> uuid.UUID | None:
    """Insert an article into the database and return its ID."""
    query = """
    WITH new_row AS (
        INSERT INTO articles_v2 (title, url, summary, published_at, source)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (url) DO NOTHING
        RETURNING id
    )
    SELECT id FROM new_row
    UNION ALL
    SELECT id FROM articles_v2 WHERE url = %s
    LIMIT 1;
    """
    cursor.execute(
        query,
        (
            article["title"],
            article["url"],
            article["summary"],
            article["published_at"],
            article["source"],
            article["url"],  # Used for the fallback UNION ALL selector
        ),
    )
    result = cursor.fetchone()
    return result["id"] if result else None


def link_article_to_ticker(
    cursor: Any, article_id: uuid.UUID, ticker_id: uuid.UUID
) -> bool:
    """Link an article to a ticker in the database."""
    cursor.execute(
        """
        INSERT INTO article_tickers (article_id, ticker_id)
        VALUES (%s, %s)
        ON CONFLICT DO NOTHING;
        """,
        (article_id, ticker_id),
    )
    return cursor.rowcount > 0
