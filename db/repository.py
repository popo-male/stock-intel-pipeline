from typing import Any

from db.connection import get_db_connection


def setup_database() -> None:
    conn = get_db_connection()
    with conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tickers (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL UNIQUE,
                    name TEXT,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS articles_v2 (
                    id SERIAL PRIMARY KEY,
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
                    article_id INT REFERENCES articles_v2(id) ON DELETE CASCADE,
                    ticker_id INT REFERENCES tickers(id) ON DELETE CASCADE,
                    PRIMARY KEY (article_id, ticker_id)
                );
                """
            )

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_articles_v2_published_at ON articles_v2 (published_at DESC);"
            )
            cursor.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_article_tickers_reverse ON article_tickers (ticker_id, article_id);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_articles_v2_unsc_jsonb ON articles_v2 (id) WHERE sentiment_score IS NULL;"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_articles_v2_unsum_jsonb ON articles_v2 (id) WHERE bullets IS NULL;"
            )

    conn.close()


def insert_ticker(cursor: Any, symbol: str) -> int:
    """Insert a ticker into the database and return its ID."""
    cursor.execute(
        """
        INSERT INTO tickers (symbol)
        VALUES (%s)
        ON CONFLICT (symbol) DO UPDATE SET symbol = EXCLUDED.symbol
        RETURNING id;
        """,
        (symbol,),
    )
    result = cursor.fetchone()
    if not result:
        raise ValueError(f"Failed to retrieve or generate ID for ticker: {symbol}")
    return result["id"]


def insert_article(cursor: Any, article: dict[str, Any]) -> int:
    """Insert an article into the database and return its ID."""
    cursor.execute(
        """
        INSERT INTO articles_v2 (title, url, summary, published_at, source)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (url) DO NOTHING
        RETURNING id;
        """,
        (
            article["title"],
            article["url"],
            article["summary"],
            article["published_at"],
            article["source"],
        ),
    )
    result = cursor.fetchone()
    return result["id"] if result else None  # type: ignore


def link_article_to_ticker(cursor: Any, article_id: int, ticker_id: int) -> None:
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
