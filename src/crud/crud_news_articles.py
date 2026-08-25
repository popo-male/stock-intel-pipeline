"""
CRUD operations for news_articles, article_tickers, and news_sentiments tables (functions only).
"""

import json
import uuid
from datetime import datetime
from typing import Any

import pandas as pd

from src.core.database import get_db


def upsert_ticker(
    symbol: str, name: str | None = None, is_active: bool = True
) -> uuid.UUID:
    """Inserts or updates a ticker record and returns its UUID."""
    query = """
    INSERT INTO tickers (symbol, name, is_active)
    VALUES (%s, %s, %s)
    ON CONFLICT (symbol) DO UPDATE SET
        name = COALESCE(EXCLUDED.name, tickers.name),
        is_active = EXCLUDED.is_active
    RETURNING id;
    """
    with get_db() as conn, conn.cursor() as cursor:
        cursor.execute(query, (symbol, name, is_active))
        result = cursor.fetchone()
        return result["id"]


def sync_tickers(config: Any | None = None) -> int:
    """
    Synchronizes watchlist and index tickers from config.yaml into the tickers table.
    Prevents duplicate records using ON CONFLICT (symbol).
    """
    if config is None:
        from src.core.config import load_config

        config = load_config()

    tickers_to_sync: list[tuple[str, str | None, bool]] = []
    for item in config.market.watchlist:
        tickers_to_sync.append((item.symbol, item.name, True))
    for idx in config.market.indices:
        tickers_to_sync.append((idx.symbol, idx.name, True))

    if not tickers_to_sync:
        return 0

    query = """
    INSERT INTO tickers (symbol, name, is_active)
    VALUES (%s, %s, %s)
    ON CONFLICT (symbol) DO UPDATE SET
        name = COALESCE(EXCLUDED.name, tickers.name),
        is_active = EXCLUDED.is_active;
    """
    with get_db() as conn, conn.cursor() as cursor:
        cursor.executemany(query, tickers_to_sync)
        return len(tickers_to_sync)


def get_registered_tickers(active_only: bool = True) -> set[str]:
    """Returns a set of registered ticker symbols in the tickers table."""
    query = "SELECT symbol FROM tickers"
    if active_only:
        query += " WHERE is_active = TRUE"
    with get_db() as conn, conn.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
        return {row["symbol"] for row in rows}


def is_ticker_registered(symbol: str, active_only: bool = True) -> bool:
    """Checks if a single ticker symbol is registered in the tickers table."""
    query = "SELECT 1 FROM tickers WHERE symbol = %s"
    if active_only:
        query += " AND is_active = TRUE"
    with get_db() as conn, conn.cursor() as cursor:
        cursor.execute(query, (symbol,))
        return cursor.fetchone() is not None


def insert_article(article: dict[str, Any]) -> uuid.UUID | None:
    """
    Inserts an article if not exists by URL, returning its UUID.
    """
    query = """
    WITH new_row AS (
        INSERT INTO news_articles (title, url, summary, published_at, source)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (url) DO NOTHING
        RETURNING id
    )
    SELECT id FROM new_row
    UNION ALL
    SELECT id FROM news_articles WHERE url = %s
    LIMIT 1;
    """
    with get_db() as conn, conn.cursor() as cursor:
        cursor.execute(
            query,
            (
                article["title"],
                article["url"],
                article.get("summary", ""),
                article["published_at"],
                article.get("source", "Yahoo Finance"),
                article["url"],
            ),
        )
        result = cursor.fetchone()
        return result["id"] if result else None


def link_article_to_ticker(article_id: uuid.UUID | str, ticker: str) -> bool:
    """Links an article to a ticker symbol and resolves ticker_id if present."""
    query = """
    INSERT INTO article_tickers (article_id, ticker_id, ticker_symbol)
    VALUES (%s, (SELECT id FROM tickers WHERE symbol = %s LIMIT 1), %s)
    ON CONFLICT (article_id, ticker_symbol) DO NOTHING;
    """
    with get_db() as conn, conn.cursor() as cursor:
        cursor.execute(query, (str(article_id), ticker, ticker))
        return cursor.rowcount > 0


def get_unscored_articles(limit: int = 200) -> list[dict[str, Any]]:
    """
    Fetches articles that have not yet had sentiment calculated.
    """
    query = """
    SELECT na.id as article_id, at.ticker_symbol as ticker, na.title, na.summary, na.published_at
    FROM news_articles na
    JOIN article_tickers at ON na.id = at.article_id
    LEFT JOIN news_sentiments ns ON na.id = ns.article_id AND at.ticker_symbol = ns.ticker
    WHERE ns.id IS NULL
    ORDER BY na.published_at DESC
    LIMIT %s;
    """
    with get_db() as conn, conn.cursor() as cursor:
        cursor.execute(query, (limit,))
        return cursor.fetchall()


def insert_sentiment(
    article_id: uuid.UUID | str,
    ticker: str,
    published_at: datetime,
    sentiment_score: float,
    sentiment_label: str,
    model_version: str = "ProsusAI/finbert",
) -> None:
    """Inserts an evaluated sentiment result."""
    query = """
    INSERT INTO news_sentiments (
        article_id, ticker, published_at, sentiment_score, sentiment_label, model_version
    ) VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (article_id, ticker)
    DO UPDATE SET
        sentiment_score = EXCLUDED.sentiment_score,
        sentiment_label = EXCLUDED.sentiment_label,
        model_version = EXCLUDED.model_version;
    """
    with get_db() as conn, conn.cursor() as cursor:
        cursor.execute(
            query,
            (
                str(article_id),
                ticker,
                published_at,
                sentiment_score,
                sentiment_label,
                model_version,
            ),
        )


def get_unsummarized_articles(limit: int = 50) -> list[dict[str, Any]]:
    """Fetches articles missing LLM bullet points or keywords."""
    query = """
    SELECT id, title, summary
    FROM news_articles
    WHERE bullets IS NULL
    ORDER BY published_at DESC
    LIMIT %s;
    """
    with get_db() as conn, conn.cursor() as cursor:
        cursor.execute(query, (limit,))
        return cursor.fetchall()


def update_article_insights(
    article_id: uuid.UUID | str,
    bullets: list[str],
    keywords: list[str],
) -> None:
    """Updates LLM extracted bullet points and keywords in news_articles."""
    query = """
    UPDATE news_articles
    SET bullets = %s, keywords = %s
    WHERE id = %s;
    """
    with get_db() as conn, conn.cursor() as cursor:
        cursor.execute(
            query,
            (json.dumps(bullets), json.dumps(keywords), str(article_id)),
        )


def get_sentiments_df(
    ticker: str,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> pd.DataFrame:
    """
    Fetches all news sentiments for a ticker within a time range as a DataFrame.
    """
    query = "SELECT article_id, ticker, published_at, sentiment_score, sentiment_label FROM news_sentiments WHERE ticker = %s"
    params: list[Any] = [ticker]

    if start_time:
        query += " AND published_at >= %s"
        params.append(start_time)
    if end_time:
        query += " AND published_at <= %s"
        params.append(end_time)

    query += " ORDER BY published_at ASC"

    with get_db() as conn, conn.cursor() as cursor:
        cursor.execute(query, params)
        rows = cursor.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"])
    df["sentiment_score"] = df["sentiment_score"].astype(float)
    return df
