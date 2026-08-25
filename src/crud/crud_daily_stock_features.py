"""
CRUD operations for daily_stock_features table (functions only).
"""

from datetime import date
from typing import Any

import pandas as pd

from src.core.database import get_db


def upsert_daily_features(df: pd.DataFrame) -> int:
    """
    Upserts a DataFrame of calculated daily stock features into daily_stock_features.
    """
    if df.empty:
        return 0

    records = df.to_dict(orient="records")
    query = """
    INSERT INTO daily_stock_features (
        date, ticker, ticker_encoded, open, high, low, close, adj_close, volume,
        daily_return, price_change_pct, volume_change_pct, ma5, ma10, ma20, rsi, macd, macd_signal,
        spy_return, qqq_return, avg_sentiment, max_sentiment, min_sentiment,
        positive_news_count, negative_news_count, neutral_news_count, news_count,
        avg_sentiment_t_1, news_count_t_1, avg_sentiment_3d, news_count_3d
    ) VALUES (
        %(date)s, %(ticker)s, %(ticker_encoded)s, %(open)s, %(high)s, %(low)s, %(close)s, %(adj_close)s, %(volume)s,
        %(daily_return)s, %(price_change_pct)s, %(volume_change_pct)s, %(ma5)s, %(ma10)s, %(ma20)s, %(rsi)s, %(macd)s, %(macd_signal)s,
        %(spy_return)s, %(qqq_return)s, %(avg_sentiment)s, %(max_sentiment)s, %(min_sentiment)s,
        %(positive_news_count)s, %(negative_news_count)s, %(neutral_news_count)s, %(news_count)s,
        %(avg_sentiment_t_1)s, %(news_count_t_1)s, %(avg_sentiment_3d)s, %(news_count_3d)s
    )
    ON CONFLICT (date, ticker)
    DO UPDATE SET
        ticker_encoded = EXCLUDED.ticker_encoded,
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        adj_close = EXCLUDED.adj_close,
        volume = EXCLUDED.volume,
        daily_return = EXCLUDED.daily_return,
        price_change_pct = EXCLUDED.price_change_pct,
        volume_change_pct = EXCLUDED.volume_change_pct,
        ma5 = EXCLUDED.ma5,
        ma10 = EXCLUDED.ma10,
        ma20 = EXCLUDED.ma20,
        rsi = EXCLUDED.rsi,
        macd = EXCLUDED.macd,
        macd_signal = EXCLUDED.macd_signal,
        spy_return = EXCLUDED.spy_return,
        qqq_return = EXCLUDED.qqq_return,
        avg_sentiment = EXCLUDED.avg_sentiment,
        max_sentiment = EXCLUDED.max_sentiment,
        min_sentiment = EXCLUDED.min_sentiment,
        positive_news_count = EXCLUDED.positive_news_count,
        negative_news_count = EXCLUDED.negative_news_count,
        neutral_news_count = EXCLUDED.neutral_news_count,
        news_count = EXCLUDED.news_count,
        avg_sentiment_t_1 = EXCLUDED.avg_sentiment_t_1,
        news_count_t_1 = EXCLUDED.news_count_t_1,
        avg_sentiment_3d = EXCLUDED.avg_sentiment_3d,
        news_count_3d = EXCLUDED.news_count_3d;
    """
    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(query, records)
    return len(records)


def get_latest_feature_vector(ticker: str, target_date: str | date | None = None) -> dict[str, Any] | None:
    """
    Retrieves the latest single feature row for inference on Day T.
    """
    query = "SELECT * FROM daily_stock_features WHERE ticker = %s"
    params: list[Any] = [ticker]
    if target_date:
        query += " AND date = %s"
        params.append(target_date)
    query += " ORDER BY date DESC LIMIT 1"

    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()
