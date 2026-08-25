"""
CRUD operations for model_training table (functions only).
"""

from typing import Any

import pandas as pd

from src.core.database import get_db


def upsert_training_dataset(df: pd.DataFrame) -> int:
    """
    Upserts a full training dataset with target labels into model_training.
    """
    if df.empty:
        return 0

    records = df.to_dict(orient="records")
    query = """
    INSERT INTO model_training (
        date, ticker, ticker_encoded, open, high, low, close, adj_close, volume,
        daily_return, price_change_pct, volume_change_pct, ma5, ma10, ma20, rsi, macd, macd_signal,
        spy_return, qqq_return, avg_sentiment, max_sentiment, min_sentiment,
        positive_news_count, negative_news_count, neutral_news_count, news_count,
        avg_sentiment_t_1, news_count_t_1, avg_sentiment_3d, news_count_3d,
        target, target_class, target_return, split_type
    ) VALUES (
        %(date)s, %(ticker)s, %(ticker_encoded)s, %(open)s, %(high)s, %(low)s, %(close)s, %(adj_close)s, %(volume)s,
        %(daily_return)s, %(price_change_pct)s, %(volume_change_pct)s, %(ma5)s, %(ma10)s, %(ma20)s, %(rsi)s, %(macd)s, %(macd_signal)s,
        %(spy_return)s, %(qqq_return)s, %(avg_sentiment)s, %(max_sentiment)s, %(min_sentiment)s,
        %(positive_news_count)s, %(negative_news_count)s, %(neutral_news_count)s, %(news_count)s,
        %(avg_sentiment_t_1)s, %(news_count_t_1)s, %(avg_sentiment_3d)s, %(news_count_3d)s,
        %(target)s, %(target_class)s, %(target_return)s, %(split_type)s
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
        news_count_3d = EXCLUDED.news_count_3d,
        target = EXCLUDED.target,
        target_class = EXCLUDED.target_class,
        target_return = EXCLUDED.target_return,
        split_type = EXCLUDED.split_type;
    """
    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(query, records)
    return len(records)


def get_training_data_df(split_type: str | None = None) -> pd.DataFrame:
    """Retrieves model training dataset as a DataFrame."""
    query = "SELECT * FROM model_training"
    params: list[Any] = []
    if split_type:
        query += " WHERE split_type = %s"
        params.append(split_type)
    query += " ORDER BY date ASC, ticker ASC"

    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

    return pd.DataFrame(rows) if rows else pd.DataFrame()
