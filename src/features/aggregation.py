"""
Sentiment Aggregation Module.
Maps news headlines to trading days using the overnight precision window and rolling metrics.
"""

from datetime import date, datetime, time, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytz

DEFAULT_METRICS: dict[str, Any] = {
    "avg_sentiment": 0.0,
    "max_sentiment": 0.0,
    "min_sentiment": 0.0,
    "positive_news_count": 0,
    "negative_news_count": 0,
    "neutral_news_count": 0,
    "news_count": 0,
}


def extract_window_metrics(
    df_window: pd.DataFrame,
    pos_threshold: float = 0.05,
    neg_threshold: float = -0.05,
) -> dict[str, Any]:
    """Calculates statistical metrics from a subset of windowed articles."""
    if df_window.empty:
        return DEFAULT_METRICS.copy()

    scores = df_window["sentiment_score"].to_numpy()
    return {
        "avg_sentiment": float(np.mean(scores)),
        "max_sentiment": float(np.max(scores)),
        "min_sentiment": float(np.min(scores)),
        "positive_news_count": int(np.sum(scores > pos_threshold)),
        "negative_news_count": int(np.sum(scores < neg_threshold)),
        "neutral_news_count": int(
            np.sum((scores >= neg_threshold) & (scores <= pos_threshold))
        ),
        "news_count": len(scores),
    }


def aggregate_news_sentiment(
    ticker: str,
    trading_dates: list[date | pd.Timestamp],
    sentiments_df: pd.DataFrame,
    pos_threshold: float = 0.05,
    neg_threshold: float = -0.05,
    market_tz: str = "America/New_York",
) -> pd.DataFrame:
    """
    Maps news headlines to trading days using the precision overnight window:
    Window for Day T: From Day T-1 at 16:00 EST to Day T at 09:30 EST.
    Articles over weekends/holidays are aggregated into the next trading day.
    """
    if not trading_dates:
        return pd.DataFrame()

    tz = pytz.timezone(market_tz)
    sorted_dates = sorted([pd.to_datetime(d).date() for d in trading_dates])

    # Prepare timezone-aware news data
    df_news = (
        sentiments_df.copy()
        if not sentiments_df.empty and "sentiment_score" in sentiments_df.columns
        else pd.DataFrame()
    )
    if not df_news.empty:
        if df_news["published_at"].dt.tz is None:
            df_news["published_at"] = df_news["published_at"].dt.tz_localize("UTC")
        df_news["published_est"] = df_news["published_at"].dt.tz_convert(tz)

    # Build records
    records: list[dict[str, Any]] = []
    for i, target_date in enumerate(sorted_dates):
        prior_date = (
            sorted_dates[i - 1] if i > 0 else (target_date - timedelta(days=1))
        )
        window_start = tz.localize(datetime.combine(prior_date, time(16, 0, 0)))
        window_end = tz.localize(datetime.combine(target_date, time(9, 30, 0)))

        if not df_news.empty:
            mask = (df_news["published_est"] >= window_start) & (
                df_news["published_est"] < window_end
            )
            metrics = extract_window_metrics(
                df_news[mask],
                pos_threshold=pos_threshold,
                neg_threshold=neg_threshold,
            )
        else:
            metrics = DEFAULT_METRICS.copy()

        records.append(
            {"date": pd.to_datetime(target_date), "ticker": ticker, **metrics}
        )

    res_df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    # Add lag and rolling window features
    res_df["avg_sentiment_t_1"] = res_df["avg_sentiment"].shift(1).fillna(0.0)
    res_df["news_count_t_1"] = res_df["news_count"].shift(1).fillna(0).astype(int)
    res_df["avg_sentiment_3d"] = (
        res_df["avg_sentiment"].rolling(3, min_periods=1).mean().fillna(0.0)
    )
    res_df["news_count_3d"] = (
        res_df["news_count"].rolling(3, min_periods=1).sum().fillna(0).astype(int)
    )

    return res_df
