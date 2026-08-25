"""
Sentiment Aggregation Module for precision news windows and rolling sentiment features.
"""

from datetime import date, datetime, time, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytz


class SentimentAggregator:
    """Aggregates article sentiments into point-in-time and rolling features for each trading day."""

    def __init__(
        self,
        pos_threshold: float = 0.05,
        neg_threshold: float = -0.05,
    ):
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.market_tz = pytz.timezone("America/New_York")

    def aggregate_for_trading_days(
        self,
        ticker: str,
        trading_dates: list[date | pd.Timestamp],
        sentiments_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Maps news headlines to trading days using the precision window:
        Window for Day T: From Day T-1 at 16:00 EST to Day T at 09:30 EST.
        Articles over weekends/holidays are aggregated into the next trading day.
        """
        if not trading_dates:
            return pd.DataFrame()

        sorted_dates = sorted([pd.to_datetime(d).date() for d in trading_dates])

        records: list[dict[str, Any]] = []

        if sentiments_df.empty or "sentiment_score" not in sentiments_df.columns:
            for dt in sorted_dates:
                records.append({
                    "date": dt,
                    "ticker": ticker,
                    "avg_sentiment": 0.0,
                    "max_sentiment": 0.0,
                    "min_sentiment": 0.0,
                    "positive_news_count": 0,
                    "negative_news_count": 0,
                    "neutral_news_count": 0,
                    "news_count": 0,
                })
        else:
            df_news = sentiments_df.copy()
            # timezone conversion: ensure published_at is in UTC, then convert to market timezone (EST)
            if df_news["published_at"].dt.tz is None:
                df_news["published_at"] = df_news["published_at"].dt.tz_localize("UTC")
            df_news["published_est"] = df_news["published_at"].dt.tz_convert(self.market_tz)

            for i, target_date in enumerate(sorted_dates):
                prior_date = sorted_dates[i - 1] if i > 0 else (target_date - timedelta(days=1))
                window_start = self.market_tz.localize(datetime.combine(prior_date, time(16, 0, 0)))
                window_end = self.market_tz.localize(datetime.combine(target_date, time(9, 30, 0)))

                mask = (df_news["published_est"] >= window_start) & (df_news["published_est"] < window_end)
                window_news = df_news[mask]

                if window_news.empty:
                    records.append({
                        "date": target_date,
                        "ticker": ticker,
                        "avg_sentiment": 0.0,
                        "max_sentiment": 0.0,
                        "min_sentiment": 0.0,
                        "positive_news_count": 0,
                        "negative_news_count": 0,
                        "neutral_news_count": 0,
                        "news_count": 0,
                    })
                else:
                    scores = window_news["sentiment_score"].values
                    pos_count = int(np.sum(scores > self.pos_threshold))
                    neg_count = int(np.sum(scores < self.neg_threshold))
                    neutral_count = int(np.sum((scores >= self.neg_threshold) & (scores <= self.pos_threshold)))

                    records.append({
                        "date": target_date,
                        "ticker": ticker,
                        "avg_sentiment": float(np.mean(scores)),
                        "max_sentiment": float(np.max(scores)),
                        "min_sentiment": float(np.min(scores)),
                        "positive_news_count": pos_count,
                        "negative_news_count": neg_count,
                        "neutral_news_count": neutral_count,
                        "news_count": len(scores),
                    })

        res_df = pd.DataFrame(records)
        res_df["date"] = pd.to_datetime(res_df["date"])
        res_df = res_df.sort_values("date").reset_index(drop=True)

        res_df["avg_sentiment_t_1"] = res_df["avg_sentiment"].shift(1).fillna(0.0)
        res_df["news_count_t_1"] = res_df["news_count"].shift(1).fillna(0).astype(int)

        res_df["avg_sentiment_3d"] = res_df["avg_sentiment"].rolling(window=3, min_periods=1).mean().fillna(0.0)
        res_df["news_count_3d"] = res_df["news_count"].rolling(window=3, min_periods=1).sum().fillna(0).astype(int)

        return res_df
