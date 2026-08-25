"""
Master Feature Generator and Dataset Pipeline.
Combines historical market prices and news sentiment via Left Join,
generates target labels, executes validation checks, and saves to database and Parquet.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import AppConfig, load_config
from src.core.logger import logger
from src.crud.crud_daily_stock_features import upsert_daily_features
from src.crud.crud_market_prices import get_market_prices_df
from src.crud.crud_model_training import upsert_training_dataset
from src.crud.crud_news_articles import get_sentiments_df
from src.features.sentiment_agg import SentimentAggregator
from src.features.technical import compute_price_features


class FeatureGenerator:
    """Master pipeline for feature engineering and training dataset synthesis."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        self.aggregator = SentimentAggregator(
            pos_threshold=self.config.nlp.bullish_threshold,
            neg_threshold=self.config.nlp.bearish_threshold,
        )

    def _get_benchmark_returns(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetches and calculates daily returns for benchmark ETFs (e.g. SPY, QQQ)."""
        benchmark_returns: dict[str, pd.DataFrame] = {}

        for bench_symbol in self.config.market.index_symbols:
            df = get_market_prices_df(bench_symbol, start_date=start_date, end_date=end_date)
            if not df.empty:
                df = df.sort_values("trade_date").reset_index(drop=True)
                # Lagged Return for Day T is return at Day T-1
                df[f"{bench_symbol}_return"] = (
                    df["adj_close"].shift(1) - df["adj_close"].shift(2)
                ) / df["adj_close"].shift(2)
                benchmark_returns[bench_symbol] = df[["trade_date", f"{bench_symbol}_return"]]
            else:
                logger.warning(f"No market data found for benchmark {bench_symbol}")

        return benchmark_returns

    def generate_ticker_features(
        self,
        ticker: str,
        benchmark_dfs: dict[str, pd.DataFrame],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Generates full feature set for a single ticker.
        """
        logger.info(f"Generating feature matrix for ticker: {ticker}")
        price_df = get_market_prices_df(ticker, start_date=start_date, end_date=end_date)
        if price_df.empty:
            logger.warning(f"No price history found for {ticker}")
            return pd.DataFrame()

        # 1. Technical & Price Features on Price Master Table
        feat_df = compute_price_features(
            price_df,
            rsi_period=self.config.features.rsi_period,
            macd_fast=self.config.features.macd_fast,
            macd_slow=self.config.features.macd_slow,
            macd_signal=self.config.features.macd_signal,
        )
        if feat_df.empty:
            return pd.DataFrame()

        # Rename trade_date to date for standardization
        feat_df = feat_df.rename(columns={"trade_date": "date"})

        # 2. Join Benchmark Returns (SPY / QQQ)
        for bench_symbol in self.config.market.index_symbols:
            if bench_symbol in benchmark_dfs and not benchmark_dfs[bench_symbol].empty:
                bench_df = benchmark_dfs[bench_symbol].rename(columns={"trade_date": "date"})
                feat_df = pd.merge(feat_df, bench_df, on="date", how="left")
                feat_df[f"{bench_symbol}_return"] = feat_df[f"{bench_symbol}_return"].fillna(0.0)
            else:
                feat_df[f"{bench_symbol}_return"] = 0.0

        feat_df["spy_return"] = feat_df.get("SPY_return", 0.0)
        feat_df["qqq_return"] = feat_df.get("QQQ_return", 0.0)

        # 3. News Sentiment Aggregation
        sentiments_df = get_sentiments_df(ticker)
        sentiment_features = self.aggregator.aggregate_for_trading_days(
            ticker=ticker,
            trading_dates=feat_df["date"].tolist(),
            sentiments_df=sentiments_df,
        )

        # 4. Master Table LEFT JOIN (Price Master + News Features)
        if not sentiment_features.empty:
            feat_df = pd.merge(feat_df, sentiment_features, on=["date", "ticker"], how="left")
        else:
            feat_df["avg_sentiment"] = 0.0
            feat_df["max_sentiment"] = 0.0
            feat_df["min_sentiment"] = 0.0
            feat_df["positive_news_count"] = 0
            feat_df["negative_news_count"] = 0
            feat_df["neutral_news_count"] = 0
            feat_df["news_count"] = 0
            feat_df["avg_sentiment_t_1"] = 0.0
            feat_df["news_count_t_1"] = 0
            feat_df["avg_sentiment_3d"] = 0.0
            feat_df["news_count_3d"] = 0

        # Fill any missing sentiment metrics with neutral defaults
        feat_df["avg_sentiment"] = feat_df["avg_sentiment"].fillna(0.0)
        feat_df["max_sentiment"] = feat_df["max_sentiment"].fillna(0.0)
        feat_df["min_sentiment"] = feat_df["min_sentiment"].fillna(0.0)
        feat_df["positive_news_count"] = feat_df["positive_news_count"].fillna(0).astype(int)
        feat_df["negative_news_count"] = feat_df["negative_news_count"].fillna(0).astype(int)
        feat_df["neutral_news_count"] = feat_df["neutral_news_count"].fillna(0).astype(int)
        feat_df["news_count"] = feat_df["news_count"].fillna(0).astype(int)
        feat_df["avg_sentiment_t_1"] = feat_df["avg_sentiment_t_1"].fillna(0.0)
        feat_df["news_count_t_1"] = feat_df["news_count_t_1"].fillna(0).astype(int)
        feat_df["avg_sentiment_3d"] = feat_df["avg_sentiment_3d"].fillna(0.0)
        feat_df["news_count_3d"] = feat_df["news_count_3d"].fillna(0).astype(int)

        # 5. Categorical Identity Feature
        feat_df["ticker_encoded"] = self.config.ticker_encodings.get(ticker, -1)

        # 6. Target Formulation
        feat_df["target_return"] = (feat_df["close"] - feat_df["feat_close"]) / feat_df["feat_close"]

        # Binary Target: 1 (UP) if Close_T > Close_{T-1}, else 0 (DOWN)
        up_code = self.config.target_definitions.up_code
        down_code = self.config.target_definitions.down_code
        feat_df["target"] = np.where(feat_df["close"] > feat_df["feat_close"], up_code, down_code)

        # Multi-class Target: UP (> +1%), DOWN (< -1%), NEUTRAL (between -1% and +1%)
        threshold = self.config.features.target_threshold_pct / 100.0
        dir_up = self.config.target_definitions.direction_up
        dir_down = self.config.target_definitions.direction_down
        dir_neutral = self.config.target_definitions.direction_neutral

        conditions = [
            feat_df["target_return"] > threshold,
            feat_df["target_return"] < -threshold,
        ]
        choices = [dir_up, dir_down]
        feat_df["target_class"] = np.select(conditions, choices, default=dir_neutral)

        # Map lagged prices to standardized schema columns
        feat_df["open"] = feat_df["feat_open"]
        feat_df["high"] = feat_df["feat_high"]
        feat_df["low"] = feat_df["feat_low"]
        feat_df["close"] = feat_df["feat_close"]
        feat_df["adj_close"] = feat_df["feat_adj_close"]
        feat_df["volume"] = feat_df["feat_volume"].fillna(0).astype(int)

        # 7. Boundary Truncation
        cold_start = self.config.features.cold_start_rows
        if len(feat_df) > cold_start:
            feat_df = feat_df.iloc[cold_start:].reset_index(drop=True)

        return feat_df

    def validate_dataset(self, df: pd.DataFrame) -> None:
        """Executes the validation protocol."""
        logger.info("Executing dataset validation protocol...")

        duplicates = df.duplicated(subset=["date", "ticker"]).sum()
        if duplicates > 0:
            raise ValueError(f"Validation failed: {duplicates} duplicate [date, ticker] rows found.")

        if df["ticker"].isnull().any():
            raise ValueError("Validation failed: Found null values in ticker column.")

        key_tech_cols = ["ma5", "ma10", "ma20", "rsi", "macd", "daily_return"]
        nan_counts = df[key_tech_cols].isnull().sum().to_dict()
        for col, cnt in nan_counts.items():
            if cnt > 0:
                logger.warning(f"Feature column '{col}' contains {cnt} NaN values. Imputing median.")
                df[col] = df[col].fillna(df[col].median())

        target_dist = df["target"].value_counts(normalize=True).to_dict()
        logger.info(f"Target class distribution: {target_dist}")

    def generate_and_save_dataset(
        self,
        tickers: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        output_parquet: str | None = None,
    ) -> pd.DataFrame:
        """
        Executes end-to-end dataset generation:
        1. Generates features for all tickers
        2. Assigns train / val / test splits
        3. Upserts to daily_stock_features and model_training tables
        4. Exports final Parquet dataset
        """
        if tickers is None:
            tickers = self.config.market.watchlist_symbols

        if start_date is None:
            start_date = self.config.training.start_date
        if end_date is None:
            end_date = self.config.training.end_date

        logger.info(f"Generating full training dataset for {tickers} from {start_date} to {end_date}")

        benchmark_dfs = self._get_benchmark_returns(start_date=start_date, end_date=end_date)

        ticker_dfs: list[pd.DataFrame] = []
        for ticker in tickers:
            df = self.generate_ticker_features(
                ticker=ticker,
                benchmark_dfs=benchmark_dfs,
                start_date=start_date,
                end_date=end_date,
            )
            if not df.empty:
                ticker_dfs.append(df)

        if not ticker_dfs:
            logger.warning("No feature data generated for any ticker.")
            return pd.DataFrame()

        master_df = pd.concat(ticker_dfs, ignore_index=True)
        master_df = master_df.sort_values(["date", "ticker"]).reset_index(drop=True)

        self.validate_dataset(master_df)

        train_ratio = self.config.training.train_split
        val_ratio = self.config.training.val_split

        dates = sorted(master_df["date"].unique())
        n_dates = len(dates)
        train_end_idx = int(n_dates * train_ratio)
        val_end_idx = int(n_dates * (train_ratio + val_ratio))

        train_dates = set(dates[:train_end_idx])
        val_dates = set(dates[train_end_idx:val_end_idx])

        def assign_split(row_date):
            if row_date in train_dates:
                return "train"
            elif row_date in val_dates:
                return "validation"
            else:
                return "test"

        master_df["split_type"] = master_df["date"].apply(assign_split)

        feature_cols = [
            "date", "ticker", "ticker_encoded", "open", "high", "low", "close", "adj_close", "volume",
            "daily_return", "price_change_pct", "volume_change_pct", "ma5", "ma10", "ma20", "rsi", "macd", "macd_signal",
            "spy_return", "qqq_return", "avg_sentiment", "max_sentiment", "min_sentiment",
            "positive_news_count", "negative_news_count", "neutral_news_count", "news_count",
            "avg_sentiment_t_1", "news_count_t_1", "avg_sentiment_3d", "news_count_3d"
        ]

        daily_features_df = master_df[feature_cols].copy()
        daily_features_df["date"] = pd.to_datetime(daily_features_df["date"]).dt.date
        upsert_daily_features(daily_features_df)
        logger.info(f"Upserted {len(daily_features_df)} rows into daily_stock_features.")

        training_cols = feature_cols + ["target", "target_class", "target_return", "split_type"]
        training_df = master_df[training_cols].copy()
        training_df["date"] = pd.to_datetime(training_df["date"]).dt.date
        upsert_training_dataset(training_df)
        logger.info(f"Upserted {len(training_df)} rows into model_training.")

        parquet_path = output_parquet or self.config.training.output_parquet
        out_file = Path(parquet_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        master_df.to_parquet(out_file, index=False)
        logger.info(f"Successfully exported final training dataset to {out_file} ({len(master_df)} rows).")

        return master_df
