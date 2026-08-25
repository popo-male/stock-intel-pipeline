"""
Master Feature Generator and Dataset Pipeline.
Pure functional implementation for feature engineering and training dataset synthesis.
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
from src.features.aggregation import aggregate_news_sentiment
from src.features.technical import compute_price_features

FEATURE_COLUMNS = [
    "date",
    "ticker",
    "ticker_encoded",
    # Metadata price fields (lagged Day T-1 close/OHLV for reference & backtesting)
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    # Technical & Momentum Features
    "daily_return",
    "price_change_pct",
    "volume_change_pct",
    "ma5",
    "ma10",
    "ma20",
    "rsi",
    "macd",
    "macd_signal",
    # Scale-Invariant Relative Features
    "close_to_ma5",
    "close_to_ma10",
    "close_to_ma20",
    "high_low_spread",
    "open_close_spread",
    # Benchmark Features
    "spy_return",
    "qqq_return",
    # News Sentiment Features
    "avg_sentiment",
    "max_sentiment",
    "min_sentiment",
    "positive_news_count",
    "negative_news_count",
    "neutral_news_count",
    "news_count",
    "avg_sentiment_t_1",
    "news_count_t_1",
    "avg_sentiment_3d",
    "news_count_3d",
]

SENTIMENT_FILL_DEFAULTS = {
    "avg_sentiment": 0.0,
    "max_sentiment": 0.0,
    "min_sentiment": 0.0,
    "positive_news_count": 0,
    "negative_news_count": 0,
    "neutral_news_count": 0,
    "news_count": 0,
    "avg_sentiment_t_1": 0.0,
    "news_count_t_1": 0,
    "avg_sentiment_3d": 0.0,
    "news_count_3d": 0,
}


def compute_benchmark_returns(
    index_symbols: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Calculates daily returns for benchmark ETFs (e.g. SPY, QQQ) lagged by 1 day."""
    benchmarks: dict[str, pd.DataFrame] = {}
    for symbol in index_symbols:
        df = get_market_prices_df(symbol, start_date=start_date, end_date=end_date)
        if df.empty:
            logger.warning(f"No market data found for benchmark {symbol}")
            continue

        df = df.sort_values("trade_date").reset_index(drop=True)
        df[f"{symbol}_return"] = (
            df["adj_close"].shift(1) - df["adj_close"].shift(2)
        ) / df["adj_close"].shift(2)
        benchmarks[symbol] = df[["trade_date", f"{symbol}_return"]].rename(
            columns={"trade_date": "date"}
        )

    return benchmarks


def apply_targets(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Formulates binary and multiclass directional targets for Day T price movement."""
    ref_close = df["feat_close"]
    today_close = df["raw_close"]
    df["target_return"] = (today_close - ref_close) / ref_close

    # Binary target
    df["target"] = np.where(
        today_close > ref_close,
        config.target_definitions.up_code,
        config.target_definitions.down_code,
    )

    # Multi-class target
    threshold = config.features.target_threshold_pct / 100.0
    conditions = [df["target_return"] > threshold, df["target_return"] < -threshold]
    choices = [
        config.target_definitions.direction_up,
        config.target_definitions.direction_down,
    ]
    df["target_class"] = np.select(
        conditions,
        choices,
        default=config.target_definitions.direction_neutral,
    )
    return df


def compute_ticker_features(
    ticker: str,
    config: AppConfig,
    benchmark_dfs: dict[str, pd.DataFrame],
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Generates feature matrix for a single ticker."""
    logger.info(f"Generating feature matrix for ticker: {ticker}")
    price_df = get_market_prices_df(
        ticker, start_date=start_date, end_date=end_date
    )
    if price_df.empty:
        return pd.DataFrame()

    # 1. Technical features (Day T features represent information up to Day T-1)
    feat_df = compute_price_features(
        price_df,
        rsi_period=config.features.rsi_period,
        macd_fast=config.features.macd_fast,
        macd_slow=config.features.macd_slow,
        macd_signal=config.features.macd_signal,
    ).rename(columns={"trade_date": "date"})

    if feat_df.empty:
        return pd.DataFrame()

    # Preserve raw Day T close before price mapping for target calculation
    feat_df["raw_close"] = feat_df["close"]

    # Scale-invariant relative features
    feat_df["close_to_ma5"] = (feat_df["feat_close"] / feat_df["ma5"]) - 1.0
    feat_df["close_to_ma10"] = (feat_df["feat_close"] / feat_df["ma10"]) - 1.0
    feat_df["close_to_ma20"] = (feat_df["feat_close"] / feat_df["ma20"]) - 1.0
    feat_df["high_low_spread"] = (
        feat_df["feat_high"] - feat_df["feat_low"]
    ) / feat_df["feat_close"]
    feat_df["open_close_spread"] = (
        feat_df["feat_close"] - feat_df["feat_open"]
    ) / feat_df["feat_open"]

    # 2. Join benchmark returns
    for symbol in config.market.index_symbols:
        col_name = f"{symbol.lower()}_return"
        if symbol in benchmark_dfs:
            feat_df = feat_df.merge(benchmark_dfs[symbol], on="date", how="left")
        if col_name in feat_df.columns:
            feat_df[col_name] = feat_df[col_name].fillna(0.0)
        else:
            feat_df[col_name] = 0.0

    # 3. Sentiment features join
    sentiments_df = get_sentiments_df(ticker)
    sentiment_features = aggregate_news_sentiment(
        ticker=ticker,
        trading_dates=feat_df["date"].tolist(),
        sentiments_df=sentiments_df,
        pos_threshold=config.nlp.bullish_threshold,
        neg_threshold=config.nlp.bearish_threshold,
    )

    if not sentiment_features.empty:
        feat_df = feat_df.merge(
            sentiment_features, on=["date", "ticker"], how="left"
        )

    feat_df = feat_df.fillna(SENTIMENT_FILL_DEFAULTS)

    # Standardize int counts
    int_cols = [
        "positive_news_count",
        "negative_news_count",
        "neutral_news_count",
        "news_count",
        "news_count_t_1",
        "news_count_3d",
    ]
    feat_df[int_cols] = feat_df[int_cols].astype(int)

    # 4. Identity and lagged price mapping (to avoid lookahead bias)
    feat_df["ticker_encoded"] = config.ticker_encodings.get(ticker, -1)
    for col in ["open", "high", "low", "close", "adj_close"]:
        feat_df[col] = feat_df[f"feat_{col}"]
    feat_df["volume"] = feat_df["feat_volume"].fillna(0).astype(int)

    # 5. Cold-start truncation
    return feat_df.iloc[config.features.cold_start_rows :].reset_index(
        drop=True
    )


def validate_features(df: pd.DataFrame) -> None:
    """Executes consistency checks on feature dataset."""
    logger.info("Executing dataset validation protocol...")

    if df.duplicated(subset=["date", "ticker"]).any():
        raise ValueError("Validation failed: Duplicate [date, ticker] rows found.")
    if df["ticker"].isnull().any():
        raise ValueError("Validation failed: Found null values in ticker column.")

    key_tech_cols = ["ma5", "ma10", "ma20", "rsi", "macd", "daily_return"]
    for col in key_tech_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())


def generate_features(
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    config: AppConfig | None = None,
) -> pd.DataFrame:
    """
    Feature Engineering Process:
    1. Computes technical indicators, scale-invariant spreads, benchmark returns, and news sentiment.
    2. Validates and saves clean feature matrix into daily_stock_features table.
    """
    cfg = config or load_config()
    watchlist = tickers or cfg.market.watchlist_symbols
    s_date = start_date or cfg.training.start_date
    e_date = end_date or cfg.training.end_date

    logger.info(
        f"Starting Feature Engineering for {watchlist} from {s_date} to {e_date}"
    )
    benchmark_dfs = compute_benchmark_returns(cfg.market.index_symbols, s_date, e_date)

    ticker_dfs = [
        compute_ticker_features(t, cfg, benchmark_dfs, s_date, e_date)
        for t in watchlist
    ]
    ticker_dfs = [df for df in ticker_dfs if not df.empty]

    if not ticker_dfs:
        logger.warning("No feature data generated.")
        return pd.DataFrame()

    master_df = (
        pd.concat(ticker_dfs, ignore_index=True)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    validate_features(master_df)

    daily_features_df = master_df[FEATURE_COLUMNS].copy()
    daily_features_df["date"] = pd.to_datetime(daily_features_df["date"]).dt.date
    upsert_daily_features(daily_features_df)
    logger.info(
        f"Successfully stored {len(daily_features_df)} feature rows in 'daily_stock_features'."
    )

    return master_df


def generate_training_data(
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    output_parquet: str | None = None,
    config: AppConfig | None = None,
) -> pd.DataFrame:
    """
    Training Dataset Generation Process:
    1. Generates features and computes directional target labels (Day T movement).
    2. Assigns chronological train / validation / test splits.
    3. Upserts to model_training table and exports final Parquet file.
    """
    cfg = config or load_config()
    watchlist = tickers or cfg.market.watchlist_symbols
    s_date = start_date or cfg.training.start_date
    e_date = end_date or cfg.training.end_date

    logger.info(
        f"Starting Training Dataset Generation for {watchlist} from {s_date} to {e_date}"
    )
    benchmark_dfs = compute_benchmark_returns(cfg.market.index_symbols, s_date, e_date)

    ticker_dfs = [
        compute_ticker_features(t, cfg, benchmark_dfs, s_date, e_date)
        for t in watchlist
    ]
    ticker_dfs = [df for df in ticker_dfs if not df.empty]

    if not ticker_dfs:
        logger.warning("No data generated for training dataset.")
        return pd.DataFrame()

    # Apply target formulation to each ticker dataframe
    processed_dfs = []
    for df in ticker_dfs:
        df = apply_targets(df, cfg)
        processed_dfs.append(df)

    master_df = (
        pd.concat(processed_dfs, ignore_index=True)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    validate_features(master_df)

    # Assign chronological splits
    unique_dates = np.array(sorted(master_df["date"].unique()))
    train_cutoff = unique_dates[
        int(len(unique_dates) * cfg.training.train_split)
    ]
    val_cutoff = unique_dates[
        int(
            len(unique_dates)
            * (cfg.training.train_split + cfg.training.val_split)
        )
    ]

    conditions = [master_df["date"] < train_cutoff, master_df["date"] < val_cutoff]
    master_df["split_type"] = np.select(
        conditions, ["train", "validation"], default="test"
    )

    training_cols = FEATURE_COLUMNS + [
        "target",
        "target_class",
        "target_return",
        "split_type",
    ]
    training_df = master_df[training_cols].copy()
    training_df["date"] = pd.to_datetime(training_df["date"]).dt.date
    upsert_training_dataset(training_df)

    # Export Parquet
    out_file = Path(output_parquet or cfg.training.output_parquet)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_parquet(out_file, index=False)
    logger.info(
        f"Successfully saved {len(training_df)} rows to 'model_training' and exported {out_file}."
    )

    return training_df
