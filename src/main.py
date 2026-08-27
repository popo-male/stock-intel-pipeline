#!/usr/bin/env python3
"""
Stock Intelligence Platform - Unified Ingestion, Feature Engineering & Prediction Pipeline.

CLI Usage:
  1. Database Initialization (Idempotent & Safe):
     uv run src/main.py init-db

  2. Market Data Ingestion:
     uv run src/main.py fetch-market --ticker AAPL --start 20240101 --end 20260101
     uv run src/main.py fetch-market --start 20240101 --end 20250101
     uv run src/main.py fetch-market

  3. News & Article Data Ingestion:
     uv run src/main.py fetch-news --ticker AAPL --start 20240101 --end 20260101
     uv run src/main.py fetch-news --start 20240101 --end 20250101
     uv run src/main.py fetch-news

  4. Generate Training Dataset:
     uv run src/main.py training-data
     uv run src/main.py training-data --output data/training_dataset.parquet

  5. Run Short-Term Trend Prediction:
     uv run src/main.py predict
     uv run src/main.py run --ticker AAPL
     uv run src/main.py predict --ticker AAPL
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is in sys.path when running as `uv run src/main.py`
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.config import load_config
from src.core.database import close_db, init_db
from src.core.logger import logger
from src.features.generator import generate_features, generate_training_data
from src.inference.predict import predict_single_stock, predict_watchlist
from src.nlp.analyzer import SentimentAnalyzer
from src.nlp.llm_processor import LLMProcessor
from src.scraper.fetcher import NewsFetcher
from src.scraper.market import MarketCollector


def normalize_date(date_str: str | None) -> str | None:
    """
    Normalizes dates in 'YYYYMMDD' or 'YYYY-MM-DD' formats to 'YYYY-MM-DD'.
    """
    if not date_str:
        return None
    d = date_str.strip()
    if len(d) == 8 and d.isdigit():
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"
    try:
        dt = datetime.strptime(d, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        raise ValueError(
            f"Invalid date format: '{date_str}'. Expected YYYYMMDD (e.g. 20240101) or YYYY-MM-DD (e.g. 2024-01-01)."
        )


def parse_tickers_arg(
    ticker: str | None, tickers: str | None = None
) -> list[str] | None:
    """Parses single ticker or comma-separated tickers into a list."""
    if ticker:
        return [t.strip().upper() for t in ticker.split(",") if t.strip()]
    if tickers:
        return [t.strip().upper() for t in tickers.split(",") if t.strip()]
    return None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stock Intelligence Platform CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Pipeline execution command"
    )

    # 1. Command: init-db
    subparsers.add_parser(
        "init-db",
        help="Safely initialize database tables and indexes from source/ SQL files (idempotent, does not overwrite existing data)",
    )

    # 2. Command: fetch-market
    market_parser = subparsers.add_parser(
        "fetch-market", help="Fetch market prices from Yahoo Finance"
    )
    market_parser.add_argument(
        "--ticker",
        "--tickers",
        dest="ticker",
        type=str,
        default=None,
        help="Ticker symbol(s) (e.g. AAPL or AAPL,MSFT)",
    )
    market_parser.add_argument(
        "--start",
        "--start-date",
        dest="start",
        type=str,
        default=None,
        help="Start date (YYYYMMDD or YYYY-MM-DD)",
    )
    market_parser.add_argument(
        "--end",
        "--end-date",
        dest="end",
        type=str,
        default=None,
        help="End date (YYYYMMDD or YYYY-MM-DD)",
    )
    market_parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target single date (YYYYMMDD or YYYY-MM-DD)",
    )
    market_parser.add_argument(
        "--live", action="store_true", help="Fetch live intraday price snapshot"
    )

    # 3. Command: fetch-news
    news_parser = subparsers.add_parser(
        "fetch-news", help="Fetch financial news articles via RSS and run NLP sentiment"
    )
    news_parser.add_argument(
        "--ticker",
        "--tickers",
        dest="ticker",
        type=str,
        default=None,
        help="Ticker symbol(s) (e.g. AAPL or AAPL,MSFT)",
    )
    news_parser.add_argument(
        "--start",
        "--start-date",
        dest="start",
        type=str,
        default=None,
        help="Start date (YYYYMMDD or YYYY-MM-DD)",
    )
    news_parser.add_argument(
        "--end",
        "--end-date",
        dest="end",
        type=str,
        default=None,
        help="End date (YYYYMMDD or YYYY-MM-DD)",
    )
    news_parser.add_argument(
        "--skip-nlp", action="store_true", help="Skip FinBERT sentiment scoring"
    )
    news_parser.add_argument(
        "--skip-llm", action="store_true", help="Skip LLM summary extraction"
    )

    # 4. Command: generate-features (alias: features)
    feature_parser = subparsers.add_parser(
        "generate-features",
        aliases=["features"],
        help="Execute feature engineering and store results in daily_stock_features table",
    )
    feature_parser.add_argument(
        "--ticker",
        "--tickers",
        dest="ticker",
        type=str,
        default=None,
        help="Ticker symbol(s)",
    )
    feature_parser.add_argument(
        "--start",
        "--start-date",
        dest="start",
        type=str,
        default=None,
        help="Start date (YYYYMMDD or YYYY-MM-DD)",
    )
    feature_parser.add_argument(
        "--end",
        "--end-date",
        dest="end",
        type=str,
        default=None,
        help="End date (YYYYMMDD or YYYY-MM-DD)",
    )
    feature_parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Single target date (YYYYMMDD or YYYY-MM-DD)",
    )
    feature_parser.add_argument(
        "--latest",
        action="store_true",
        help="Generate features only for the latest recent trading days",
    )

    # 5. Command: training-data (alias: generate-training-data)
    train_parser = subparsers.add_parser(
        "training-data",
        aliases=["generate-training-data"],
        help="Generate training dataset with target labels and train/val/test splits (DB & Parquet)",
    )
    train_parser.add_argument(
        "--output",
        "--output-parquet",
        dest="output",
        type=str,
        default=None,
        help="Parquet output file path",
    )
    train_parser.add_argument(
        "--ticker",
        "--tickers",
        dest="ticker",
        type=str,
        default=None,
        help="Ticker symbol(s)",
    )
    train_parser.add_argument(
        "--start",
        "--start-date",
        dest="start",
        type=str,
        default=None,
        help="Start date (YYYYMMDD or YYYY-MM-DD)",
    )
    train_parser.add_argument(
        "--end",
        "--end-date",
        dest="end",
        type=str,
        default=None,
        help="End date (YYYYMMDD or YYYY-MM-DD)",
    )
    train_parser.add_argument(
        "--latest",
        action="store_true",
        help="Generate training dataset only for the latest recent trading days",
    )

    # 6. Command: predict / run
    pred_parser = subparsers.add_parser(
        "predict",
        aliases=["run"],
        help="Run short-term trend prediction (all tickers or specified ticker)",
    )
    pred_parser.add_argument(
        "--ticker",
        "--tickers",
        dest="ticker",
        type=str,
        default=None,
        help="Target ticker symbol (e.g. AAPL)",
    )
    pred_parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Prediction target date (YYYYMMDD or YYYY-MM-DD)",
    )

    # Complete pipeline
    subparsers.add_parser("run-all", help="Execute complete end-to-end pipeline")

    return parser.parse_args()


def run_market_ingestion(args: argparse.Namespace) -> None:
    logger.info("=== Running Market Data Ingestion ===")
    config = load_config()
    tickers = parse_tickers_arg(args.ticker)
    start_date = normalize_date(args.start)
    end_date = normalize_date(args.end)
    single_date = normalize_date(args.date)

    if single_date:
        start_date = single_date
        end_date = single_date

    # Default to live / current day price snapshot when no historical date range is specified
    is_live = args.live or (not start_date and not end_date and not single_date)

    collector = MarketCollector(config)
    collector.collect(
        tickers=tickers,
        target_date=single_date,
        start_date=start_date,
        end_date=end_date,
        live=is_live,
    )


def run_news_ingestion(args: argparse.Namespace) -> None:
    logger.info("=== Running News Ingestion & NLP Intelligence ===")
    config = load_config()
    tickers = parse_tickers_arg(args.ticker)
    start_date = normalize_date(args.start)
    end_date = normalize_date(args.end)

    # 1. Scrape RSS Feeds
    fetcher = NewsFetcher(config)
    fetcher.fetch_all(tickers=tickers, start_date=start_date, end_date=end_date)

    # 2. FinBERT Sentiment Scoring
    if not getattr(args, "skip_nlp", False):
        logger.info("Executing FinBERT sentiment analysis...")
        analyzer = SentimentAnalyzer(config)
        analyzer.process_unscored_articles()

    # 3. LLM Insights & Bullet Extraction
    if not getattr(args, "skip_llm", False):
        logger.info("Executing LLM insights extraction...")
        llm = LLMProcessor()
        llm.process_unsummarized_articles()


def run_feature_generation(args: argparse.Namespace) -> None:
    logger.info("=== Running Feature Engineering ===")
    config = load_config()
    tickers = parse_tickers_arg(args.ticker)
    start_date = normalize_date(getattr(args, "start", None))
    end_date = normalize_date(getattr(args, "end", None))
    single_date = normalize_date(getattr(args, "date", None))
    latest = getattr(args, "latest", False)

    if single_date:
        start_date = single_date
        end_date = single_date
    elif latest:
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")

    generate_features(
        tickers=tickers or config.market.watchlist_symbols,
        start_date=start_date or config.training.start_date,
        end_date=end_date or config.training.end_date,
        config=config,
    )


def run_training_data_generation(args: argparse.Namespace) -> None:
    logger.info("=== Running Training Dataset Generation ===")
    config = load_config()
    tickers = parse_tickers_arg(args.ticker)
    start_date = normalize_date(getattr(args, "start", None))
    end_date = normalize_date(getattr(args, "end", None))
    output_path = getattr(args, "output", None)
    latest = getattr(args, "latest", False)

    if latest:
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")

    generate_training_data(
        tickers=tickers or config.market.watchlist_symbols,
        start_date=start_date or config.training.start_date,
        end_date=end_date or config.training.end_date,
        output_parquet=output_path or config.training.output_parquet,
        config=config,
    )


def run_prediction_pipeline(args: argparse.Namespace) -> None:
    logger.info("=== Running Stock Trend Direction Prediction ===")
    config = load_config()
    tickers = parse_tickers_arg(args.ticker)
    pred_date = normalize_date(getattr(args, "date", None))

    if tickers and len(tickers) == 1:
        res = predict_single_stock(tickers[0], target_date=pred_date, config=config)
        logger.info(
            f"Stored prediction for {res['ticker']} on {res['prediction_date']}: "
            f"Direction={res['target_direction']}, Confidence={res['confidence_score'] * 100:.1f}%, "
            f"Probabilities=(UP:{res['probability_up']*100:.1f}%, DOWN:{res['probability_down']*100:.1f}%, NEUTRAL:{res['probability_neutral']*100:.1f}%)"
        )
    else:
        results = predict_watchlist(
            tickers=tickers, target_date=pred_date, config=config
        )
        total = len(tickers or config.market.watchlist_symbols)
        logger.info(
            f"Completed and stored predictions for {len(results)}/{total} tickers on {pred_date or 'latest available'}."
        )


def main() -> None:
    args = parse_arguments()
    command = args.command

    if not command:
        print("No command specified. Use --help to view available commands:")
        sys.exit(1)

    try:
        if command == "init-db":
            init_db()
        elif command == "fetch-market":
            init_db()
            run_market_ingestion(args)
        elif command in ("fetch-news"):
            init_db()
            run_news_ingestion(args)
        elif command in ("generate-features", "features"):
            init_db()
            run_feature_generation(args)
        elif command in ("training-data", "generate-training-data"):
            init_db()
            run_training_data_generation(args)
        elif command in ("predict", "run"):
            init_db()
            run_prediction_pipeline(args)
        elif command == "run-all":
            init_db()
            run_market_ingestion(args)
            run_news_ingestion(args)
            run_feature_generation(args)
            run_training_data_generation(args)
            run_prediction_pipeline(args)
            logger.info("=== Full Pipeline Execution Completed Successfully ===")
    except Exception as exc:
        logger.error(
            f"Fatal error during execution of '{command}': {exc}", exc_info=True
        )
        sys.exit(1)
    finally:
        close_db()


if __name__ == "__main__":
    main()
