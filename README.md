# Stock Intelligence Platform - Ingestion & Prediction Pipeline

A professional, data-driven platform that ingests daily stock market data and financial news, performs sentiment analysis and precision window aggregations, engineers technical & macro features, and generates explainable short-term stock trend predictions.

---

## 📌 Table of Contents

- [Overview & Architecture](#-overview--architecture)
- [📚 Detailed Documentation Index](#-detailed-documentation-index)
- [Standardized Data Layer](#-standardized-data-layer)
- [Quick Start with Local Database (Docker Compose)](#-quick-start-with-local-database-docker-compose)
- [Configuration Guide](#-configuration-guide)
- [CLI Command Reference](#-cli-command-reference)
  - [1. Database Initialization (`init-db`)](#1-database-initialization-init-db)
  - [2. Market Data Ingestion (`fetch-market`)](#2-market-data-ingestion-fetch-market)
  - [3. News & Sentiment Ingestion (`fetch-news`)](#3-news--sentiment-ingestion-fetch-news)
  - [4. Training Dataset Generation (`training-data`)](#4-training-dataset-generation-training-data)
  - [5. Trend Prediction (`predict` / `run`)](#5-trend-prediction-predict--run)
  - [6. Full Pipeline Execution (`run-all`)](#6-full-pipeline-execution-run-all)
- [Project Directory Structure](#-project-directory-structure)

---

## 📚 Detailed Documentation Index

For in-depth mathematical formulations, workflows, and module-specific guides, refer to the dedicated documents in [`docs/`](docs/):

1. 🗄️ [**Database Design & Schema Architecture**](docs/database_design.md) — 4-tier standardized database layers, table schemas, and function-only CRUD architecture.
2. 📡 [**Data Collector Architecture**](docs/data_collector.md) — Yahoo Finance market prices and multi-threaded parallel RSS feed ingestion.
3. 🧠 [**News Sentiment Analysis & LLM Insights**](docs/news_sentiment.md) — FinBERT compound scoring $[-1.0, 1.0]$ and Groq LLM executive summaries.
4. ⚙️ [**Feature Engineering & Dataset Generation**](docs/feature_engineering.md) — Precision news window rules, TA indicators (MA, RSI, MACD), Left-Join architecture, and Parquet export.
5. 🔮 [**Prediction Engine & Explainability**](docs/prediction.md) — Next-day trend inference, sigmoid probability calibration, and factor attribution.

---

## 🏛️ Overview & Architecture

The system combines structured daily price movements from Yahoo Finance and unstructured financial news articles into a unified feature matrix, training a classifier to predict whether a stock will move **UP** or **DOWN** on the next trading day.

```
                                  ┌─────────────────────────────────────────────────┐
                                  │                  DATA SOURCES                   │
                                  │   Yahoo Finance (Prices & RSS News Feeds)       │
                                  └────────────────────────┬────────────────────────┘
                                                           │
                                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. RAW DATA LAYER (source/schema.sql)                                                                            │
│   • tickers          : Tracked assets (Magnificent 7 + Indices) with categorical encoding IDs                    │
│   • market_prices    : Raw daily OHLCV, adjusted close, volume, price change %, and market cap                   │
│   • news_articles    : Raw scraped news articles, body text, and LLM bullet point highlights                     │
│   • article_tickers  : Many-to-many link between news articles and tickers                                       │
│   • news_sentiments  : Article sentiment scores (s ∈ [-1.0, 1.0]) & labels generated via FinBERT                 │
└──────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. PROCESSED / FEATURE LAYER (source/schema.sql)                                                                 │
│   • daily_stock_features : Market features (MA5/10/20, RSI-14, MACD, SPY/QQQ returns) joined with Precision      │
│                            News Windows (T-1 4:00 PM EST to T 9:30 AM EST, weekend aggregation, 3-day rolling)   │
└──────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. MODEL TRAINING LAYER (source/schema.sql)                                                                      │
│   • model_training   : Master dataset joined with future direction target labels                                 │
│                        (Binary UP/DOWN and Multi-Class UP/NEUTRAL/DOWN) with train/validation/test partitions    │
│   • Parquet Export   : data/training_dataset.parquet                                                             │
└──────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. PREDICTION LAYER (source/schema.sql)                                                                          │
│   • stock_predictions: Direction predictions (UP / DOWN / NEUTRAL), confidence scores, probabilities,            │
│                        and human-readable explainability driver breakdowns                                       │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🗄️ Standardized Data Layer

All SQL DDL definitions are centrally maintained in [`source/schema.sql`](source/schema.sql):

| Layer | Table Name | Description |
|---|---|---|
| **Raw Data** | `tickers` | Asset registry and categorical integer IDs (`AAPL: 0, MSFT: 1, ...`). |
| **Raw Data** | `market_prices` | Raw daily OHLCV prices, volume, and market cap from Yahoo Finance. |
| **Raw Data** | `news_articles` | Raw headlines, body text summaries, and extracted LLM bullet points. |
| **Raw Data** | `article_tickers` | Many-to-many relationship join linking articles to stocks. |
| **Raw Data** | `news_sentiments` | FinBERT sentiment scores ($[-1.0, 1.0]$) and labels per article. |
| **Processed** | `daily_stock_features` | Engineered indicators + precision window news sentiment aggregations. |
| **Training** | `model_training` | Master dataset joined with directional targets and train/val/test splits. |
| **Prediction** | `stock_predictions` | Predicted direction, confidence %, probability distribution, and explanations. |

---

## 🐳 Quick Start with Local Database (Docker Compose)

To test and run without touching production databases, launch the local PostgreSQL container:

```bash
# 1. Start local PostgreSQL 16 container
docker compose up -d

# 2. Check container status
docker compose ps

# 3. Stop container when finished
docker compose down
```

The local test database will be available at:
```
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/stock_db
```

Copy the sample environment file:
```bash
cp .env.example .env
```

---

## ⚙️ Configuration Guide

All watchlist assets, benchmark indices, thresholds, and hyperparameters are declared in [`config.yaml`](config.yaml) and loaded dynamically:

```yaml
market:
  watchlist:
    - symbol: "AAPL"
      name: "Apple Inc."
      encoding: 0
    - symbol: "MSFT"
      name: "Microsoft Corporation"
      encoding: 1
    # ... AMZN, GOOGL, NVDA, META, TSLA
  indices:
    - symbol: "SPY"
      name: "SPDR S&P 500 ETF Trust"
    - symbol: "QQQ"
      name: "Invesco QQQ Trust"

training:
  start_date: "2023-01-01"
  end_date: "2025-12-31"
  output_parquet: "data/training_dataset.parquet"
```

---

## 🚀 CLI Command Reference

Execute commands using `uv run src/main.py <command>`. Both `YYYYMMDD` (e.g. `20240101`) and `YYYY-MM-DD` (e.g. `2024-01-01`) date formats are supported.

### 1. Database Initialization (`init-db`)

Initializes all database tables, constraints, and indexes from `source/*.sql`.

> **Idempotent & Safe**: Running `init-db` multiple times will **not** raise errors, drop tables, or overwrite existing data.

```bash
uv run src/main.py init-db
```

---

### 2. Market Data Ingestion (`fetch-market`)

Extracts historical daily OHLCV prices from Yahoo Finance.

```bash
# Fetch specific ticker across date range
uv run src/main.py fetch-market --ticker AAPL --start 20240101 --end 20260101

# Fetch all tickers in config.yaml across date range
uv run src/main.py fetch-market --start 20240101 --end 20250101

# Fetch all tickers in config.yaml using default config time range
uv run src/main.py fetch-market

# Fetch a single specific day (e.g. today or specific historical date)
uv run src/main.py fetch-market --ticker AAPL --start 20260824 --end 20260824

# Fetch live intraday snapshot
uv run src/main.py fetch-market --live
```

> **Q: If I want to fetch the current day, do I run `--start 20260824 --end 20260824`?**  
> **A:** **Yes.** When `--start` and `--end` are set to the same date, the pipeline extracts data specifically for that single day. If the market is currently active and you want the live intraday quote, use `--live`.

---

### 3. News & Sentiment Ingestion (`fetch-news`)

Scrapes Yahoo Finance RSS feeds, computes FinBERT sentiment scores, and extracts LLM insights.

```bash
# Ingest news for specific ticker across date range
uv run src/main.py fetch-news --ticker AAPL --start 20240101 --end 20260101

# Ingest news for all tickers in config.yaml across date range
uv run src/main.py fetch-news --start 20240101 --end 20250101

# Ingest all latest RSS articles for all tickers in config.yaml
uv run src/main.py fetch-news

# Filter single day news (e.g. current day)
uv run src/main.py fetch-news --start 20260824 --end 20260824

# Skip LLM summary extraction (faster news + FinBERT scoring only)
uv run src/main.py fetch-news --skip-llm
```

---

### 4. Training Dataset Generation (`training-data`)

Executes technical feature calculation (MA5/10/20, RSI-14, MACD), precision news window aggregation, master Left Join, target labeling, validation, and Parquet export.

```bash
# Generate training dataset using default output path (data/training_dataset.parquet)
uv run src/main.py training-data

# Specify custom Parquet output file
uv run src/main.py training-data --output data/custom_training_dataset.parquet

# Generate for a custom date range
uv run src/main.py training-data --start 20230101 --end 20251231
```

---

### 5. Trend Prediction (`predict` / `run`)

Predicts next-day direction (**UP**, **DOWN**, or **NEUTRAL**) with calibrated confidence scores and human-readable explanations.

```bash
# Predict for all tickers in config.yaml
uv run src/main.py predict

# Predict for a specific ticker
uv run src/main.py run --ticker AAPL

# Predict for a specific ticker on a specific target date
uv run src/main.py run --ticker AAPL --date 20250115
```

**Sample Output**:
```
============================================================
PREDICTION RESULT: AAPL on 2026-08-24
Predicted Direction : UP (Class 1)
Confidence Score    : 57.6%
Probabilities       : UP: 57.6% | DOWN: 30.6% | NEUTRAL: 11.8%
Explanation         : Prediction: UP (Confidence: 57.6%). Key driving factors for AAPL: RSI is neutral-to-moderately bearish at 50.0. MACD line (12.86) sits above signal line (5.76), confirming positive momentum.
============================================================
```

---

### 6. Full Pipeline Execution (`run-all`)

Runs the end-to-end workflow sequentially:
$\text{Market Ingestion} \rightarrow \text{News Ingestion} \rightarrow \text{NLP Scoring} \rightarrow \text{Feature Generation} \rightarrow \text{Model Prediction}$

```bash
uv run src/main.py run-all
```

---

## 📁 Project Directory Structure

```
article-ingestion-pipeline/
├── docker-compose.yaml                 # Local PostgreSQL 16 test container
├── config.yaml                         # Central configuration (watchlist, encodings, thresholds)
├── pyproject.toml                      # Python dependencies
├── .env.example                        # Local test environment template
├── README.md                           # Comprehensive documentation
├── source/                             # Database DDL directory (.sql only)
│   └── schema.sql                      # Standardized 4-tier SQL DDL definitions & indexes
└── src/
    ├── main.py                         # Primary CLI Entrypoint with rich arguments
    ├── core/
    │   ├── config.py                   # YAML configuration loader (dynamic from config.yaml)
    │   ├── database.py                 # DB connection pool (get_db) & SQL init (init_db)
    │   ├── logger.py                   # Dual standard console & structured JSON file logging
    │   └── settings.py                 # Environment variables (.env)
    ├── crud/
    │   ├── crud_market_prices.py       # Functions for market_prices table
    │   ├── crud_news_articles.py       # Functions for news_articles, sentiments, & tickers
    │   ├── crud_daily_stock_features.py# Functions for daily_stock_features table
    │   ├── crud_model_training.py      # Functions for model_training table
    │   └── crud_stock_predictions.py   # Functions for stock_predictions table
    ├── scraper/
    │   ├── base.py                     # Base scraper class
    │   ├── market.py                   # Yahoo Finance market collector (live & historical)
    │   ├── yahoo_rss.py                # RSS news scraper (with date window filtering)
    │   ├── fetcher.py                  # Parallel news ingestor
    │   └── helpers.py                  # HTML text cleaning
    ├── nlp/
    │   ├── analyzer.py                 # FinBERT sentiment scoring (60% title / 40% summary)
    │   └── llm_processor.py            # LLM bullet points and keyword extractor
    ├── features/
    │   ├── technical.py                # Technical indicators (MA, RSI, MACD)
    │   ├── sentiment_agg.py            # Precision news window aggregator
    │   └── generator.py                # Master Left-Join, target labeling, & Parquet export
    └── inference/
        ├── predictor.py                # Trend predictor & confidence scorer
        └── explainability.py           # Factor attribution generator
```
