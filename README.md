# Stock Intelligence Platform - Ingestion & Prediction Pipeline

A high-performance, data-driven platform that ingests daily stock market data and financial news, performs sentiment analysis and precision window aggregations, engineers scale-invariant technical & macro features, and generates explainable short-term stock trend predictions with rule-based signals.

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
  - [4. Feature Engineering (`generate-features`)](#4-feature-engineering-generate-features)
  - [5. Training Dataset Generation (`training-data`)](#5-training-dataset-generation-training-data)
  - [6. Trend Prediction (`predict` / `run`)](#6-trend-prediction-predict--run)
  - [7. Full Pipeline Execution (`run-all`)](#7-full-pipeline-execution-run-all)
- [Project Directory Structure](#-project-directory-structure)

---

## 📚 Detailed Documentation Index

For in-depth mathematical formulations, workflows, and module-specific guides, refer to the dedicated documents in [`docs/`](docs/):

1. 🏛️ [**System Architecture & Pipeline Overview**](docs/architecture.md) — End-to-end data flow, execution sequence, and design principles.
2. 🗄️ [**Database Design & Schema Architecture**](docs/database_design.md) — 4-tier standardized database layers, JSONB signals, and function-only CRUD architecture.
3. 📡 [**Data Collector Architecture**](docs/data_collector.md) — Yahoo Finance market prices and multi-threaded parallel RSS feed ingestion.
4. 🧠 [**News Sentiment Analysis & LLM Insights**](docs/news_sentiment.md) — FinBERT compound scoring $[-1.0, 1.0]$ and Groq LLM executive summaries.
5. ⚙️ [**Feature Engineering & Scale-Invariance**](docs/feature_engineering.md) — Precision news window rules, TA indicators, scale-invariant relative spreads, and Left-Join store.
6. 🎯 [**Training Dataset Pipeline**](docs/training_pipeline.md) — Target return labeling, chronological train/val/test partitions, and Parquet export.
7. 🔮 [**Prediction Engine & Rule-Based Signals**](docs/prediction.md) — Next-day trend inference, probability calibration, and structured JSONB technical signals.

---

## 🏛️ Overview & Architecture

The system combines structured daily price movements from Yahoo Finance and unstructured financial news articles into a unified feature matrix, isolating pre-market information to predict whether a stock will move **UP** or **DOWN** on the next trading day.

```
                                  ┌─────────────────────────────────────────────────┐
                                  │                  DATA SOURCES                   │
                                  │   Yahoo Finance (Prices & RSS News Feeds)       │
                                  └────────────────────────┬────────────────────────┘
                                                           │
                                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. RAW DATA LAYER (source/schema.sql)                                                                            │
│   • tickers          : Tracked assets (Magnificent 7 + Indices) auto-seeded from config.yaml                     │
│   • market_prices    : Raw daily OHLCV, adjusted close, volume, price change %, and market cap                   │
│   • news_articles    : Raw scraped news articles, body text, and LLM bullet point highlights                     │
│   • article_tickers  : Many-to-many link between news articles and tickers                                       │
│   • news_sentiments  : Article sentiment scores (s ∈ [-1.0, 1.0]) & labels generated via FinBERT                 │
└──────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. PROCESSED / FEATURE STORE LAYER (source/schema.sql)                                                           │
│   • daily_stock_features : Technical indicators (MA, RSI, MACD), scale-invariant spreads, macro returns          │
│                            joined with Precision News Windows (T-1 4:00 PM EST to T 9:30 AM EST, 3d rolling)     │
└──────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
┌──────────────────────────────────────────────┐    ┌──────────────────────────────────────────────┐
│ 3. MODEL TRAINING LAYER                      │    │ 4. PREDICTION LAYER                          │
│   • model_training   : Features + target     │    │   • stock_predictions: Direction (UP/DOWN),  │
│                        labels & train/val/   │    │                        confidence %, probs,  │
│                        test splits           │    │                        and JSONB signals     │
│   • Parquet Export   : training_dataset.     │    │   • Real-time inference without lookahead    │
│                        parquet               │    │     bias                                     │
└──────────────────────────────────────────────┘    └──────────────────────────────────────────────┘
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
| **Processed** | `daily_stock_features` | Engineered indicators + scale-invariant spreads + precision news window aggregations. |
| **Training** | `model_training` | Master dataset joined with directional targets and train/val/test splits. |
| **Prediction** | `stock_predictions` | Predicted direction, confidence %, probability distribution, and structured JSONB signals. |

---

## 🐳 Quick Start with Local Database (Docker Compose)

To test and run locally without touching production databases:

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
  start_date: "2026-01-01"
  end_date: "2026-08-25"
  output_parquet: "data/training_dataset.parquet"
```

---

## 🚀 CLI Command Reference

Execute commands using `uv run src/main.py <command>`. Both `YYYYMMDD` (e.g. `20260101`) and `YYYY-MM-DD` (e.g. `2026-01-01`) date formats are supported.

### 1. Database Initialization (`init-db`)

Initializes all database tables, constraints, and indexes from `source/*.sql` and synchronizes tickers from `config.yaml`.

> **Idempotent & Safe**: Running `init-db` multiple times will **not** drop tables or overwrite existing market data.

```bash
uv run src/main.py init-db
```

---

### 2. Market Data Ingestion (`fetch-market`)

Extracts historical daily OHLCV prices and volume from Yahoo Finance.

```bash
# Fetch specific ticker across date range
uv run src/main.py fetch-market --ticker AAPL --start 20260101 --end 20260825

# Fetch all tickers in config.yaml across date range
uv run src/main.py fetch-market --start 20260101 --end 20260825

# Fetch all tickers using default config time range
uv run src/main.py fetch-market

# Fetch a single specific day
uv run src/main.py fetch-market --ticker AAPL --date 20260825

# Fetch live intraday snapshot
uv run src/main.py fetch-market --live
```

---

### 3. News & Sentiment Ingestion (`fetch-news`)

Scrapes Yahoo Finance RSS feeds, computes FinBERT sentiment scores, and extracts LLM insights.

```bash
# Ingest news for specific ticker across date range
uv run src/main.py fetch-news --ticker AAPL --start 20260101 --end 20260825

# Ingest news for all tickers across date range
uv run src/main.py fetch-news --start 20260101 --end 20260825

# Ingest all latest RSS articles for all tickers
uv run src/main.py fetch-news

# Skip LLM summary extraction (faster news + FinBERT scoring only)
uv run src/main.py fetch-news --skip-llm
```

---

### 4. Feature Engineering (`generate-features` / `features`)

Calculates scale-invariant technical indicators, benchmark returns, and overnight news sentiment aggregations, and saves directly into the `daily_stock_features` table.

```bash
# Generate daily feature store for all tickers
uv run src/main.py generate-features --start 20260101 --end 20260825

# Generate daily feature store for a single ticker
uv run src/main.py generate-features --ticker AAPL --start 20260101 --end 20260825
```

---

### 5. Training Dataset Generation (`training-data`)

Computes directional future target labels (Day $T$ movement), assigns chronological train/validation/test splits, upserts to the `model_training` table, and exports the final Parquet file.

```bash
# Generate training dataset using default output path (data/training_dataset.parquet)
uv run src/main.py training-data

# Specify custom Parquet output file
uv run src/main.py training-data --output data/custom_training_dataset.parquet

# Generate for a custom date range
uv run src/main.py training-data --start 20260101 --end 20260825
```

---

### 6. Trend Prediction (`predict` / `run`)

Predicts next-day direction (**UP**, **DOWN**, or **NEUTRAL**), calibrated confidence scores, and rule-based technical signals (stored as `JSONB` in `stock_predictions`).

```bash
# Predict for all tickers in watchlist
uv run src/main.py predict

# Predict for a specific ticker
uv run src/main.py predict --ticker AAPL

# Predict for a specific historical or target date
uv run src/main.py predict --ticker AAPL --date 20260825
```

---

### 7. Full Pipeline Execution (`run-all`)

Runs the end-to-end workflow sequentially:
$\text{Market Ingestion} \rightarrow \text{News Ingestion} \rightarrow \text{Feature Engineering} \rightarrow \text{Training Dataset Generation} \rightarrow \text{Prediction}$

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
├── README.md                           # Master repository documentation
├── source/                             # Database DDL directory (.sql only)
│   └── schema.sql                      # Standardized 4-tier SQL DDL definitions & indexes
├── docs/                               # Detailed architectural and mathematical docs
│   ├── architecture.md                 # System overview and execution flow
│   ├── database_design.md              # 4-tier database schema & JSONB signal design
│   ├── data_collector.md               # Market price & RSS news collectors
│   ├── news_sentiment.md               # FinBERT scoring & LLM insights
│   ├── feature_engineering.md          # Technical features, scale invariance & news windows
│   ├── training_pipeline.md            # Target labeling, splits, and Parquet export
│   └── prediction.md                   # Prediction engine & rule-based signals
└── src/
    ├── main.py                         # Primary CLI Entrypoint
    ├── core/
    │   ├── config.py                   # YAML configuration loader
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
    │   ├── technical.py                # Pure technical indicators (MA, RSI, MACD)
    │   ├── aggregation.py              # Pure precision news window aggregator
    │   └── generator.py                # Pure functional feature engineering & training dataset generator
    └── inference/
        └── predict.py                  # Pure functional trend prediction & rule-based signal generator
```

