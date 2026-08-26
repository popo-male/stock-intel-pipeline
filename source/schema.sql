-- ====================================================================
-- Stock Intelligence Platform - Database Schema with TimescaleDB
-- ====================================================================
-- Extension: TimescaleDB enabled
-- Chunk Time Interval: 1 Month (INTERVAL '1 month')
-- Compression Policy: 2 Months (INTERVAL '2 month')
-- Retention Policy: None (historical data stored indefinitely)
-- ====================================================================

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ==========================================
-- 1. RAW DATA LAYER
-- ==========================================

-- Tickers Registry (Standard PostgreSQL Table)
CREATE TABLE IF NOT EXISTS tickers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(10) NOT NULL UNIQUE,
    name TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Raw Market Prices (Timescale Hypertable)
CREATE TABLE IF NOT EXISTS market_prices (
    ticker VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    open NUMERIC(14, 4),
    high NUMERIC(14, 4),
    low NUMERIC(14, 4),
    close NUMERIC(14, 4),
    adj_close NUMERIC(14, 4),
    volume BIGINT,
    price_change_pct NUMERIC(8, 4),
    market_cap BIGINT,
    extracted_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_market_prices_ticker_date ON market_prices (ticker, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_market_prices_trade_date ON market_prices (trade_date DESC);

-- Convert to Hypertable (1 Month Chunks)
SELECT create_hypertable('market_prices', 'trade_date', chunk_time_interval => INTERVAL '1 month', if_not_exists => TRUE);

-- Columnar Compression Policy (2 Months)
ALTER TABLE market_prices SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'ticker',
    timescaledb.compress_orderby = 'trade_date DESC'
);
SELECT add_compression_policy('market_prices', INTERVAL '2 month', if_not_exists => TRUE);


-- Raw News Articles (Standard PostgreSQL Table for URL Deduplication)
CREATE TABLE IF NOT EXISTS news_articles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    url TEXT NOT NULL UNIQUE,
    summary TEXT,
    published_at TIMESTAMPTZ NOT NULL,
    source VARCHAR(100) DEFAULT 'Yahoo Finance',
    bullets JSONB,
    keywords JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_news_articles_published_at ON news_articles (published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_articles_unsummarized ON news_articles (id) WHERE bullets IS NULL;

-- Article <-> Ticker Join Table (Standard PostgreSQL Table)
CREATE TABLE IF NOT EXISTS article_tickers (
    article_id UUID NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
    ticker_id UUID REFERENCES tickers(id) ON DELETE SET NULL,
    ticker_symbol VARCHAR(10) NOT NULL,
    PRIMARY KEY (article_id, ticker_symbol)
);

CREATE INDEX IF NOT EXISTS idx_article_tickers_symbol ON article_tickers (ticker_symbol);
CREATE INDEX IF NOT EXISTS idx_article_tickers_article_id ON article_tickers (article_id);

-- Article-Level Sentiment Results (Timescale Hypertable)
CREATE TABLE IF NOT EXISTS news_sentiments (
    article_id UUID NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
    ticker VARCHAR(10) NOT NULL,
    published_at TIMESTAMPTZ NOT NULL,
    sentiment_score REAL NOT NULL,
    sentiment_label VARCHAR(20) NOT NULL,
    model_version VARCHAR(50) DEFAULT 'ProsusAI/finbert',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (article_id, ticker, published_at)
);

CREATE INDEX IF NOT EXISTS idx_news_sentiments_ticker_pub ON news_sentiments (ticker, published_at DESC);

-- Convert to Hypertable (1 Month Chunks)
SELECT create_hypertable('news_sentiments', 'published_at', chunk_time_interval => INTERVAL '1 month', if_not_exists => TRUE);

-- Columnar Compression Policy (2 Months)
ALTER TABLE news_sentiments SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'ticker',
    timescaledb.compress_orderby = 'published_at DESC'
);
SELECT add_compression_policy('news_sentiments', INTERVAL '2 month', if_not_exists => TRUE);


-- ==========================================
-- 2. PROCESSED / FEATURE LAYER
-- ==========================================

-- Combined Daily Stock Features (Timescale Hypertable)
CREATE TABLE IF NOT EXISTS daily_stock_features (
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    ticker_encoded INT NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    adj_close DOUBLE PRECISION,
    volume BIGINT,
    daily_return DOUBLE PRECISION,
    price_change_pct DOUBLE PRECISION,
    volume_change_pct DOUBLE PRECISION,
    ma5 DOUBLE PRECISION,
    ma10 DOUBLE PRECISION,
    ma20 DOUBLE PRECISION,
    rsi DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    close_to_ma5 DOUBLE PRECISION,
    close_to_ma10 DOUBLE PRECISION,
    close_to_ma20 DOUBLE PRECISION,
    high_low_spread DOUBLE PRECISION,
    open_close_spread DOUBLE PRECISION,
    spy_return DOUBLE PRECISION,
    qqq_return DOUBLE PRECISION,
    avg_sentiment DOUBLE PRECISION DEFAULT 0.0,
    max_sentiment DOUBLE PRECISION DEFAULT 0.0,
    min_sentiment DOUBLE PRECISION DEFAULT 0.0,
    positive_news_count INT DEFAULT 0,
    negative_news_count INT DEFAULT 0,
    neutral_news_count INT DEFAULT 0,
    news_count INT DEFAULT 0,
    avg_sentiment_t_1 DOUBLE PRECISION DEFAULT 0.0,
    news_count_t_1 INT DEFAULT 0,
    avg_sentiment_3d DOUBLE PRECISION DEFAULT 0.0,
    news_count_3d INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_daily_stock_features_ticker_date ON daily_stock_features (ticker, date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_stock_features_date ON daily_stock_features (date DESC);

-- Convert to Hypertable (1 Month Chunks)
SELECT create_hypertable('daily_stock_features', 'date', chunk_time_interval => INTERVAL '1 month', if_not_exists => TRUE);

-- Columnar Compression Policy (2 Months)
ALTER TABLE daily_stock_features SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'ticker',
    timescaledb.compress_orderby = 'date DESC'
);
SELECT add_compression_policy('daily_stock_features', INTERVAL '2 month', if_not_exists => TRUE);


-- ==========================================
-- 3. MODEL TRAINING LAYER
-- ==========================================

-- Model Training Dataset (Timescale Hypertable)
CREATE TABLE IF NOT EXISTS model_training (
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    ticker_encoded INT NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    adj_close DOUBLE PRECISION,
    volume BIGINT,
    daily_return DOUBLE PRECISION,
    price_change_pct DOUBLE PRECISION,
    volume_change_pct DOUBLE PRECISION,
    ma5 DOUBLE PRECISION,
    ma10 DOUBLE PRECISION,
    ma20 DOUBLE PRECISION,
    rsi DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    close_to_ma5 DOUBLE PRECISION,
    close_to_ma10 DOUBLE PRECISION,
    close_to_ma20 DOUBLE PRECISION,
    high_low_spread DOUBLE PRECISION,
    open_close_spread DOUBLE PRECISION,
    spy_return DOUBLE PRECISION,
    qqq_return DOUBLE PRECISION,
    avg_sentiment DOUBLE PRECISION DEFAULT 0.0,
    max_sentiment DOUBLE PRECISION DEFAULT 0.0,
    min_sentiment DOUBLE PRECISION DEFAULT 0.0,
    positive_news_count INT DEFAULT 0,
    negative_news_count INT DEFAULT 0,
    neutral_news_count INT DEFAULT 0,
    news_count INT DEFAULT 0,
    avg_sentiment_t_1 DOUBLE PRECISION DEFAULT 0.0,
    news_count_t_1 INT DEFAULT 0,
    avg_sentiment_3d DOUBLE PRECISION DEFAULT 0.0,
    news_count_3d INT DEFAULT 0,
    target INT NOT NULL,
    target_class VARCHAR(10) NOT NULL,
    target_return DOUBLE PRECISION,
    split_type VARCHAR(20) DEFAULT 'train',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_model_training_ticker_date ON model_training (ticker, date DESC);
CREATE INDEX IF NOT EXISTS idx_model_training_split ON model_training (split_type);

-- Convert to Hypertable (1 Month Chunks)
SELECT create_hypertable('model_training', 'date', chunk_time_interval => INTERVAL '1 month', if_not_exists => TRUE);

-- Columnar Compression Policy (2 Months)
ALTER TABLE model_training SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'ticker',
    timescaledb.compress_orderby = 'date DESC'
);
SELECT add_compression_policy('model_training', INTERVAL '2 month', if_not_exists => TRUE);


-- ==========================================
-- 4. PREDICTION LAYER
-- ==========================================

-- Short-Term Direction Predictions with Structured Signals (Timescale Hypertable)
CREATE TABLE IF NOT EXISTS stock_predictions (
    id UUID DEFAULT gen_random_uuid(),
    ticker VARCHAR(10) NOT NULL,
    prediction_date DATE NOT NULL,
    target_direction VARCHAR(10) NOT NULL,
    predicted_class INT NOT NULL,
    confidence_score DOUBLE PRECISION NOT NULL,
    probability_up DOUBLE PRECISION NOT NULL,
    probability_down DOUBLE PRECISION NOT NULL,
    probability_neutral DOUBLE PRECISION DEFAULT 0.0,
    signal JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, prediction_date)
);

CREATE INDEX IF NOT EXISTS idx_stock_predictions_ticker_date ON stock_predictions (ticker, prediction_date DESC);
CREATE INDEX IF NOT EXISTS idx_stock_predictions_date ON stock_predictions (prediction_date DESC);
CREATE INDEX IF NOT EXISTS idx_stock_predictions_signal ON stock_predictions USING GIN (signal);

-- Convert to Hypertable (1 Month Chunks)
SELECT create_hypertable('stock_predictions', 'prediction_date', chunk_time_interval => INTERVAL '1 month', if_not_exists => TRUE);

-- Columnar Compression Policy (2 Months)
ALTER TABLE stock_predictions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'ticker',
    timescaledb.compress_orderby = 'prediction_date DESC'
);
SELECT add_compression_policy('stock_predictions', INTERVAL '2 month', if_not_exists => TRUE);
