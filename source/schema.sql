-- ==========================================
-- 1. RAW DATA LAYER
-- ==========================================

-- Tickers Registry
CREATE TABLE IF NOT EXISTS tickers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(10) NOT NULL UNIQUE,
    name TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Raw Market Prices
CREATE TABLE IF NOT EXISTS market_prices (
    id BIGSERIAL PRIMARY KEY,
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
    CONSTRAINT uq_market_prices_ticker_date UNIQUE (ticker, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_market_prices_ticker_date ON market_prices (ticker, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_market_prices_trade_date ON market_prices (trade_date DESC);

-- Raw News Articles
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

-- Article <-> Ticker Join Table
CREATE TABLE IF NOT EXISTS article_tickers (
    article_id UUID NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
    ticker_id UUID REFERENCES tickers(id) ON DELETE SET NULL,
    ticker_symbol VARCHAR(10) NOT NULL,
    PRIMARY KEY (article_id, ticker_symbol)
);

CREATE INDEX IF NOT EXISTS idx_article_tickers_symbol ON article_tickers (ticker_symbol);
CREATE INDEX IF NOT EXISTS idx_article_tickers_article_id ON article_tickers (article_id);

-- Article-Level Sentiment Results
CREATE TABLE IF NOT EXISTS news_sentiments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
    ticker VARCHAR(10) NOT NULL,
    published_at TIMESTAMPTZ NOT NULL,
    sentiment_score REAL NOT NULL,
    sentiment_label VARCHAR(20) NOT NULL,
    model_version VARCHAR(50) DEFAULT 'ProsusAI/finbert',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_news_sentiments_article_ticker UNIQUE (article_id, ticker)
);

CREATE INDEX IF NOT EXISTS idx_news_sentiments_ticker_pub ON news_sentiments (ticker, published_at DESC);


-- ==========================================
-- 2. PROCESSED / FEATURE LAYER
-- ==========================================

-- Combined Daily Stock Features (Market + News Sentiment Precision Windows)
CREATE TABLE IF NOT EXISTS daily_stock_features (
    id BIGSERIAL PRIMARY KEY,
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
    CONSTRAINT uq_daily_stock_features_date_ticker UNIQUE (date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_daily_stock_features_ticker_date ON daily_stock_features (ticker, date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_stock_features_date ON daily_stock_features (date DESC);


-- ==========================================
-- 3. MODEL TRAINING LAYER
-- ==========================================

-- Model Training Dataset (Features joined with future Direction Target)
CREATE TABLE IF NOT EXISTS model_training (
    id BIGSERIAL PRIMARY KEY,
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
    CONSTRAINT uq_model_training_date_ticker UNIQUE (date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_model_training_ticker_date ON model_training (ticker, date DESC);
CREATE INDEX IF NOT EXISTS idx_model_training_split ON model_training (split_type);


-- ==========================================
-- 4. PREDICTION LAYER
-- ==========================================

-- Short-Term Direction Predictions with Structured Signals & Confidence Scoring
CREATE TABLE IF NOT EXISTS stock_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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
    CONSTRAINT uq_stock_predictions_ticker_date UNIQUE (ticker, prediction_date)
);

CREATE INDEX IF NOT EXISTS idx_stock_predictions_ticker_date ON stock_predictions (ticker, prediction_date DESC);
CREATE INDEX IF NOT EXISTS idx_stock_predictions_date ON stock_predictions (prediction_date DESC);
CREATE INDEX IF NOT EXISTS idx_stock_predictions_signal ON stock_predictions USING GIN (signal);
