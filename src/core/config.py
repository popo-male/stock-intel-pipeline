from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError


class WatchlistItem(BaseModel):
    symbol: str
    name: str | None = None
    encoding: int | None = None


class IndexItem(BaseModel):
    symbol: str
    name: str | None = None


class MarketConfig(BaseModel):
    watchlist: list[WatchlistItem] = Field(default_factory=list)
    indices: list[IndexItem] = Field(default_factory=list)

    @property
    def watchlist_symbols(self) -> list[str]:
        return [item.symbol for item in self.watchlist]

    @property
    def index_symbols(self) -> list[str]:
        return [item.symbol for item in self.indices]

    @property
    def all_symbols(self) -> list[str]:
        symbols = self.watchlist_symbols.copy()
        for idx in self.index_symbols:
            if idx not in symbols:
                symbols.append(idx)
        return symbols

    @property
    def ticker_encodings(self) -> dict[str, int]:
        return {
            item.symbol: item.encoding
            for item in self.watchlist
            if item.encoding is not None
        }


class ScraperConfig(BaseModel):
    rss_base_url: str = Field(
        default="https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    )
    sleep_interval: int = Field(default=1, gt=0)
    max_workers: int = Field(default=4, gt=0)


class NLPConfig(BaseModel):
    sentiment_model: str = Field(default="ProsusAI/finbert")
    title_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    summary_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    bullish_threshold: float = Field(default=0.05)
    bearish_threshold: float = Field(default=-0.05)


class FeatureConfig(BaseModel):
    lookback_days: int = Field(default=20, gt=0)
    cold_start_rows: int = Field(default=20, gt=0)
    target_threshold_pct: float = Field(default=1.0)
    rsi_period: int = Field(default=14, gt=0)
    macd_fast: int = Field(default=12, gt=0)
    macd_slow: int = Field(default=26, gt=0)
    macd_signal: int = Field(default=9, gt=0)


class TargetDefinitions(BaseModel):
    up_code: int = Field(default=1)
    down_code: int = Field(default=0)
    direction_up: str = Field(default="UP")
    direction_down: str = Field(default="DOWN")
    direction_neutral: str = Field(default="NEUTRAL")


class TrainingConfig(BaseModel):
    start_date: str = Field(default="2023-01-01")
    end_date: str = Field(default="2025-12-31")
    output_parquet: str = Field(default="data/training_dataset.parquet")
    train_split: float = Field(default=0.8)
    val_split: float = Field(default=0.1)
    test_split: float = Field(default=0.1)


class ModelConfig(BaseModel):
    version: str = Field(default="v1.0.0")


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")
    name: str = Field(default="stock_intelligence_pipeline")
    path: str = Field(default="log/out.log")


class AppConfig(BaseModel):
    timezone: str = Field(default="Asia/Kuala_Lumpur")
    market: MarketConfig = Field(default_factory=MarketConfig)
    scraper: ScraperConfig = Field(default_factory=ScraperConfig)
    nlp: NLPConfig = Field(default_factory=NLPConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    target_definitions: TargetDefinitions = Field(default_factory=TargetDefinitions)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @property
    def ticker_encodings(self) -> dict[str, int]:
        return self.market.ticker_encodings


def load_config(config_path: Path | str | None = None) -> AppConfig:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return AppConfig()

    with config_path.open("r", encoding="utf-8") as f:
        raw_config: dict[str, Any] = yaml.safe_load(f) or {}

    try:
        return AppConfig.model_validate(raw_config)
    except ValidationError as exc:
        raise ValueError(f"Invalid configuration in {config_path}: {exc}") from exc
