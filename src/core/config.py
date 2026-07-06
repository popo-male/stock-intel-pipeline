from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError


class WatchlistItem(BaseModel):
    symbol: str
    name: str | None = None


class ScraperConfig(BaseModel):
    watchlist: list[WatchlistItem] = Field(default_factory=list)
    rss_base_url: str
    sleep_interval: int = Field(default=1, gt=0)

class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")
    name: str = Field(default="news_ingest_pipeline")
    path: str = Field(default="log/out.log")


class AppConfig(BaseModel):
    scraper: ScraperConfig = Field(default_factory=ScraperConfig)  # type: ignore
    logging: LoggingConfig = Field(default_factory=LoggingConfig)  # type: ignore


def load_config() -> AppConfig:
    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    if not config_path.exists():
        return AppConfig()

    with config_path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    try:
        return AppConfig.model_validate(raw_config)
    except ValidationError as exc:
        raise ValueError(f"Invalid configuration in {config_path}: {exc}") from exc
