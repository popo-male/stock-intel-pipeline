from abc import ABC, abstractmethod
from typing import Any

from src.core.config import AppConfig


class BaseNewsScraper(ABC):
    """Abstract Base Class specifying contracts for all news engine scrapers."""

    def __init__(self, config: AppConfig):
        self.config = config

    @abstractmethod
    def fetch_news(self, ticker: str) -> list[dict[str, Any]]:
        """Extract articles from source and yield standardized platform dictionary shapes."""
