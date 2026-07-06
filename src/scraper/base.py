from abc import ABC, abstractmethod
from typing import Any

from src.core.config import AppConfig


class BaseScraper(ABC):
    """Abstract Base Class specifying contracts for all downstream news engine ingestors."""
    def __init__(self, config: AppConfig):
        self.config = config

    @abstractmethod
    def fetch_news(self, ticker: str) -> list[dict[str, Any]]:
        """Extract elements from source target and yield uniform platform dictionary shapes."""
        pass