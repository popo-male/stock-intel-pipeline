from email.utils import parsedate_to_datetime
from typing import Any

import feedparser

from src.core.logger import logger
from src.scraper.helpers import clean_html


class YahooRSSScraper:
    def __init__(self, config):
        self.config = config

    def fetch_news(self, ticker: str) -> list[dict[str, Any]]:
        rss_url = self.config.scraper.rss_base_url.format(ticker=ticker)
        feed = feedparser.parse(rss_url)
        articles: list[dict[str, Any]] = []

        for entry in feed.entries:
            try:
                published_raw = entry.get("published", "")
                if not isinstance(published_raw, str) or not published_raw:
                    continue

                articles.append(
                    {
                        "ticker": ticker,
                        "title": str(entry.get("title", "")),
                        "url": str(entry.get("link", "")),
                        "summary": clean_html(str(entry.get("summary", ""))),
                        "published_at": parsedate_to_datetime(published_raw),
                        "source": "Yahoo Finance",
                    }
                )
            except Exception as exc:
                logger.error(
                    f"Error parsing article for {ticker}: {exc}", exc_info=True
                )

        return articles
