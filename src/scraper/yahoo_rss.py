from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser

from src.core.config import AppConfig
from src.core.logger import logger
from src.scraper.base import BaseNewsScraper
from src.scraper.helpers import clean_html


class YahooRSSScraper(BaseNewsScraper):
    """Scraper fetching financial news feeds from Yahoo Finance RSS."""

    def __init__(self, config: AppConfig):
        super().__init__(config)

    def fetch_news(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        rss_url = self.config.scraper.rss_base_url.format(ticker=ticker)
        logger.info(f"Fetching RSS feed for {ticker} from {rss_url}")
        feed = feedparser.parse(rss_url)
        articles: list[dict[str, Any]] = []

        start_dt = None
        end_dt = None
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
                hour=23, minute=59, second=59, tzinfo=UTC
            )

        raw_entries_count = len(feed.entries)

        for entry in feed.entries:
            try:
                published_raw = entry.get("published", "")
                if not isinstance(published_raw, str) or not published_raw:
                    published_at = datetime.now(UTC)
                else:
                    published_at = parsedate_to_datetime(published_raw)
                    if published_at.tzinfo is None:
                        published_at = published_at.replace(tzinfo=UTC)

                # Filter by date range if provided
                if start_dt and published_at < start_dt:
                    continue
                if end_dt and published_at > end_dt:
                    continue

                title = clean_html(str(entry.get("title", "")))
                url = str(entry.get("link", ""))
                summary = clean_html(str(entry.get("summary", "")))

                if not title or not url:
                    continue

                articles.append(
                    {
                        "ticker": ticker,
                        "title": title,
                        "url": url,
                        "summary": summary,
                        "published_at": published_at,
                        "source": "Yahoo Finance",
                    }
                )
            except Exception as exc:
                logger.error(
                    f"Error parsing RSS article entry for {ticker}: {exc}",
                    exc_info=True,
                )

        if raw_entries_count > 0 and len(articles) == 0 and (start_date or end_date):
            logger.info(
                f"[{ticker}] Feed contained {raw_entries_count} live articles, but 0 fell in range [{start_date} to {end_date}]. "
            )
        else:
            logger.info(
                f"[{ticker}] Fetched {len(articles)} raw articles matching criteria."
            )

        return articles
