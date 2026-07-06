import time
import uuid
from email.utils import parsedate_to_datetime
from typing import Any, Optional

import feedparser

from src.core.config import AppConfig
from src.core.logger import logger
from src.db.connection import get_db_connection
from src.db.repository import insert_article, insert_ticker, link_article_to_ticker
from src.scraper.helpers import clean_html


def fetch_news(ticker: str, rss_base_url: str) -> list[dict[str, Any]]:
    rss_url = rss_base_url.format(ticker=ticker)
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
            logger.error(f"Error parsing article for {ticker}: {exc}", exc_info=True)

    return articles


def run_scraper(config: AppConfig) -> None:
    total_new_articles = 0
    total_new_links = 0

    with get_db_connection() as conn:
        for ticker in config.scraper.watchlist:
            articles = fetch_news(ticker.symbol, config.scraper.rss_base_url)

            if not articles:
                logger.info(f"{ticker.symbol}: No active feed entries discovered.")
                time.sleep(config.scraper.sleep_interval)
                continue

            ticker_new_articles = 0
            ticker_new_links = 0

            try:
                with conn:
                    with conn.cursor() as cursor:
                        ticker_id = insert_ticker(cursor, ticker.symbol, ticker.name)

                        for article_data in articles:
                            article_id: Optional[uuid.UUID] = insert_article(
                                cursor, article_data
                            )

                            if article_id is None:
                                cursor.execute(
                                    "SELECT id FROM articles_v2 WHERE url = %s;",
                                    (article_data["url"],),
                                )
                                res = cursor.fetchone()
                                article_id = res["id"] if res else None

                            if article_id:
                                is_new_link = link_article_to_ticker(
                                    cursor, article_id, ticker_id
                                )
                                if is_new_link:
                                    ticker_new_links += 1
                                    if not article_id:
                                        ticker_new_articles += 1

                total_new_articles += ticker_new_articles
                total_new_links += ticker_new_links
                logger.info(
                    f"[{ticker.symbol}] Ingested {len(articles)} articles. (New: {ticker_new_articles}, Links: {ticker_new_links})"
                )

            except Exception as exc:
                logger.error(
                    f"Transaction block crash encountered for ticker {ticker.symbol}: {exc}",
                    exc_info=True,
                )

            time.sleep(config.scraper.sleep_interval)

    logger.info(
        f"Scraping phase complete. Ingested {total_new_articles} master documents, created {total_new_links} joins."
    )
