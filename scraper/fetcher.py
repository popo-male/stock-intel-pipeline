import time
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser

from core.config import AppConfig
from db.repository import upload_articles
from scraper.helpers import clean_html


def fetch_news(ticker: str, rss_base_url: str) -> list[dict[str, Any]]:
    rss_url = rss_base_url.format(ticker=ticker)
    feed = feedparser.parse(rss_url)

    articles: list[dict[str, Any]] = []

    for entry in feed.entries:
        try:
            published_raw = entry.get("published", "")
            if not isinstance(published_raw, str):
                raise ValueError("Invalid or missing published date")

            title_raw = entry.get("title", "")
            link_raw = entry.get("link", "")
            summary_raw = entry.get("summary", "")

            title = title_raw if isinstance(title_raw, str) else ""
            link = link_raw if isinstance(link_raw, str) else ""
            summary = summary_raw if isinstance(summary_raw, str) else ""

            published_date = parsedate_to_datetime(published_raw)

            articles.append(
                {
                    "ticker": ticker,
                    "title": title,
                    "url": link,
                    "summary": clean_html(summary),
                    "published_at": published_date,
                    "source": "Yahoo Finance",
                }
            )
        except Exception as exc:
            print(f"Error parsing article for {ticker}: {exc}")

    return articles


def run_scraper(config: AppConfig) -> None:
    """Iterates through the watchlist and processes all news"""
    total_new = 0

    for ticket in config.scraper.watchlist:
        articles = fetch_news(ticket, config.scraper.rss_base_url)

        if articles:
            new_count = upload_articles(articles)
            total_new += new_count
            print(
                f"Fetched {len(articles)} articles for {ticket}. Inserted {new_count} new."
            )
        else:
            print(f"{ticket}: No articles found.")

        time.sleep(config.scraper.sleep_interval)

    print(f"Scraping complete! {total_new} new articles stored.")
