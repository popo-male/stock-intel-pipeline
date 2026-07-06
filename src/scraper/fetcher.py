from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from src.core.config import AppConfig
from src.core.logger import logger
from src.db.connection import get_db_connection
from src.db.repository import insert_article, insert_ticker, link_article_to_ticker
from src.scraper.yahoo_rss import YahooRSSScraper


def process_single_ticker(ticker_symbol: str, config: AppConfig) -> tuple[int, int]:
    """Worker function executed in parallel threads to ingest news"""
    ticker_new_articles = 0
    ticker_new_links = 0

    scraper = YahooRSSScraper(config)
    articles = scraper.fetch_news(ticker_symbol)

    if not articles:
        return 0, 0

    with get_db_connection() as conn:
        try:
            with conn.transaction():
                with conn.cursor() as cursor:
                    ticker_id = insert_ticker(cursor, ticker_symbol)

                    for article_data in articles:
                        article_id: Optional[str] = insert_article(cursor, article_data)
                        is_brand_new = True

                        if article_id is None:
                            is_brand_new = False
                            cursor.execute(
                                "SELECT id FROM articles_v2 WHERE url = %s;",
                                (article_data["url"],),
                            )
                            res = cursor.fetchone()
                            article_id = str(res["id"]) if res else None

                        if article_id:
                            is_new_link = link_article_to_ticker(
                                cursor, article_id, ticker_id
                            )
                            if is_new_link:
                                ticker_new_links += 1
                                if is_brand_new:
                                    ticker_new_articles += 1

            logger.info(
                f"[{ticker_symbol}] Processed {len(articles)} articles. (New: {ticker_new_articles}, Links: {ticker_new_links})"
            )
        except Exception as exc:
            logger.error(
                f"Parallel database block crash for ticker {ticker_symbol}: {exc}",
                exc_info=True,
            )

    return ticker_new_articles, ticker_new_links


def run_scraper(config: AppConfig) -> None:
    total_new_articles = 0
    total_new_links = 0

    logger.info("Starting news scraping phase...")

    watchlist_symbols = []
    for item in config.scraper.watchlist:
        if isinstance(item, dict):
            watchlist_symbols.append(item.get("symbol"))
        elif hasattr(item, "symbol"):
            watchlist_symbols.append(item.symbol)
        else:
            watchlist_symbols.append(str(item))

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_ticker = {
            executor.submit(process_single_ticker, ticker, config): ticker
            for ticker in watchlist_symbols
        }

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                new_articles, new_links = future.result()
                total_new_articles += new_articles
                total_new_links += new_links
            except Exception as exc:
                logger.error(
                    f"Thread worker for ticker {ticker} generated an exception: {exc}"
                )

    logger.info(
        f"Scraping phase complete. Ingested {total_new_articles} master documents, created {total_new_links} joins."
    )
