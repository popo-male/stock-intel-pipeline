from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core.config import AppConfig, load_config
from src.core.logger import logger
from src.crud.crud_news_articles import (
    get_registered_tickers,
    insert_article,
    is_ticker_registered,
    link_article_to_ticker,
)
from src.scraper.yahoo_rss import YahooRSSScraper


class NewsFetcher:
    """Orchestrator for scraping financial news and storing into raw news tables."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()

    def process_single_ticker(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> tuple[int, int]:
        """Fetches and inserts news articles for a single ticker."""
        if not is_ticker_registered(ticker):
            logger.warning(
                f"[{ticker}] Ticker is not registered or active in 'tickers' table. Skipping news scraping."
            )
            return 0, 0

        scraper = YahooRSSScraper(self.config)
        articles = scraper.fetch_news(ticker, start_date=start_date, end_date=end_date)

        if not articles:
            return 0, 0

        new_articles = 0
        new_links = 0

        for article_data in articles:
            try:
                article_id = insert_article(article_data)
                if article_id:
                    is_linked = link_article_to_ticker(article_id, ticker)
                    if is_linked:
                        new_links += 1
                    new_articles += 1
            except Exception as e:
                logger.error(
                    f"Error saving article '{article_data.get('title')}' for {ticker}: {e}"
                )

        logger.info(
            f"[{ticker}] Processed {len(articles)} articles (Saved: {new_articles}, Links: {new_links})"
        )
        return new_articles, new_links

    def fetch_all(
        self,
        tickers: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> tuple[int, int]:
        """Runs parallel fetching across all target tickers."""
        if tickers is None:
            tickers = self.config.market.watchlist_symbols

        registered = get_registered_tickers()
        valid_tickers = [t for t in tickers if t in registered]
        skipped_tickers = [t for t in tickers if t not in registered]

        if skipped_tickers:
            logger.warning(
                f"Skipping unregistered tickers (not in 'tickers' table): {skipped_tickers}"
            )

        if not valid_tickers:
            logger.warning("No registered tickers found to fetch news.")
            return 0, 0

        total_articles = 0
        total_links = 0
        max_workers = getattr(self.config.scraper, "max_workers", 4)

        logger.info(
            f"Starting news scraping for tickers: {valid_tickers} with {max_workers} threads..."
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(
                    self.process_single_ticker, ticker, start_date, end_date
                ): ticker
                for ticker in valid_tickers
            }

            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    articles_cnt, links_cnt = future.result()
                    total_articles += articles_cnt
                    total_links += links_cnt
                except Exception as exc:
                    logger.error(
                        f"Thread worker for ticker {ticker} generated exception: {exc}"
                    )

        logger.info(
            f"News scraping completed: {total_articles} articles processed, {total_links} ticker links established."
        )
        return total_articles, total_links


def run_scraper(
    config: AppConfig | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    fetcher = NewsFetcher(config)
    fetcher.fetch_all(start_date=start_date, end_date=end_date)
