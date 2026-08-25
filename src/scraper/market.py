"""
Market Data Scraper and Collector Module (Yahoo Finance).
Directly extracts historical and live market prices for stocks and benchmark indices.
"""

from datetime import datetime, timedelta
from typing import Any

import pytz
import yfinance as yf

from src.core.config import AppConfig, load_config
from src.core.logger import logger
from src.crud.crud_market_prices import insert_market_prices
from src.crud.crud_news_articles import get_registered_tickers


class MarketCollector:
    """Collector for market price data (historical and live)."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        self.tz = pytz.timezone(getattr(self.config, "timezone", "Asia/Kuala_Lumpur"))

    def _get_current_time(self) -> datetime:
        """Returns current datetime localized to configured timezone (Malaysia Time)."""
        return datetime.now(self.tz)

    def fetch_live_ticker(self, ticker_symbol: str) -> dict[str, Any]:
        """Fetches latest intraday/current price data for a ticker."""
        logger.info(f"Extracting live data for {ticker_symbol}")
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.fast_info

        current_price = float(info.last_price or 0.0)
        open_price = float(info.open or current_price)
        price_change_pct = 0.0
        if open_price > 0 and current_price > 0:
            price_change_pct = ((current_price - open_price) / open_price) * 100

        now = self._get_current_time()
        trade_date = now.strftime("%Y-%m-%d")

        return {
            "ticker": ticker_symbol,
            "trade_date": trade_date,
            "open": open_price,
            "high": float(info.day_high or current_price),
            "low": float(info.day_low or current_price),
            "close": current_price,
            "adj_close": current_price,
            "price_change_pct": round(price_change_pct, 4),
            "volume": int(info.last_volume or 0),
            "market_cap": int(getattr(info, "market_cap", 0) or 0),
            "extracted_at": now.strftime("%Y-%m-%d %H:%M:%S%z"),
        }

    def fetch_historical_ticker(
        self,
        ticker_symbol: str,
        start_date: str,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetches historical daily market data for a given date range.
        Dates in format YYYY-MM-DD.
        """
        logger.info(
            f"Extracting historical market data for {ticker_symbol} from {start_date} to {end_date or 'today'}"
        )
        ticker = yf.Ticker(ticker_symbol)

        if end_date is None:
            end_dt = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=1)
            end_date = end_dt.strftime("%Y-%m-%d")
        else:
            if start_date == end_date:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
                end_date = end_dt.strftime("%Y-%m-%d")

        hist = ticker.history(start=start_date, end=end_date, auto_adjust=False)
        if hist.empty:
            logger.warning(f"No historical data returned for {ticker_symbol} between {start_date} and {end_date}")
            return []

        records: list[dict[str, Any]] = []
        market_cap = getattr(ticker.fast_info, "market_cap", None) or 0
        extracted_at_str = self._get_current_time().strftime("%Y-%m-%d %H:%M:%S%z")

        for index, row in hist.iterrows():
            trade_dt = index.to_pydatetime() if hasattr(index, "to_pydatetime") else index
            trade_date_str = trade_dt.strftime("%Y-%m-%d")

            open_val = float(row["Open"])
            close_val = float(row["Close"])
            adj_close_val = float(row.get("Adj Close", close_val))
            high_val = float(row["High"])
            low_val = float(row["Low"])
            volume_val = int(row["Volume"])

            price_change_pct = 0.0
            if open_val > 0:
                price_change_pct = ((close_val - open_val) / open_val) * 100

            records.append({
                "ticker": ticker_symbol,
                "trade_date": trade_date_str,
                "open": open_val,
                "high": high_val,
                "low": low_val,
                "close": close_val,
                "adj_close": adj_close_val,
                "price_change_pct": round(price_change_pct, 4),
                "volume": volume_val,
                "market_cap": int(market_cap),
                "extracted_at": extracted_at_str,
            })

        return records

    def collect(
        self,
        tickers: list[str] | None = None,
        target_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        live: bool = False,
    ) -> int:
        """
        Main collection execution method.
        Collects for Magnificent 7 stocks and benchmark indices (SPY, QQQ).
        """
        if tickers is None:
            tickers = self.config.market.all_symbols

        registered = get_registered_tickers()
        valid_tickers = [t for t in tickers if t in registered]
        skipped_tickers = [t for t in tickers if t not in registered]

        if skipped_tickers:
            logger.warning(
                f"Skipping unregistered tickers (not in 'tickers' table): {skipped_tickers}"
            )

        if not valid_tickers:
            logger.warning("No registered tickers found to collect market data.")
            return 0

        total_saved = 0
        logger.info(f"Starting market collection for tickers: {valid_tickers}")

        for ticker in valid_tickers:
            try:
                if live or (not target_date and not start_date):
                    record = self.fetch_live_ticker(ticker)
                    insert_market_prices([record])
                    total_saved += 1
                    logger.info(f"[{ticker}] Live data collected successfully.")
                elif target_date:
                    records = self.fetch_historical_ticker(ticker, start_date=target_date)
                    saved = insert_market_prices(records)
                    total_saved += saved
                    logger.info(f"[{ticker}] Historical data for {target_date} collected: {saved} rows.")
                elif start_date:
                    records = self.fetch_historical_ticker(
                        ticker, start_date=start_date, end_date=end_date or self.config.training.end_date
                    )
                    saved = insert_market_prices(records)
                    total_saved += saved
                    logger.info(f"[{ticker}] Historical data collected: {saved} rows.")
            except Exception as e:
                logger.error(f"Error collecting market data for {ticker}: {e}", exc_info=True)

        logger.info(f"Market collection completed. Total records saved/updated: {total_saved}")
        return total_saved
