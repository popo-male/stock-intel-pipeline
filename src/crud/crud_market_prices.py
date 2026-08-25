"""
CRUD operations for market_prices table (functions only).
"""

from datetime import date
from typing import Any

import pandas as pd

from src.core.database import get_db


def insert_market_prices(records: list[dict[str, Any]]) -> int:
    """
    Batch upserts market price records into market_prices table.
    """
    if not records:
        return 0

    query = """
    INSERT INTO market_prices (
        ticker, trade_date, open, high, low, close, adj_close, volume, price_change_pct, market_cap, extracted_at
    ) VALUES (
        %(ticker)s, %(trade_date)s, %(open)s, %(high)s, %(low)s, %(close)s, %(adj_close)s, %(volume)s, %(price_change_pct)s, %(market_cap)s, %(extracted_at)s
    )
    ON CONFLICT (ticker, trade_date)
    DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        adj_close = EXCLUDED.adj_close,
        volume = EXCLUDED.volume,
        price_change_pct = EXCLUDED.price_change_pct,
        market_cap = EXCLUDED.market_cap,
        extracted_at = EXCLUDED.extracted_at;
    """
    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(query, records)
            return len(records)


def get_market_prices_df(
    ticker: str,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
) -> pd.DataFrame:
    """
    Retrieves historical market prices for a given ticker as a pandas DataFrame.
    """
    query = "SELECT ticker, trade_date, open, high, low, close, adj_close, volume, price_change_pct, market_cap FROM market_prices WHERE ticker = %s"
    params: list[Any] = [ticker]

    if start_date:
        query += " AND trade_date >= %s"
        params.append(start_date)
    if end_date:
        query += " AND trade_date <= %s"
        params.append(end_date)

    query += " ORDER BY trade_date ASC"

    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for col in ["open", "high", "low", "close", "adj_close", "price_change_pct"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(float)
    return df


def get_latest_trade_date(ticker: str) -> date | None:
    """Returns the most recent trade date available for a ticker."""
    query = "SELECT MAX(trade_date) as max_date FROM market_prices WHERE ticker = %s"
    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (ticker,))
            row = cursor.fetchone()
            return row["max_date"] if row and row["max_date"] else None
