"""
Technical indicator computation module.
Calculates technical indicators on Day T-1 price data to eliminate lookahead bias.
"""

import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates standard Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series]:
    """Calculates MACD line and MACD Signal line."""
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line, macd_signal


def compute_price_features(
    df: pd.DataFrame,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> pd.DataFrame:
    """
    Computes all price, volume, volatility, and technical features for a ticker's price history.
    Expects df sorted by trade_date ASC with columns: ['trade_date', 'open', 'high', 'low', 'close', 'adj_close', 'volume'].

    All features for Day T row represent data available up to Day T-1.
    """
    if df.empty or len(df) < 5:
        return pd.DataFrame()

    df = df.sort_values("trade_date").reset_index(drop=True).copy()

    # Raw metrics for Day T-1 (lagged by 1 row)
    df["feat_open"] = df["open"].shift(1)
    df["feat_high"] = df["high"].shift(1)
    df["feat_low"] = df["low"].shift(1)
    df["feat_close"] = df["close"].shift(1)
    df["feat_adj_close"] = df["adj_close"].shift(1)
    df["feat_volume"] = df["volume"].shift(1)

    # Daily Return (T-1 Close vs T-2 Close on adj_close)
    df["daily_return"] = (df["adj_close"].shift(1) - df["adj_close"].shift(2)) / df["adj_close"].shift(2)

    # Intraday Price Change % (T-1 Close vs T-1 Open)
    df["price_change_pct"] = (df["close"].shift(1) - df["open"].shift(1)) / df["open"].shift(1).replace(0, np.nan)

    # Volume Change % (T-1 Volume vs T-2 Volume)
    df["volume_change_pct"] = (df["volume"].shift(1) - df["volume"].shift(2)) / df["volume"].shift(2).replace(0, np.nan)

    # Moving Averages on T-1 adj_close
    df["ma5"] = df["adj_close"].shift(1).rolling(window=5).mean()
    df["ma10"] = df["adj_close"].shift(1).rolling(window=10).mean()
    df["ma20"] = df["adj_close"].shift(1).rolling(window=20).mean()

    # RSI & MACD on lagged adj_close
    lagged_adj_close = df["adj_close"].shift(1)
    df["rsi"] = compute_rsi(lagged_adj_close, period=rsi_period)
    macd_line, macd_sig = compute_macd(lagged_adj_close, fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal)
    df["macd"] = macd_line
    df["macd_signal"] = macd_sig

    return df
