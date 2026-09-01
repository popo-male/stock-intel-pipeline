"""
Stock Direction Prediction & Explainability Service.
Provides functional prediction orchestration and rule-based explanations.
"""

from datetime import date, datetime
from typing import Any

from src.core.config import AppConfig, load_config
from src.core.logger import logger
from src.crud.crud_daily_stock_features import get_latest_feature_vector
from src.crud.crud_stock_predictions import save_prediction
from src.inference.model import StockDirectionModel, get_model


def predict_direction(
    features: dict[str, Any], config: AppConfig | None = None
) -> tuple[float, float, float]:
    """
    Computes probabilities using the global StockDirectionModel instance.
    Returns: (probability_up, probability_down, probability_neutral)
    """
    model = get_model(config)
    res = model.predict(features)
    return (
        res["probability_up"],
        res["probability_down"],
        res["probability_neutral"],
    )


def get_signal(features: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluates rule-based technical indicators and sentiment metrics to produce structured signals.
    These signals are stored as JSONB for UI badges, filtering, and analytical queries.
    """
    rsi = float(features.get("rsi") or 50.0)
    macd = float(features.get("macd") or 0.0)
    macd_signal = float(features.get("macd_signal") or 0.0)
    ma5 = float(features.get("ma5") or 0.0)
    ma10 = float(features.get("ma10") or 0.0)
    ma20 = float(features.get("ma20") or 0.0)
    close_to_ma5 = float(features.get("close_to_ma5") or 0.0)
    close_to_ma20 = float(features.get("close_to_ma20") or 0.0)
    high_low_spread = float(features.get("high_low_spread") or 0.0)
    spy_return = float(features.get("spy_return") or 0.0) * 100.0
    qqq_return = float(features.get("qqq_return") or 0.0) * 100.0

    avg_sent = float(features.get("avg_sentiment") or 0.0)
    sent_3d = float(features.get("avg_sentiment_3d") or 0.0)
    news_cnt = int(features.get("news_count") or 0)
    pos_cnt = int(features.get("positive_news_count") or 0)
    neg_cnt = int(features.get("negative_news_count") or 0)

    # 1. RSI Rule
    if rsi >= 70.0:
        rsi_info = {
            "value": round(rsi, 2),
            "rule": "rsi >= 70",
            "signal": "Overbought",
            "sentiment": "Bearish",
        }
    elif rsi <= 30.0:
        rsi_info = {
            "value": round(rsi, 2),
            "rule": "rsi <= 30",
            "signal": "Oversold",
            "sentiment": "Bullish",
        }
    elif rsi > 50.0:
        rsi_info = {
            "value": round(rsi, 2),
            "rule": "50 < rsi < 70",
            "signal": "Bullish Momentum",
            "sentiment": "Bullish",
        }
    else:
        rsi_info = {
            "value": round(rsi, 2),
            "rule": "30 < rsi <= 50",
            "signal": "Bearish Momentum",
            "sentiment": "Bearish",
        }

    # 2. MACD Rule
    macd_diff = macd - macd_signal
    if macd > macd_signal:
        macd_info = {
            "macd": round(macd, 2),
            "signal_line": round(macd_signal, 2),
            "diff": round(macd_diff, 2),
            "rule": "macd > signal",
            "signal": "Golden Cross / Bullish Momentum",
            "sentiment": "Bullish",
        }
    else:
        macd_info = {
            "macd": round(macd, 2),
            "signal_line": round(macd_signal, 2),
            "diff": round(macd_diff, 2),
            "rule": "macd <= signal",
            "signal": "Death Cross / Bearish Momentum",
            "sentiment": "Bearish",
        }

    # 3. Moving Average Trend Rule
    if ma5 > 0 and ma20 > 0:
        ma_signal = (
            "Bullish (MA5 > MA20)" if ma5 > ma20 else "Bearish (MA5 < MA20)"
        )
        ma_info = {
            "ma5": round(ma5, 2),
            "ma10": round(ma10, 2),
            "ma20": round(ma20, 2),
            "close_to_ma5_pct": round(close_to_ma5 * 100.0, 2),
            "close_to_ma20_pct": round(close_to_ma20 * 100.0, 2),
            "rule": "ma5 > ma20" if ma5 > ma20 else "ma5 <= ma20",
            "signal": ma_signal,
            "sentiment": "Bullish" if ma5 > ma20 else "Bearish",
        }
    else:
        ma_info = {
            "ma5": round(ma5, 2),
            "ma20": round(ma20, 2),
            "signal": "Insufficient History",
            "sentiment": "Neutral",
        }

    # 4. Market / Macro Beta Rule
    if spy_return > 0.2:
        macro_signal = "Strong Market Tailwind"
        macro_sent = "Bullish"
    elif spy_return < -0.2:
        macro_signal = "Market Headwind"
        macro_sent = "Bearish"
    else:
        macro_signal = "Market Neutral"
        macro_sent = "Neutral"

    macro_info = {
        "spy_return_pct": round(spy_return, 2),
        "qqq_return_pct": round(qqq_return, 2),
        "signal": macro_signal,
        "sentiment": macro_sent,
    }

    # 5. Overnight News Sentiment Rule
    if news_cnt > 0:
        if avg_sent > 0.05:
            sent_label = "Positive Pre-Market News"
            sent_direction = "Bullish"
        elif avg_sent < -0.05:
            sent_label = "Negative Pre-Market News"
            sent_direction = "Bearish"
        else:
            sent_label = "Neutral Pre-Market News"
            sent_direction = "Neutral"

        sentiment_info = {
            "avg_sentiment": round(avg_sent, 4),
            "avg_sentiment_3d": round(sent_3d, 4),
            "news_count": news_cnt,
            "positive_news_count": pos_cnt,
            "negative_news_count": neg_cnt,
            "signal": sent_label,
            "sentiment": sent_direction,
        }
    else:
        sentiment_info = {
            "avg_sentiment": 0.0,
            "avg_sentiment_3d": round(sent_3d, 4),
            "news_count": 0,
            "signal": "No Overnight News",
            "sentiment": "Neutral",
        }

    # 6. Volatility Spread
    volatility_info = {
        "high_low_spread_pct": round(high_low_spread * 100.0, 2),
        "signal": "Elevated Volatility"
        if high_low_spread > 0.03
        else "Normal Volatility",
    }

    return {
        "rsi": rsi_info,
        "macd": macd_info,
        "moving_averages": ma_info,
        "macro": macro_info,
        "sentiment": sentiment_info,
        "volatility": volatility_info,
    }


def predict_single_stock(
    ticker: str,
    target_date: str | date | None = None,
    config: AppConfig | None = None,
    model: StockDirectionModel | None = None,
) -> dict[str, Any]:
    """Runs end-to-end direction prediction, structured signal extraction, and DB persistence for one ticker."""
    cfg = config or load_config()
    pred_date = str(target_date or datetime.now().date())

    features = get_latest_feature_vector(ticker, target_date=pred_date)
    if not features:
        raise ValueError(
            f"Cannot generate prediction: No feature records exist for {ticker} on or before {pred_date}. "
            f"Please run market collection and feature engineering first."
        )

    # 1. Execute ML/Heuristic model inference via StockDirectionModel
    model_engine = model or get_model(cfg)
    pred_res = model_engine.predict(features)

    # 2. Extract structured technical and sentiment explainability signals
    signals = get_signal(features=features)

    # 3. Persist prediction to database
    pred_id = save_prediction(
        ticker=ticker,
        prediction_date=pred_date,
        target_direction=pred_res["target_direction"],
        predicted_class=pred_res["predicted_class"],
        confidence_score=pred_res["confidence_score"],
        probability_up=pred_res["probability_up"],
        probability_down=pred_res["probability_down"],
        probability_neutral=pred_res["probability_neutral"],
        signal=signals,
    )

    return {
        "id": str(pred_id),
        "ticker": ticker,
        "prediction_date": pred_date,
        "target_direction": pred_res["target_direction"],
        "predicted_class": pred_res["predicted_class"],
        "confidence_score": pred_res["confidence_score"],
        "probability_up": pred_res["probability_up"],
        "probability_down": pred_res["probability_down"],
        "probability_neutral": pred_res["probability_neutral"],
        "signal": signals,
    }


def predict_watchlist(
    tickers: list[str] | None = None,
    target_date: str | date | None = None,
    config: AppConfig | None = None,
) -> list[dict[str, Any]]:
    """Runs batch predictions across target tickers with minimal summary logging."""
    cfg = config or load_config()
    watchlist = tickers or cfg.market.watchlist_symbols
    pred_date = str(target_date or datetime.now().date())
    predictions: list[dict[str, Any]] = []

    model_engine = get_model(cfg)

    for ticker in watchlist:
        try:
            res = predict_single_stock(
                ticker, target_date=pred_date, config=cfg, model=model_engine
            )
            predictions.append(res)
        except Exception as e:
            logger.error(f"[{ticker}] Prediction failed: {e}")

    logger.info(
        f"Generated predictions for {len(predictions)}/{len(watchlist)} tickers on {pred_date}."
    )
    return predictions
