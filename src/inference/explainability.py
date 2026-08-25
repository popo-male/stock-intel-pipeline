"""
Explainability Engine for attributing model predictions to technical and sentiment drivers.
"""

from typing import Any


class ExplainabilityEngine:
    """Generates human-readable explanations and feature attributions for stock direction predictions."""

    @staticmethod
    def generate_explanation(
        ticker: str,
        direction: str,
        confidence: float,
        features: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """
        Synthesizes technical indicators and news sentiment into a coherent rationale and importance dictionary.
        """
        reasons: list[str] = []
        importance: dict[str, Any] = {}

        rsi = float(features.get("rsi") or 50.0)
        macd = float(features.get("macd") or 0.0)
        macd_signal = float(features.get("macd_signal") or 0.0)
        ma5 = float(features.get("ma5") or 0.0)
        ma20 = float(features.get("ma20") or 0.0)
        daily_return = float(features.get("daily_return") or 0.0) * 100
        spy_return = float(features.get("spy_return") or 0.0) * 100

        avg_sent = float(features.get("avg_sentiment") or 0.0)
        sent_3d = float(features.get("avg_sentiment_3d") or 0.0)
        news_cnt = int(features.get("news_count") or 0)
        pos_cnt = int(features.get("positive_news_count") or 0)
        neg_cnt = int(features.get("negative_news_count") or 0)

        # 1. Technical Drivers
        if rsi > 70:
            reasons.append(f"RSI is overbought at {rsi:.1f}, suggesting potential consolidation or pullback pressure.")
            importance["rsi"] = {"value": rsi, "signal": "Bearish (Overbought)"}
        elif rsi < 30:
            reasons.append(f"RSI is oversold at {rsi:.1f}, presenting strong upside mean-reversion opportunity.")
            importance["rsi"] = {"value": rsi, "signal": "Bullish (Oversold)"}
        else:
            trend_str = "moderately bullish" if rsi > 50 else "moderately bearish"
            reasons.append(f"RSI is neutral-to-{trend_str} at {rsi:.1f}.")
            importance["rsi"] = {"value": rsi, "signal": "Neutral"}

        # MACD
        if macd > macd_signal:
            reasons.append(f"MACD line ({macd:.2f}) sits above signal line ({macd_signal:.2f}), confirming positive momentum.")
            importance["macd"] = {"value": macd, "signal": "Bullish Momentum"}
        else:
            reasons.append(f"MACD line ({macd:.2f}) trades below signal line ({macd_signal:.2f}), reflecting downward momentum.")
            importance["macd"] = {"value": macd, "signal": "Bearish Momentum"}

        # Moving Averages
        if ma5 > 0 and ma20 > 0:
            if ma5 > ma20:
                reasons.append(f"Short-term MA5 (${ma5:.2f}) remains above medium-term MA20 (${ma20:.2f}), supporting an uptrend.")
                importance["moving_averages"] = {"ma5": ma5, "ma20": ma20, "signal": "Bullish Trend"}
            else:
                reasons.append(f"Short-term MA5 (${ma5:.2f}) remains below medium-term MA20 (${ma20:.2f}), indicating a downtrend.")
                importance["moving_averages"] = {"ma5": ma5, "ma20": ma20, "signal": "Bearish Trend"}

        # Macro Benchmark
        if abs(spy_return) > 0.1:
            spy_direction = "gain" if spy_return > 0 else "decline"
            reasons.append(f"Broader market ETF (SPY) recorded a {spy_direction} of {spy_return:+.2f}%.")
            importance["spy_return"] = {"value": spy_return, "signal": "Tailwind" if spy_return > 0 else "Headwind"}

        # 2. News Sentiment Drivers
        if news_cnt > 0:
            sent_label = "positive" if avg_sent > 0.05 else ("negative" if avg_sent < -0.05 else "neutral")
            reasons.append(
                f"News sentiment is {sent_label} (avg score: {avg_sent:+.2f} across {news_cnt} articles, {pos_cnt} positive / {neg_cnt} negative). 3-day trailing sentiment is {sent_3d:+.2f}."
            )
            importance["sentiment"] = {
                "avg_sentiment": avg_sent,
                "avg_sentiment_3d": sent_3d,
                "news_count": news_cnt,
                "positive_count": pos_cnt,
                "negative_count": neg_cnt,
                "signal": "Bullish Sentiment" if avg_sent > 0.05 else ("Bearish Sentiment" if avg_sent < -0.05 else "Neutral"),
            }
        else:
            reasons.append("Zero company headlines in pre-market window; sentiment defaulted to neutral.")
            importance["sentiment"] = {"news_count": 0, "signal": "No News"}

        summary = (
            f"Prediction: {direction} (Confidence: {confidence * 100:.1f}%). "
            f"Key driving factors for {ticker}: " + " ".join(reasons)
        )

        return summary, importance
