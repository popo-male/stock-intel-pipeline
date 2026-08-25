"""
Inference Engine for next-day stock price direction prediction.
"""

from datetime import date, datetime
from typing import Any

import numpy as np

from src.core.config import AppConfig, load_config
from src.core.logger import logger
from src.crud.crud_daily_stock_features import get_latest_feature_vector
from src.crud.crud_stock_predictions import save_prediction
from src.inference.explainability import ExplainabilityEngine


class StockPredictor:
    """Predictor for next trading day stock trend direction."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        self.model_version = self.config.model.version

    def _compute_direction_probabilities(self, features: dict[str, Any]) -> tuple[float, float, float]:
        """
        Inference logic combining normalized technical indicators and sentiment scores into calibrated probabilities.
        """
        rsi = float(features.get("rsi") or 50.0)
        macd = float(features.get("macd") or 0.0)
        macd_sig = float(features.get("macd_signal") or 0.0)
        daily_return = float(features.get("daily_return") or 0.0)
        spy_return = float(features.get("spy_return") or 0.0)

        avg_sentiment = float(features.get("avg_sentiment") or 0.0)
        sentiment_3d = float(features.get("avg_sentiment_3d") or 0.0)

        # Normalize signals
        rsi_signal = np.clip((50.0 - rsi) / 20.0, -1.0, 1.0)
        macd_diff = macd - macd_sig
        macd_signal = np.tanh(macd_diff * 2.0)
        ret_signal = np.tanh(daily_return * 10.0)
        spy_signal = np.tanh(spy_return * 15.0)

        intraday_sent = np.clip(avg_sentiment, -1.0, 1.0)
        macro_sent = np.clip(sentiment_3d, -1.0, 1.0)

        tech_score = (rsi_signal * 0.3) + (macd_signal * 0.4) + (ret_signal * 0.15) + (spy_signal * 0.15)
        sent_score = (intraday_sent * 0.7) + (macro_sent * 0.3)
        composite_score = (tech_score * 0.55) + (sent_score * 0.45)

        prob_up = float(1.0 / (1.0 + np.exp(-3.0 * composite_score)))
        prob_down = 1.0 - prob_up

        neutral_margin = max(0.0, 1.0 - (abs(composite_score) * 2.5))
        prob_neutral = float(neutral_margin * 0.25)

        prob_up = prob_up * (1.0 - prob_neutral)
        prob_down = prob_down * (1.0 - prob_neutral)

        return round(prob_up, 4), round(prob_down, 4), round(prob_neutral, 4)

    def predict_ticker(
        self,
        ticker: str,
        target_date: str | date | None = None,
    ) -> dict[str, Any]:
        """
        Runs direction prediction for a single ticker on a given prediction date.
        """
        pred_date = target_date or datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Running trend prediction for {ticker} on {pred_date}...")

        features = get_latest_feature_vector(ticker, target_date=pred_date)
        if not features:
            logger.warning(
                f"No feature vector found in daily_stock_features for {ticker} on {pred_date}. Using available latest features."
            )
            features = get_latest_feature_vector(ticker)

        if not features:
            raise ValueError(
                f"Cannot generate prediction: No feature records exist for {ticker}. "
                f"Please run market collection and feature generation first."
            )

        prob_up, prob_down, prob_neutral = self._compute_direction_probabilities(features)

        dir_up = self.config.target_definitions.direction_up
        dir_down = self.config.target_definitions.direction_down
        dir_neutral = self.config.target_definitions.direction_neutral
        up_code = self.config.target_definitions.up_code
        down_code = self.config.target_definitions.down_code

        if prob_up >= prob_down and prob_up >= prob_neutral:
            direction = dir_up
            predicted_class = up_code
            confidence = prob_up
        elif prob_down >= prob_up and prob_down >= prob_neutral:
            direction = dir_down
            predicted_class = down_code
            confidence = prob_down
        else:
            direction = dir_neutral
            predicted_class = down_code
            confidence = prob_neutral

        explanation, importance = ExplainabilityEngine.generate_explanation(
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            features=features,
        )

        prediction_id = save_prediction(
            ticker=ticker,
            prediction_date=pred_date,
            target_direction=direction,
            predicted_class=predicted_class,
            confidence_score=confidence,
            probability_up=prob_up,
            probability_down=prob_down,
            probability_neutral=prob_neutral,
            explanation=explanation,
            feature_importance=importance,
            model_version=self.model_version,
        )

        result = {
            "id": str(prediction_id),
            "ticker": ticker,
            "prediction_date": str(pred_date),
            "target_direction": direction,
            "predicted_class": predicted_class,
            "confidence_score": confidence,
            "probability_up": prob_up,
            "probability_down": prob_down,
            "probability_neutral": prob_neutral,
            "explanation": explanation,
            "feature_importance": importance,
            "model_version": self.model_version,
        }

        logger.info(f"[{ticker}] Prediction complete: {direction} (Confidence: {confidence * 100:.1f}%)")
        return result

    def predict_all(
        self,
        tickers: list[str] | None = None,
        target_date: str | date | None = None,
    ) -> list[dict[str, Any]]:
        """Runs predictions across all target tickers."""
        if tickers is None:
            tickers = self.config.market.watchlist_symbols

        results: list[dict[str, Any]] = []
        for ticker in tickers:
            try:
                pred = self.predict_ticker(ticker, target_date=target_date)
                results.append(pred)
            except Exception as e:
                logger.error(f"Prediction failed for {ticker}: {e}")

        return results
