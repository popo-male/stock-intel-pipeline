"""
Stock Direction Model Engine.
Encapsulates ML model loading, feature preprocessing, and inference.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.core.config import AppConfig, load_config
from src.core.logger import logger


class StockDirectionModel:
    """
    Encapsulates the trained ML model for stock directional trend predictions.
    Supports primary joblib bundle loading, fallback native XGBoost JSON loading,
    and fallback heuristic inference.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        model_path: str | Path | None = None,
        fallback_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        threshold: float | None = None,
        auto_load: bool = True,
    ) -> None:
        self.config = config or load_config()
        self.model_path = Path(model_path or self.config.model.model_path)
        self.fallback_path = Path(
            fallback_path or self.config.model.fallback_model_path
        )
        self.metadata_path = Path(
            metadata_path or self.config.model.metadata_path
        )
        self.threshold = (
            threshold
            if threshold is not None
            else self.config.model.decision_threshold
        )

        self.model: xgb.XGBClassifier | None = None
        self.features: list[str] = []
        self.metadata: dict[str, Any] = {}
        self._is_loaded: bool = False

        if auto_load:
            self.load()

    @property
    def is_loaded(self) -> bool:
        """Returns True if the ML model is successfully loaded in memory."""
        return self._is_loaded and self.model is not None

    def load(self) -> bool:
        """
        Loads the trained model into memory.
        1. Tries primary joblib bundle.
        2. Tries fallback native XGBoost JSON with metadata.json.
        """
        # 1. Primary loader: joblib model bundle
        if self.model_path.exists():
            try:
                bundle = joblib.load(self.model_path)
                if isinstance(bundle, dict) and "model" in bundle and "features" in bundle:
                    self.model = bundle["model"]
                    self.features = list(bundle["features"])
                    if "threshold" in bundle:
                        self.threshold = float(bundle["threshold"])
                    self._is_loaded = True
                    logger.info(
                        f"Successfully loaded ML model bundle from '{self.model_path}' "
                        f"({len(self.features)} features, threshold={self.threshold})."
                    )
                    return True
            except Exception as exc:
                logger.warning(
                    f"Failed to load joblib model from '{self.model_path}': {exc}. "
                    f"Attempting fallback..."
                )

        # 2. Fallback loader: native XGBoost JSON + metadata.json
        if self.fallback_path.exists() and self.metadata_path.exists():
            try:
                with open(self.metadata_path, encoding="utf-8") as f:
                    self.metadata = json.load(f)
                self.features = list(self.metadata.get("features", []))
                if "calibrated_threshold" in self.metadata:
                    self.threshold = float(self.metadata["calibrated_threshold"])

                clf = xgb.XGBClassifier()
                clf.load_model(str(self.fallback_path))
                self.model = clf
                self._is_loaded = True
                logger.info(
                    f"Successfully loaded fallback XGBoost model from '{self.fallback_path}' "
                    f"({len(self.features)} features, threshold={self.threshold})."
                )
                return True
            except Exception as exc:
                logger.error(
                    f"Failed to load fallback XGBoost model from '{self.fallback_path}': {exc}"
                )

        logger.warning(
            "No ML model could be loaded. Model class will use heuristic inference."
        )
        self._is_loaded = False
        return False

    def _prepare_input_dataframe(
        self, features: dict[str, Any] | pd.DataFrame
    ) -> pd.DataFrame:
        """Aligns input features with the exact columns required by the model."""
        if isinstance(features, pd.DataFrame):
            df = features.copy()
            for col in self.features:
                if col not in df.columns:
                    df[col] = 0.0
            return df[self.features].fillna(0.0)

        # Convert dictionary to DataFrame
        row: dict[str, Any] = {}
        for col in self.features:
            val = features.get(col)
            if val is None or pd.isna(val):
                row[col] = 0.0
            else:
                row[col] = float(val) if isinstance(val, (int, float, np.number)) else 0.0

        return pd.DataFrame([row])[self.features]

    def predict(
        self, features: dict[str, Any], fallback_on_error: bool = True
    ) -> dict[str, Any]:
        """
        Generates direction prediction, probabilities, class, and confidence score.
        Gracefully falls back to heuristic scoring if ML model is not loaded or errors out.
        """
        if self.is_loaded:
            try:
                input_df = self._prepare_input_dataframe(features)
                proba = self.model.predict_proba(input_df)[0]
                prob_down = float(proba[0])
                prob_up = float(proba[1])
                prob_neutral = 0.0

                # Calibrated threshold evaluation
                if prob_up >= self.threshold:
                    direction = self.config.target_definitions.direction_up
                    predicted_class = self.config.target_definitions.up_code
                    confidence = prob_up
                else:
                    direction = self.config.target_definitions.direction_down
                    predicted_class = self.config.target_definitions.down_code
                    confidence = prob_down

                return {
                    "probability_up": round(prob_up, 4),
                    "probability_down": round(prob_down, 4),
                    "probability_neutral": round(prob_neutral, 4),
                    "predicted_class": predicted_class,
                    "target_direction": direction,
                    "confidence_score": round(confidence, 4),
                    "inference_type": "ml",
                }
            except Exception as exc:
                if not fallback_on_error:
                    raise
                logger.warning(
                    f"ML inference error: {exc}. Falling back to heuristic prediction."
                )

        # Heuristic scoring fallback
        return self.predict_heuristic(features)

    def predict_heuristic(self, features: dict[str, Any]) -> dict[str, Any]:
        """
        Computes deterministic heuristic probabilities from normalized technical and sentiment signals.
        """
        rsi = float(features.get("rsi") or 50.0)
        macd = float(features.get("macd") or 0.0)
        macd_sig = float(features.get("macd_signal") or 0.0)
        daily_return = float(features.get("daily_return") or 0.0)
        spy_return = float(features.get("spy_return") or 0.0)
        avg_sent = float(features.get("avg_sentiment") or 0.0)
        sent_3d = float(features.get("avg_sentiment_3d") or 0.0)

        # Normalize technical and sentiment indicators to [-1, 1]
        rsi_signal = np.clip((rsi - 50.0) / 20.0, -1.0, 1.0)
        macd_signal = np.tanh((macd - macd_sig) * 2.0)
        ret_signal = np.tanh(daily_return * 10.0)
        spy_signal = np.tanh(spy_return * 15.0)
        intraday_sent = np.clip(avg_sent, -1.0, 1.0)
        macro_sent = np.clip(sent_3d, -1.0, 1.0)

        tech_score = (
            (rsi_signal * 0.3)
            + (macd_signal * 0.4)
            + (ret_signal * 0.15)
            + (spy_signal * 0.15)
        )
        sent_score = (intraday_sent * 0.7) + (macro_sent * 0.3)
        composite = (tech_score * 0.55) + (sent_score * 0.45)

        prob_up_raw = float(1.0 / (1.0 + np.exp(-3.0 * composite)))
        prob_neutral = float(max(0.0, 1.0 - (abs(composite) * 2.5)) * 0.25)
        prob_down = (1.0 - prob_up_raw) * (1.0 - prob_neutral)
        prob_up = prob_up_raw * (1.0 - prob_neutral)

        # Heuristic class resolution
        if prob_up >= prob_down and prob_up >= prob_neutral:
            direction = self.config.target_definitions.direction_up
            predicted_class = self.config.target_definitions.up_code
            confidence = prob_up
        elif prob_down >= prob_up and prob_down >= prob_neutral:
            direction = self.config.target_definitions.direction_down
            predicted_class = self.config.target_definitions.down_code
            confidence = prob_down
        else:
            direction = self.config.target_definitions.direction_neutral
            predicted_class = getattr(
                self.config.target_definitions, "neutral_code", -1
            )
            confidence = prob_neutral

        return {
            "probability_up": round(prob_up, 4),
            "probability_down": round(prob_down, 4),
            "probability_neutral": round(prob_neutral, 4),
            "predicted_class": predicted_class,
            "target_direction": direction,
            "confidence_score": round(confidence, 4),
            "inference_type": "heuristic",
        }


# Singleton model cache instance
_GLOBAL_MODEL: StockDirectionModel | None = None


def get_model(config: AppConfig | None = None) -> StockDirectionModel:
    """Retrieves or initializes the global singleton StockDirectionModel instance."""
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None:
        _GLOBAL_MODEL = StockDirectionModel(config=config)
    return _GLOBAL_MODEL
