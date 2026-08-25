"""
CRUD operations for stock_predictions table (pure functions).
"""

import json
import uuid
from datetime import date
from typing import Any

from src.core.database import get_db


def save_prediction(
    ticker: str,
    prediction_date: str | date,
    target_direction: str,
    predicted_class: int,
    confidence_score: float,
    probability_up: float,
    probability_down: float,
    probability_neutral: float = 0.0,
    signal: dict[str, Any] | None = None,
) -> uuid.UUID:
    """
    Saves or updates a stock trend prediction and its structured technical signals in stock_predictions.
    """
    query = """
    INSERT INTO stock_predictions (
        ticker, prediction_date, target_direction, predicted_class,
        confidence_score, probability_up, probability_down, probability_neutral,
        signal
    ) VALUES (
        %(ticker)s, %(prediction_date)s, %(target_direction)s, %(predicted_class)s,
        %(confidence_score)s, %(probability_up)s, %(probability_down)s, %(probability_neutral)s,
        %(signal)s
    )
    ON CONFLICT (ticker, prediction_date)
    DO UPDATE SET
        target_direction = EXCLUDED.target_direction,
        predicted_class = EXCLUDED.predicted_class,
        confidence_score = EXCLUDED.confidence_score,
        probability_up = EXCLUDED.probability_up,
        probability_down = EXCLUDED.probability_down,
        probability_neutral = EXCLUDED.probability_neutral,
        signal = EXCLUDED.signal,
        created_at = CURRENT_TIMESTAMP
    RETURNING id;
    """
    params = {
        "ticker": ticker,
        "prediction_date": prediction_date,
        "target_direction": target_direction,
        "predicted_class": predicted_class,
        "confidence_score": confidence_score,
        "probability_up": probability_up,
        "probability_down": probability_down,
        "probability_neutral": probability_neutral,
        "signal": json.dumps(signal or {}),
    }

    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            res = cursor.fetchone()
            return res["id"]


def get_latest_predictions(
    prediction_date: str | date | None = None, limit: int = 20
) -> list[dict[str, Any]]:
    """Retrieves recent stock predictions with structured signals."""
    query = "SELECT * FROM stock_predictions"
    params: list[Any] = []
    if prediction_date:
        query += " WHERE prediction_date = %s"
        params.append(prediction_date)
    query += " ORDER BY prediction_date DESC, confidence_score DESC LIMIT %s"
    params.append(limit)

    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()

