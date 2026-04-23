from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Any, cast

from db.connection import get_db_connection


class Analyzer:
    def __init__(self, sentiment_analyzer: SentimentIntensityAnalyzer | None = None):
        self.sentiment_analyzer = sentiment_analyzer or SentimentIntensityAnalyzer()

    def calculate_sentiment(self, text: str) -> dict:
        """
        Returns a dictionary with sentiment scores.
        Compound score ranges from -1 (Extremely Negative) to +1 (Extremely Positive).
        """
        if not text:
            return {"compound": 0, "label": "Neutral"}

        scores = self.sentiment_analyzer.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            label = "Bullish"
        elif compound <= -0.05:
            label = "Bearish"
        else:
            label = "Neutral"

        return {"compound": compound, "label": label}

    def process_unscored_articles(self) -> None:
        """Fetches articles without sentiment scores and updates sentiment values."""
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "ALTER TABLE articles ADD COLUMN IF NOT EXISTS sentiment_score REAL"
        )
        cursor.execute(
            "ALTER TABLE articles ADD COLUMN IF NOT EXISTS sentiment_label TEXT"
        )

        cursor.execute(
            "SELECT id, title, summary FROM articles WHERE sentiment_score IS NULL"
        )
        unscored_articles = cursor.fetchall()

        if not unscored_articles:
            conn.close()
            return

        update_count = 0
        for article in unscored_articles:
            row = cast(dict[str, Any], article)
            full_text = f"{row['title']} {row['summary']}"

            sentiment = self.calculate_sentiment(full_text)

            cursor.execute(
                """
                UPDATE articles
                SET sentiment_score = %s, sentiment_label = %s
                WHERE id = %s
            """,
                (sentiment["compound"], sentiment["label"], row["id"]),
            )

            update_count += 1

        conn.commit()
        conn.close()
        print(f"Successfully updated sentiment for {update_count} articles.")


def calculate_sentiment(text: str) -> dict:
    return Analyzer().calculate_sentiment(text)


def process_unscored_articles() -> None:
    Analyzer().process_unscored_articles()
