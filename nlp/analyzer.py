import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Any, cast

from db.connection import get_db_connection


class Analyzer:
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.labels = ["Bullish", "Bearish", "Neutral"]

    def analyze_text(self, text: str) -> dict[str, Any]:
        if not text.strip():
            return {"score": 0.0, "label": "Neutral"}

        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        # convert raw logits into probabilities using softmax
        probabilities = (
            torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()
        )

        pos_score, neg_score, neu_score = (
            probabilities[0],
            probabilities[1],
            probabilities[2],
        )
        compound_score = pos_score - neg_score

        max_idx = probabilities.index(max(probabilities))
        label = self.labels[max_idx]

        return {"score": round(compound_score, 4), "label": label}

    def calculate_sentiment(self, title: str, summary: str) -> dict[str, Any]:
        """
        Calculates distinct sentiment profiles for title and summary.
        Applies a 60% headline / 40% summary weighted distribution.
        """
        title_result = self.analyze_text(title)
        summary_result = self.analyze_text(summary)

        final_score = (title_result["score"] * 0.6) + (summary_result["score"] * 0.4)

        if final_score >= 0.15:
            final_label = "Bullish"
        elif final_score <= -0.15:
            final_label = "Bearish"
        else:
            final_label = "Neutral"

        return {"score": round(final_score, 4), "label": final_label}

    def process_unscored_articles(self) -> None:
        """Fetches articles without sentiment scores and updates sentiment values."""
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, title, summary FROM articles_v2 WHERE sentiment_score IS NULL"
        )
        unscored_articles = cursor.fetchall()

        if not unscored_articles:
            print("No unscored articles remaining.")
            conn.close()
            return

        print(f"Analyzing sentiment for {len(unscored_articles)} articles...")
        update_count = 0

        for article in unscored_articles:
            row = cast(dict[str, Any], article)

            # Execute weighted calculation
            sentiment = self.calculate_sentiment(
                title=row.get("title", ""), summary=row.get("summary", "")
            )

            try:
                # Update individual record immediately using an isolated transaction
                with conn:
                    with conn.cursor() as update_cursor:
                        update_cursor.execute(
                            """
                            UPDATE articles_v2
                            SET sentiment_score = %s, sentiment_label = %s
                            WHERE id = %s
                            """,
                            (sentiment["score"], sentiment["label"], row["id"]),
                        )
                update_count += 1
            except Exception as exc:
                print(
                    f"Failed to commit sentiment update for article ID {row['id']}: {exc}"
                )

        conn.close()
        print(f"Successfully updated sentiment metrics for {update_count} articles.")


def process_unscored_articles() -> None:
    Analyzer().process_unscored_articles()
