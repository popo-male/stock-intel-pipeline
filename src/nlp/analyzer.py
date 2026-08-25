"""
Sentiment Analysis Engine using FinBERT for financial text sentiment classification.
"""

from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.core.config import AppConfig, load_config
from src.core.logger import logger
from src.crud.crud_news_articles import get_unscored_articles, insert_sentiment


class SentimentAnalyzer:
    """FinBERT sentiment analyzer for financial news headlines and summaries."""

    def __init__(self, config: AppConfig | None = None, model_name: str | None = None):
        self.config = config or load_config()
        self.model_name = model_name or self.config.nlp.sentiment_model
        self.labels = ["Bullish", "Bearish", "Neutral"]
        self._tokenizer = None
        self._model = None

    def _load_model(self) -> None:
        """Lazy load tokenizer and model."""
        if self._model is None or self._tokenizer is None:
            logger.info(f"Loading FinBERT model: {self.model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()

    def analyze_text(self, text: str) -> dict[str, Any]:
        """Analyzes a single text string and returns compound score and label."""
        if not text or not text.strip():
            return {"score": 0.0, "label": "Neutral"}

        try:
            self._load_model()
            inputs = self._tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            with torch.no_grad():
                outputs = self._model(**inputs)

            probabilities = (
                torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()
            )

            compound_score = probabilities[0] - probabilities[1]
            max_idx = probabilities.index(max(probabilities))
            label = self.labels[max_idx]

            return {"score": round(compound_score, 4), "label": label}
        except Exception as e:
            logger.error(f"Error during FinBERT text analysis: {e}")
            return {"score": 0.0, "label": "Neutral"}

    def calculate_sentiment(self, title: str, summary: str) -> dict[str, Any]:
        """
        Calculates weighted sentiment score (60% headline, 40% summary).
        """
        title_wt = self.config.nlp.title_weight
        summary_wt = self.config.nlp.summary_weight

        title_res = self.analyze_text(title)
        summary_res = self.analyze_text(summary)

        final_score = (title_res["score"] * title_wt) + (summary_res["score"] * summary_wt)

        if final_score >= self.config.nlp.bullish_threshold:
            final_label = "Bullish"
        elif final_score <= self.config.nlp.bearish_threshold:
            final_label = "Bearish"
        else:
            final_label = "Neutral"

        return {"score": round(final_score, 4), "label": final_label}

    def process_unscored_articles(self, limit: int = 200) -> int:
        """Fetches unscored articles and saves calculated sentiments to news_sentiments table."""
        unscored = get_unscored_articles(limit=limit)
        if not unscored:
            logger.info("No unscored news articles found in database queue.")
            return 0

        logger.info(f"Analyzing sentiment for {len(unscored)} news records...")
        processed_count = 0

        for row in unscored:
            try:
                sentiment = self.calculate_sentiment(
                    title=row.get("title", ""), summary=row.get("summary", "")
                )
                insert_sentiment(
                    article_id=row["article_id"],
                    ticker=row["ticker"],
                    published_at=row["published_at"],
                    sentiment_score=sentiment["score"],
                    sentiment_label=sentiment["label"],
                    model_version=self.model_name,
                )
                processed_count += 1
            except Exception as e:
                logger.error(f"Failed to calculate/save sentiment for article ID {row.get('article_id')}: {e}")

        logger.info(f"Successfully processed sentiment for {processed_count} articles.")
        return processed_count


Analyzer = SentimentAnalyzer


def process_unscored_articles() -> None:
    SentimentAnalyzer().process_unscored_articles()
