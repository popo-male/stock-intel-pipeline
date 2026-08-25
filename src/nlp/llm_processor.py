"""
LLM Processor for generating structured bullet summaries and keyword tags for news articles.
"""

import time

from openai import OpenAI
from pydantic import BaseModel, Field

from src.core.logger import logger
from src.core.settings import settings
from src.crud.crud_news_articles import (
    get_unsummarized_articles,
    update_article_insights,
)


class OutputSchema(BaseModel):
    bullets: list[str] = Field(
        ...,
        description="An array of 2 to 3 short bullet points summarizing the key facts.",
    )
    keywords: list[str] = Field(
        ...,
        description="An array of 3 to 5 highly relevant financial or tech keywords or tags.",
    )


class LLMProcessor:
    """Processor extracting structured insights using LLM API."""

    def __init__(self, client: OpenAI | None = None):
        self._client = client

    @property
    def client(self) -> OpenAI | None:
        if self._client is None and settings.LLM_API_KEY:
            try:
                self._client = OpenAI(
                    base_url=settings.LLM_BASE_URL,
                    api_key=settings.LLM_API_KEY,
                    timeout=30.0,
                    max_retries=3,
                )
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        return self._client

    def generate_insights(self, title: str, summary: str) -> OutputSchema | None:
        """Uses LLM to generate structured bullet points and keywords."""
        if not self.client or not settings.LLM_API_KEY:
            logger.debug("LLM API key not configured. Skipping LLM insights generation.")
            return None

        prompt = f"Analyze the following financial news article.\nTitle: {title}\nContent: {summary}"
        system_instruction = (
            "You are a financial analyst backend service. Extract key metadata details from the user prompt. "
            "Your response must be a valid JSON object matching this schema exactly:\n"
            "{\n"
            '  "bullets": ["string", "string"],\n'
            '  "keywords": ["string", "string"]\n'
            "}"
        )

        try:
            response = self.client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            raw_content = response.choices[0].message.content
            if not raw_content:
                return None

            return OutputSchema.model_validate_json(raw_content)
        except Exception as exc:
            logger.error(f"Error generating insights using {settings.LLM_MODEL}: {exc}")
            return None

    def process_unsummarized_articles(self, limit: int = 50) -> int:
        """
        Queries articles missing summaries and updates bullets and keywords in news_articles.
        """
        if not settings.LLM_API_KEY:
            logger.info("LLM_API_KEY is not set. Skipping LLM article summarization.")
            return 0

        unsummarized = get_unsummarized_articles(limit=limit)
        if not unsummarized:
            logger.info("No articles waiting for LLM summarization.")
            return 0

        logger.info(f"Generating LLM insights for {len(unsummarized)} articles...")
        update_count = 0

        for article in unsummarized:
            insights = self.generate_insights(
                title=article.get("title", ""), summary=article.get("summary", "")
            )
            if insights:
                try:
                    update_article_insights(
                        article_id=article["id"],
                        bullets=insights.bullets,
                        keywords=insights.keywords,
                    )
                    update_count += 1
                except Exception as e:
                    logger.error(f"Failed to update LLM insights for ID {article['id']}: {e}")

            time.sleep(0.5)

        logger.info(f"Successfully processed LLM insights for {update_count} articles.")

        if settings.STRICT_LLM_FAILURE and unsummarized and update_count == 0:
            raise RuntimeError("Strict mode: LLM insight extraction failed across all targets.")

        return update_count


def process_unsummarized_articles() -> None:
    LLMProcessor().process_unsummarized_articles()
