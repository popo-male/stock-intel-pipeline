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
        api_key = settings.effective_llm_api_key
        if self._client is None and api_key:
            try:
                self._client = OpenAI(
                    base_url=settings.LLM_BASE_URL,
                    api_key=api_key,
                    timeout=30.0,
                    max_retries=2,
                )
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI/Groq client: {e}")
        return self._client

    def generate_insights(self, title: str, summary: str) -> OutputSchema | None:
        """Uses LLM to generate structured bullet points and keywords."""
        if not self.client or not settings.effective_llm_api_key:
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
            err_str = str(exc)
            if "401" in err_str or "invalid_api_key" in err_str.lower() or "unauthorized" in err_str.lower():
                logger.warning(
                    f"LLM API Authentication failed (401 Unauthorized): {err_str}. "
                    "Please verify your LLM_API_KEY / GROQ_API_KEY secret."
                )
                self._auth_failed = True
            else:
                logger.error(f"Error generating insights using {settings.LLM_MODEL}: {err_str}")
            return None

    def process_unsummarized_articles(self, limit: int = 50) -> int:
        """
        Queries articles missing summaries and updates bullets and keywords in news_articles.
        """
        if not settings.effective_llm_api_key:
            logger.info("LLM API key is not set. Skipping LLM article summarization.")
            return 0

        unsummarized = get_unsummarized_articles(limit=limit)
        if not unsummarized:
            logger.info("No articles waiting for LLM summarization.")
            return 0

        logger.info(f"Generating LLM insights for {len(unsummarized)} articles...")
        update_count = 0

        for article in unsummarized:
            if getattr(self, "_auth_failed", False):
                logger.warning("Aborting remaining LLM summarization due to authentication failure.")
                break

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
