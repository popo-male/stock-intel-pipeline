import time
import json
from typing import Any, cast
from pydantic import BaseModel, Field

from openai import OpenAI
from src.core.settings import settings
from src.db.connection import get_db_connection
from src.core.logger import logger


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
    def __init__(self, client: OpenAI | None = None):
        self.client = client or OpenAI(
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            timeout=30.0,
            max_retries=1,
        )

    def generate_insights(self, title: str, summary: str) -> OutputSchema | None:
        """Uses LLM to generate bullet points and extract keywords."""
        prompt = f"Analyze the following financial news article.\nTitle: {title}\nContent: {summary}"
        try:
            response = self.client.beta.chat.completions.parse(
                model=settings.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analyst backend service. Extract key metadata details.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=OutputSchema,
                temperature=0.1,
            )
            return response.choices[0].message.parsed
        except Exception as exc:
            logger.error(
                f"Error extracting structured insights using {settings.LLM_MODEL}: {exc}"
            )
            return None

    def process_unsummarized_articles(self) -> None:
        """
        Queries articles missing summaries using partial indexing.
        Updates JSONB structures using single-row runtime transactions.
        """
        unsummarized = []

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id, title, summary FROM articles_v2 WHERE bullets IS NULL"
                )
                unsummarized = cursor.fetchall()

            if not unsummarized:
                logger.info("No text assets waiting for summary extractions.")
                return

            logger.info(
                f"Generating structured AI highlights for {len(unsummarized)} elements..."
            )
            update_count = 0

            for article in unsummarized:
                row = cast(dict[str, Any], article)
                insights = self.generate_insights(
                    row.get("title", ""), row.get("summary", "")
                )

                if insights:
                    try:
                        with conn:
                            with conn.cursor() as update_cursor:
                                update_cursor.execute(
                                    """
                                    UPDATE articles_v2
                                    SET bullets = %s, keywords = %s
                                    WHERE id = %s
                                    """,
                                    (
                                        json.dumps(insights.bullets),
                                        json.dumps(insights.keywords),
                                        row["id"],
                                    ),
                                )
                        update_count += 1
                    except Exception as exc:
                        logger.error(
                            f"Failed to commit metadata context updates on ID {row['id']}: {exc}"
                        )

                time.sleep(1.0)

            logger.info(
                f"Successfully processed structured AI summaries for {update_count} elements."
            )

            if settings.STRICT_LLM_FAILURE and unsummarized and update_count == 0:
                raise RuntimeError(
                    "Critical Error: LLM extraction engine failed completely across active processing targets."
                )


# Functional entry points matching original app architecture patterns
def process_unsummarized_articles() -> None:
    LLMProcessor().process_unsummarized_articles()
