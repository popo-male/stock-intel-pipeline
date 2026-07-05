import time
import json
from typing import Any, cast
from pydantic import BaseModel, Field

from openai import OpenAI
from core.settings import settings
from db.connection import get_db_connection


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
                        "content": "You are a financial analyst backend service. Extract key metadata details from the user prompt structure.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=OutputSchema,
                temperature=0.1,
            )

            return response.choices[0].message.parsed

        except Exception as exc:
            print(
                f"Error generating structured insights with model {settings.LLM_MODEL}: {exc}"
            )
            return None

    def process_unsummarized_articles(self) -> None:
        """
        Queries articles missing summaries using partial indexing.
        Updates JSONB structures using single-row runtime transactions.
        """
        conn = get_db_connection()
        cursor = conn.cursor()

        # Queries optimization leverages native JSONB partial indices
        cursor.execute(
            "SELECT id, title, summary FROM articles_v2 WHERE bullets IS NULL"
        )
        unsummarized = cursor.fetchall()

        if not unsummarized:
            print("No unsummarized articles remaining.")
            conn.close()
            return

        print(f"Generating structured AI insights for {len(unsummarized)} articles...")
        update_count = 0

        for article in unsummarized:
            row = cast(dict[str, Any], article)

            # The API response is returned as a fully typed object instead of a text string
            insights = self.generate_insights(
                row.get("title", ""), row.get("summary", "")
            )

            if insights:
                try:
                    # Leverage single-row atomicity via isolated transaction manager blocks
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
                    print(
                        f"Failed to commit AI insights for article ID {row['id']}: {exc}"
                    )

            time.sleep(1.0)  # Throttling delay for rate-limits protection

        conn.close()
        print(
            f"Successfully generated structured insights for {update_count} articles."
        )

        if settings.STRICT_LLM_FAILURE and unsummarized and update_count == 0:
            raise RuntimeError(
                "LLM parsing completely failed or timed out across current active processing targets."
            )


# Functional entry points matching original app architecture patterns
def process_unsummarized_articles() -> None:
    LLMProcessor().process_unsummarized_articles()
