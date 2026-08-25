from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str | None = None

    # LLM Settings
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "openai/gpt-oss-120b"
    STRICT_LLM_FAILURE: bool = False

    # Optional Watchlist Override
    WATCHLIST: str | None = None

    @property
    def watchlist_symbols(self) -> list[str]:
        if not self.WATCHLIST:
            return []
        return [item.strip() for item in self.WATCHLIST.split(",") if item.strip()]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
