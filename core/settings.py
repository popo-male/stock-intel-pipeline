from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # DB
    DATABASE_URL: str | None = None
    DB_HOST: str | None = None
    DB_PORT: int | None = None
    DB_NAME: str | None = None
    DB_USER: str | None = None
    DB_PASSWORD: str | None = None

    # LLM
    LLM_BASE_URL: str
    LLM_API_KEY: str
    LLM_MODEL: str
    STRICT_LLM_FAILURE: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()  # pyright: ignore[reportCallIssue]
