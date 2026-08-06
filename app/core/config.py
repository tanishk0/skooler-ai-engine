from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Values are read from the environment or .env."""

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    frontend_url: str = "http://localhost:3000"
    max_upload_size_mb: int = 15
    chunk_size: int = 1200
    chunk_overlap: int = 150
    llm_timeout_seconds: int = 45

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
