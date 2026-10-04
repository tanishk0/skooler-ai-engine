from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Values are read from the environment or .env."""

    # Active provider selection: "auto", "gemini", or "openai"
    llm_provider: str = "auto"

    # Gemini configuration
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"

    # OpenAI configuration
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None

    # Generic model override (overrides provider-specific model when set)
    llm_model: str | None = None

    frontend_url: str = "http://localhost:3000"
    max_upload_size_mb: int = 15
    chunk_size: int = 1200
    chunk_overlap: int = 150
    llm_timeout_seconds: int = 45

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def active_provider(self) -> str:
        """Resolve the effective provider: respects explicit setting or autodetects from keys."""
        provider = (self.llm_provider or "auto").strip().lower()
        if provider in ("gemini", "google"):
            return "gemini"
        if provider == "openai":
            return "openai"
        if provider == "auto":
            if self.openai_api_key and not self.gemini_api_key:
                return "openai"
            if self.gemini_api_key and not self.openai_api_key:
                return "gemini"
            # Default to gemini when both or neither are provided for backwards compatibility
            return "gemini"
        return provider


@lru_cache
def get_settings() -> Settings:
    return Settings()
