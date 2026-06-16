"""Environment-driven settings for the demo service.

Reads `.env` when present; all values have safe defaults so the service
starts (in mock mode) with no configuration at all.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    app_env: str = "local"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    artifact_dir: str = "artifacts/index"
    llm_mode: str = "mock"  # mock | gemini | gemma | groq
    model_name: str = "mock-local"

    gemini_api_key: str = ""
    groq_api_key: str = ""
    llm_timeout_seconds: float = 90.0

    default_top_k: int = 5
    default_retrieval_mode: str = "hybrid"  # hybrid | dense | bm25
    enable_debug: bool = True
    enable_semantic_gate: bool = False  # experimental; needs dense backend


@lru_cache
def get_settings() -> Settings:
    return Settings()
