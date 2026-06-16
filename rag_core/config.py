"""Environment-driven settings for the demo service.

Reads `.env` when present; all values have safe defaults so the service
starts (in mock mode) with no configuration at all.
"""

from functools import lru_cache
from typing import Literal

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

    corpus_profile: Literal["sample", "public_226"] = "sample"
    artifact_dir: str = "artifacts/index"
    public_226_artifact_dir: str = "data/public_corpus_226"
    llm_mode: str = "mock"  # mock | gemini | gemma | groq | ollama
    model_name: str = "mock-local"

    gemini_api_key: str = ""
    groq_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    llm_timeout_seconds: float = 300.0

    default_top_k: int = 5
    default_retrieval_mode: str = "hybrid"  # hybrid | dense | bm25
    enable_debug: bool = True
    enable_semantic_gate: bool = False  # experimental; needs dense backend

    @property
    def resolved_artifact_dir(self) -> str:
        if self.corpus_profile == "public_226":
            return self.public_226_artifact_dir
        return self.artifact_dir


@lru_cache
def get_settings() -> Settings:
    return Settings()
