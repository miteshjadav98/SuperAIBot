"""Central configuration for the Super Bot platform.

A single ``settings`` singleton, read once from the project ``.env`` (one level
above ``backend/``). Every module imports this instead of reaching into
``os.environ`` directly, so there is exactly one place that knows about config.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/core/settings.py -> backend/ -> project root (.env lives here)
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",  # the .env carries course/tracing keys we don't model
        case_sensitive=False,
    )

    # Which provider get_chat_model() uses when none is passed explicitly.
    default_llm_provider: str = "azure"

    # Azure OpenAI (the provider the four agents use today).
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_version: str = "2024-02-01"
    azure_openai_deployment: str = "gpt-4.1"
    azure_openai_embedding_deployment: str = "text-embedding-3-small"

    # Other providers (optional — only needed if you switch default_llm_provider).
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"

    # Tools.
    tavily_api_key: Optional[str] = None

    # Redis (checkpointer / session memory).
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None

    # Super Bot router fallback when classification is uncertain.
    default_agent_id: str = "personal_chef"


settings = Settings()
