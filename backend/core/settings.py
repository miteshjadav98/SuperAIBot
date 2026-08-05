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

    # Azure AI Document Intelligence (OCR + layout for PDFs; reads scanned pages
    # MarkItDown can't). Free (F0) tier: 500 pages/month, 20 calls/minute — the
    # monthly page budget below guards it and ingestion falls back to MarkItDown
    # when it's unset or exhausted. Leave endpoint/key empty to disable.
    azure_docintel_endpoint: Optional[str] = None
    azure_docintel_key: Optional[str] = None
    azure_docintel_model: str = "prebuilt-layout"
    azure_docintel_monthly_page_budget: int = 500  # 0 disables the cap

    # Tools.
    tavily_api_key: Optional[str] = None

    # Weather (Personal Assistant). "openmeteo" is free and needs no key or
    # signup — see tools/weather/. A provider is one module plus one line in
    # tools/weather/__init__.py, so OpenWeather or the weather-mcp server slot
    # in without touching the agent.
    weather_provider: str = "openmeteo"
    weather_timeout_seconds: float = 8.0

    # Tasks (Personal Assistant). "internal" is the built-in MongoDB list; it
    # falls back to an in-process list when MONGODB_URI is unset. "memory"
    # forces that in-process list. Notion/Google Tasks are documented seams.
    todo_provider: str = "internal"

    # Email. "mock" is a working in-memory mailbox (no credentials needed) —
    # see tools/email/. "gmail" is a documented seam, not yet implemented.
    email_provider: str = "mock"
    email_address: Optional[str] = None  # the mailbox owner's own address
    gmail_client_id: Optional[str] = None
    gmail_client_secret: Optional[str] = None

    # MongoDB Atlas (users, PDF vector search, gateway chat memory).
    mongodb_uri: Optional[str] = None
    mongodb_db: str = "superbot"

    # Auth (JWT for the gateway + the LangGraph server's custom auth).
    auth_secret: str = "change-me-in-.env"
    access_token_ttl_hours: int = 72

    # Super Bot router fallback when classification is uncertain. Defaults to
    # the Super Bot's own voice (superbot.state.ASSISTANT_AGENT_ID) rather than
    # a specialist — an uncertain turn should sound like the assistant, not like
    # a personal chef who was handed a question about weddings.
    default_agent_id: str = "assistant"

    # What happens when a second request arrives on a thread that is still
    # running: "reject" (409, the default) or "enqueue" (run it after).
    run_concurrency_policy: str = "reject"

    # USD per MILLION tokens, as JSON, e.g.
    #   {"gpt-4.1": {"input": 2.0, "output": 8.0}}
    # Empty by default: rates depend on provider, region and contract, so
    # tokens are always recorded but cost is only computed once you set yours.
    model_pricing: Optional[str] = None


settings = Settings()
