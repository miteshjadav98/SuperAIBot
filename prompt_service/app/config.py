"""Environment-driven configuration.

Everything the service needs comes from environment variables (or a local
`.env` file). No values are hard-coded so the same image runs in dev,
staging, and production.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # MongoDB Atlas connection string, e.g.
    # mongodb+srv://user:pass@cluster0.xxxxx.mongodb.net
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "prompt_management"

    # Path prefix when served behind a reverse proxy that strips it, e.g.
    # ROOT_PATH=/prompt-api for nginx `location /prompt-api/ { proxy_pass ...:8020/; }`.
    # Makes Swagger UI (/docs) and its "Try it out" calls work on the proxy URL.
    root_path: str = ""

    # Optional shared secret. When set, every /prompts endpoint requires the
    # X-API-Key header (Swagger UI gets an Authorize button for it). Leave
    # unset for local development. STRONGLY recommended when publicly hosted.
    prompt_api_key: str | None = None

    # Optional. When unset, the service uses an in-process TTL cache instead,
    # which is fine for a single instance. Set REDIS_URL for multi-instance
    # deployments so cache invalidation is shared.
    redis_url: str | None = None
    cache_ttl_seconds: int = 300

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
