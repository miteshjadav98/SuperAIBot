"""Cache abstraction: Redis when configured, in-process TTL cache otherwise.

The service only depends on the `CacheBackend` protocol, so swapping the
implementation (or disabling caching in tests) requires no service changes.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

from app.config import Settings

logger = logging.getLogger(__name__)


class CacheBackend(Protocol):
    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, ttl_seconds: int) -> None: ...

    async def delete(self, key: str) -> None: ...

    async def close(self) -> None: ...


class InMemoryCache:
    """Single-process TTL cache. Suitable for one instance or for tests.

    For multi-instance deployments use Redis so an invalidation performed by
    one instance is seen by all of them.
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, str]] = {}

    async def get(self, key: str) -> str | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() >= expires_at:
            self._store.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self._store[key] = (time.monotonic() + ttl_seconds, value)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def close(self) -> None:
        self._store.clear()


class RedisCache:
    def __init__(self, url: str) -> None:
        import redis.asyncio as redis  # imported lazily: redis is optional

        self._client = redis.from_url(url, decode_responses=True)

    async def get(self, key: str) -> str | None:
        try:
            return await self._client.get(key)
        except Exception:  # noqa: BLE001 — a cache outage must never break reads
            logger.warning("Redis GET failed for %s; treating as miss", key)
            return None

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        try:
            await self._client.set(key, value, ex=ttl_seconds)
        except Exception:  # noqa: BLE001
            logger.warning("Redis SET failed for %s; skipping cache write", key)

    async def delete(self, key: str) -> None:
        # Deliberately NOT swallowed into a silent no-op without logging:
        # a failed invalidation means stale data may be served until TTL.
        try:
            await self._client.delete(key)
        except Exception:  # noqa: BLE001
            logger.error("Redis DELETE failed for %s; stale cache until TTL", key)

    async def close(self) -> None:
        await self._client.aclose()


def build_cache(settings: Settings) -> CacheBackend:
    if settings.redis_url:
        logger.info("Using Redis cache at %s", settings.redis_url)
        return RedisCache(settings.redis_url)
    logger.info("REDIS_URL not set; using in-process TTL cache")
    return InMemoryCache()


def active_prompt_key(prompt_id: str) -> str:
    return f"prompt:active:{prompt_id}"
