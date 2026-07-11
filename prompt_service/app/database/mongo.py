"""Async MongoDB (Motor) client lifecycle and index management."""

import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import Settings

logger = logging.getLogger(__name__)

PROMPTS_COLLECTION = "prompts"
REGISTRY_COLLECTION = "prompt_registry"
AUDIT_COLLECTION = "prompt_audit"


def create_client(settings: Settings) -> AsyncIOMotorClient:
    return AsyncIOMotorClient(
        settings.mongodb_uri,
        serverSelectionTimeoutMS=5000,
        uuidRepresentation="standard",
    )


async def ensure_indexes(db: AsyncIOMotorDatabase) -> None:
    """Create indexes idempotently at startup.

    The unique (prompt_id, version) index is what makes concurrent version
    creation safe: two writers racing for the same version number cannot both
    succeed, and the loser retries with the next number.
    """
    await db[PROMPTS_COLLECTION].create_index(
        [("prompt_id", 1), ("version", -1)], unique=True
    )
    await db[REGISTRY_COLLECTION].create_index([("prompt_id", 1)], unique=True)
    # Audit log is queried per prompt, newest first.
    await db[AUDIT_COLLECTION].create_index([("prompt_id", 1), ("timestamp", -1)])
    logger.info("MongoDB indexes ensured")
