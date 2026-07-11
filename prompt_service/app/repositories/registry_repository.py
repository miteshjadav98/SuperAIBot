"""Data access for the `prompt_registry` collection.

One document per prompt_id, tracking both:
- current_version: what consumers should use (changed by publish/rollback)
- latest_version:  highest version ever created (never decreases)

Keeping latest_version here means a rollback to v2 followed by a new version
still allocates v4 (not a colliding v3) without scanning the prompts
collection.
"""

from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database.mongo import REGISTRY_COLLECTION
from app.models.prompt import RegistryEntry


class RegistryRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db[REGISTRY_COLLECTION]

    async def create(self, prompt_id: str) -> RegistryEntry:
        """Insert the registry row for a brand-new prompt (version 1).

        Raises DuplicateKeyError if the prompt_id is already registered.
        """
        entry = RegistryEntry(
            prompt_id=prompt_id,
            current_version=1,
            latest_version=1,
            updated_at=datetime.now(timezone.utc),
        )
        await self._col.insert_one(entry.model_dump())
        return entry

    async def get(self, prompt_id: str) -> RegistryEntry | None:
        doc = await self._col.find_one({"prompt_id": prompt_id}, {"_id": 0})
        return RegistryEntry.model_validate(doc) if doc else None

    async def list_all(self) -> list[RegistryEntry]:
        cursor = self._col.find({}, {"_id": 0}).sort("prompt_id", 1)
        return [RegistryEntry.model_validate(doc) async for doc in cursor]

    async def publish_version(self, prompt_id: str, version: int) -> RegistryEntry | None:
        """Point current_version at `version` after a new version is created,
        advancing latest_version with $max so it never moves backwards."""
        doc = await self._col.find_one_and_update(
            {"prompt_id": prompt_id},
            {
                "$set": {
                    "current_version": version,
                    "updated_at": datetime.now(timezone.utc),
                },
                "$max": {"latest_version": version},
            },
            projection={"_id": 0},
            return_document=True,
        )
        return RegistryEntry.model_validate(doc) if doc else None

    async def set_current_version(self, prompt_id: str, version: int) -> RegistryEntry | None:
        """Rollback path: repoint current_version only. latest_version and the
        historical documents are untouched."""
        doc = await self._col.find_one_and_update(
            {"prompt_id": prompt_id},
            {
                "$set": {
                    "current_version": version,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
            projection={"_id": 0},
            return_document=True,
        )
        return RegistryEntry.model_validate(doc) if doc else None
