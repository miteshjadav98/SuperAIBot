"""Data access for the `prompts` collection (immutable version documents).

Repositories know Mongo; they raise pymongo errors upward and return domain
models. No business rules live here — that's the service layer's job.
"""

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database.mongo import PROMPTS_COLLECTION
from app.models.prompt import PromptVersion


class PromptRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db[PROMPTS_COLLECTION]

    async def insert_version(self, version: PromptVersion) -> None:
        """Insert one immutable version document.

        Raises pymongo.errors.DuplicateKeyError when (prompt_id, version)
        already exists — the service layer uses this for optimistic
        concurrency control.
        """
        doc = version.model_dump(exclude={"id"})
        await self._col.insert_one(doc)

    async def get_version(self, prompt_id: str, version: int) -> PromptVersion | None:
        doc = await self._col.find_one({"prompt_id": prompt_id, "version": version})
        return self._to_model(doc) if doc else None

    async def list_versions(self, prompt_id: str) -> list[PromptVersion]:
        cursor = self._col.find({"prompt_id": prompt_id}).sort("version", 1)
        return [self._to_model(doc) async for doc in cursor]

    async def exists(self, prompt_id: str) -> bool:
        return await self._col.count_documents({"prompt_id": prompt_id}, limit=1) > 0

    @staticmethod
    def _to_model(doc: dict) -> PromptVersion:
        doc["_id"] = str(doc["_id"])
        return PromptVersion.model_validate(doc)
