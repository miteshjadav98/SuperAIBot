"""Append-only audit log (`prompt_audit`).

Audit writes are best-effort: an audit failure is logged loudly but never
fails the business operation that triggered it.
"""

import logging

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database.mongo import AUDIT_COLLECTION
from app.models.prompt import AuditEntry

logger = logging.getLogger(__name__)


class AuditRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db[AUDIT_COLLECTION]

    async def record(self, entry: AuditEntry) -> None:
        try:
            await self._col.insert_one(entry.model_dump())
        except Exception:  # noqa: BLE001
            logger.exception(
                "Audit write failed for %s v%s (%s)",
                entry.prompt_id,
                entry.version,
                entry.action,
            )

    async def list_for_prompt(self, prompt_id: str, limit: int = 50) -> list[AuditEntry]:
        cursor = (
            self._col.find({"prompt_id": prompt_id}, {"_id": 0})
            .sort("timestamp", -1)
            .limit(limit)
        )
        return [AuditEntry.model_validate(doc) async for doc in cursor]
