"""Shared fakes: in-memory repositories that mimic Mongo behaviour
(including DuplicateKeyError on the unique indexes), so the service layer is
tested without a database."""

from datetime import datetime, timezone

import pytest
from pymongo.errors import DuplicateKeyError

from app.models.prompt import AuditEntry, PromptVersion, RegistryEntry
from app.services.prompt_service import PromptService
from app.utils.cache import InMemoryCache


class FakePromptRepository:
    def __init__(self):
        self.docs: dict[tuple[str, int], PromptVersion] = {}

    async def insert_version(self, version: PromptVersion) -> None:
        key = (version.prompt_id, version.version)
        if key in self.docs:
            raise DuplicateKeyError("dup (prompt_id, version)")
        self.docs[key] = version

    async def get_version(self, prompt_id, version):
        return self.docs.get((prompt_id, version))

    async def list_versions(self, prompt_id):
        return sorted(
            (v for (pid, _), v in self.docs.items() if pid == prompt_id),
            key=lambda v: v.version,
        )

    async def exists(self, prompt_id):
        return any(pid == prompt_id for pid, _ in self.docs)


class FakeRegistryRepository:
    def __init__(self):
        self.entries: dict[str, RegistryEntry] = {}

    async def create(self, prompt_id):
        if prompt_id in self.entries:
            raise DuplicateKeyError("dup prompt_id")
        entry = RegistryEntry(
            prompt_id=prompt_id,
            current_version=1,
            latest_version=1,
            updated_at=datetime.now(timezone.utc),
        )
        self.entries[prompt_id] = entry
        return entry

    async def get(self, prompt_id):
        return self.entries.get(prompt_id)

    async def list_all(self):
        return sorted(self.entries.values(), key=lambda e: e.prompt_id)

    async def publish_version(self, prompt_id, version):
        entry = self.entries.get(prompt_id)
        if entry is None:
            return None
        updated = entry.model_copy(
            update={
                "current_version": version,
                "latest_version": max(entry.latest_version, version),
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self.entries[prompt_id] = updated
        return updated

    async def set_current_version(self, prompt_id, version):
        entry = self.entries.get(prompt_id)
        if entry is None:
            return None
        updated = entry.model_copy(
            update={
                "current_version": version,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self.entries[prompt_id] = updated
        return updated


class FakeAuditRepository:
    def __init__(self):
        self.entries: list[AuditEntry] = []

    async def record(self, entry: AuditEntry) -> None:
        self.entries.append(entry)

    async def list_for_prompt(self, prompt_id, limit=50):
        return [e for e in self.entries if e.prompt_id == prompt_id][:limit]


@pytest.fixture
def repos():
    return FakePromptRepository(), FakeRegistryRepository(), FakeAuditRepository()


@pytest.fixture
def cache():
    return InMemoryCache()


@pytest.fixture
def service(repos, cache):
    prompts, registry, audit = repos
    return PromptService(prompts, registry, audit, cache, cache_ttl_seconds=300)
