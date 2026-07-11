"""Business rules for prompt versioning.

Invariants enforced here:
- Version documents are immutable: updates always insert a new document.
- The registry's current_version is the only definition of "active".
- New versions are allocated as latest_version + 1 under a unique index, with
  retry on DuplicateKeyError, so concurrent writers can never overwrite or
  duplicate a version.
- The active-prompt cache is invalidated on every publish and rollback.
"""

import logging
from datetime import datetime, timezone

from pymongo.errors import DuplicateKeyError

from app.models.prompt import AuditAction, AuditEntry, PromptVersion, RegistryEntry
from app.repositories.audit_repository import AuditRepository
from app.repositories.prompt_repository import PromptRepository
from app.repositories.registry_repository import RegistryRepository
from app.schemas.prompt import PromptCreateRequest, PromptUpdateRequest
from app.utils.cache import CacheBackend, active_prompt_key
from app.utils.exceptions import (
    PromptAlreadyExistsError,
    PromptNotFoundError,
    VersionConflictError,
    VersionNotFoundError,
)

logger = logging.getLogger(__name__)

_VERSION_ALLOCATION_RETRIES = 3


class PromptService:
    def __init__(
        self,
        prompts: PromptRepository,
        registry: RegistryRepository,
        audit: AuditRepository,
        cache: CacheBackend,
        cache_ttl_seconds: int = 300,
    ) -> None:
        self._prompts = prompts
        self._registry = registry
        self._audit = audit
        self._cache = cache
        self._cache_ttl = cache_ttl_seconds

    # -- Commands -------------------------------------------------------------

    async def create_prompt(self, request: PromptCreateRequest) -> PromptVersion:
        """Create a new prompt: version 1 plus its registry row."""
        version = PromptVersion(
            prompt_id=request.prompt_id,
            version=1,
            name=request.name,
            content=request.content,
            description=request.description or "Initial version",
            created_by=request.created_by,
            created_at=datetime.now(timezone.utc),
            metadata=request.metadata,
        )
        try:
            # Registry first: its unique prompt_id index is the authoritative
            # existence check, so two concurrent creates cannot both succeed.
            await self._registry.create(request.prompt_id)
        except DuplicateKeyError:
            raise PromptAlreadyExistsError(request.prompt_id)

        await self._prompts.insert_version(version)
        await self._audit_log(
            request.prompt_id, 1, AuditAction.CREATE_PROMPT, request.created_by
        )
        logger.info("Created prompt '%s' v1", request.prompt_id)
        return version

    async def create_new_version(
        self, prompt_id: str, request: PromptUpdateRequest
    ) -> PromptVersion:
        """Append a new immutable version and publish it.

        Never modifies existing documents. Name/metadata are inherited from
        the latest version when the request omits them, so callers can PUT
        just new content.
        """
        registry = await self._require_registry(prompt_id)
        latest = await self._prompts.get_version(prompt_id, registry.latest_version)
        if latest is None:  # registry exists but the document is missing
            raise VersionNotFoundError(prompt_id, registry.latest_version)

        next_version = registry.latest_version + 1
        for attempt in range(_VERSION_ALLOCATION_RETRIES):
            version = PromptVersion(
                prompt_id=prompt_id,
                version=next_version,
                name=request.name or latest.name,
                content=request.content,
                description=request.description,
                created_by=request.created_by,
                created_at=datetime.now(timezone.utc),
                metadata=request.metadata if request.metadata is not None else latest.metadata,
            )
            try:
                await self._prompts.insert_version(version)
                break
            except DuplicateKeyError:
                # Another writer took this number; move to the next one.
                logger.warning(
                    "Version %s of '%s' taken concurrently (attempt %s); retrying",
                    next_version, prompt_id, attempt + 1,
                )
                next_version += 1
        else:
            raise VersionConflictError(prompt_id)

        await self._registry.publish_version(prompt_id, next_version)
        await self._cache.delete(active_prompt_key(prompt_id))
        await self._audit_log(
            prompt_id, next_version, AuditAction.CREATE_VERSION, request.created_by
        )
        logger.info("Published prompt '%s' v%s", prompt_id, next_version)
        return version

    async def rollback_prompt(
        self, prompt_id: str, version: int, performed_by: str = "system"
    ) -> tuple[RegistryEntry, int]:
        """Repoint current_version at an existing historical version.

        Returns (updated registry entry, the version that was active before).
        History is never modified — rolling back is just moving the pointer.
        """
        registry = await self._require_registry(prompt_id)
        target = await self._prompts.get_version(prompt_id, version)
        if target is None:
            raise VersionNotFoundError(prompt_id, version)

        previous = registry.current_version
        updated = await self._registry.set_current_version(prompt_id, version)
        if updated is None:  # registry row vanished between reads
            raise PromptNotFoundError(prompt_id)

        await self._cache.delete(active_prompt_key(prompt_id))
        await self._audit_log(
            prompt_id, version, AuditAction.ROLLBACK, performed_by,
            detail=f"from v{previous}",
        )
        logger.info("Rolled back '%s' from v%s to v%s", prompt_id, previous, version)
        return updated, previous

    # -- Queries --------------------------------------------------------------

    async def get_active_prompt(self, prompt_id: str) -> PromptVersion:
        """Cache-first read of the version the registry points at."""
        key = active_prompt_key(prompt_id)
        cached = await self._cache.get(key)
        if cached is not None:
            return PromptVersion.model_validate_json(cached)

        registry = await self._require_registry(prompt_id)
        version = await self._prompts.get_version(prompt_id, registry.current_version)
        if version is None:
            raise VersionNotFoundError(prompt_id, registry.current_version)

        await self._cache.set(key, version.model_dump_json(), self._cache_ttl)
        return version

    async def get_prompt_history(
        self, prompt_id: str
    ) -> tuple[RegistryEntry, list[PromptVersion]]:
        registry = await self._require_registry(prompt_id)
        versions = await self._prompts.list_versions(prompt_id)
        return registry, versions

    async def get_prompt_version(self, prompt_id: str, version: int) -> PromptVersion:
        await self._require_registry(prompt_id)
        doc = await self._prompts.get_version(prompt_id, version)
        if doc is None:
            raise VersionNotFoundError(prompt_id, version)
        return doc

    async def list_prompts(self) -> list[RegistryEntry]:
        return await self._registry.list_all()

    async def get_registry(self, prompt_id: str) -> RegistryEntry:
        return await self._require_registry(prompt_id)

    # -- Internals ------------------------------------------------------------

    async def _require_registry(self, prompt_id: str) -> RegistryEntry:
        registry = await self._registry.get(prompt_id)
        if registry is None:
            raise PromptNotFoundError(prompt_id)
        return registry

    async def _audit_log(
        self,
        prompt_id: str,
        version: int,
        action: AuditAction,
        performed_by: str,
        detail: str = "",
    ) -> None:
        await self._audit.record(
            AuditEntry(
                prompt_id=prompt_id,
                version=version,
                action=action,
                performed_by=performed_by,
                timestamp=datetime.now(timezone.utc),
                detail=detail,
            )
        )
