"""Unit tests for PromptService against in-memory fakes.

Run from prompt_service/:  pytest -q
"""

import pytest

from app.models.prompt import AuditAction, PromptVersion
from app.schemas.prompt import PromptCreateRequest, PromptUpdateRequest
from app.utils.cache import active_prompt_key
from app.utils.exceptions import (
    PromptAlreadyExistsError,
    PromptNotFoundError,
    VersionNotFoundError,
)

pytestmark = pytest.mark.asyncio


def _create_req(prompt_id="rag_system_prompt", content="You are a RAG assistant."):
    return PromptCreateRequest(
        prompt_id=prompt_id,
        name="RAG System Prompt",
        content=content,
        created_by="admin",
        metadata={"model": "gpt-4.1", "temperature": 0.2},
    )


async def test_create_prompt_makes_version_1_and_registry(service, repos):
    _, registry_repo, audit_repo = repos
    version = await service.create_prompt(_create_req())

    assert version.version == 1
    entry = await registry_repo.get("rag_system_prompt")
    assert entry.current_version == 1 and entry.latest_version == 1
    assert audit_repo.entries[0].action == AuditAction.CREATE_PROMPT


async def test_create_duplicate_prompt_rejected(service):
    await service.create_prompt(_create_req())
    with pytest.raises(PromptAlreadyExistsError):
        await service.create_prompt(_create_req())


async def test_update_appends_new_version_and_keeps_history(service, repos):
    prompt_repo, registry_repo, _ = repos
    await service.create_prompt(_create_req(content="v1 content"))

    v2 = await service.create_new_version(
        "rag_system_prompt",
        PromptUpdateRequest(content="v2 content", created_by="admin"),
    )

    assert v2.version == 2
    # Immutability: v1 still exists, unmodified.
    v1 = await prompt_repo.get_version("rag_system_prompt", 1)
    assert v1.content == "v1 content"
    entry = await registry_repo.get("rag_system_prompt")
    assert entry.current_version == 2 and entry.latest_version == 2


async def test_update_inherits_name_and_metadata(service):
    await service.create_prompt(_create_req())
    v2 = await service.create_new_version(
        "rag_system_prompt", PromptUpdateRequest(content="new")
    )
    assert v2.name == "RAG System Prompt"
    assert v2.metadata == {"model": "gpt-4.1", "temperature": 0.2}


async def test_update_unknown_prompt_raises(service):
    with pytest.raises(PromptNotFoundError):
        await service.create_new_version("nope", PromptUpdateRequest(content="x"))


async def test_version_race_retries_next_number(service, repos):
    """If another writer grabs version N concurrently, the service retries
    with N+1 instead of failing or overwriting."""
    prompt_repo, _, _ = repos
    await service.create_prompt(_create_req())

    # Simulate a concurrent writer that already inserted v2 without the
    # registry knowing yet (registry.latest_version is still 1).
    v1 = await prompt_repo.get_version("rag_system_prompt", 1)
    await prompt_repo.insert_version(
        PromptVersion(**{**v1.model_dump(), "version": 2, "id": None})
    )

    v3 = await service.create_new_version(
        "rag_system_prompt", PromptUpdateRequest(content="mine")
    )
    assert v3.version == 3


async def test_rollback_moves_pointer_only(service, repos):
    prompt_repo, registry_repo, audit_repo = repos
    await service.create_prompt(_create_req(content="v1"))
    await service.create_new_version(
        "rag_system_prompt", PromptUpdateRequest(content="v2")
    )

    entry, previous = await service.rollback_prompt("rag_system_prompt", 1, "admin")

    assert previous == 2
    assert entry.current_version == 1
    assert entry.latest_version == 2  # latest never decreases
    assert (await prompt_repo.get_version("rag_system_prompt", 2)).content == "v2"
    assert audit_repo.entries[-1].action == AuditAction.ROLLBACK


async def test_rollback_to_missing_version_raises(service):
    await service.create_prompt(_create_req())
    with pytest.raises(VersionNotFoundError):
        await service.rollback_prompt("rag_system_prompt", 99)


async def test_new_version_after_rollback_does_not_collide(service):
    await service.create_prompt(_create_req(content="v1"))
    await service.create_new_version(
        "rag_system_prompt", PromptUpdateRequest(content="v2")
    )
    await service.rollback_prompt("rag_system_prompt", 1)

    v3 = await service.create_new_version(
        "rag_system_prompt", PromptUpdateRequest(content="v3")
    )
    assert v3.version == 3  # latest_version+1, not current_version+1


async def test_get_active_uses_cache_and_invalidation(service, cache):
    await service.create_prompt(_create_req(content="v1"))

    first = await service.get_active_prompt("rag_system_prompt")
    assert first.content == "v1"
    assert await cache.get(active_prompt_key("rag_system_prompt")) is not None

    # Publishing v2 must invalidate, so the next read sees v2 not cached v1.
    await service.create_new_version(
        "rag_system_prompt", PromptUpdateRequest(content="v2")
    )
    assert await cache.get(active_prompt_key("rag_system_prompt")) is None
    assert (await service.get_active_prompt("rag_system_prompt")).content == "v2"

    # Rollback invalidates too.
    await service.rollback_prompt("rag_system_prompt", 1)
    assert (await service.get_active_prompt("rag_system_prompt")).content == "v1"


async def test_get_active_unknown_prompt_raises(service):
    with pytest.raises(PromptNotFoundError):
        await service.get_active_prompt("ghost")


async def test_history_sorted_ascending(service):
    await service.create_prompt(_create_req(content="v1"))
    for i in (2, 3):
        await service.create_new_version(
            "rag_system_prompt", PromptUpdateRequest(content=f"v{i}")
        )
    registry, versions = await service.get_prompt_history("rag_system_prompt")
    assert [v.version for v in versions] == [1, 2, 3]
    assert registry.current_version == 3
