"""HTTP endpoints for prompt management.

Thin layer: validate input (Pydantic), call the service, shape the response.
Domain exceptions are translated to HTTP errors by handlers in app.main.
"""

from fastapi import APIRouter, Depends, status

from app.api.deps import get_prompt_service, require_api_key
from app.models.prompt import PromptVersion
from app.schemas.prompt import (
    ErrorResponse,
    PromptCreateRequest,
    PromptUpdateRequest,
    PromptVersionListResponse,
    PromptVersionResponse,
    RegistryResponse,
    RollbackRequest,
    RollbackResponse,
)
from app.services.prompt_service import PromptService

# require_api_key is a no-op until PROMPT_API_KEY is set on the server.
router = APIRouter(
    prefix="/prompts", tags=["prompts"], dependencies=[Depends(require_api_key)]
)

_NOT_FOUND = {status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}}
_CONFLICT = {status.HTTP_409_CONFLICT: {"model": ErrorResponse}}


def _to_response(version: PromptVersion, active_version: int) -> PromptVersionResponse:
    return PromptVersionResponse(
        prompt_id=version.prompt_id,
        version=version.version,
        name=version.name,
        content=version.content,
        description=version.description,
        created_by=version.created_by,
        created_at=version.created_at,
        metadata=version.metadata,
        is_active=version.version == active_version,
    )


@router.get("", response_model=list[RegistryResponse], summary="List all prompts")
async def list_prompts(service: PromptService = Depends(get_prompt_service)):
    """All registry entries: every prompt with its current and latest version."""
    return await service.list_prompts()


@router.post(
    "",
    response_model=PromptVersionResponse,
    status_code=status.HTTP_201_CREATED,
    responses=_CONFLICT,
    summary="Create a new prompt (version 1)",
)
async def create_prompt(
    request: PromptCreateRequest,
    service: PromptService = Depends(get_prompt_service),
):
    version = await service.create_prompt(request)
    return _to_response(version, active_version=1)


@router.put(
    "/{prompt_id}",
    response_model=PromptVersionResponse,
    status_code=status.HTTP_201_CREATED,
    responses=_NOT_FOUND,
    summary="Publish a new version (immutable append)",
)
async def create_new_version(
    prompt_id: str,
    request: PromptUpdateRequest,
    service: PromptService = Depends(get_prompt_service),
):
    """Never overwrites: inserts version N+1 and points the registry at it.
    Omitted name/metadata are inherited from the latest version."""
    version = await service.create_new_version(prompt_id, request)
    return _to_response(version, active_version=version.version)


@router.get(
    "/{prompt_id}",
    response_model=PromptVersionResponse,
    responses=_NOT_FOUND,
    summary="Get the active version (cache-first)",
)
async def get_active_prompt(
    prompt_id: str, service: PromptService = Depends(get_prompt_service)
):
    version = await service.get_active_prompt(prompt_id)
    return _to_response(version, active_version=version.version)


@router.get(
    "/{prompt_id}/versions",
    response_model=PromptVersionListResponse,
    responses=_NOT_FOUND,
    summary="Full version history",
)
async def get_prompt_history(
    prompt_id: str, service: PromptService = Depends(get_prompt_service)
):
    registry, versions = await service.get_prompt_history(prompt_id)
    return PromptVersionListResponse(
        prompt_id=prompt_id,
        current_version=registry.current_version,
        total=len(versions),
        versions=[_to_response(v, registry.current_version) for v in versions],
    )


@router.get(
    "/{prompt_id}/versions/{version}",
    response_model=PromptVersionResponse,
    responses=_NOT_FOUND,
    summary="Get one specific version",
)
async def get_prompt_version(
    prompt_id: str,
    version: int,
    service: PromptService = Depends(get_prompt_service),
):
    doc = await service.get_prompt_version(prompt_id, version)
    registry = await service.get_registry(prompt_id)
    return _to_response(doc, registry.current_version)


@router.post(
    "/{prompt_id}/rollback/{version}",
    response_model=RollbackResponse,
    responses=_NOT_FOUND,
    summary="Roll back to a historical version",
)
async def rollback_prompt(
    prompt_id: str,
    version: int,
    request: RollbackRequest | None = None,
    service: PromptService = Depends(get_prompt_service),
):
    """Moves the registry pointer only; no historical document is touched."""
    performed_by = request.performed_by if request else "system"
    registry, previous = await service.rollback_prompt(prompt_id, version, performed_by)
    return RollbackResponse(
        prompt_id=prompt_id,
        rolled_back_to=registry.current_version,
        previous_version=previous,
        updated_at=registry.updated_at,
    )
