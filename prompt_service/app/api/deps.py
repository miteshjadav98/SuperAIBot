"""FastAPI dependency providers.

The Mongo client and cache are created once at startup (see app.main) and
stored on app.state; per-request we only assemble cheap wrapper objects.
Tests can override `get_prompt_service` to inject fakes.
"""

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from app.config import get_settings
from app.repositories.audit_repository import AuditRepository
from app.repositories.prompt_repository import PromptRepository
from app.repositories.registry_repository import RegistryRepository
from app.services.prompt_service import PromptService


# Declared via fastapi.security so Swagger UI renders an "Authorize" button
# where testers paste the key once and it is sent with every request.
_api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="Required only when PROMPT_API_KEY is configured on the server.",
)


async def require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    expected = get_settings().prompt_api_key
    if expected is None:  # auth disabled (local development)
        return
    if api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )


def get_prompt_service(request: Request) -> PromptService:
    db = request.app.state.db
    return PromptService(
        prompts=PromptRepository(db),
        registry=RegistryRepository(db),
        audit=AuditRepository(db),
        cache=request.app.state.cache,
        cache_ttl_seconds=get_settings().cache_ttl_seconds,
    )
