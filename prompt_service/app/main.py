"""Prompt Management Service — FastAPI entrypoint.

Run locally:
    uvicorn app.main:app --reload --port 8020
Swagger UI:
    http://localhost:8020/docs
    (on the Azure VM: https://aibot.miteklabs.tech/prompt-api/docs)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.api.routes import router as prompts_router
from app.config import get_settings
from app.database.mongo import create_client, ensure_indexes
from app.utils.cache import build_cache
from app.utils.exceptions import (
    PromptAlreadyExistsError,
    PromptNotFoundError,
    VersionConflictError,
    VersionNotFoundError,
)
from app.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)

    client = create_client(settings)
    app.state.db = client[settings.mongodb_db]
    await ensure_indexes(app.state.db)

    app.state.cache = build_cache(settings)
    logger.info("Prompt service started (db=%s)", settings.mongodb_db)
    yield

    await app.state.cache.close()
    client.close()


app = FastAPI(
    title="Prompt Management Service",
    description=(
        "Versioned, immutable prompt storage on MongoDB Atlas. "
        "Updates always create a new version; the registry decides which "
        "version is active; rollback just moves the pointer."
    ),
    version="1.0.0",
    lifespan=lifespan,
    # Behind nginx the service is mounted under a stripped prefix; root_path
    # tells Swagger/OpenAPI the public base URL so /docs works when hosted.
    root_path=get_settings().root_path,
)

app.include_router(prompts_router)


# --- Domain error -> HTTP mapping --------------------------------------------

def _error(status_code: int, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": str(exc)})


@app.exception_handler(PromptNotFoundError)
async def prompt_not_found(_: Request, exc: PromptNotFoundError):
    return _error(status.HTTP_404_NOT_FOUND, exc)


@app.exception_handler(VersionNotFoundError)
async def version_not_found(_: Request, exc: VersionNotFoundError):
    return _error(status.HTTP_404_NOT_FOUND, exc)


@app.exception_handler(PromptAlreadyExistsError)
async def prompt_exists(_: Request, exc: PromptAlreadyExistsError):
    return _error(status.HTTP_409_CONFLICT, exc)


@app.exception_handler(VersionConflictError)
async def version_conflict(_: Request, exc: VersionConflictError):
    return _error(status.HTTP_409_CONFLICT, exc)


@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8020, reload=True)
