"""Request/response schemas — the public API contract.

Kept separate from `app.models` so internal storage can evolve without
breaking API consumers.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# prompt_id is a caller-chosen slug, e.g. "rag_system_prompt".
_PROMPT_ID_PATTERN = r"^[a-z0-9][a-z0-9_\-]{1,98}[a-z0-9]$"


class PromptCreateRequest(BaseModel):
    prompt_id: str = Field(
        pattern=_PROMPT_ID_PATTERN,
        description="Stable slug identifying the prompt across versions",
        examples=["rag_system_prompt"],
    )
    name: str = Field(min_length=1, max_length=200, examples=["RAG System Prompt"])
    content: str = Field(min_length=1, description="The prompt text itself")
    description: str = Field(default="", max_length=2000)
    created_by: str = Field(default="system", max_length=100)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form settings, e.g. model and temperature",
        examples=[{"model": "gpt-4.1", "temperature": 0.2}],
    )


class PromptUpdateRequest(BaseModel):
    """Creates a NEW version — never mutates an existing one."""

    content: str = Field(min_length=1)
    name: str | None = Field(default=None, max_length=200)
    description: str = Field(default="", max_length=2000)
    created_by: str = Field(default="system", max_length=100)
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Omit to inherit metadata from the latest version",
    )


class RollbackRequest(BaseModel):
    performed_by: str = Field(default="system", max_length=100)


class PromptVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    prompt_id: str
    version: int
    name: str
    content: str
    description: str
    created_by: str
    created_at: datetime
    metadata: dict[str, Any]
    is_active: bool = Field(
        description="True when this version is the registry's current_version"
    )


class PromptVersionListResponse(BaseModel):
    prompt_id: str
    current_version: int
    total: int
    versions: list[PromptVersionResponse]


class RegistryResponse(BaseModel):
    prompt_id: str
    current_version: int
    latest_version: int
    updated_at: datetime


class RollbackResponse(BaseModel):
    prompt_id: str
    rolled_back_to: int
    previous_version: int
    updated_at: datetime


class ErrorResponse(BaseModel):
    detail: str
