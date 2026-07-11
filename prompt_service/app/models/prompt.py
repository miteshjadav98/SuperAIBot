"""Domain models — the internal shape of documents, independent of the API.

Design note: version documents are immutable. There is deliberately no
`status` field on a version; which version is "active" is decided solely by
`prompt_registry.current_version` so there is a single source of truth.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AuditAction(str, Enum):
    CREATE_PROMPT = "CREATE_PROMPT"
    CREATE_VERSION = "CREATE_VERSION"
    ROLLBACK = "ROLLBACK"


class PromptVersion(BaseModel):
    """One immutable version of a prompt (a document in `prompts`)."""

    model_config = ConfigDict(populate_by_name=True)

    id: str | None = Field(default=None, alias="_id")
    prompt_id: str
    version: int
    name: str
    content: str
    description: str = ""
    created_by: str = "system"
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class RegistryEntry(BaseModel):
    """Pointer to the active version of a prompt (a document in
    `prompt_registry`)."""

    prompt_id: str
    current_version: int
    latest_version: int
    updated_at: datetime


class AuditEntry(BaseModel):
    """One row in the append-only `prompt_audit` collection."""

    prompt_id: str
    version: int
    action: AuditAction
    performed_by: str
    timestamp: datetime
    detail: str = ""
