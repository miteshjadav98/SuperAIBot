"""Custom auth for the LangGraph dev server (:2024).

Registered via ``langgraph.json`` (``"auth": {"path": "./core/lg_auth.py:auth"}``).
Every request from the chat UI must carry ``Authorization: Bearer <JWT>`` issued
by the FastAPI gateway's ``/auth/login`` / ``/auth/register``. Threads are
stamped with ``metadata.owner`` on creation and all reads/searches are filtered
by it, so each user only ever sees their own chat history.
"""

from __future__ import annotations

import jwt as pyjwt
from langgraph_sdk import Auth

from core.security import decode_access_token

auth = Auth()


@auth.authenticate
async def authenticate(authorization: str | None) -> Auth.types.MinimalUserDict:
    if not authorization:
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Missing Authorization header"
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Expected 'Authorization: Bearer <token>'"
        )

    try:
        payload = decode_access_token(token)
    except pyjwt.InvalidTokenError:
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Invalid or expired token"
        )

    return {
        "identity": payload["sub"],
        "email": payload.get("email", ""),
        "is_authenticated": True,
    }


@auth.on
async def owner_only(ctx: Auth.types.AuthContext, value: dict):
    """Stamp new resources with the owner and filter every access by it."""
    filters = {"owner": ctx.user.identity}
    metadata = value.setdefault("metadata", {})
    metadata.update(filters)
    return filters
