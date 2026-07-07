"""Password hashing and JWT helpers shared by the gateway and lg_auth."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from core.settings import settings

_ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(user_id: str, email: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "iat": now,
        "exp": now + timedelta(hours=settings.access_token_ttl_hours),
    }
    return jwt.encode(payload, settings.auth_secret, algorithm=_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Return the token payload; raises jwt.InvalidTokenError if invalid/expired."""
    return jwt.decode(token, settings.auth_secret, algorithms=[_ALGORITHM])
