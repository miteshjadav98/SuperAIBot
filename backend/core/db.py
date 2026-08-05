"""MongoDB Atlas access — one lazily-created client for the whole backend.

Used for three things: the ``users`` collection (auth), the ``pdf_chunks``
collection (Atlas Vector Search for the PDF chatbot), and gateway chat
checkpoints. Everything degrades gracefully when ``MONGODB_URI`` is unset so
the platform still boots before the Atlas cluster exists.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from core.settings import settings


def mongo_configured() -> bool:
    return bool(settings.mongodb_uri)


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    if not settings.mongodb_uri:
        raise RuntimeError(
            "MONGODB_URI is not set. Add your Atlas connection string to .env "
            "(see README: MongoDB Atlas setup)."
        )
    return MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=10_000)


def get_db() -> Database:
    return get_client()[settings.mongodb_db]


@lru_cache(maxsize=1)
def users_collection() -> Collection:
    users = get_db()["users"]
    users.create_index("email", unique=True)
    return users


def pdf_chunks_collection() -> Optional[Collection]:
    if not mongo_configured():
        return None
    return get_db()["pdf_chunks"]


def agent_memory_collection() -> Optional[Collection]:
    """Backing collection for :class:`core.store.MongoDBStore` — the platform's
    long-term memory, shared by every agent."""
    if not mongo_configured():
        return None
    return get_db()["agent_memory"]


def run_metrics_collection() -> Optional[Collection]:
    """One row per graph invocation — tokens, latency, cost. See
    :mod:`core.telemetry`."""
    if not mongo_configured():
        return None
    return get_db()["run_metrics"]


def usage_counters_collection() -> Optional[Collection]:
    """Small counters keyed by ``<service>:<YYYY-MM>`` — used to track the Azure
    Document Intelligence free-tier monthly page budget across restarts."""
    if not mongo_configured():
        return None
    return get_db()["usage_counters"]
