"""Shared test fixtures.

Two things make this suite fast and hermetic:

* ``MONGODB_URI`` is cleared before any platform module is imported, so nothing
  reaches for Atlas. Prompt loading falls back to code defaults instantly
  instead of waiting out a 10-second server-selection timeout per prompt.
* :func:`store` hands back a ``MongoDBStore`` backed by ``mongomock``, so the
  store is exercised through real pymongo calls without a running database.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

BACKEND = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.settings import settings  # noqa: E402

settings.mongodb_uri = None


@pytest.fixture
def store():
    """A MongoDBStore over an in-process Mongo."""
    import mongomock

    from core.store import MongoDBStore

    return MongoDBStore(mongomock.MongoClient()["test"]["agent_memory"])


@pytest.fixture
def memory(store, monkeypatch):
    """The memory module, wired to the mock store."""
    import core.memory as memory_module

    monkeypatch.setattr(memory_module, "_store", store)
    return memory_module
