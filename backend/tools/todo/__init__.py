"""Task backends, chosen by ``settings.todo_provider``.

    from tools.todo import get_provider
    tasks = get_provider().list(owner)

``internal`` (MongoDB) is the default and degrades to the in-process list when
Mongo isn't configured — the same "still usable before Atlas exists" rule the
rest of the platform follows. Adding Notion or Google Tasks: implement
:class:`~tools.todo.base.TodoProvider`, add a line to ``_PROVIDERS``, set
``TODO_PROVIDER`` in .env.
"""

from __future__ import annotations

import importlib
import logging
from functools import lru_cache

from core import db
from core.settings import settings
from tools.todo.base import (
    PRIORITIES,
    Priority,
    Todo,
    TodoError,
    TodoProvider,
    due_today,
    find_duplicate,
    high_priority,
    normalise_title,
    overdue,
    pending,
    sort_for_display,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PRIORITIES",
    "Priority",
    "Todo",
    "TodoError",
    "TodoProvider",
    "due_today",
    "find_duplicate",
    "get_provider",
    "high_priority",
    "normalise_title",
    "overdue",
    "pending",
    "sort_for_display",
]

_PROVIDERS = {
    "internal": "tools.todo.internal:MongoTodoProvider",
    "memory": "tools.todo.memory:InMemoryTodoProvider",
}


def _build(target: str) -> TodoProvider:
    module_path, _, class_name = target.partition(":")
    return getattr(importlib.import_module(module_path), class_name)()


@lru_cache(maxsize=1)
def get_provider() -> TodoProvider:
    """The configured provider, built once per process."""
    configured = settings.todo_provider.lower()

    if configured == "internal" and not db.mongo_configured():
        logger.warning("MONGODB_URI unset — tasks are in-process and will not persist")
        return _build(_PROVIDERS["memory"])

    target = _PROVIDERS.get(configured)
    if target is None:
        raise TodoError(
            f"Unknown todo provider '{settings.todo_provider}'. "
            f"Known providers: {', '.join(sorted(_PROVIDERS))}."
        )
    return _build(target)
