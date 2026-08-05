"""The built-in task list: MongoDB, one document per task, scoped by owner.

The only backend that needs no OAuth dance, which is why it is the MVP's. It
also fixes the shape the external ones (Notion, Microsoft To Do, Google Tasks)
have to map onto.

Storage notes:

* ``due_date`` is stored as an ISO ``YYYY-MM-DD`` string, not a datetime. A task
  is due on a *day*; storing midnight-in-some-timezone invites the classic
  off-by-one where "due today" flips depending on where the server runs.
* Ids are the Mongo ``_id`` hex, truncated for display by nobody — the model
  passes back exactly what it was given.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Optional

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import ReturnDocument

from core import db
from tools.todo.base import Priority, Todo, TodoError

logger = logging.getLogger(__name__)


def _to_todo(doc: dict) -> Todo:
    raw_due = doc.get("due_date")
    return Todo(
        id=str(doc["_id"]),
        title=doc.get("title", ""),
        priority=doc.get("priority", "normal"),
        done=bool(doc.get("done", False)),
        due_date=date.fromisoformat(raw_due) if raw_due else None,
    )


def _object_id(todo_id: str) -> ObjectId:
    try:
        return ObjectId(todo_id)
    except (InvalidId, TypeError) as exc:
        raise TodoError(f"'{todo_id}' is not a task id.") from exc


class MongoTodoProvider:
    """Per-user tasks in the ``todos`` collection."""

    name = "internal"

    def __init__(self, collection=None) -> None:
        self._collection = collection if collection is not None else db.todos_collection()
        if self._collection is None:
            raise TodoError("MongoDB is not configured, so there is no task list.")

    def list(self, owner: str, *, include_done: bool = False) -> list[Todo]:
        query: dict = {"owner": owner}
        if not include_done:
            query["done"] = False
        try:
            docs = self._collection.find(query).sort("created_at", 1)
            return [_to_todo(d) for d in docs]
        except Exception as exc:  # noqa: BLE001
            raise TodoError(f"Couldn't read the task list: {exc}") from exc

    def add(
        self,
        owner: str,
        title: str,
        *,
        priority: Priority = "normal",
        due_date: Optional[date] = None,
    ) -> Todo:
        title = (title or "").strip()
        if not title:
            raise TodoError("A task needs a title.")

        doc = {
            "owner": owner,
            "title": title,
            "priority": priority,
            "done": False,
            "due_date": due_date.isoformat() if due_date else None,
            "created_at": datetime.now(timezone.utc),
            "completed_at": None,
        }
        try:
            doc["_id"] = self._collection.insert_one(doc).inserted_id
        except Exception as exc:  # noqa: BLE001
            raise TodoError(f"Couldn't save that task: {exc}") from exc
        return _to_todo(doc)

    def complete(self, owner: str, todo_id: str) -> Todo:
        # Owner is part of the filter, not checked afterwards: a task belonging
        # to someone else simply does not match, so one user can never complete
        # (or learn of) another's task by guessing an id.
        try:
            doc = self._collection.find_one_and_update(
                {"_id": _object_id(todo_id), "owner": owner},
                {"$set": {"done": True, "completed_at": datetime.now(timezone.utc)}},
                return_document=ReturnDocument.AFTER,
            )
        except TodoError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise TodoError(f"Couldn't update that task: {exc}") from exc

        if doc is None:
            raise TodoError(f"No open task with id '{todo_id}'.")
        return _to_todo(doc)

    def delete(self, owner: str, todo_id: str) -> bool:
        try:
            result = self._collection.delete_one(
                {"_id": _object_id(todo_id), "owner": owner}
            )
        except TodoError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise TodoError(f"Couldn't delete that task: {exc}") from exc
        return result.deleted_count > 0
