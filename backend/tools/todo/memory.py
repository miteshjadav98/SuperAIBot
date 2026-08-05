"""In-process task list — the working stand-in when Mongo isn't configured.

Same role as ``tools/email/mock.py``: the platform boots and the agent is fully
usable before Atlas exists, and the test suite exercises the real tool layer
without a database. Tasks live for the life of the process and no longer, which
is the honest behaviour for something with no storage behind it.
"""

from __future__ import annotations

from datetime import date
from itertools import count
from typing import Optional

from tools.todo.base import Priority, Todo, TodoError


class InMemoryTodoProvider:
    """Per-owner task lists in a dict. Not shared between processes."""

    name = "memory"

    def __init__(self) -> None:
        self._tasks: dict[str, list[Todo]] = {}
        self._ids = count(1)

    def list(self, owner: str, *, include_done: bool = False) -> list[Todo]:
        tasks = self._tasks.get(owner, [])
        return [t for t in tasks if include_done or not t.done]

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

        task = Todo(
            id=str(next(self._ids)),
            title=title,
            priority=priority,
            done=False,
            due_date=due_date,
        )
        self._tasks.setdefault(owner, []).append(task)
        return task

    def _find(self, owner: str, todo_id: str) -> Todo:
        for task in self._tasks.get(owner, []):
            if task.id == todo_id:
                return task
        raise TodoError(f"No task with id '{todo_id}'.")

    def complete(self, owner: str, todo_id: str) -> Todo:
        task = self._find(owner, todo_id)
        task.done = True
        return task

    def delete(self, owner: str, todo_id: str) -> bool:
        task = self._find(owner, todo_id)
        self._tasks[owner].remove(task)
        return True
