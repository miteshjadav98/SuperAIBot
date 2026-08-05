"""The task-source contract, plus the filters the briefing is actually made of.

The provider protocol keeps Notion, Microsoft To Do and Google Tasks a new file
away. The *filters* below are deliberately not part of it: "what's overdue" is
the same question whatever stores the task, and answering it here means it is
pure, testable, and identical across every backend. A provider that can push the
filter down to its own API can still do so — it just has to agree with these.

Ownership: every method takes an ``owner`` (the authenticated user id). Nothing
here reads identity for itself — the caller passes the one it was given, the
same rule :mod:`core.memory` follows, and for the same reason: whoever controls
the owner controls whose tasks are read.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Optional, Protocol, runtime_checkable

Priority = Literal["high", "normal", "low"]

PRIORITIES: tuple[Priority, ...] = ("high", "normal", "low")


class TodoError(RuntimeError):
    """An expected task-source failure — surfaced to the model, not raised."""


@dataclass
class Todo:
    """One task, in the subset of fields every backend can supply."""

    id: str
    title: str
    priority: Priority
    done: bool
    due_date: Optional[date] = None

    def is_overdue(self, today: Optional[date] = None) -> bool:
        if self.done or self.due_date is None:
            return False
        return self.due_date < (today or date.today())

    def is_due_today(self, today: Optional[date] = None) -> bool:
        return not self.done and self.due_date == (today or date.today())

    def describe(self) -> str:
        """One line for the model: id first, so it can act on it afterwards."""
        bits = [f"[{self.id}] {self.title}"]
        if self.priority != "normal":
            bits.append(f"({self.priority} priority)")
        if self.due_date:
            bits.append(f"due {self.due_date.isoformat()}")
        return " ".join(bits)


@runtime_checkable
class TodoProvider(Protocol):
    """What a task backend must do. Synchronous: every backend here is a
    blocking client (pymongo today, REST SDKs later), and the tool layer
    offloads to a worker thread rather than making each one fake async."""

    name: str

    def list(self, owner: str, *, include_done: bool = False) -> list[Todo]: ...

    def add(
        self,
        owner: str,
        title: str,
        *,
        priority: Priority = "normal",
        due_date: Optional[date] = None,
    ) -> Todo: ...

    def complete(self, owner: str, todo_id: str) -> Todo: ...

    def delete(self, owner: str, todo_id: str) -> bool: ...


# --- Duplicates --------------------------------------------------------------


def normalise_title(title: str) -> str:
    """Comparison key for "is this the same task?" — case, spacing and trailing
    punctuation insensitive, so "Review PR" and "review pr." are one task."""
    return " ".join(title.lower().split()).strip(" .!,;:")


def find_duplicate(title: str, tasks: list[Todo]) -> Optional[Todo]:
    """An existing *open* task with the same title, if there is one.

    Checked in the tool layer rather than inside a provider, so every backend
    (Mongo today, Notion or Google Tasks later) gets the same behaviour without
    implementing it. The trade-off is a read before each write and no protection
    against two simultaneous adds — worth it to keep the provider contract at
    four methods.
    """
    key = normalise_title(title)
    if not key:
        return None
    return next((t for t in tasks if not t.done and normalise_title(t.title) == key), None)


# --- Filters -----------------------------------------------------------------


def overdue(tasks: list[Todo], today: Optional[date] = None) -> list[Todo]:
    return [t for t in tasks if t.is_overdue(today)]


def due_today(tasks: list[Todo], today: Optional[date] = None) -> list[Todo]:
    return [t for t in tasks if t.is_due_today(today)]


def high_priority(tasks: list[Todo]) -> list[Todo]:
    return [t for t in tasks if not t.done and t.priority == "high"]


def pending(tasks: list[Todo]) -> list[Todo]:
    return [t for t in tasks if not t.done]


def sort_for_display(tasks: list[Todo], today: Optional[date] = None) -> list[Todo]:
    """Most urgent first: overdue, then due today, then by priority, then by date.

    The order the briefing shows tasks in *is* the recommendation — a user reads
    the top of the list, so what lands there matters more than the prose below.
    """
    rank = {p: i for i, p in enumerate(PRIORITIES)}
    far_future = date.max

    return sorted(
        tasks,
        key=lambda t: (
            not t.is_overdue(today),
            not t.is_due_today(today),
            rank.get(t.priority, 1),
            t.due_date or far_future,
            t.title.lower(),
        ),
    )
