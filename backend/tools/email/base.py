"""The email provider contract.

Everything above this line is provider-agnostic: the agent, its tools, and the
approval policy know only about :class:`EmailProvider`. Swapping Gmail for
Outlook, Exchange or plain IMAP is a new implementation of this protocol plus
one settings value — no agent code changes.

Design notes:

* **Ids are opaque strings.** Gmail message ids, IMAP UIDs and mock counters
  have nothing in common, so nothing above this layer may parse them.
* **Methods raise :class:`EmailError` for anything expected** (message not
  found, not connected, quota). The tool layer turns those into readable
  ``ToolMessage`` text so the model can recover, rather than crashing the run.
* **Provider methods are synchronous.** Every current backend has a blocking
  client library; the tool layer offloads to a worker thread. Making the
  protocol async would force every implementation to fake it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable


class EmailError(RuntimeError):
    """An expected email failure — surfaced to the model, not raised at the graph."""


@dataclass
class EmailMessage:
    """One message, in the subset of fields every provider can supply."""

    id: str
    sender: str
    to: list[str]
    subject: str
    body: str
    date: datetime
    labels: list[str] = field(default_factory=list)
    unread: bool = False

    def summary(self) -> str:
        """One line for list views — keeps search results out of the context
        window until the model actually opens a message."""
        flag = "•" if self.unread else " "
        return f"{flag} [{self.id}] {self.date:%Y-%m-%d} from {self.sender} — {self.subject}"

    def full(self) -> str:
        """The rendering used when a message is actually read."""
        return (
            f"From: {self.sender}\n"
            f"To: {', '.join(self.to)}\n"
            f"Date: {self.date:%Y-%m-%d %H:%M}\n"
            f"Subject: {self.subject}\n"
            f"Labels: {', '.join(self.labels) or 'none'}\n\n"
            f"{self.body}"
        )


@dataclass
class Contact:
    name: str
    email: str

    def __str__(self) -> str:
        return f"{self.name} <{self.email}>"


@runtime_checkable
class EmailProvider(Protocol):
    """What every mailbox backend must implement.

    Deliberately small. Attachments, threading and label management are absent
    because a partially-supported operation across four providers is worse than
    an honestly missing one — and every extra tool costs tool-selection accuracy
    in the agent above.
    """

    name: str

    def is_connected(self) -> bool:
        """Whether this mailbox is usable. Drives which tools the agent offers."""

    def search(self, query: str, limit: int = 10) -> list[EmailMessage]:
        """Messages matching a natural-language-ish query. Empty list, not an
        error, when nothing matches."""

    def get(self, message_id: str) -> EmailMessage:
        """One message. Raises :class:`EmailError` if the id is unknown."""

    def send(self, to: list[str], subject: str, body: str) -> str:
        """Send a new message. Returns the new message id."""

    def reply(self, message_id: str, body: str) -> str:
        """Reply to a message, quoting its recipients and subject."""

    def create_draft(self, to: list[str], subject: str, body: str) -> str:
        """Save an unsent draft. Returns its id."""

    def archive(self, message_id: str) -> str:
        """Remove from the inbox, keep in the mailbox."""

    def trash(self, message_id: str) -> str:
        """Move to trash."""

    def contacts(self, query: str) -> list[Contact]:
        """Known correspondents matching ``query``."""
