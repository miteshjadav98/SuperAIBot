"""An in-memory mailbox — the default provider.

Why a mock is the default rather than a stub that returns one hardcoded string
(which is what this agent used to do): the agent is only exercisable if the
mailbox has *state*. Sending has to make a message appear in Sent; trashing has
to make it disappear from search; replying has to thread onto something real.
Without that, "the agent works" is unfalsifiable, and every approval flow is
theatre.

So this is a real mailbox implementation that happens to live in a dict. It is
per-user, seeded with plausible messages on first use, and it is the reason the
whole email surface can be unit-tested with no network and no credentials.

It is **not** persistent: mailboxes are per-process and reset on restart. That
is deliberate — persisting a fake inbox to Mongo would create a second, subtly
different mail store to reason about.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from itertools import count

from tools.email.base import Contact, EmailError, EmailMessage

_SEED_SENDERS = [
    ("Jane Okafor", "jane@example.com"),
    ("Priya Nair", "priya@example.com"),
    ("Accounts", "billing@vendor.example"),
    ("Standup Bot", "noreply@teamtools.example"),
]


def _seed(owner_address: str) -> list[EmailMessage]:
    now = datetime.now(timezone.utc)
    ids = count(1)

    def message(sender, subject, body, days, labels, unread):
        name, address = sender
        return EmailMessage(
            id=f"m{next(ids)}",
            sender=f"{name} <{address}>",
            to=[owner_address],
            subject=subject,
            body=body,
            date=now - timedelta(days=days, hours=3),
            labels=labels,
            unread=unread,
        )

    return [
        message(
            _SEED_SENDERS[0],
            "Coffee next week?",
            "Hi! I'm going to be in town Tuesday to Thursday next week and "
            "wondered if you had time for a coffee. Any morning works for me.\n\n"
            "— Jane",
            1, ["inbox"], True,
        ),
        message(
            _SEED_SENDERS[1],
            "Re: Goa trip dates",
            "I checked with work — I can take the 14th to the 20th off. "
            "Shall we look at flights that land on the 14th?\n\n— Priya",
            2, ["inbox", "travel"], True,
        ),
        message(
            _SEED_SENDERS[2],
            "Invoice INV-4021 is due",
            "Invoice INV-4021 for 12,400 INR is due on the 28th. "
            "Please arrange payment at your convenience.",
            4, ["inbox", "finance"], False,
        ),
        message(
            _SEED_SENDERS[3],
            "Standup summary — Thursday",
            "3 items moved to done. 1 blocker: the staging deploy is waiting "
            "on a database migration review.",
            5, ["inbox", "work"], False,
        ),
    ]


class MockEmailProvider:
    """A working mailbox backed by a dict. Satisfies ``EmailProvider``."""

    name = "mock"

    def __init__(self, owner_address: str = "you@example.com"):
        self.owner_address = owner_address
        self._messages: list[EmailMessage] = _seed(owner_address)
        self._ids = count(len(self._messages) + 1)

    def _next_id(self) -> str:
        return f"m{next(self._ids)}"

    def _find(self, message_id: str) -> EmailMessage:
        for message in self._messages:
            if message.id == message_id:
                return message
        raise EmailError(
            f"No message with id '{message_id}'. Search first and use an id from the results."
        )

    # --- EmailProvider -------------------------------------------------------

    def is_connected(self) -> bool:
        return True

    def search(self, query: str, limit: int = 10) -> list[EmailMessage]:
        """Substring match over sender, subject, body and labels.

        Not clever on purpose: a fake mailbox with fuzzy search would hide
        retrieval bugs that a real provider would expose.
        """
        terms = [t for t in re.split(r"\s+", query.lower().strip()) if t]
        visible = [m for m in self._messages if "trash" not in m.labels]

        if not terms:
            matches = [m for m in visible if "inbox" in m.labels]
        else:
            matches = [
                m
                for m in visible
                if all(
                    t in f"{m.sender} {m.subject} {m.body} {' '.join(m.labels)}".lower()
                    for t in terms
                )
            ]
        return sorted(matches, key=lambda m: m.date, reverse=True)[:limit]

    def get(self, message_id: str) -> EmailMessage:
        message = self._find(message_id)
        message.unread = False  # reading marks read, as a real mailbox would
        return message

    def send(self, to: list[str], subject: str, body: str) -> str:
        message = EmailMessage(
            id=self._next_id(),
            sender=self.owner_address,
            to=list(to),
            subject=subject,
            body=body,
            date=datetime.now(timezone.utc),
            labels=["sent"],
        )
        self._messages.append(message)
        return message.id

    def reply(self, message_id: str, body: str) -> str:
        original = self._find(message_id)
        subject = original.subject
        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"
        return self.send([original.sender], subject, body)

    def create_draft(self, to: list[str], subject: str, body: str) -> str:
        message = EmailMessage(
            id=self._next_id(),
            sender=self.owner_address,
            to=list(to),
            subject=subject,
            body=body,
            date=datetime.now(timezone.utc),
            labels=["draft"],
        )
        self._messages.append(message)
        return message.id

    def archive(self, message_id: str) -> str:
        message = self._find(message_id)
        message.labels = [label for label in message.labels if label != "inbox"]
        if "archived" not in message.labels:
            message.labels.append("archived")
        return message.id

    def trash(self, message_id: str) -> str:
        message = self._find(message_id)
        message.labels = [label for label in message.labels if label != "inbox"]
        if "trash" not in message.labels:
            message.labels.append("trash")
        return message.id

    def contacts(self, query: str) -> list[Contact]:
        term = query.lower().strip()
        known = [Contact(name, address) for name, address in _SEED_SENDERS]
        if not term:
            return known
        return [c for c in known if term in c.name.lower() or term in c.email.lower()]
