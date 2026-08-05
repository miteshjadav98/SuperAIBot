"""Email provider selection.

One place decides which backend a user's mailbox uses, so no agent, tool or test
ever names a concrete provider. Selection is configuration
(``EMAIL_PROVIDER`` in ``.env``), never a code change.

Mailboxes are **per user**: the mock provider holds real state (sending appends
to Sent, trashing hides a message), so one shared instance would leak one user's
mail into another's session. The cache is keyed by the authenticated user id.
"""

from __future__ import annotations

from core.settings import settings
from tools.email.base import Contact, EmailError, EmailMessage, EmailProvider
from tools.email.gmail import GmailProvider
from tools.email.mock import MockEmailProvider

__all__ = [
    "Contact",
    "EmailError",
    "EmailMessage",
    "EmailProvider",
    "get_email_provider",
    "reset_mailboxes",
]

_mailboxes: dict[str, EmailProvider] = {}


def _build(owner: str) -> EmailProvider:
    provider = (settings.email_provider or "mock").lower()

    if provider == "gmail":
        return GmailProvider(owner_address=settings.email_address or "")
    if provider == "mock":
        return MockEmailProvider(owner_address=settings.email_address or "you@example.com")

    raise EmailError(
        f"Unknown EMAIL_PROVIDER {provider!r}. Supported: mock, gmail."
    )


def get_email_provider(owner: str | None) -> EmailProvider | None:
    """The mailbox for ``owner``, or ``None`` when there is no signed-in user.

    Returning ``None`` rather than a shared fallback mailbox is the same rule
    long-term memory follows: without an authenticated identity there is no
    correct mailbox to use, and guessing means showing one user another's mail.
    """
    if not owner:
        return None
    if owner not in _mailboxes:
        _mailboxes[owner] = _build(owner)
    return _mailboxes[owner]


def reset_mailboxes() -> None:
    """Drop all cached mailboxes. For tests, and for switching provider at runtime."""
    _mailboxes.clear()
