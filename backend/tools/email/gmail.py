"""Gmail provider — the seam, not an implementation.

This file exists so that "add Gmail" is *implementing one class against a
contract that already has a working reference implementation and a test suite*,
rather than a refactor of the agent. The protocol, the tools, the approval
policy and the tests are all provider-agnostic already; only the methods below
are missing.

**It is deliberately not implemented.** Shipping untested Gmail API calls that
look finished is worse than an honest stub: nobody can tell the difference until
it silently mislabels or misfiles someone's real mail. Every method therefore
fails loudly with instructions instead of pretending.

To make it real:

1. ``pip install google-api-python-client google-auth-oauthlib``
2. Google Cloud console → enable the Gmail API → create an **OAuth client ID**
   (application type: Desktop or Web, depending on how you run the gateway).
3. Put the credentials in ``.env``::

       EMAIL_PROVIDER=gmail
       GMAIL_CLIENT_ID=...
       GMAIL_CLIENT_SECRET=...

4. Add a per-user OAuth consent + token-refresh flow. Tokens are per-user
   secrets: store them keyed by the authenticated user id (the platform already
   has one — see ``core/memory.owner_from_config``), never in a shared global.
5. Implement the methods below in terms of ``users().messages()``. Map Gmail's
   ``id`` straight onto :attr:`EmailMessage.id` — ids are opaque above this
   layer, so nothing else needs to change.

Scopes: ``gmail.readonly`` covers search/get; sending needs ``gmail.send``;
archive/trash need ``gmail.modify``. Request the narrowest set the deployment
actually uses — ``gmail.modify`` on a personal mailbox is a large amount of
trust to ask for.
"""

from __future__ import annotations

from tools.email.base import Contact, EmailError, EmailMessage

_NOT_IMPLEMENTED = (
    "The Gmail provider is not implemented yet. Set EMAIL_PROVIDER=mock to use "
    "the built-in mailbox, or implement tools/email/gmail.py (see its docstring)."
)


class GmailProvider:
    """Satisfies ``EmailProvider`` structurally; every call fails loudly."""

    name = "gmail"

    def __init__(self, owner_address: str = "", credentials: dict | None = None):
        self.owner_address = owner_address
        self.credentials = credentials or {}

    def is_connected(self) -> bool:
        # Never claims a usable mailbox, so the agent advertises no email tools
        # rather than offering ones that would fail on use.
        return False

    def search(self, query: str, limit: int = 10) -> list[EmailMessage]:
        raise EmailError(_NOT_IMPLEMENTED)

    def get(self, message_id: str) -> EmailMessage:
        raise EmailError(_NOT_IMPLEMENTED)

    def send(self, to: list[str], subject: str, body: str) -> str:
        raise EmailError(_NOT_IMPLEMENTED)

    def reply(self, message_id: str, body: str) -> str:
        raise EmailError(_NOT_IMPLEMENTED)

    def create_draft(self, to: list[str], subject: str, body: str) -> str:
        raise EmailError(_NOT_IMPLEMENTED)

    def archive(self, message_id: str) -> str:
        raise EmailError(_NOT_IMPLEMENTED)

    def trash(self, message_id: str) -> str:
        raise EmailError(_NOT_IMPLEMENTED)

    def contacts(self, query: str) -> list[Contact]:
        raise EmailError(_NOT_IMPLEMENTED)
