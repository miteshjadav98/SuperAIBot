"""Email provider, approval policy, and the agent's tool surface."""

from __future__ import annotations

import pytest

from core.approval import approval_middleware, is_gated, requires_approval
from tools.email import get_email_provider, reset_mailboxes
from tools.email.base import EmailError, EmailProvider
from tools.email.gmail import GmailProvider
from tools.email.mock import MockEmailProvider


@pytest.fixture
def mailbox():
    return MockEmailProvider(owner_address="me@example.com")


# --- Provider contract -------------------------------------------------------


@pytest.mark.parametrize("provider", [MockEmailProvider(), GmailProvider()])
def test_providers_satisfy_the_protocol(provider):
    """Structural check: the seam and the mock are interchangeable to callers."""
    assert isinstance(provider, EmailProvider)


def test_mock_mailbox_is_seeded_and_searchable(mailbox):
    assert len(mailbox.search("")) == 4
    assert len(mailbox.search("coffee")) == 1
    assert mailbox.search("no such thing") == []


def test_search_matches_all_terms(mailbox):
    assert len(mailbox.search("jane coffee")) == 1
    assert mailbox.search("jane invoice") == []


def test_reading_marks_a_message_read(mailbox):
    unread = [m for m in mailbox.search("") if m.unread]
    assert unread

    mailbox.get(unread[0].id)

    assert mailbox.get(unread[0].id).unread is False


def test_unknown_message_id_raises_a_recoverable_error(mailbox):
    with pytest.raises(EmailError, match="No message with id"):
        mailbox.get("does-not-exist")


# --- State actually changes: this is what a hardcoded stub could not do ------


def test_sending_appends_to_the_mailbox(mailbox):
    before = len(mailbox.search(""))

    sent_id = mailbox.send(["jane@example.com"], "Hello", "Body text")

    assert mailbox.get(sent_id).labels == ["sent"]
    assert len(mailbox.search("")) == before  # sent mail is not in the inbox


def test_reply_threads_onto_the_original(mailbox):
    original = mailbox.search("coffee")[0]

    reply_id = mailbox.reply(original.id, "Tuesday works!")

    reply = mailbox.get(reply_id)
    assert reply.subject.startswith("Re:")
    assert reply.to == [original.sender]


def test_reply_does_not_double_prefix_a_subject(mailbox):
    first = mailbox.reply(mailbox.search("coffee")[0].id, "one")
    second = mailbox.reply(first, "two")

    assert mailbox.get(second).subject.count("Re:") == 1


def test_archiving_removes_from_the_inbox_but_keeps_the_message(mailbox):
    """Archive is not delete: gone from the inbox listing, still findable."""
    target = mailbox.search("coffee")[0]

    mailbox.archive(target.id)

    assert target.id not in {m.id for m in mailbox.search("")}
    assert [m.id for m in mailbox.search("coffee")] == [target.id]
    assert "archived" in mailbox.get(target.id).labels


def test_trashing_hides_the_message(mailbox):
    target = mailbox.search("invoice")[0]

    mailbox.trash(target.id)

    assert mailbox.search("invoice") == []


def test_drafts_are_not_sent(mailbox):
    draft_id = mailbox.create_draft(["jane@example.com"], "Later", "Not yet")

    assert mailbox.get(draft_id).labels == ["draft"]


def test_contacts_lookup(mailbox):
    assert [c.email for c in mailbox.contacts("priya")] == ["priya@example.com"]
    assert mailbox.contacts("nobody") == []


# --- Provider selection ------------------------------------------------------


def test_mailboxes_are_per_user():
    """A shared mailbox would show one user another's mail."""
    reset_mailboxes()

    alice = get_email_provider("alice")
    alice.send(["x@example.com"], "Alice only", "secret")

    assert get_email_provider("bob").search("Alice only") == []
    assert get_email_provider("alice") is alice  # cached, so state persists


def test_no_signed_in_user_gets_no_mailbox():
    """Same rule as memory: without identity there is no correct mailbox."""
    assert get_email_provider(None) is None
    assert get_email_provider("") is None


def test_gmail_seam_is_honest_about_being_unimplemented():
    """It must not look like a working provider."""
    gmail = GmailProvider()

    assert gmail.is_connected() is False
    with pytest.raises(EmailError, match="not implemented"):
        gmail.search("anything")


# --- Approval policy ---------------------------------------------------------


def test_destructive_email_tools_are_gated():
    from agents.email_agent import TOOLS

    gated = {t.name for t in TOOLS if is_gated(t.name)}

    assert gated == {"send_email", "reply_to_email", "archive_email", "trash_email"}


def test_read_only_tools_are_not_gated():
    """Approval fatigue is a real failure mode — reads must not prompt."""
    for name in ("search_emails", "read_email", "search_contacts", "create_draft"):
        assert not is_gated(name)


def test_middleware_only_interrupts_marked_tools():
    from agents.email_agent import TOOLS

    middleware = approval_middleware(TOOLS)

    assert set(middleware.interrupt_on) == {
        "send_email",
        "reply_to_email",
        "archive_email",
        "trash_email",
    }


def test_marking_a_tool_gates_it_automatically():
    """The point of the decorator: a new destructive tool cannot ship ungated
    because someone forgot to edit a map."""
    from langchain.tools import tool

    @requires_approval(describe=lambda args: f"Delete {args['target']}")
    @tool
    def delete_everything(target: str) -> str:
        """Delete a thing."""
        return "gone"

    middleware = approval_middleware([delete_everything])
    config = middleware.interrupt_on["delete_everything"]

    assert "approve" in config["allowed_decisions"]
    assert config["description"]({"args": {"target": "inbox"}}, None, None) == "Delete inbox"


def test_allow_edit_false_removes_the_edit_decision():
    from agents.email_agent import TOOLS

    interrupt_on = approval_middleware(TOOLS).interrupt_on

    assert "edit" in interrupt_on["send_email"]["allowed_decisions"]
    assert "edit" not in interrupt_on["trash_email"]["allowed_decisions"]


def test_overrides_can_ungate_a_tool():
    from agents.email_agent import TOOLS

    middleware = approval_middleware(TOOLS, overrides={"send_email": False})

    # The middleware drops auto-approved entries rather than storing False.
    assert "send_email" not in middleware.interrupt_on
    assert "trash_email" in middleware.interrupt_on
