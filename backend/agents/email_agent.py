"""Email agent — a ReAct agent over a provider-agnostic mailbox.

Every tool below delegates to an :class:`~tools.email.base.EmailProvider`
resolved from the *authenticated* user. Nothing here knows whether that mailbox
is Gmail, Outlook or the built-in mock, which is what makes the provider
swappable by configuration alone.

Three things this replaces, and why:

* **In-chat authentication is gone.** The agent used to ask for an email address
  and password and compare them against a dataclass holding
  ``password = "password123"``. The platform already authenticates users (JWT →
  ``configurable.owner``), so a second, weaker login was both redundant and an
  active anti-pattern — it trained users to type credentials into a chat box.
  The mailbox is now resolved from the session identity.

* **The hardcoded inbox is gone.** ``check_inbox()`` returned a string literal,
  so no tool could observe the effect of any other tool. The mock provider is a
  real mailbox with state, which is what makes send/reply/archive testable.

* **The hand-maintained approval map is gone.** Destructive tools declare
  themselves with ``@requires_approval`` (see :mod:`core.approval`), so adding
  one later cannot silently ship without a gate.

Tool count is kept to eight. Attachment and label management are absent
deliberately: tool-selection accuracy degrades past roughly a dozen tools, and
each one added makes the rest harder to choose between.
"""

from __future__ import annotations

import asyncio

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.tools import ToolRuntime, tool
from typing import Awaitable, Callable

from core.approval import approval_middleware, requires_approval
from core.base_agent import AgentManifest
from core.memory import MemoryMiddleware, current_owner, owner_from_config
from core.prompts import get_prompt
from llm.factory import get_chat_model
from tools.email import EmailError, get_email_provider

model = get_chat_model()


def _mailbox(runtime: ToolRuntime):
    """The signed-in user's mailbox, or an error the model can act on."""
    provider = get_email_provider(owner_from_config(runtime.config))
    if provider is None:
        raise EmailError("No signed-in user, so there is no mailbox to open.")
    if not provider.is_connected():
        raise EmailError(
            f"The '{provider.name}' mailbox is not connected. "
            "Tell the user their email account needs to be set up first."
        )
    return provider


async def _run(work) -> str:
    """Run a blocking mailbox call off the event loop, turning expected email
    failures into text the model can recover from.

    Returning the error as a normal tool result — rather than raising — lets the
    model correct itself ("that id doesn't exist, let me search again") instead
    of aborting the run.
    """
    try:
        return await asyncio.to_thread(work)
    except EmailError as exc:
        return f"Email error: {exc}"


# --- Read-only tools ---------------------------------------------------------


@tool
async def search_emails(query: str, runtime: ToolRuntime, limit: int = 10) -> str:
    """Search the mailbox and return one line per match.

    Use plain keywords — a sender name, a subject word, a label like 'finance'.
    Leave the query empty to list the current inbox. Returns message ids; pass
    one to read_email to see the full message. Always search before acting on a
    message, because ids are not guessable.
    """

    def work():
        results = _mailbox(runtime).search(query, limit=limit)
        if not results:
            return "No matching messages."
        return "\n".join(m.summary() for m in results)

    return await _run(work)


@tool
async def read_email(message_id: str, runtime: ToolRuntime) -> str:
    """Read one message in full, including its body. Marks it as read.

    Takes a message id from search_emails — never a guess.
    """
    return await _run(lambda: _mailbox(runtime).get(message_id).full())


@tool
async def search_contacts(query: str, runtime: ToolRuntime) -> str:
    """Look up a correspondent's email address by name.

    Use this before sending whenever the user names a person rather than giving
    an address ("email Priya about the trip").
    """

    def work():
        found = _mailbox(runtime).contacts(query)
        return "\n".join(str(c) for c in found) if found else "No matching contacts."

    return await _run(work)


@tool
async def create_draft(to: str, subject: str, body: str, runtime: ToolRuntime) -> str:
    """Save a draft without sending it. Not gated by approval, because a draft
    has no external effect — prefer this when the user is still deciding.

    `to` is a comma-separated list of addresses.
    """

    def work():
        recipients = [a.strip() for a in to.split(",") if a.strip()]
        draft_id = _mailbox(runtime).create_draft(recipients, subject, body)
        return f"Draft saved (id {draft_id})."

    return await _run(work)


# --- Tools with external or destructive effect -------------------------------
# Each declares its own approval requirement; core.approval turns those
# declarations into the interrupt policy.


@requires_approval(
    describe=lambda args: (
        f"Send an email\n\nTo: {args.get('to')}\n"
        f"Subject: {args.get('subject')}\n\n{args.get('body')}"
    )
)
@tool
async def send_email(to: str, subject: str, body: str, runtime: ToolRuntime) -> str:
    """Send a new email. Requires human approval before it goes out.

    `to` is a comma-separated list of addresses. Resolve names to addresses with
    search_contacts first. Write the full final message — the human approving it
    sees exactly what will be sent.
    """

    def work():
        recipients = [a.strip() for a in to.split(",") if a.strip()]
        if not recipients:
            raise EmailError("No recipient address was given.")
        return f"Sent (id {_mailbox(runtime).send(recipients, subject, body)})."

    return await _run(work)


@requires_approval(
    describe=lambda args: f"Reply to message {args.get('message_id')}\n\n{args.get('body')}"
)
@tool
async def reply_to_email(message_id: str, body: str, runtime: ToolRuntime) -> str:
    """Reply to an existing message. Requires human approval before it goes out.

    Read the message first so the reply actually answers it.
    """

    def work():
        return f"Reply sent (id {_mailbox(runtime).reply(message_id, body)})."

    return await _run(work)


@requires_approval(
    describe=lambda args: f"Archive message {args.get('message_id')} (removes it from the inbox)",
    allow_edit=False,
)
@tool
async def archive_email(message_id: str, runtime: ToolRuntime) -> str:
    """Remove a message from the inbox, keeping it in the mailbox."""

    def work():
        return f"Archived {_mailbox(runtime).archive(message_id)}."

    return await _run(work)


@requires_approval(
    describe=lambda args: f"Move message {args.get('message_id')} to trash",
    allow_edit=False,
)
@tool
async def trash_email(message_id: str, runtime: ToolRuntime) -> str:
    """Move a message to trash."""

    def work():
        return f"Moved {_mailbox(runtime).trash(message_id)} to trash."

    return await _run(work)


TOOLS = [
    search_emails,
    read_email,
    search_contacts,
    create_draft,
    send_email,
    reply_to_email,
    archive_email,
    trash_email,
]


# --- Availability gating -----------------------------------------------------


@wrap_model_call
async def only_offer_a_usable_mailbox(
    request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
) -> ModelResponse:
    """Hide the email tools when the user has no connected mailbox.

    Replaces the old authenticated/unauthenticated tool switch. The condition is
    now "is there a mailbox to act on", which is a real state the platform knows,
    rather than a password typed into the chat. Note this narrows the tool list
    to *email* tools only — it preserves anything other middleware contributed
    (such as memory's `remember`) instead of replacing the list wholesale.
    """
    # ModelRequest.runtime carries no run config, so identity comes from the
    # same contextvar the memory layer reads.
    provider = get_email_provider(current_owner())
    if provider is not None and provider.is_connected():
        return await handler(request)

    email_tool_names = {t.name for t in TOOLS}
    kept = [t for t in request.tools if getattr(t, "name", None) not in email_tool_names]
    return await handler(request.override(tools=kept))


_DEFAULT_SYSTEM_PROMPT = """You are an email assistant working on the user's own \
mailbox.

How to work:
- Search before you act. Message ids come from search_emails; never invent one.
- Read a message before replying to it.
- Resolve names to addresses with search_contacts before sending.
- Draft when the user is still deciding; send only when they have decided.
- Sending, replying, archiving and trashing need the user's approval. Write the \
complete final text before asking — they approve exactly what you propose.
- Never ask for a password. The user is already signed in.
- Quote message ids and subjects when you summarise, so the user can follow up."""


agent = create_agent(
    model,
    tools=TOOLS,
    system_prompt=get_prompt(
        "email_agent_system", _DEFAULT_SYSTEM_PROMPT, name="Email Agent — System Prompt"
    ),
    middleware=[
        MemoryMiddleware(),
        only_offer_a_usable_mailbox,
        approval_middleware(TOOLS),
    ],
)


MANIFEST = AgentManifest(
    id="email_agent",
    label="Email Agent",
    emoji="✉️",
    description=(
        "Works with the user's mailbox: searches and reads mail, looks up "
        "contacts, drafts and sends replies, and archives or trashes messages — "
        "with human approval required before anything is sent or destroyed. Use "
        "for anything about email: reading, searching, drafting, writing to "
        "someone, or tidying an inbox."
    ),
    agent_type="langchain",
    builder=lambda: agent,
    capabilities=["email", "contacts"],
)
