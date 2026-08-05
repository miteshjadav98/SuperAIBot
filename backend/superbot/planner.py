"""Planner — turns a user request into a validated task DAG.

The planner replaces the old single-shot classifier. Instead of "which one agent
handles this?" it answers "which agents, in what order, and which of them can run
at the same time?" — so a request like *"find flights to Goa and email the
shortlist to Priya"* becomes two tasks instead of one misrouted one.

One of the agents it can pick is the Super Bot itself
(:data:`~superbot.state.ASSISTANT_AGENT_ID`), which is what a greeting or a
"what can you do?" should get — forcing those onto a domain specialist is how
the platform ended up introducing itself as a personal chef.

Two deliberate constraints:

* **The plan is produced once, up front.** No replanning loop. A stale plan
  costs one bad answer; an unbounded replan loop costs your token budget.
* **Model output is validated, not trusted.** The LLM proposes ids, agents and
  dependencies; :func:`_validate` is what decides which of those survive.

Planning is never allowed to hard-fail: any error falls back to the cheap
single-agent classifier in :mod:`superbot.router`, which is the pre-v2 behaviour.
"""

from __future__ import annotations

from langchain_core.messages import AnyMessage

from core.prompts import get_prompt
from core.registry import registry
from llm.factory import get_chat_model
from superbot.router import route
from superbot.state import (
    ASSISTANT_AGENT_ID,
    CONTEXT_MESSAGES,
    MAX_TASKS,
    TASK_MAX_ATTEMPTS,
    Plan,
    Task,
    TaskSpec,
)

_SYSTEM_DEFAULT = """You are the supervisor for Super Bot, a personal AI \
assistant. You answer the user yourself unless one of your specialists is \
genuinely better placed to. Break the user's request into the smallest set of \
tasks that fully answers it, and assign each task to the agent best suited to it.

Available agents (id [capabilities]: what it does):
{agents}
{assistant}

Rules:
- Prefer ONE task. Only split when the request genuinely needs different agents \
or clearly separate pieces of work.
- Never emit more than {max_tasks} tasks.
- Each instruction must be self-contained. An agent cannot see the other tasks, \
the plan, or their outputs, so never write "the above", "that flight", or \
"the previous result".
- Use depends_on ONLY when a task truly needs an earlier task's output. Tasks \
with no dependency between them run in parallel, which is faster — leave \
depends_on empty whenever you can.
- agent_id must be copied exactly from the list above. Never invent one.
- Use a specialist only for work genuinely inside its domain. Greetings, thanks, \
"who are you", "what can you do", chit-chat, and anything no specialist covers \
go to "{assistant_id}" — never make a specialist answer outside its domain.

Routing across turns:
{continuity}"""

_ASSISTANT_ENTRY = (
    f"- {ASSISTANT_AGENT_ID} [general]: Super Bot answering in its own voice. "
    "Greetings, small talk, questions about what Super Bot is or can do, "
    "follow-ups answerable from the conversation itself, and general questions "
    "none of the specialists above cover."
)

_FIRST_TURN = "- This is the start of the conversation; route on the request alone."


def _continuity(previous: str | None) -> str:
    """The rule that keeps a conversation on one agent — and lets the user leave.

    Without this the supervisor re-plans every turn from scratch, so a short
    follow-up ("yes", "what about the venue?") loses the agent that answered the
    question it follows up on.
    """
    if not previous:
        return _FIRST_TURN
    return (
        f"- The previous turn was handled by: {previous}.\n"
        "- A follow-up on that same subject stays with that same agent, even when "
        'the message is short or elliptical ("yes", "what about the venue?", '
        '"make it cheaper") — resolve it against the conversation below.\n'
        "- Switch as soon as the subject changes, or when the user asks for "
        "something else outright (\"let's plan the wedding now\"). The previous "
        "agent is a tie-breaker, never a lock-in."
    )


def _history(messages: list[AnyMessage], limit: int = CONTEXT_MESSAGES) -> str:
    """A compact transcript tail, so the planner can resolve "send it to her"."""
    recent = messages[-(limit + 1) : -1] if len(messages) > 1 else []
    lines = []
    for message in recent:
        role = getattr(message, "type", "user")
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.strip():
            lines.append(f"{role}: {content[:400]}")
    return "\n".join(lines)


def _fill(template: str, values: dict[str, str]) -> str:
    """Format a prompt without letting an edited one break the turn.

    Prompts are editable from the Prompt Management UI, so the stored text can
    carry a placeholder this code doesn't supply. Falling back to the code
    default costs one edit; raising would cost the whole run.
    """
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError) as exc:
        print(f"[planner] stored prompt failed to format ({exc}); using code default")
        return _SYSTEM_DEFAULT.format(**values)


def _validate(specs: list[TaskSpec]) -> list[Task]:
    """Keep only what is safely executable, and attach the retry budget.

    Dependencies may only point *backwards* in the emitted order. That single
    rule drops forward references and makes cycles unrepresentable, so the
    executor cannot deadlock on a plan the model got wrong.
    """
    valid_agents = {*registry.ids(), ASSISTANT_AGENT_ID}
    tasks: list[Task] = []
    seen: set[str] = set()

    for spec in specs[:MAX_TASKS]:
        if spec.agent_id not in valid_agents or spec.id in seen or not spec.instruction.strip():
            continue
        tasks.append(
            Task(
                id=spec.id,
                agent_id=spec.agent_id,
                instruction=spec.instruction.strip(),
                depends_on=[d for d in spec.depends_on if d in seen],
                max_attempts=TASK_MAX_ATTEMPTS,
            )
        )
        seen.add(spec.id)

    return tasks


def single_task_plan(agent_id: str, instruction: str) -> list[Task]:
    """A one-task plan. Used for the manual-agent-override fast path and as the
    fallback when planning fails — both skip the planner LLM call entirely."""
    return [
        Task(
            id="t1",
            agent_id=agent_id,
            instruction=instruction,
            depends_on=[],
            max_attempts=TASK_MAX_ATTEMPTS,
        )
    ]


async def make_plan(
    query: str,
    messages: list[AnyMessage],
    previous_agents: str | None = None,
) -> list[Task]:
    """Plan ``query`` into validated tasks, falling back to a single agent.

    ``previous_agents`` is who handled the last turn (the state's ``routed_to``).
    It is what makes a follow-up stay with the agent that answered before it.
    """
    if not registry.ids():
        return single_task_plan(ASSISTANT_AGENT_ID, query)

    # A new prompt id, not an edit of the old one: `get_prompt` seeds a prompt
    # once and then always serves the stored version, so an already-deployed
    # `superbot_planner_system` would have masked every change made here.
    template = get_prompt(
        "superbot_supervisor_system",
        _SYSTEM_DEFAULT,
        name="Super Bot — Supervisor",
    )
    system = _fill(
        template,
        {
            "agents": registry.catalog(),
            "assistant": _ASSISTANT_ENTRY,
            "assistant_id": ASSISTANT_AGENT_ID,
            "max_tasks": str(MAX_TASKS),
            "continuity": _continuity(previous_agents),
        },
    )

    history = _history(messages)
    user = f"Conversation so far:\n{history}\n\nRequest: {query}" if history else query

    try:
        # `langsmith:nostream` keeps the planner's tokens out of the UI stream.
        # It emits structured JSON, so streaming it would spray a half-built
        # object into the chat before any agent has run. It still appears in
        # LangSmith traces — this hides it from the user, not from you.
        model = get_chat_model(temperature=0).with_structured_output(Plan).with_config(
            tags=["langsmith:nostream"]
        )
        plan: Plan = await model.ainvoke(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        tasks = _validate(plan.tasks)
    except Exception as exc:  # noqa: BLE001 — planning must never hard-fail
        print(f"[planner] planning failed, falling back to single agent: {exc}")
        tasks = []

    if not tasks:
        return single_task_plan(await route(query), query)
    return tasks
