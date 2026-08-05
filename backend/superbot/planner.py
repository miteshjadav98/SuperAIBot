"""Planner — turns a user request into a validated task DAG.

The planner replaces the old single-shot classifier. Instead of "which one agent
handles this?" it answers "which agents, in what order, and which of them can run
at the same time?" — so a request like *"find flights to Goa and email the
shortlist to Priya"* becomes two tasks instead of one misrouted one.

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
from core.settings import settings
from llm.factory import get_chat_model
from superbot.router import route
from superbot.state import MAX_TASKS, TASK_MAX_ATTEMPTS, Plan, Task, TaskSpec

_SYSTEM_DEFAULT = """You are the planner for a multi-agent platform. Break the \
user's request into the smallest set of tasks that fully answers it, and assign \
each task to the agent best suited to it.

Available agents (id [capabilities]: what it does):
{agents}

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
- If nothing fits well, emit a single task for "{default}"."""


def _history(messages: list[AnyMessage], limit: int = 4) -> str:
    """A compact transcript tail, so the planner can resolve "send it to her"."""
    recent = messages[-limit:-1] if len(messages) > 1 else []
    lines = []
    for message in recent:
        role = getattr(message, "type", "user")
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.strip():
            lines.append(f"{role}: {content[:400]}")
    return "\n".join(lines)


def _validate(specs: list[TaskSpec]) -> list[Task]:
    """Keep only what is safely executable, and attach the retry budget.

    Dependencies may only point *backwards* in the emitted order. That single
    rule drops forward references and makes cycles unrepresentable, so the
    executor cannot deadlock on a plan the model got wrong.
    """
    valid_agents = set(registry.ids())
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


async def make_plan(query: str, messages: list[AnyMessage]) -> list[Task]:
    """Plan ``query`` into validated tasks, falling back to a single agent."""
    if not registry.ids():
        return []

    system = get_prompt(
        "superbot_planner_system",
        _SYSTEM_DEFAULT,
        name="Super Bot — Planner",
    ).format(
        agents=registry.catalog(),
        max_tasks=MAX_TASKS,
        default=settings.default_agent_id,
    )

    history = _history(messages)
    user = f"Conversation so far:\n{history}\n\nRequest: {query}" if history else query

    try:
        model = get_chat_model(temperature=0).with_structured_output(Plan)
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
