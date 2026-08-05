"""Personal Assistant — the daily briefing agent.

Answers "good morning" with the day itself: weather, what's due, and one or two
concrete suggestions drawn from both. Everything else it does (adding a task,
checking tomorrow's weather) is the same tools used conversationally.

Three things worth knowing before changing this:

**The briefing fans out in code, not in the model.** ``daily_briefing`` gathers
its sources with ``asyncio.gather`` rather than hoping the model emits parallel
tool calls. One tool call, one round trip, sources that fail independently —
and adding Calendar means adding a coroutine to ``_gather``, not re-teaching the
model to call three tools at once.

**Recommendations are rules, not a second LLM call.** "Rain at 40% — carry an
umbrella" is a fact about the data, so it is computed in :func:`_suggestions`
where it is deterministic, free, and testable. The model's job is to say it
nicely, not to derive it.

**A dead source degrades the briefing, never fails it.** Each section reports
its own failure in place; the rest of the day still gets delivered. Same policy
as the Super Bot's fan-out, for the same reason.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Optional

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool

from core.base_agent import AgentManifest
from core.memory import MemoryMiddleware, owner_from_config, recall, remember_fact
from core.prompts import get_prompt
from llm.factory import get_chat_model
from tools.todo import (
    PRIORITIES,
    Todo,
    TodoError,
    due_today,
    find_duplicate,
    get_provider as get_todo_provider,
    high_priority,
    overdue,
    sort_for_display,
)
from tools.weather import Weather, WeatherError, get_provider as get_weather_provider

TASKS_IN_BRIEFING = 5
"""How many tasks the briefing lists. A dashboard is a glance, not a backlog —
past about five the user stops reading and the prompt keeps growing."""

_NO_OWNER = "No signed-in user, so there is no task list to read."

HOME_CITY_PREFIX = "Home city:"
"""How the user's city is stored in shared memory.

A stable prefix rather than free text, because this is written *and read back*
by code. The first live run proved why it can't be left to the model: told "I
live in Ahmedabad", it saved "does not know user's city; must ask city before
briefing" — a note to itself, in a store every agent reads, that would have
outlived the conversation. Facts the platform depends on are written by the
platform.
"""


def _remembered_city(owner: str) -> Optional[str]:
    for memory in recall(owner):
        if memory.lower().startswith(HOME_CITY_PREFIX.lower()):
            city = memory[len(HOME_CITY_PREFIX) :].strip()
            if city:
                return city
    return None


def _remember_city(owner: str, city: str) -> None:
    remember_fact(owner, f"{HOME_CITY_PREFIX} {city}")


# --- Task helpers ------------------------------------------------------------


def _parse_due(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise TodoError(f"'{value}' isn't a date. Use YYYY-MM-DD.") from exc


def _format_tasks(tasks: list[Todo], today: Optional[date] = None) -> str:
    if not tasks:
        return "No open tasks."
    return "\n".join(f"- {t.describe()}" for t in sort_for_display(tasks, today))


async def _load_tasks(owner: str) -> list[Todo]:
    """Open tasks for ``owner``. pymongo is blocking — keep it off the loop."""
    return await asyncio.to_thread(get_todo_provider().list, owner)


# --- Briefing ----------------------------------------------------------------


def _suggestions(
    weather: Optional[Weather], tasks: list[Todo], today: Optional[date] = None
) -> list[str]:
    """Rule-based advice, in the order a person would act on it.

    Deliberately few and specific. A briefing that suggests six things every
    morning is one the user stops reading — each rule here has to earn its line.
    """
    today = today or date.today()
    out: list[str] = []

    late = overdue(tasks, today)
    if late:
        titles = ", ".join(t.title for t in late[:2])
        out.append(
            f"{len(late)} task{'s' if len(late) > 1 else ''} already overdue "
            f"({titles}) — clear those before anything new."
        )

    urgent = [t for t in high_priority(tasks) if t not in late]
    if urgent:
        out.append(f"Start with the high-priority one: {urgent[0].title}.")

    today_tasks = due_today(tasks, today)
    if today_tasks and not late and not urgent:
        out.append(f"{len(today_tasks)} due today — none overdue, so you're ahead.")

    if weather is not None:
        if weather.rain_probability >= 40:
            out.append(
                f"{weather.rain_probability}% chance of rain — carry an umbrella."
            )
        if weather.high_c >= 38:
            out.append(f"It hits {weather.high_c:.0f}°C today — keep water on hand.")
        for alert in weather.alerts:
            out.append(f"Weather warning: {alert}")

    if not tasks and weather is not None:
        out.append("Nothing on your task list — a good day to get ahead on something.")

    return out


async def _gather(owner: str, location: str) -> tuple[Optional[Weather], str, list[Todo], str]:
    """Fetch every briefing source at once.

    Returns each source's value *and* its error text, because a briefing with a
    broken weather feed should still show the tasks. Adding Calendar is one more
    coroutine in this gather and one more section in the caller.
    """
    weather_task = asyncio.create_task(get_weather_provider().current(location))
    tasks_task = asyncio.create_task(_load_tasks(owner))

    weather: Optional[Weather] = None
    weather_error = ""
    tasks: list[Todo] = []
    tasks_error = ""

    results = await asyncio.gather(weather_task, tasks_task, return_exceptions=True)

    weather_result, tasks_result = results
    if isinstance(weather_result, BaseException):
        weather_error = (
            str(weather_result)
            if isinstance(weather_result, WeatherError)
            else f"Weather is unavailable ({type(weather_result).__name__})."
        )
    else:
        weather = weather_result

    if isinstance(tasks_result, BaseException):
        tasks_error = (
            str(tasks_result)
            if isinstance(tasks_result, TodoError)
            else f"The task list is unavailable ({type(tasks_result).__name__})."
        )
    else:
        tasks = tasks_result

    return weather, weather_error, tasks, tasks_error


def _compose(
    weather: Optional[Weather],
    weather_error: str,
    tasks: list[Todo],
    tasks_error: str,
    today: Optional[date] = None,
) -> str:
    """The briefing as facts for the model to present. Pure — no I/O, no model."""
    today = today or date.today()
    lines = [f"DATE: {today.strftime('%A, %d %B %Y')}", ""]

    lines.append("WEATHER:")
    if weather is not None:
        lines.append(f"- {weather.location}: {weather.summary()}")
        lines.append(f"- Feels like {weather.feels_like_c:.0f}°C, wind {weather.wind_kph:.0f} km/h")
    else:
        lines.append(f"- Unavailable. {weather_error}")

    lines.append("")
    lines.append("TASKS:")
    if tasks_error:
        lines.append(f"- Unavailable. {tasks_error}")
    else:
        shown = sort_for_display(tasks, today)[:TASKS_IN_BRIEFING]
        lines.append(
            f"- {len(tasks)} open, {len(overdue(tasks, today))} overdue, "
            f"{len(due_today(tasks, today))} due today, "
            f"{len(high_priority(tasks))} high priority"
        )
        lines.extend(f"- {t.describe()}" for t in shown)
        if len(tasks) > len(shown):
            lines.append(f"- (+{len(tasks) - len(shown)} more not shown)")

    lines.append("")
    lines.append("CALENDAR:")
    lines.append("- Not connected yet. Say nothing about meetings.")

    suggestions = _suggestions(weather, tasks, today)
    lines.append("")
    lines.append("SUGGESTIONS (already worked out — present these, don't invent others):")
    if suggestions:
        lines.extend(f"- {s}" for s in suggestions)
    else:
        lines.append("- None. Say something brief and human instead.")

    return "\n".join(lines)


# --- Tools -------------------------------------------------------------------


@tool
async def daily_briefing(runtime: ToolRuntime, location: str = "") -> str:
    """Build the user's daily dashboard: today's weather, their open tasks, and
    concrete suggestions drawn from both.

    Call this for a greeting ("hi", "good morning"), for "what's planned for
    today?", or whenever the user wants an overview of their day. One call
    fetches everything — do not call the weather and task tools separately for
    a briefing.

    Leave `location` empty: the user's city is remembered for them. Only pass it
    when they have just told you where they are, or asked to change it — it is
    saved automatically, so never call `remember` for a city yourself.
    """
    owner = owner_from_config(runtime.config)
    if not owner:
        return _NO_OWNER

    given = (location or "").strip()
    # Remembering is a write this code owns, not a tool call the model has to
    # get right — see HOME_CITY_PREFIX.
    city = given or await asyncio.to_thread(_remembered_city, owner)
    if not city:
        return (
            "NO CITY ON FILE. Ask the user which city they're in, then call this "
            "tool again with that city as `location`. Do not guess one."
        )
    if given:
        await asyncio.to_thread(_remember_city, owner, given)

    weather, weather_error, tasks, tasks_error = await _gather(owner, city)
    return _compose(weather, weather_error, tasks, tasks_error)


@tool
async def get_weather(location: str) -> str:
    """Current weather for a city: temperature, conditions, today's high and low,
    and the chance of rain. Use for a direct weather question — the daily
    briefing already includes it."""
    try:
        weather = await get_weather_provider().current(location)
    except WeatherError as exc:
        return str(exc)
    return (
        f"{weather.location}: {weather.summary()}. "
        f"Feels like {weather.feels_like_c:.0f}°C, wind {weather.wind_kph:.0f} km/h."
    )


@tool
async def list_tasks(runtime: ToolRuntime, include_done: bool = False) -> str:
    """The user's tasks, most urgent first (overdue, then due today, then by
    priority). Set include_done to also show completed ones."""
    owner = owner_from_config(runtime.config)
    if not owner:
        return _NO_OWNER
    try:
        tasks = await asyncio.to_thread(
            get_todo_provider().list, owner, include_done=include_done
        )
    except TodoError as exc:
        return str(exc)
    return _format_tasks(tasks)


@tool
async def add_task(
    title: str,
    runtime: ToolRuntime,
    priority: str = "normal",
    due_date: str = "",
) -> str:
    """Add a task to the user's list.

    `priority` is one of high, normal, low. `due_date` is YYYY-MM-DD — resolve
    relative dates ("tomorrow", "friday") to a real date before calling.
    """
    owner = owner_from_config(runtime.config)
    if not owner:
        return _NO_OWNER

    priority = priority.lower().strip()
    if priority not in PRIORITIES:
        return f"Priority must be one of: {', '.join(PRIORITIES)}."

    provider = get_todo_provider()
    try:
        # Read before write: the model re-adds a task it already added often
        # enough ("add review PR" twice in one conversation) that the list fills
        # with near-duplicates. One extra read is cheaper than a list the user
        # has to clean up by hand.
        existing = find_duplicate(title, await asyncio.to_thread(provider.list, owner))
        if existing is not None:
            return f"Already on your list, not added again: {existing.describe()}"

        task = await asyncio.to_thread(
            provider.add,
            owner,
            title,
            priority=priority,
            due_date=_parse_due(due_date),
        )
    except TodoError as exc:
        return str(exc)
    return f"Added: {task.describe()}"


@tool
async def complete_task(task_id: str, runtime: ToolRuntime) -> str:
    """Mark a task done. `task_id` is the id shown in square brackets by
    `list_tasks` — if you don't have it, list the tasks first."""
    owner = owner_from_config(runtime.config)
    if not owner:
        return _NO_OWNER
    try:
        task = await asyncio.to_thread(get_todo_provider().complete, owner, task_id)
    except TodoError as exc:
        return str(exc)
    return f"Done: {task.title}"


# --- Agent -------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = """You are the user's personal assistant. You open \
their day and keep their task list.

When they greet you, say "good morning", or ask what's planned for today, call \
`daily_briefing` once and present it as a dashboard. Reply in markdown, in \
exactly this shape — the blank lines and the `-` bullets matter, because \
without them the whole dashboard renders as one paragraph:

Good morning{, name} 👋

**🌤 Weather**

<temperature | condition | high/low | chance of rain>

**✅ Tasks**

- <task>
- <task>

**💡 Recommendation**

<the suggestions from the tool, in a sentence or two>

Rules that matter:
- Never invent a number, a task, or a forecast. Every fact comes from a tool. If \
a section came back unavailable, say so in one short line and move on.
- The suggestions are already worked out for you. Present them; don't add \
advice the data doesn't support.
- Call `daily_briefing` with no location. Their city is remembered for them. If \
it replies that no city is on file, ask which city they're in and call it again \
with that city — it is saved automatically. Never call `remember` for a city.
- `remember` is for durable facts the user tells you about themselves. Never use \
it to record what you don't know, or what you intend to do next.
- Calendar is not connected yet. Never mention meetings, and never imply you \
checked a calendar.
- Outside a briefing, just be useful and brief: add tasks, complete them, \
answer a weather question. Resolve "tomorrow" or "friday" to a real date \
yourself before calling a tool.
- Today's date is provided by the briefing tool. For anything date-related \
outside it, ask rather than guess."""

agent = create_agent(
    model=get_chat_model(),
    tools=[daily_briefing, get_weather, list_tasks, add_task, complete_task],
    system_prompt=get_prompt(
        "personal_assistant_system",
        _DEFAULT_SYSTEM_PROMPT,
        name="Personal Assistant — System Prompt",
    ),
    middleware=[MemoryMiddleware()],  # remembers the user's city, and their name
)

MANIFEST = AgentManifest(
    id="personal_assistant",
    label="Personal Assistant",
    emoji="🌤",
    description=(
        "The user's daily briefing and task list: today's weather, what's due, "
        "what's overdue, and concrete suggestions drawn from both. Use when the "
        "user greets Super Bot (hi, hello, good morning), asks what's planned "
        "for today, asks about the weather, or wants to add, list or complete a "
        "task or reminder."
    ),
    agent_type="langchain",
    builder=lambda: agent,
    capabilities=["briefing", "weather", "tasks"],
)
