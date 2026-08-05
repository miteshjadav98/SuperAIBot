"""Personal Assistant — task filtering, suggestions, and briefing composition.

No LLM calls and no network: this covers the deterministic half of the agent,
which is where the briefing's correctness actually lives. The model only
presents what these functions decide.
"""

from __future__ import annotations

from datetime import date

import pytest

from agents.personal_assistant import (
    HOME_CITY_PREFIX,
    _compose,
    _format_tasks,
    _parse_due,
    _remember_city,
    _remembered_city,
    _suggestions,
)
from tools.todo.base import (
    Todo,
    TodoError,
    due_today,
    find_duplicate,
    high_priority,
    overdue,
    sort_for_display,
)
from tools.todo.memory import InMemoryTodoProvider
from tools.weather.base import Weather

TODAY = date(2026, 8, 5)


def todo(title, *, priority="normal", due=None, done=False, todo_id=None):
    return Todo(
        id=todo_id or title[:4],
        title=title,
        priority=priority,
        done=done,
        due_date=due,
    )


def weather(**overrides):
    defaults = dict(
        location="Ahmedabad, Gujarat, India",
        temperature_c=31.0,
        feels_like_c=33.0,
        condition="Sunny",
        high_c=34.0,
        low_c=26.0,
        rain_probability=20,
        wind_kph=12.0,
        alerts=[],
    )
    return Weather(**{**defaults, **overrides})


# --- Filters -----------------------------------------------------------------


def test_overdue_is_strictly_before_today():
    """A task due today is not late yet — the user still has the day."""
    tasks = [
        todo("yesterday", due=date(2026, 8, 4)),
        todo("today", due=TODAY),
        todo("tomorrow", due=date(2026, 8, 6)),
    ]

    assert [t.title for t in overdue(tasks, TODAY)] == ["yesterday"]
    assert [t.title for t in due_today(tasks, TODAY)] == ["today"]


def test_a_completed_task_is_never_overdue():
    """Otherwise finishing a late task keeps nagging about it forever."""
    assert overdue([todo("done", due=date(2026, 1, 1), done=True)], TODAY) == []


def test_a_task_with_no_due_date_is_never_late():
    assert overdue([todo("someday")], TODAY) == []
    assert due_today([todo("someday")], TODAY) == []


def test_high_priority_ignores_completed_tasks():
    tasks = [todo("a", priority="high", done=True), todo("b", priority="high")]

    assert [t.title for t in high_priority(tasks)] == ["b"]


def test_display_order_is_overdue_then_today_then_priority():
    """The order the briefing shows tasks in is itself the recommendation."""
    tasks = [
        todo("low, no date", priority="low"),
        todo("high, no date", priority="high"),
        todo("due today", due=TODAY),
        todo("overdue", due=date(2026, 7, 1)),
    ]

    ordered = [t.title for t in sort_for_display(tasks, TODAY)]

    assert ordered == ["overdue", "due today", "high, no date", "low, no date"]


# --- Suggestions -------------------------------------------------------------
# Rules, not a model call: they must be exactly this predictable.


def test_overdue_tasks_lead_the_suggestions():
    out = _suggestions(weather(), [todo("file taxes", due=date(2026, 7, 1))], TODAY)

    assert "overdue" in out[0]
    assert "file taxes" in out[0]


def test_rain_suggests_an_umbrella_only_past_the_threshold():
    assert any("umbrella" in s for s in _suggestions(weather(rain_probability=40), [], TODAY))
    assert not any("umbrella" in s for s in _suggestions(weather(rain_probability=39), [], TODAY))


def test_heat_warning_is_included():
    assert any("water" in s for s in _suggestions(weather(high_c=41.0), [], TODAY))


def test_weather_alerts_are_passed_through():
    out = _suggestions(weather(alerts=["Cyclone warning"]), [], TODAY)

    assert any("Cyclone warning" in s for s in out)


def test_suggestions_survive_a_missing_forecast():
    """Weather can be down; the task advice still has to arrive."""
    out = _suggestions(None, [todo("ship it", due=date(2026, 7, 1))], TODAY)

    assert out and "overdue" in out[0]


def test_an_empty_day_still_says_something():
    assert _suggestions(weather(), [], TODAY)


# --- Briefing composition ----------------------------------------------------


def test_briefing_reports_the_counts_it_was_given():
    tasks = [
        todo("overdue one", due=date(2026, 7, 1)),
        todo("due now", due=TODAY),
        todo("urgent", priority="high"),
    ]

    brief = _compose(weather(), "", tasks, "", TODAY)

    assert "3 open, 1 overdue, 1 due today, 1 high priority" in brief
    assert "Ahmedabad" in brief


def test_briefing_degrades_when_weather_is_down():
    """One dead source must not cost the user the rest of their day."""
    brief = _compose(None, "The weather service didn't respond in time.", [todo("a")], "", TODAY)

    assert "Unavailable" in brief
    assert "[a] a" in brief or "] a" in brief


def test_briefing_degrades_when_the_task_list_is_down():
    brief = _compose(weather(), "", [], "Couldn't read the task list.", TODAY)

    assert "Couldn't read the task list." in brief
    assert "Sunny" in brief


def test_briefing_caps_the_task_list():
    from agents.personal_assistant import TASKS_IN_BRIEFING

    tasks = [todo(f"task {i}", todo_id=f"id{i}") for i in range(TASKS_IN_BRIEFING + 3)]

    brief = _compose(weather(), "", tasks, "", TODAY)

    assert "+3 more not shown" in brief


def test_briefing_tells_the_model_not_to_invent_meetings():
    """Calendar isn't connected — the dashboard must not imply that it is."""
    assert "Not connected yet" in _compose(weather(), "", [], "", TODAY)


# --- Tool input handling -----------------------------------------------------


def test_due_dates_are_parsed_strictly():
    assert _parse_due("2026-08-05") == TODAY
    assert _parse_due("") is None

    with pytest.raises(TodoError):
        _parse_due("next friday")


def test_empty_task_list_reads_as_a_sentence():
    assert _format_tasks([]) == "No open tasks."


# --- Duplicate tasks ---------------------------------------------------------


def test_the_same_title_is_a_duplicate_whatever_the_casing_or_spacing():
    tasks = [todo("Review PR")]

    assert find_duplicate("review  pr", tasks) is not None
    assert find_duplicate("REVIEW PR.", tasks) is not None


def test_a_different_task_is_not_a_duplicate():
    assert find_duplicate("review PR", [todo("reply to recruiter")]) is None


def test_a_completed_task_does_not_block_adding_it_again():
    """Finishing "water the plants" must not stop you adding it next week."""
    assert find_duplicate("water the plants", [todo("water the plants", done=True)]) is None


def test_an_empty_title_matches_nothing():
    assert find_duplicate("   ", [todo("something")]) is None


# --- Remembering the user's city ---------------------------------------------
# Written and read by code, never by the model: asked to save "Ahmedabad", a
# live model instead saved "does not know user's city; must ask city before
# briefing" — into the store every agent reads.


def test_city_round_trips_through_memory(memory):
    _remember_city("alice", "Ahmedabad")

    assert _remembered_city("alice") == "Ahmedabad"


def test_no_city_on_file_reads_as_none(memory):
    assert _remembered_city("alice") is None


def test_the_city_is_found_among_unrelated_memories(memory):
    memory.remember_fact("alice", "is vegetarian")
    _remember_city("alice", "Pune")
    memory.remember_fact("alice", "prefers short answers")

    assert _remembered_city("alice") == "Pune"


def test_one_user_s_city_is_not_another_s(memory):
    _remember_city("alice", "Ahmedabad")

    assert _remembered_city("bob") is None


def test_a_saved_city_is_stored_under_the_shared_prefix(memory):
    """Other agents read this store too — the fact has to read as a fact."""
    _remember_city("alice", "Ahmedabad")

    assert memory.recall("alice") == [f"{HOME_CITY_PREFIX} Ahmedabad"]


# --- Provider contract -------------------------------------------------------


def test_in_memory_provider_keeps_users_apart():
    """The owner argument is the isolation boundary, not a filter for later."""
    provider = InMemoryTodoProvider()
    provider.add("alice", "alice's task")
    provider.add("bob", "bob's task")

    assert [t.title for t in provider.list("alice")] == ["alice's task"]


def test_completing_a_task_hides_it_from_the_open_list():
    provider = InMemoryTodoProvider()
    task = provider.add("alice", "ship it")

    provider.complete("alice", task.id)

    assert provider.list("alice") == []
    assert [t.title for t in provider.list("alice", include_done=True)] == ["ship it"]


def test_a_task_needs_a_title():
    with pytest.raises(TodoError):
        InMemoryTodoProvider().add("alice", "   ")


def test_completing_someone_else_s_task_fails():
    provider = InMemoryTodoProvider()
    task = provider.add("alice", "private")

    with pytest.raises(TodoError):
        provider.complete("bob", task.id)


# --- The MongoDB provider, over mongomock ------------------------------------
# The default in production, so its round-trip is worth exercising for real.


@pytest.fixture
def mongo_todos():
    import mongomock

    from tools.todo.internal import MongoTodoProvider

    return MongoTodoProvider(mongomock.MongoClient()["test"]["todos"])


def test_mongo_round_trips_a_task_with_a_due_date(mongo_todos):
    """Dates are stored as ISO day strings — the value must survive unchanged,
    with no timezone shifting it across a day boundary."""
    mongo_todos.add("alice", "renew passport", priority="high", due_date=TODAY)

    (task,) = mongo_todos.list("alice")

    assert (task.title, task.priority, task.due_date) == ("renew passport", "high", TODAY)


def test_mongo_scopes_every_read_to_the_owner(mongo_todos):
    mongo_todos.add("alice", "alice's")
    mongo_todos.add("bob", "bob's")

    assert [t.title for t in mongo_todos.list("bob")] == ["bob's"]


def test_mongo_will_not_complete_another_user_s_task(mongo_todos):
    """Owner is part of the query, so a guessed id matches nothing."""
    task = mongo_todos.add("alice", "private")

    with pytest.raises(TodoError):
        mongo_todos.complete("bob", task.id)

    assert mongo_todos.list("alice")[0].done is False


def test_mongo_rejects_a_malformed_id_without_raising_into_the_graph(mongo_todos):
    with pytest.raises(TodoError):
        mongo_todos.complete("alice", "not-an-id")
