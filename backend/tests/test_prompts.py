"""Prompt declaration and publishing.

The publisher writes to the same collections the Prompt Management service owns,
so the rules that matter are: versions are immutable, the pointer moves, and a
rollback target survives. All of that is exercised here over ``mongomock``.
"""

from __future__ import annotations

import mongomock
import pytest

from core import prompts as prompts_module


@pytest.fixture
def store(monkeypatch):
    """`core.prompts` wired to an in-process Mongo."""
    database = mongomock.MongoClient()["test"]

    monkeypatch.setattr(prompts_module, "_cache", {})
    monkeypatch.setattr("core.db.mongo_configured", lambda: True)
    monkeypatch.setattr("core.db.get_db", lambda: database)
    return database


# --- Declaration -------------------------------------------------------------


def test_a_prompt_declares_its_default_at_import_time():
    """The publisher can only see prompts that have been declared — one loaded
    inside a function would otherwise stay invisible until a turn ran."""
    prompt = prompts_module.Prompt("test_declared", "hello", name="Test")

    assert prompts_module.declared_defaults()["test_declared"] == ("Test", "hello")
    assert prompt.default == "hello"


def test_get_prompt_declares_what_it_was_asked_for(store):
    prompts_module.get_prompt("test_via_get", "the default", name="Via get")

    assert prompts_module.declared_defaults()["test_via_get"] == ("Via get", "the default")


# --- Reading the active version ----------------------------------------------


def test_active_content_is_none_before_anything_is_stored(store):
    """Unlike get_prompt, the publisher's read never seeds and never falls back
    to a code default — it has to report what is actually in the database."""
    assert prompts_module.active_content("nothing_here") is None
    assert store["prompts"].count_documents({}) == 0


def test_active_content_follows_the_registry_pointer(store):
    prompts_module.publish("p", "v1 text", name="P")
    prompts_module.publish("p", "v2 text", name="P")

    assert prompts_module.active_content("p") == "v2 text"


# --- Publishing --------------------------------------------------------------


def test_publishing_increments_the_version(store):
    assert prompts_module.publish("p", "first", name="P") == 1
    assert prompts_module.publish("p", "second", name="P") == 2


def test_publishing_leaves_the_previous_version_intact(store):
    """The rollback target. If a publish overwrote in place, a bad prompt would
    be unrecoverable."""
    prompts_module.publish("p", "first", name="P")
    prompts_module.publish("p", "second", name="P")

    v1 = store["prompts"].find_one({"prompt_id": "p", "version": 1})

    assert v1["content"] == "first"
    assert store["prompts"].count_documents({"prompt_id": "p"}) == 2


def test_publishing_moves_both_registry_pointers(store):
    prompts_module.publish("p", "first", name="P")
    prompts_module.publish("p", "second", name="P")

    entry = store["prompt_registry"].find_one({"prompt_id": "p"})

    assert (entry["current_version"], entry["latest_version"]) == (2, 2)


def test_publishing_writes_an_audit_row(store):
    prompts_module.publish("p", "first", name="P")
    prompts_module.publish("p", "second", name="P")

    actions = [d["action"] for d in store["prompt_audit"].find({"prompt_id": "p"})]

    assert actions == ["CREATE_PROMPT", "CREATE_VERSION"]


def test_a_published_prompt_is_what_agents_then_load(store):
    """The point of the whole exercise: the new version reaches the agents."""
    prompts_module.get_prompt("p", "the code default", name="P")  # seeds v1
    prompts_module.publish("p", "the edited version", name="P")

    assert prompts_module.get_prompt("p", "the code default", name="P") == "the edited version"


def test_publishing_without_mongo_fails_loudly(monkeypatch):
    """Every other prompt path degrades silently to the code default. This one
    must not: a publish that quietly did nothing would be reported as a deploy
    step that succeeded."""
    monkeypatch.setattr("core.db.mongo_configured", lambda: False)

    with pytest.raises(RuntimeError):
        prompts_module.publish("p", "content")
