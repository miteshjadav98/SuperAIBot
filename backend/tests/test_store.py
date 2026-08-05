"""MongoDBStore — the BaseStore contract, against a real pymongo API surface."""

from __future__ import annotations

import pytest

NS = ("memories", "user-42")
OTHER = ("memories", "user-99")


def test_put_get_roundtrip(store):
    store.put(NS, "k1", {"text": "Prefers vegetarian food"})

    item = store.get(NS, "k1")
    assert item is not None
    assert item.value["text"] == "Prefers vegetarian food"
    assert item.namespace == NS
    assert store.get(NS, "missing") is None


def test_namespaces_are_isolated(store):
    """The security-critical property: one user's namespace never returns
    another's items."""
    store.put(NS, "k1", {"text": "mine"})
    store.put(OTHER, "k1", {"text": "theirs"})

    assert [i.value["text"] for i in store.search(NS)] == ["mine"]
    assert [i.value["text"] for i in store.search(OTHER)] == ["theirs"]


def test_prefix_search_does_not_match_sibling_namespace(store):
    """("memories", "user-4") must not be swept up by a search for
    ("memories", "user-42") — the prefix regex has to respect separators."""
    store.put(NS, "k", {"text": "user-42"})
    store.put(("memories", "user-4"), "k", {"text": "user-4"})

    assert [i.value["text"] for i in store.search(NS)] == ["user-42"]
    assert len(store.search(("memories",))) == 2


def test_put_same_key_updates_rather_than_duplicates(store):
    store.put(NS, "k1", {"text": "first"})
    store.put(NS, "k1", {"text": "second"})

    assert len(store.search(NS)) == 1
    assert store.get(NS, "k1").value["text"] == "second"


def test_delete_removes_the_item(store):
    store.put(NS, "k1", {"text": "temporary"})
    store.delete(NS, "k1")

    assert store.get(NS, "k1") is None
    assert store.search(NS) == []


@pytest.mark.parametrize(
    ("filt", "expected"),
    [
        ({"kind": "location"}, 1),
        ({"kind": {"$ne": "location"}}, 1),
        (None, 2),
    ],
)
def test_value_filters(store, filt, expected):
    store.put(NS, "a", {"text": "Lives in Pune", "kind": "location"})
    store.put(NS, "b", {"text": "Vegetarian", "kind": "diet"})

    assert len(store.search(NS, filter=filt)) == expected


def test_search_is_newest_first_and_paginates(store):
    for i in range(3):
        store.put(NS, f"k{i}", {"text": f"fact {i}"})

    assert store.search(NS, limit=1)[0].value["text"] == "fact 2"
    assert len(store.search(NS, limit=2, offset=2)) == 1


def test_list_namespaces(store):
    """Regression: Mongo's distinct() unwinds array fields, so listing has to
    go through the joined path or it returns single labels, not namespaces."""
    store.put(NS, "k", {"text": "x"})
    store.put(OTHER, "k", {"text": "y"})

    assert store.list_namespaces() == [NS, OTHER]
    assert store.list_namespaces(prefix=NS) == [NS]
    assert store.list_namespaces(max_depth=1) == [("memories",)]


async def test_async_api(store):
    await store.aput(NS, "k1", {"text": "written async"})

    item = await store.aget(NS, "k1")
    assert item.value["text"] == "written async"
    assert len(await store.asearch(NS)) == 1

    await store.adelete(NS, "k1")
    assert await store.aget(NS, "k1") is None
