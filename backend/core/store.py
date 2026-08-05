"""A MongoDB-backed LangGraph :class:`~langgraph.store.base.BaseStore`.

A *checkpointer* remembers what happened in one conversation (keyed by
``thread_id``). A *store* remembers what is true about a user, across every
conversation (keyed by a namespace tuple). This module is the store.

**Why hand-write one.** LangGraph ships ``InMemoryStore`` (lost on restart) and
``PostgresStore``. There is no first-party Mongo store. Since this platform
already runs one MongoDB Atlas cluster — users, PDF chunks, chat checkpoints and
prompts all live there — adding Postgres purely for memory would mean a second
datastore to operate, back up and secure. Implementing ``BaseStore`` over the
cluster we already have is the cheaper trade, and because it *is* a ``BaseStore``
it can be handed to ``builder.compile(store=...)`` or swapped for
``PostgresStore`` later without touching a single caller.

**Data model.** One document per stored item, in the ``agent_memory``
collection::

    {
      "path":       "memories\\x1fuser-42",     # namespace, joined — indexed prefix
      "namespace":  ["memories", "user-42"],
      "key":        "b6f1...",                   # unique within the namespace
      "value":      {"text": "Prefers vegetarian food"},
      "created_at": ..., "updated_at": ..., "expires_at": ...   # optional TTL
    }

``path`` exists so a namespace-prefix search is an indexed regex anchored at the
start of the string, rather than an array comparison Mongo cannot index as well.

**Not implemented: semantic search.** ``SearchOp.query`` is ignored. Below a few
dozen memories per user, loading them all beats embedding them — retrieval only
adds a recall failure mode and cost. The seam is marked below; Atlas Vector
Search is already used by the PDF chatbot when this becomes worth doing.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

# Unit separator: legal in a namespace label in theory, absent from one in
# practice, and never produced by the ids we namespace on.
_SEP = "\x1f"

# Comparison operators BaseStore defines for SearchOp.filter. They are spelled
# identically in MongoDB, so translation is a key rename, not a rewrite.
_COMPARISON_OPERATORS = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte"}


def _path(namespace: tuple[str, ...]) -> str:
    return _SEP.join(namespace)


def _prefix_query(prefix: tuple[str, ...]) -> dict:
    """Match a namespace and everything nested beneath it."""
    if not prefix:
        return {}
    anchored = re.escape(_path(prefix))
    return {"path": {"$regex": f"^{anchored}({re.escape(_SEP)}|$)"}}


def _translate_filter(filt: dict[str, Any] | None) -> dict:
    """Map a BaseStore value-filter onto the ``value.*`` sub-document."""
    query: dict[str, Any] = {}
    for field, condition in (filt or {}).items():
        if isinstance(condition, dict) and condition and set(condition) <= _COMPARISON_OPERATORS:
            query[f"value.{field}"] = condition
        else:
            query[f"value.{field}"] = {"$eq": condition}
    return query


def _matches(namespace: tuple[str, ...], pattern: tuple[str, ...], *, suffix: bool) -> bool:
    """Wildcard-aware prefix/suffix match used by ``list_namespaces``."""
    if len(pattern) > len(namespace):
        return False
    window = namespace[-len(pattern):] if suffix else namespace[: len(pattern)]
    return all(p == "*" or p == n for p, n in zip(pattern, window))


class MongoDBStore(BaseStore):
    """``BaseStore`` over a single MongoDB collection.

    ``BaseStore`` requires only ``batch``/``abatch``; the ``get``/``put``/
    ``search``/``delete`` convenience methods are inherited and route through
    them. pymongo is synchronous, so ``abatch`` offloads to a worker thread
    rather than blocking the event loop on an Atlas round-trip.
    """

    def __init__(self, collection):
        self._collection = collection
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        # Identity of an item is (namespace, key) — enforce it in the database,
        # not just in application code.
        self._collection.create_index([("path", 1), ("key", 1)], unique=True)
        self._collection.create_index([("path", 1), ("updated_at", -1)])
        # Mongo reaps expired documents itself; TTL costs us no sweeper job.
        self._collection.create_index("expires_at", expireAfterSeconds=0)

    # --- Op handlers ---------------------------------------------------------

    def _get(self, op: GetOp) -> Item | None:
        doc = self._collection.find_one({"path": _path(op.namespace), "key": op.key})
        return self._to_item(doc) if doc else None

    def _put(self, op: PutOp) -> None:
        # BaseStore encodes deletion as a put with value=None.
        if op.value is None:
            self._collection.delete_one({"path": _path(op.namespace), "key": op.key})
            return

        now = datetime.now(timezone.utc)
        document = {
            "path": _path(op.namespace),
            "namespace": list(op.namespace),
            "key": op.key,
            "value": op.value,
            "updated_at": now,
        }
        if op.ttl is not None:
            document["expires_at"] = now + timedelta(minutes=op.ttl)

        self._collection.update_one(
            {"path": document["path"], "key": op.key},
            {"$set": document, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )

    def _search(self, op: SearchOp) -> list[SearchItem]:
        query = {**_prefix_query(op.namespace_prefix), **_translate_filter(op.filter)}
        # NOTE: op.query (semantic search) is deliberately unsupported — see the
        # module docstring. Newest first is the right default for observations.
        # _id breaks ties on updated_at. Two items written in the same clock
        # tick would otherwise have an arbitrary relative order, which makes
        # skip/limit pagination unstable — the same item can appear on two
        # pages, or none. ObjectIds increase monotonically, so this orders ties
        # by insertion.
        cursor = (
            self._collection.find(query)
            .sort([("updated_at", -1), ("_id", -1)])
            .skip(op.offset)
            .limit(op.limit)
        )
        return [
            SearchItem(
                namespace=tuple(doc["namespace"]),
                key=doc["key"],
                value=doc["value"],
                created_at=doc.get("created_at"),
                updated_at=doc.get("updated_at"),
            )
            for doc in cursor
        ]

    def _list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        # distinct() over the *array* field would unwind it and hand back
        # individual labels ("memories", "user-42") rather than whole
        # namespaces. The joined path is the only field that survives distinct
        # intact — and it is indexed.
        namespaces = {
            tuple(path.split(_SEP)) for path in self._collection.distinct("path") if path
        }

        for condition in op.match_conditions or ():
            namespaces = {
                ns
                for ns in namespaces
                if _matches(ns, tuple(condition.path), suffix=condition.match_type == "suffix")
            }

        if op.max_depth is not None:
            namespaces = {ns[: op.max_depth] for ns in namespaces}

        return sorted(namespaces)[op.offset : op.offset + op.limit]

    def _to_item(self, doc: dict) -> Item:
        return Item(
            value=doc["value"],
            key=doc["key"],
            namespace=tuple(doc["namespace"]),
            created_at=doc.get("created_at"),
            updated_at=doc.get("updated_at"),
        )

    # --- BaseStore contract --------------------------------------------------

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Results are positional — one per op, in the order given."""
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._get(op))
            elif isinstance(op, PutOp):
                results.append(self._put(op))
            elif isinstance(op, SearchOp):
                results.append(self._search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._list_namespaces(op))
            else:  # pragma: no cover — new op type in a future langgraph
                raise NotImplementedError(f"MongoDBStore cannot handle {type(op).__name__}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.to_thread(self.batch, list(ops))
