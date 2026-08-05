"""Run-concurrency policy for graph invocations.

The bug this closes: a checkpointed graph keyed by ``thread_id`` assumes one run
at a time on that thread. Two concurrent requests on the same thread both read
the same checkpoint, both append to ``messages``, and both write back — so one
run's messages vanish, or the two interleave into a conversation that never
happened. Nothing in LangGraph prevents this; the *server* solves it for its own
runs, but a graph invoked directly from a web handler has no such protection.

Anyone embedding LangGraph in their own service has to choose a policy:

============  ==========================================  ==========================
Policy        Behaviour                                   Fits
============  ==========================================  ==========================
``reject``    Refuse the second run                       Transactional endpoints
``enqueue``   Run it after the first finishes             Task assistants
``interrupt`` Stop the first, run the second              Chat UIs (user changed mind)
``rollback``  Discard the first, run the second           Chat where partial output is noise
============  ==========================================  ==========================

Implemented here: ``reject`` (default) and ``enqueue``. The other two require
cancelling an in-flight run and reconciling its partial checkpoint — worth doing
when the UI needs it, and not worth faking before then.

``reject`` is the default because this gateway is request/response: telling the
caller immediately is more honest than making them wait an unbounded time behind
someone else's run.

**Scope: one process.** These are in-memory primitives, so they serialise runs
within a single gateway worker. Multiple workers or replicas need a shared lock
(a Mongo ``findOneAndUpdate`` with a TTL would fit the existing stack). The
gateway runs single-process today; :func:`thread_run` is the single place that
would change.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager

from core.settings import settings


class RunBusy(RuntimeError):
    """Raised under the ``reject`` policy when a thread is already running."""


# Threads with a run in flight. A set (not a lock) because the check-and-add
# below has no await between its two steps, which makes it atomic on the event
# loop — two coroutines cannot both observe "free" and both claim it.
_active: set[str] = set()

# Per-thread locks for `enqueue`, plus the number of coroutines interested in
# each, so the dict does not grow forever across many threads.
_locks: dict[str, asyncio.Lock] = {}
_waiting: defaultdict[str, int] = defaultdict(int)


@asynccontextmanager
async def thread_run(thread_id: str, policy: str | None = None):
    """Hold the right to run on ``thread_id`` for the duration of the block.

    Args:
        thread_id: The checkpointer thread being written to.
        policy: ``reject`` or ``enqueue``. Defaults to
            ``settings.run_concurrency_policy``.

    Raises:
        RunBusy: Under ``reject``, when a run is already in flight.
        ValueError: For an unknown policy — better to fail at the first request
            than to silently run with no protection at all.
    """
    policy = (policy or settings.run_concurrency_policy).lower()

    if policy == "reject":
        if thread_id in _active:
            raise RunBusy(
                "This conversation is already generating a response. "
                "Wait for it to finish before sending another message."
            )
        _active.add(thread_id)
        try:
            yield
        finally:
            _active.discard(thread_id)
        return

    if policy == "enqueue":
        _waiting[thread_id] += 1
        lock = _locks.setdefault(thread_id, asyncio.Lock())
        try:
            async with lock:
                _active.add(thread_id)
                try:
                    yield
                finally:
                    _active.discard(thread_id)
        finally:
            _waiting[thread_id] -= 1
            if _waiting[thread_id] <= 0:
                _waiting.pop(thread_id, None)
                _locks.pop(thread_id, None)
        return

    raise ValueError(
        f"Unknown run concurrency policy {policy!r}. Supported: reject, enqueue."
    )


def is_running(thread_id: str) -> bool:
    """Whether a run is currently in flight on ``thread_id``."""
    return thread_id in _active


def reset() -> None:
    """Drop all tracking. For tests."""
    _active.clear()
    _locks.clear()
    _waiting.clear()
