"""Run-concurrency policy.

The failure this guards: two runs on one thread_id share a checkpoint, so both
read the same messages and both write back — losing one side or interleaving
them into a conversation that never happened.
"""

from __future__ import annotations

import asyncio

import pytest

from core.concurrency import RunBusy, is_running, reset, thread_run


@pytest.fixture(autouse=True)
def clean():
    reset()
    yield
    reset()


async def test_a_single_run_is_allowed():
    async with thread_run("t1", policy="reject"):
        assert is_running("t1")

    assert not is_running("t1")


async def test_reject_refuses_a_concurrent_run_on_the_same_thread():
    async with thread_run("t1", policy="reject"):
        with pytest.raises(RunBusy):
            async with thread_run("t1", policy="reject"):
                pass


async def test_different_threads_run_concurrently():
    """Isolation is per thread — one user's run must not block another's."""
    async with thread_run("t1", policy="reject"):
        async with thread_run("t2", policy="reject"):
            assert is_running("t1") and is_running("t2")


async def test_the_thread_is_released_after_a_failure():
    """A crashed run must not wedge the thread forever."""
    with pytest.raises(ValueError):
        async with thread_run("t1", policy="reject"):
            raise ValueError("boom")

    assert not is_running("t1")
    async with thread_run("t1", policy="reject"):
        pass  # usable again


async def test_reject_is_atomic_under_a_real_race():
    """Both coroutines start together; exactly one must win."""
    outcomes = []

    async def run():
        try:
            async with thread_run("t1", policy="reject"):
                await asyncio.sleep(0.02)
                outcomes.append("ran")
        except RunBusy:
            outcomes.append("rejected")

    await asyncio.gather(run(), run(), run())

    assert outcomes.count("ran") == 1
    assert outcomes.count("rejected") == 2


async def test_enqueue_serialises_instead_of_rejecting():
    order = []

    async def run(label):
        async with thread_run("t1", policy="enqueue"):
            order.append(f"start-{label}")
            await asyncio.sleep(0.01)
            order.append(f"end-{label}")

    await asyncio.gather(run("a"), run("b"))

    # No interleaving: each run completes before the next begins.
    assert order in (
        ["start-a", "end-a", "start-b", "end-b"],
        ["start-b", "end-b", "start-a", "end-a"],
    ), order


async def test_enqueue_does_not_leak_locks():
    """Per-thread locks must be reclaimed, or the dict grows without bound."""
    from core.concurrency import _locks

    for i in range(20):
        async with thread_run(f"thread-{i}", policy="enqueue"):
            pass

    assert _locks == {}


async def test_unknown_policy_fails_loudly():
    """Silently running with no protection would be the worst outcome."""
    with pytest.raises(ValueError, match="Unknown run concurrency policy"):
        async with thread_run("t1", policy="whatever"):
            pass
