"""Super Bot planner/executor — plan validation and DAG scheduling.

No LLM calls: these cover the deterministic half of the graph, which is where
the failure modes that matter live (deadlocks, runaway loops, bad plans).
"""

from __future__ import annotations

import pytest

from superbot.executor import _ready, fan_out
from superbot.planner import _validate
from superbot.state import TaskResult, TaskSpec


def task(task_id, depends_on=(), agent_id="personal_chef", max_attempts=2):
    return {
        "id": task_id,
        "agent_id": agent_id,
        "instruction": f"do {task_id}",
        "depends_on": list(depends_on),
        "max_attempts": max_attempts,
    }


def result(task_id, ok=True):
    return TaskResult(task_id=task_id, agent_id="personal_chef", ok=ok, output=f"out-{task_id}")


@pytest.fixture
def diamond():
    """t1 and t2 are independent; t3 waits for both."""
    return [task("t1"), task("t2"), task("t3", depends_on=["t1", "t2"])]


# --- Scheduling --------------------------------------------------------------


def test_independent_tasks_are_ready_together(diamond):
    """Both run in one superstep — this is the parallelism."""
    assert [t["id"] for t in _ready(diamond, [])] == ["t1", "t2"]


def test_dependent_task_waits_for_every_dependency(diamond):
    assert [t["id"] for t in _ready(diamond, [result("t1")])] == ["t2"]
    assert [t["id"] for t in _ready(diamond, [result("t1"), result("t2")])] == ["t3"]


def test_nothing_is_ready_once_all_tasks_succeed(diamond):
    done = [result("t1"), result("t2"), result("t3")]

    assert _ready(diamond, done) == []


def test_failed_task_is_retried_within_its_budget():
    plan = [task("t1", max_attempts=2)]

    assert [t["id"] for t in _ready(plan, [result("t1", ok=False)])] == ["t1"]
    assert _ready(plan, [result("t1", ok=False), result("t1", ok=False)]) == []


def test_task_blocked_by_a_dead_dependency_terminates(diamond):
    """t1 has spent its budget, so t3 can never run. The scheduler must return
    empty (ending the run) rather than spinning waiting for it."""
    exhausted = [result("t1", ok=False), result("t1", ok=False), result("t2")]

    assert _ready(diamond, exhausted) == []


# --- Fan-out -----------------------------------------------------------------


def test_fan_out_sends_one_worker_per_ready_task(diamond):
    sends = fan_out({"messages": [], "plan": diamond, "results": [], "layers": 1})

    assert [s.node for s in sends] == ["execute", "execute"]


def test_fan_out_passes_dependency_output_downstream(diamond):
    state = {
        "messages": [],
        "plan": diamond,
        "results": [result("t1"), result("t2")],
        "layers": 1,
    }

    sends = fan_out(state)

    assert len(sends) == 1
    assert sends[0].arg["upstream"] == "out-t1\n\nout-t2"


def test_solo_flag_marks_a_single_task_plan_as_streamable():
    """A one-task plan skips synthesis, so that agent's tokens are the final
    answer and must reach the UI."""
    sends = fan_out({"messages": [], "plan": [task("t1")], "results": [], "layers": 1})

    assert sends[0].arg["solo"] is True


def test_solo_is_false_when_several_agents_run(diamond):
    """Streaming two agents at once would interleave two half-written answers."""
    sends = fan_out({"messages": [], "plan": diamond, "results": [], "layers": 1})

    assert all(s.arg["solo"] is False for s in sends)


def test_fan_out_stops_at_the_layer_cap(diamond):
    """Bound on the dispatch->execute loop, independent of plan shape."""
    state = {"messages": [], "plan": diamond, "results": [], "layers": 99}

    assert fan_out(state) == "synthesize"


def test_fan_out_finishes_when_nothing_is_ready():
    assert fan_out({"messages": [], "plan": [], "results": [], "layers": 1}) == "synthesize"


# --- Plan validation ---------------------------------------------------------
# The planner's output is model-generated and therefore untrusted.


def spec(task_id, agent_id="personal_chef", depends_on=(), instruction="x"):
    return TaskSpec(
        id=task_id, agent_id=agent_id, instruction=instruction, depends_on=list(depends_on)
    )


def test_tasks_for_unknown_agents_are_dropped():
    tasks = _validate([spec("a"), spec("b", agent_id="does_not_exist")])

    assert [t["id"] for t in tasks] == ["a"]


def test_duplicate_task_ids_are_dropped():
    tasks = _validate([spec("a"), spec("a", instruction="dupe")])

    assert len(tasks) == 1


def test_cycles_are_unrepresentable():
    """Dependencies may only point backwards, so a -> b -> a collapses to a
    runnable plan instead of a deadlock."""
    tasks = _validate([spec("a", depends_on=["b"]), spec("b", depends_on=["a"])])

    assert [t["id"] for t in tasks] == ["a", "b"]
    assert tasks[0]["depends_on"] == []
    assert tasks[1]["depends_on"] == ["a"]


def test_plan_is_capped():
    from superbot.state import MAX_TASKS

    tasks = _validate([spec(f"t{i}") for i in range(MAX_TASKS + 5)])

    assert len(tasks) == MAX_TASKS


def test_blank_instructions_are_dropped():
    assert _validate([spec("a", instruction="   ")]) == []
