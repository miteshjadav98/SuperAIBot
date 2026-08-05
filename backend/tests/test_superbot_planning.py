"""Super Bot planner/executor — plan validation and DAG scheduling.

No LLM calls: these cover the deterministic half of the graph, which is where
the failure modes that matter live (deadlocks, runaway loops, bad plans).
"""

from __future__ import annotations

import pytest

from superbot.executor import _ready, fan_out
from superbot.planner import _continuity, _history, _validate
from superbot.state import (
    ASSISTANT_AGENT_ID,
    CONTEXT_MESSAGES,
    TaskResult,
    TaskSpec,
    accumulate_results,
)


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


# --- Turn boundaries ---------------------------------------------------------
# State is checkpointed per thread, not per turn, so results have to be cleared
# between turns. They are not, a second turn is answered by the first turn's
# output — the whole conversation freezes on its first answer.


def test_results_accumulate_within_a_turn():
    """Parallel workers in one superstep merge, rather than overwriting."""
    merged = accumulate_results([result("t1")], [result("t2")])

    assert [r["task_id"] for r in merged] == ["t1", "t2"]


def test_a_new_turn_clears_the_previous_turn_s_results():
    assert accumulate_results([result("t1"), result("t2")], None) == []


def test_a_reused_task_id_does_not_inherit_the_previous_turn_s_success():
    """Task ids restart at 't1' every turn. With last turn's result still in
    state, this turn's t1 looks done, nothing dispatches, and the user gets the
    previous answer again."""
    stale = accumulate_results([result("t1")], None)

    assert [t["id"] for t in _ready([task("t1")], stale)] == ["t1"]


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


def test_the_supervisor_may_route_a_task_to_itself():
    """The Super Bot's own voice is a valid destination even though it is not a
    registry agent — without it, a greeting has to be forced onto a specialist."""
    tasks = _validate([spec("a", agent_id=ASSISTANT_AGENT_ID)])

    assert [t["agent_id"] for t in tasks] == [ASSISTANT_AGENT_ID]


# --- Cross-turn routing ------------------------------------------------------


def test_continuity_names_the_previous_agent():
    """What keeps a short follow-up with the agent that answered before it."""
    assert "wedding_planner" in _continuity("wedding_planner")


def test_continuity_on_the_first_turn_has_no_previous_agent():
    assert "previous turn was handled by" not in _continuity(None)


def test_history_excludes_the_current_message():
    """The live request is passed separately; repeating it here would double it."""
    class Message:
        def __init__(self, type_, content):
            self.type = type_
            self.content = content

    messages = [Message("human", "older"), Message("ai", "reply"), Message("human", "now")]

    history = _history(messages)

    assert "older" in history and "reply" in history and "now" not in history


def test_history_is_bounded():
    class Message:
        def __init__(self, content):
            self.type = "human"
            self.content = content

    messages = [Message(f"m{i}") for i in range(50)]

    assert len(_history(messages).splitlines()) == CONTEXT_MESSAGES
