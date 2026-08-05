"""The Super Bot graph — a planner/executor over the agent registry.

Registered as the ``superbot`` assistant. Where v1 was a router (``classify ->
delegate``, exactly one agent per turn), v2 plans:

    plan -> dispatch --Send x N--> execute --+
              ^                              |
              |                              |
              +------------------------------+
              |
              +--> synthesize -> END

``plan`` produces a validated task DAG, ``dispatch`` fans out each layer of
independent tasks in parallel, and ``synthesize`` merges the results. Requests
that span agents ("book the flights and email the itinerary") now work; requests
that don't still cost one planner call plus one agent call, because a
single-task plan short-circuits synthesis.

Manual agent selection is unchanged: pass ``config.configurable.agent_id`` (or
``agent_id`` in state) and the run becomes a one-task plan for that agent with no
planner call at all.

Note: nested human-in-the-loop interrupts (e.g. the email agent's send approval)
are still best driven by selecting that agent directly — an interrupt inside a
fanned-out worker propagates to this parent run, so resuming targets the Super
Bot thread rather than the sub-agent's.

Assembly only — the logic lives in ``planner.py``/``executor.py``, the state
contract in ``state.py``.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from superbot.executor import dispatch, execute, fan_out, plan, synthesize
from superbot.state import SuperBotState

_builder = StateGraph(SuperBotState)

_builder.add_node("plan", plan)
_builder.add_node("dispatch", dispatch)
_builder.add_node("execute", execute)
_builder.add_node("synthesize", synthesize)

_builder.add_edge(START, "plan")
_builder.add_edge("plan", "dispatch")
# Explicit destinations so Studio and static analysis can see where this goes.
_builder.add_conditional_edges("dispatch", fan_out, ["execute", "synthesize"])
_builder.add_edge("execute", "dispatch")  # next layer, or fall through to synthesis
_builder.add_edge("synthesize", END)

agent = _builder.compile()
