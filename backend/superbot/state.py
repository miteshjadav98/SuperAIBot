"""State schema for the Super Bot planner/executor graph.

One module owns "what the state is", so nodes never disagree about the contract.

The load-bearing detail is ``results``: several ``execute`` workers write it in
the *same* superstep (that's the point of the parallel fan-out), so it needs a
reducer. Without one LangGraph raises ``InvalidUpdateError`` rather than guessing
which worker's write wins.

Everything stored here is JSON-serialisable, because the whole state is
checkpointed on every superstep. Task results are plain strings by the time they
land — sub-agent message histories stay inside the sub-agent.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import MessagesState
from pydantic import BaseModel, Field

# --- Bounds. Every one of these exists because the alternative is unbounded. ---

MAX_TASKS = 6
"""Most tasks one plan may contain. Caps fan-out width (cost per turn)."""

MAX_LAYERS = 5
"""Most dispatch rounds per run. Caps DAG depth, so the dispatch->execute loop
can never spin forever even on a malformed plan."""

TASK_MAX_ATTEMPTS = 2
"""Attempts per task before it is recorded as failed and the run moves on.
A code-level policy, deliberately not something the planner LLM decides."""

CONTEXT_MESSAGES = 6
"""How many recent messages ride along in each worker's payload. Send payloads
are checkpointed per worker, so unbounded history here costs N x history."""


class TaskSpec(BaseModel):
    """One planned step, exactly as the planner LLM emits it.

    Pydantic rather than TypedDict on purpose: this crosses a trust boundary
    (model output), so it is validated at runtime, not just type-checked.
    """

    id: str = Field(description="Short unique id for this task, e.g. 't1'.")
    agent_id: str = Field(description="Exact id of the agent that should run this task.")
    instruction: str = Field(
        description="Self-contained instruction for that agent. It cannot see the "
        "other tasks, so do not refer to them."
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Ids of tasks whose output this one needs. Empty means it can "
        "run immediately, in parallel with other independent tasks.",
    )


class Plan(BaseModel):
    """The planner's structured output. A DAG of tasks, not a script."""

    tasks: list[TaskSpec] = Field(description="Tasks to run, in rough execution order.")


class Task(TypedDict):
    """A validated task as it lives in state (with its retry budget attached)."""

    id: str
    agent_id: str
    instruction: str
    depends_on: list[str]
    max_attempts: int


class TaskResult(TypedDict):
    """One attempt at one task. Failures are recorded, never raised — a dead
    agent must not take down the other branches of the fan-out."""

    task_id: str
    agent_id: str
    ok: bool
    output: str


class TaskPayload(TypedDict):
    """Private input handed to an ``execute`` worker via ``Send``.

    Deliberately not the parent schema: each worker sees only its own task plus
    a trimmed slice of conversation.
    """

    task: Task
    context: list
    upstream: str  # outputs of this task's dependencies, inlined into its prompt
    solo: bool
    """True when this is the only task in the plan, so its tokens are the final
    answer and should stream straight to the UI. False when several agents run:
    their output is merged by ``synthesize``, and streaming them all would
    interleave two half-finished answers in the chat."""


class SuperBotState(MessagesState):
    agent_id: Optional[str]  # manual override from config/state; skips planning
    plan: Optional[list[Task]]  # in state so it is inspectable, editable, resumable
    results: Annotated[list[TaskResult], operator.add]  # parallel writers -> reducer
    routed_to: Optional[str]  # which agent(s) actually ran; for observability
    layers: int  # dispatch rounds so far; bounded by MAX_LAYERS
