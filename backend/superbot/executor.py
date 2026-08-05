"""Executor — runs a planned task DAG, one parallel layer at a time.

``dispatch`` works out which tasks are *ready* (all dependencies satisfied) and
fans them out with ``Send``. Every ready task runs concurrently in the same
superstep; their results merge back through the ``results`` reducer, and control
returns to ``dispatch`` for the next layer. When nothing is ready, the run moves
on to ``synthesize``.

    dispatch --Send x N--> execute --+
        ^                            |
        +----------------------------+
        |
        +--> synthesize -> END

Failure policy: a worker never raises into the graph. A failed task is recorded
as a result with ``ok=False``, retried while its attempt budget lasts, and
otherwise reported honestly in the final answer. One dead agent degrades the
answer; it does not abort the other branches of the fan-out.
"""

from __future__ import annotations

from collections import Counter

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Send

from core.prompts import get_prompt
from core.registry import registry
from llm.factory import get_chat_model
from superbot.planner import make_plan, single_task_plan
from superbot.state import (
    ASSISTANT_AGENT_ID,
    CONTEXT_MESSAGES,
    MAX_LAYERS,
    SuperBotState,
    Task,
    TaskPayload,
    TaskResult,
)

_SYNTHESIS_DEFAULT = """You are the Super Bot. Several specialist agents each \
handled part of the user's request; their results are below.

Write the single answer the user should see. Merge the results into one coherent \
response, keep the useful detail, and do not mention the agents, the plan, or \
that the work was split up. If a part failed, say plainly what could not be done \
and still give the user everything that succeeded."""

_ASSISTANT_DEFAULT = """You are Super Bot, a personal AI assistant.

You work with a team of specialists — a wedding planner, an email assistant, a \
personal chef, a PDF document expert and a movie recommender — and you hand work \
to them behind the scenes. This turn needs none of them, so you are answering in \
your own voice.

Be warm, direct and concise. Answer from the conversation and from what you \
know. If the user would be better served by one of the specialists, say what you \
can help with and invite them to ask — never impersonate one of them, and never \
claim to have done work you have not done."""


def _forced_agent(state: SuperBotState, config: RunnableConfig | None) -> str | None:
    """A manual pick from the frontend dropdown or the API, if it names a real
    agent. Honouring this is what keeps "choose your own agent" working."""
    configurable = (config or {}).get("configurable", {})
    forced = configurable.get("agent_id") or state.get("agent_id")
    return forced if forced and registry.get(forced) else None


def _query(messages: list[AnyMessage]) -> str:
    last = messages[-1]
    content = getattr(last, "content", last)
    return content if isinstance(content, str) else str(content)


# --- Nodes -------------------------------------------------------------------


async def plan(state: SuperBotState, config: RunnableConfig | None = None) -> dict:
    """Build the task DAG. A manual agent override skips the planner LLM call."""
    query = _query(state["messages"])

    forced = _forced_agent(state, config)
    if forced:
        tasks = single_task_plan(forced, query)
    else:
        # `routed_to` still holds the *previous* turn's agents at this point —
        # this node is what overwrites it — which is how a follow-up ("what
        # about the venue?") stays with the agent that answered before it.
        tasks = await make_plan(
            query, state["messages"], previous_agents=state.get("routed_to")
        )

    return {
        "plan": tasks,
        "layers": 0,
        # Clear last turn's results (see `accumulate_results`): this thread's
        # state outlives the turn, and stale results would satisfy this turn's
        # dependencies — the run would answer with the previous turn's output.
        "results": None,
        "routed_to": ",".join(sorted({t["agent_id"] for t in tasks})) or None,
    }


def dispatch(state: SuperBotState) -> dict:
    """Advance the layer counter. The routing itself happens in :func:`fan_out`,
    because conditional edges choose destinations but cannot write state."""
    return {"layers": state.get("layers", 0) + 1}


def _ready(plan: list[Task], results: list[TaskResult]) -> list[Task]:
    """Tasks whose dependencies have all succeeded and whose attempt budget is
    not yet spent. A task blocked behind a permanently failed dependency simply
    never becomes ready — which ends the loop instead of hanging it."""
    succeeded = {r["task_id"] for r in results if r["ok"]}
    attempts = Counter(r["task_id"] for r in results)

    return [
        task
        for task in plan
        if task["id"] not in succeeded
        and attempts[task["id"]] < task["max_attempts"]
        and set(task["depends_on"]) <= succeeded
    ]


def fan_out(state: SuperBotState) -> list[Send] | str:
    """Send every ready task to its own worker, or finish."""
    if state.get("layers", 0) > MAX_LAYERS:
        return "synthesize"

    ready = _ready(state.get("plan") or [], state.get("results") or [])
    if not ready:
        return "synthesize"

    context = state["messages"][-CONTEXT_MESSAGES:-1]
    completed = {r["task_id"]: r["output"] for r in state.get("results") or [] if r["ok"]}
    solo = len(state.get("plan") or []) == 1

    return [
        Send(
            "execute",
            TaskPayload(
                task=task,
                context=context,
                upstream=_upstream(task, completed),
                solo=solo,
            ),
        )
        for task in ready
    ]


def _upstream(task: Task, completed: dict[str, str]) -> str:
    """Outputs of the tasks this one depends on, inlined into its prompt — this
    is how a dependent task actually receives its predecessor's work."""
    parts = [completed[dep] for dep in task["depends_on"] if dep in completed]
    return "\n\n".join(parts)


def _result(task: Task, ok: bool, output: str) -> dict:
    return {
        "results": [
            TaskResult(task_id=task["id"], agent_id=task["agent_id"], ok=ok, output=output)
        ]
    }


async def _answer_here(payload: TaskPayload, instruction: str, run_config: dict) -> str:
    """The Super Bot's own reply — no sub-agent, no tools, just its persona.

    This is the turn the platform used to have nowhere to put: a greeting, "what
    can you do?", a thank-you. The conversation slice rides along, so a
    follow-up here reads the same history a specialist would have seen.
    """
    system = get_prompt(
        "superbot_assistant_system", _ASSISTANT_DEFAULT, name="Super Bot — Assistant"
    )
    reply = await get_chat_model().ainvoke(
        [
            SystemMessage(content=system),
            *payload["context"],
            HumanMessage(content=instruction),
        ],
        config=run_config,
    )
    return str(getattr(reply, "content", ""))


async def execute(payload: TaskPayload, config: RunnableConfig | None = None) -> dict:
    """Run one task on one agent. Receives a private payload, not parent state."""
    task = payload["task"]
    is_assistant = task["agent_id"] == ASSISTANT_AGENT_ID
    agent = None if is_assistant else registry.get(task["agent_id"])

    if agent is None and not is_assistant:
        return _result(task, False, f"Agent '{task['agent_id']}' is not registered.")

    instruction = task["instruction"]
    if payload.get("upstream"):
        instruction = f"{instruction}\n\nUse this information from an earlier step:\n{payload['upstream']}"

    # Streaming policy. A one-task plan short-circuits synthesis, so this
    # agent's tokens *are* the final answer and must reach the UI. With several
    # agents running at once, streaming them all would interleave two
    # half-written answers, so they are silenced and only `synthesize` streams.
    # Suppressed either way in the UI only — LangSmith still records everything.
    run_config = dict(config or {})
    if not payload.get("solo", True):
        run_config["tags"] = [*run_config.get("tags", []), "langsmith:nostream"]

    try:
        if is_assistant:
            output = await _answer_here(payload, instruction, run_config)
        else:
            result = await agent.invoke(
                {"messages": [*payload["context"], HumanMessage(content=instruction)]},
                config=run_config,
            )
            messages = result.get("messages") or []
            output = getattr(messages[-1], "content", "") if messages else ""
    except Exception as exc:  # noqa: BLE001 — isolate branch failures
        return _result(task, False, f"{type(exc).__name__}: {exc}")

    if not str(output).strip():
        return _result(task, False, "The agent returned an empty response.")

    return _result(task, True, str(output))


async def synthesize(state: SuperBotState) -> dict:
    """Compose the user-facing answer.

    The single-task fast path returns the agent's answer verbatim — the common
    case must not pay for an extra LLM call just to rephrase one result.
    """
    results = state.get("results") or []
    plan = state.get("plan") or []

    if not results:
        return {"messages": [AIMessage(content="I couldn't work out how to handle that request.")]}

    # Keep the last attempt per task, in plan order.
    final: dict[str, TaskResult] = {r["task_id"]: r for r in results}
    ordered = [final[t["id"]] for t in plan if t["id"] in final]

    if len(ordered) == 1 and ordered[0]["ok"]:
        return {"messages": [AIMessage(content=ordered[0]["output"])]}

    sections = "\n\n".join(
        f"### {r['agent_id']} ({'ok' if r['ok'] else 'failed'})\n{r['output']}" for r in ordered
    )
    system = get_prompt("superbot_synthesis_system", _SYNTHESIS_DEFAULT, name="Super Bot — Synthesis")

    try:
        model = get_chat_model(temperature=0)
        answer = await model.ainvoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Request: {_query(state['messages'])}\n\nResults:\n{sections}"},
            ]
        )
        return {"messages": [AIMessage(content=answer.content)]}
    except Exception as exc:  # noqa: BLE001 — never lose completed work
        print(f"[executor] synthesis failed, returning raw results: {exc}")
        return {"messages": [AIMessage(content=sections)]}
