"""The Super Bot supervisor graph.

A two-node LangGraph graph registered as the ``superbot`` assistant:

    classify -> delegate

``classify`` picks an agent — honouring a manual override passed via
``config.configurable.agent_id`` (or state), otherwise calling the LLM
:func:`route` classifier. ``delegate`` invokes the chosen agent's compiled
graph and returns the messages it produced.

This makes "manual dropdown + LLM fallback" work: when the frontend picks a
specific assistant it bypasses the Super Bot entirely; when it picks
``superbot`` (or the API omits ``agent_id``), the LLM routes.

Note: nested human-in-the-loop interrupts (e.g. the email agent's send
approval) are best handled by selecting that agent directly. Routing through
the Super Bot returns the final response.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState

from core.registry import registry
from core.settings import settings
from superbot.router import route


class SuperBotState(MessagesState):
    agent_id: Optional[str]  # manual override, if any
    routed_to: Optional[str]  # the agent actually chosen


def _forced_agent(state: SuperBotState, config: RunnableConfig | None) -> str | None:
    configurable = (config or {}).get("configurable", {})
    forced = configurable.get("agent_id") or state.get("agent_id")
    return forced if forced and registry.get(forced) else None


async def classify(state: SuperBotState, config: RunnableConfig | None = None) -> dict:
    forced = _forced_agent(state, config)
    if forced:
        return {"routed_to": forced}

    last = state["messages"][-1]
    query = last.content if hasattr(last, "content") else str(last)
    return {"routed_to": await route(query)}


async def delegate(state: SuperBotState, config: RunnableConfig | None = None) -> dict:
    agent = registry.get(state["routed_to"]) or registry.get(settings.default_agent_id)
    incoming = state["messages"]
    result = await agent.invoke({"messages": incoming}, config=config)
    # Return only the messages the sub-agent added, so MessagesState's
    # append reducer doesn't duplicate the conversation so far.
    produced = result.get("messages", [])[len(incoming):]
    return {"messages": produced}


_builder = StateGraph(SuperBotState)
_builder.add_node("classify", classify)
_builder.add_node("delegate", delegate)
_builder.add_edge(START, "classify")
_builder.add_edge("classify", "delegate")
_builder.add_edge("delegate", END)

agent = _builder.compile()
