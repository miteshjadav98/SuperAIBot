"""The agent contract every agent on the platform satisfies.

Each agent module exposes two things:

* ``agent`` — the compiled LangGraph graph (what ``langgraph dev`` serves and
  what the frontend streams against; unchanged from before).
* ``MANIFEST`` — an :class:`AgentManifest` describing the agent so the registry,
  the Super Bot router, and the API can reason about it without importing it.

:class:`BaseAgent` is a thin wrapper around a compiled graph: it exposes the
graph plus a normalising ``invoke`` (accepting a raw string, a messages list, or
a full state dict). It does not re-implement anything LangGraph already provides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from langchain_core.messages import HumanMessage

AgentType = Literal["langchain", "langgraph"]


@dataclass
class AgentManifest:
    """Metadata + builder for one agent. The registry's unit of currency."""

    id: str
    label: str
    emoji: str
    description: str  # what the LLM router reads to pick this agent
    agent_type: AgentType
    builder: Callable[[], Any]  # returns the compiled graph
    tags: list[str] = field(default_factory=list)


class BaseAgent:
    """Wraps a compiled graph and exposes the platform agent contract."""

    def __init__(self, manifest: AgentManifest):
        self.manifest = manifest
        self._graph: Any | None = None

    @property
    def id(self) -> str:
        return self.manifest.id

    @property
    def graph(self) -> Any:
        """The compiled graph, built lazily and cached."""
        if self._graph is None:
            self._graph = self.manifest.builder()
        return self._graph

    def _to_state(self, query: str | dict | list) -> dict:
        """Accept a raw string, a messages list, or a full state dict."""
        if isinstance(query, str):
            return {"messages": [HumanMessage(content=query)]}
        if isinstance(query, list):
            return {"messages": query}
        return query

    async def invoke(self, query: str | dict | list, config: dict | None = None) -> dict:
        return await self.graph.ainvoke(self._to_state(query), config=config or {})
