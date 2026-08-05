"""Agent registry — the single source of truth for which agents exist.

To add an agent to the platform: create the agent module exposing ``MANIFEST``,
then add its import to ``_AGENT_MODULES`` below (one line). The registry, the
Super Bot router, and the FastAPI ``/agents`` endpoint all read from here.

Importing an agent module compiles its graph (and may hit the network, e.g. the
wedding planner's MCP tools), so discovery is lazy and resilient: a failing
agent is skipped with a warning rather than taking the whole platform down.
"""

from __future__ import annotations

import importlib

from core.base_agent import AgentManifest, BaseAgent

# Module paths to import. Each must expose a module-level ``MANIFEST``.
_AGENT_MODULES = [
    "agents.personal_assistant",
    "agents.personal_chef",
    "agents.email_agent",
    "agents.wedding_planner",
    "agents.pdf_chatbot",
    "agents.movie_recommender",
]


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    def discover(self) -> "AgentRegistry":
        """Import each agent module and register its manifest (idempotent)."""
        for module_path in _AGENT_MODULES:
            try:
                module = importlib.import_module(module_path)
                manifest: AgentManifest = module.MANIFEST
                self._agents[manifest.id] = BaseAgent(manifest)
            except Exception as exc:  # noqa: BLE001 — never let one agent break boot
                print(f"[registry] skipped {module_path}: {exc}")
        return self

    def all(self) -> list[BaseAgent]:
        return list(self._agents.values())

    def ids(self) -> list[str]:
        return list(self._agents.keys())

    def get(self, agent_id: str) -> BaseAgent | None:
        return self._agents.get(agent_id)

    def descriptions(self) -> str:
        """Formatted agent list for the router prompt."""
        return "\n".join(
            f"- {a.manifest.id}: {a.manifest.description}" for a in self.all()
        )

    def catalog(self) -> str:
        """Formatted agent list for the planner prompt — same as
        :meth:`descriptions` plus each agent's capabilities, which give the
        planner a compact vocabulary for matching a task to an agent."""
        lines = []
        for a in self.all():
            caps = ", ".join(a.manifest.capabilities) or "general"
            lines.append(f"- {a.manifest.id} [{caps}]: {a.manifest.description}")
        return "\n".join(lines)

    def capabilities(self) -> list[str]:
        """Every capability provided by at least one registered agent."""
        return sorted({c for a in self.all() for c in a.manifest.capabilities})

    def by_capability(self, capability: str) -> list[BaseAgent]:
        """Agents providing ``capability``. The lookup that makes adding a
        second provider for a domain (e.g. another email agent) a registration
        change rather than a routing-code change."""
        return [a for a in self.all() if capability in a.manifest.capabilities]

    def manifests(self) -> list[dict]:
        """Serializable manifest summaries for the API / frontend."""
        return [
            {
                "id": a.manifest.id,
                "label": a.manifest.label,
                "emoji": a.manifest.emoji,
                "description": a.manifest.description,
                "agent_type": a.manifest.agent_type,
                "capabilities": a.manifest.capabilities,
            }
            for a in self.all()
        ]


# Module-level singleton, discovered once on import.
registry = AgentRegistry().discover()
