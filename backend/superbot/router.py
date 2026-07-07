"""LLM intent classifier for the Super Bot.

Given a user query, pick the best agent id from the registry's descriptions.
Falls back to ``settings.default_agent_id`` when the model is unsure or returns
something not in the registry.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from core.registry import registry
from core.settings import settings
from llm.factory import get_chat_model

_SYSTEM = """You are the router for a multi-agent platform. Given the user's \
message, choose the single best agent to handle it from the list below. \
Respond with that agent's exact id and your confidence (0-1).

Available agents:
{agents}

If no agent is a good fit, choose "{default}"."""


class RouteDecision(BaseModel):
    agent_id: str = Field(description="The exact id of the chosen agent.")
    confidence: float = Field(description="Confidence 0-1 that this is the right agent.")


async def route(query: str, *, confidence_floor: float = 0.35) -> str:
    """Return the chosen agent id (validated against the registry)."""
    valid_ids = set(registry.ids())
    if not valid_ids:
        return settings.default_agent_id

    model = get_chat_model(temperature=0).with_structured_output(RouteDecision)
    prompt = _SYSTEM.format(
        agents=registry.descriptions(), default=settings.default_agent_id
    )
    try:
        decision: RouteDecision = await model.ainvoke(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ]
        )
    except Exception as exc:  # noqa: BLE001 — routing must never hard-fail
        print(f"[router] classification failed, using default: {exc}")
        return settings.default_agent_id

    if decision.agent_id in valid_ids and decision.confidence >= confidence_floor:
        return decision.agent_id
    return settings.default_agent_id
