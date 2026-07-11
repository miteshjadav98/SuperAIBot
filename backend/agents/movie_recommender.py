"""Movie Recommender agent.

A simple LangChain tool agent: given a user's taste (genres, a film they liked,
a mood), it searches the web for current, well-reviewed titles and returns a
short ranked shortlist with one-line reasons.

This is the reference example for adding an agent to the platform — see the
"Adding an agent" section of the README. It follows the same shape as every
other agent: build the model via the LLM factory, expose a compiled ``agent``
graph, and declare a ``MANIFEST``.
"""

from dotenv import load_dotenv

load_dotenv()

from typing import Any, Dict

from langchain.agents import create_agent
from langchain.tools import tool
from tavily import TavilyClient

from llm.factory import get_chat_model

model = get_chat_model()
tavily_client = TavilyClient()


@tool
def search_movies(query: str) -> Dict[str, Any]:
    """Search the web for movies, reviews, ratings, cast, or where to watch."""
    try:
        return tavily_client.search(query)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


from core.prompts import get_prompt

_DEFAULT_SYSTEM_PROMPT = """
You are a film buff who recommends movies.

Given the user's taste — favourite genres, a film or director they liked, or a
mood — use the search_movies tool to find current, well-reviewed titles. Then
return a shortlist of 3-5 movies. For each: the title, year, and a single line
on why it fits their taste. Prefer variety over near-duplicates, and mention
where to watch if the search surfaces it.
"""

system_prompt = get_prompt(
    "movie_recommender_system",
    _DEFAULT_SYSTEM_PROMPT,
    name="Movie Recommender — System Prompt",
)

agent = create_agent(
    model=model,
    tools=[search_movies],
    system_prompt=system_prompt,
)

from core.base_agent import AgentManifest

MANIFEST = AgentManifest(
    id="movie_recommender",
    label="Movie Recommender",
    emoji="🎬",
    description=(
        "Recommends movies based on the user's taste — favourite genres, a film "
        "or director they liked, or a mood. Use for anything about films, what "
        "to watch tonight, movie suggestions, or finding similar movies."
    ),
    agent_type="langchain",
    builder=lambda: agent,
    tags=["movies", "recommendation", "search"],
)
