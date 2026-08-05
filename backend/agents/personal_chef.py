from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

from llm.factory import get_chat_model

tavily_client = TavilyClient()

azure_model = get_chat_model()

@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(query)

from core.prompts import get_prompt

_DEFAULT_SYSTEM_PROMPT = """

You are a personal chef. The user will give you a list of ingredients they have left over in their house.

Using the web search tool, search the web for recipes that can be made with the ingredients they have.

Return recipe suggestions and eventually the recipe instructions to the user, if requested.

"""

system_prompt = get_prompt(
    "personal_chef_system",
    _DEFAULT_SYSTEM_PROMPT,
    name="Personal Chef — System Prompt",
)

from langchain.agents import create_agent

agent = create_agent(
    model=azure_model,
    tools=[web_search],
    system_prompt=system_prompt,
)

from core.base_agent import AgentManifest

MANIFEST = AgentManifest(
    id="personal_chef",
    label="Personal Chef",
    emoji="🍳",
    description=(
        "Suggests recipes and cooking instructions from a list of leftover or "
        "available ingredients. Use for anything about food, recipes, meals, "
        "cooking, or what to make with ingredients on hand."
    ),
    agent_type="langchain",
    builder=lambda: agent,
    capabilities=["cooking", "search"],
)