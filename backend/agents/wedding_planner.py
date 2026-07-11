import os
from typing import Dict, Any
from dotenv import load_dotenv

from langchain.agents import AgentState, create_agent
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage, ToolMessage
from langgraph.types import Command
from langchain_community.utilities import SQLDatabase

from tavily import TavilyClient

from tools.mcp import get_mcp_tools

load_dotenv()

from llm.factory import get_chat_model
from core.prompts import get_prompt

azure_model = get_chat_model()

# --- TOOLS ---

# 1. MCP Travel Tools (loaded from tools/mcp_servers.json via the shared helper)
mcp_tools = get_mcp_tools(["travel"])

# 2. Tavily Web Search
tavily_client = TavilyClient()

@tool
def web_search(query: str, search_number: int, max_search_number: int) -> Dict[str, Any]:
    """Search the web for information. You must track your search count by providing
    search_number (starting at 1) and max_search_number on every call.
    Queries must use only plain text characters. Do not use accented or special characters     
      (e.g., use 'capacite' instead of 'capacité').
    """
    if search_number > max_search_number:
        return {"message": "Search limit reached. Please summarize your findings and provide your final answer."}
    try:
        return tavily_client.search(query)
    except Exception as e:
        return {"error": str(e)}

# 3. SQL Database
# NOTE: Adjusted the path to point to resources folder relatively
db_path = os.path.join(os.path.dirname(__file__), "resources", "Chinook.db")
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

@tool
def query_playlist_db(query: str) -> str:
    """Query the database for playlist information"""
    try:
        return db.run(query)
    except Exception as e:
        return f"Error querying database: {e}"


# --- STATE ---
class WeddingState(AgentState):
    origin: str
    destination: str
    guest_count: str
    genre: str


# --- SUB-AGENTS ---
travel_agent = create_agent(
    model=azure_model,
    tools=mcp_tools,
    system_prompt=get_prompt(
        "wedding_travel_system",
        """
    You are a travel agent. Search for flights to the desired destination wedding location.
    You are not allowed to ask any more follow up questions, you must find the best flight options based on the following criteria:
    - Price (lowest, economy class)
    - Duration (shortest)
    - Date (time of year which you believe is best for a wedding at this location)
    To make things easy, only look for one ticket, one way.
    You may need to make multiple searches to iteratively find the best options.
    You will be given no extra information, only the origin and destination. It is your job to think critically about the best options.
    If the MCP tool fails, returns malformed output, or does not give you usable flight results, try the tool again.
    Once you have found the best options, let the user know your shortlist of options.
    """,
        name="Wedding Planner — Travel Sub-agent",
    ),
)

venue_agent = create_agent(
    model=azure_model,
    tools=[web_search],
    system_prompt=get_prompt(
        "wedding_venue_system",
        """
    You are a venue specialist. Search for venues in the desired location, and with the desired capacity.
    You are not allowed to ask any more follow up questions, you must find the best venue options based on the following criteria:
    - Price (lowest)
    - Capacity (exact match)
    - Reviews (highest)
    You may need to make multiple searches to iteratively find the best options.
    You have a suggested limit of 12 web searches. Count every web_search call you make.
    After 12 searches, you should stop searching and summarize the best options you have
    found so far.
    """,
        name="Wedding Planner — Venue Sub-agent",
    ),
)

playlist_agent = create_agent(
    model=azure_model,
    tools=[query_playlist_db],
    system_prompt=get_prompt(
        "wedding_playlist_system",
        """
    You are a playlist specialist. Query the sql database and curate the perfect playlist for a wedding given a genre.
    Once you have your playlist, calculate the total duration and cost of the playlist, each song has an associated price.
    If you run into errors when querying the database, try to fix them by making changes to the query.
    Do not come back empty handed, keep trying to query the db until you find a list of songs.

    This is a SQLite database. Before writing any data queries, first discover the schema.
    """,
        name="Wedding Planner — Playlist Sub-agent",
    ),
)


# --- COORDINATOR TOOLS ---
@tool
async def search_flights(runtime: ToolRuntime) -> str:
    """Travel agent searches for flights to the desired destination wedding location."""
    origin = runtime.state.get("origin", "")
    destination = runtime.state.get("destination", "")
    response = await travel_agent.ainvoke({"messages": [HumanMessage(content=f"Find flights from {origin} to {destination}")]})
    return response['messages'][-1].content

@tool
def search_venues(runtime: ToolRuntime) -> str:
    """Venue agent chooses the best venue for the given location and capacity."""
    destination = runtime.state.get("destination", "")
    capacity = runtime.state.get("guest_count", "")
    query = f"Find wedding venues in {destination} for {capacity} guests"
    response = venue_agent.invoke({"messages": [HumanMessage(content=query)]})
    return response['messages'][-1].content

@tool
def suggest_playlist(runtime: ToolRuntime) -> str:
    """Playlist agent curates the perfect playlist for the given genre."""
    genre = runtime.state.get("genre", "")
    query = f"Find {genre} tracks for wedding playlist"
    response = playlist_agent.invoke({"messages": [HumanMessage(content=query)]})
    return response['messages'][-1].content

@tool
def update_state(origin: str, destination: str, guest_count: str, genre: str, runtime: ToolRuntime) -> str:
    """Update the state when you know all of the values: origin, destination, guest_count, genre. 
    This tool must be called alone, without any other tool calls. It must complete and return to make,
    the information available to other tools."""
    return Command(update={
        "origin": origin, 
        "destination": destination, 
        "guest_count": guest_count, 
        "genre": genre, 
        "messages": [ToolMessage("Successfully updated state", tool_call_id=runtime.tool_call_id)]}
        )

# --- MAIN AGENT ---
agent = create_agent(
    model=azure_model,
    tools=[search_flights, search_venues, suggest_playlist, update_state],
    state_schema=WeddingState,
    system_prompt=get_prompt(
        "wedding_coordinator_system",
        """
    You are a wedding coordinator.
    First find all the information you need to update the state. When you have the information, update the state.
    Once that has completed and returned, you can delegate the tasks
    to your specialists for flights, venues, and playlists.
    Once you have received their answers, coordinate the perfect wedding for me.
    """,
        name="Wedding Planner — Coordinator",
    ),
)

from core.base_agent import AgentManifest

MANIFEST = AgentManifest(
    id="wedding_planner",
    label="Wedding Planner",
    emoji="💍",
    description=(
        "Coordinates a destination wedding end to end: finds flights (travel "
        "MCP), wedding venues by location and guest capacity (web search), and "
        "curates a music playlist by genre (SQL). Use for wedding planning, "
        "budgets, venues, guest logistics, or destination travel for a wedding."
    ),
    agent_type="langgraph",
    builder=lambda: agent,
    tags=["wedding", "multi-agent", "mcp", "sql", "search"],
)
