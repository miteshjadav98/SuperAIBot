# 🦜 MegaProject — Super Bot Agent Platform

A plug-and-play AI agent platform: one chat UI, a dropdown, and a **Super Bot**
that auto-routes each message to the right agent. Pick a specific agent to talk
to it directly, or pick **🤖 Super Bot (Auto)** and let the LLM router choose.

| Agent | What it does |
| --- | --- |
| 🤖 **Super Bot (Auto)** | Router/orchestrator — classifies your message and delegates to the best agent below. |
| 🍳 **Personal Chef** | Suggests recipes from your leftover ingredients (web search via Tavily). |
| ✉️ **Email Agent** | Authenticates, reads an inbox, and sends email — with **human-in-the-loop approval** before anything is sent. |
| 💍 **Wedding Planner** | Multi-agent coordinator: flights (remote MCP), venues (web search), and a playlist (SQL over `Chinook.db`). |
| 📄 **PDF Chatbot** | RAG over PDFs — **attach a PDF in the chat** (or `POST /upload`) and ask about it. Redis-backed memory. |
| 🎬 **Movie Recommender** | Suggests films from your taste (web search). The reference example for adding an agent. |

**This project is fully self-contained** — its own `.env`, agent code, database,
Python venv, and frontend. Nothing is read from any other directory.

## Architecture (Super Bot platform)

A thin platform layer turns the agents into a plug-and-play registry. Adding an
agent is one file plus three one-line registrations — no existing agent changes.

```text
Client (Next.js chat UI)
  │
  ├─ streaming chat ─────────────►  LangGraph dev server  :2024
  │                                 (each agent + the `superbot` supervisor graph)
  └─ control / non-stream chat ──►  FastAPI gateway        :8000  (api/app.py)

backend/
├─ core/
│  ├─ settings.py     # one pydantic-settings singleton (reads ../.env)
│  ├─ base_agent.py   # BaseAgent contract + AgentManifest
│  └─ registry.py     # AgentRegistry — the source of truth for which agents exist
├─ llm/
│  └─ factory.py      # get_chat_model(provider) — Azure default; openai/anthropic/gemini/ollama
├─ agents/            # each file exposes `agent` (compiled graph) + MANIFEST
├─ tools/
│  ├─ mcp.py          # get_mcp_tools() — shared MCP loader + retry interceptor
│  └─ mcp_servers.json# MCP servers, keyed by name (add a server here)
├─ superbot/
│  ├─ router.py       # LLM intent classifier over the registry
│  └─ graph.py        # supervisor graph, registered as the `superbot` assistant
├─ api/app.py         # FastAPI: /health, /agents, /chat, + RAG /upload /ask
└─ langgraph.json     # graph ids served by `langgraph dev`
```

**Routing:** picking a specific agent in the dropdown talks to it directly. Picking
**Super Bot (Auto)** (or calling `POST /chat` without `agent_id`) runs the LLM router,
which reads each agent's `MANIFEST.description` and delegates to the best match.

## Adding an agent (the plug-and-play recipe)

Adding an agent is **one new file + three one-line registrations**. No existing
agent is touched. The 🎬 **Movie Recommender** is the worked example — copy its
shape ([`backend/agents/movie_recommender.py`](backend/agents/movie_recommender.py)).

**Step 1 — Create the agent file** `backend/agents/movie_recommender.py`. Build the
model from the factory (never hardcode a provider), compile a graph as `agent`,
and declare a `MANIFEST` — the `description` is what the Super Bot router reads:

```python
from llm.factory import get_chat_model
from langchain.agents import create_agent
from core.base_agent import AgentManifest

agent = create_agent(model=get_chat_model(), tools=[...], system_prompt="...")

MANIFEST = AgentManifest(
    id="movie_recommender",
    label="Movie Recommender",
    emoji="🎬",
    description="Recommends movies based on the user's taste, mood, or films they liked. "
                "Use for anything about films or what to watch.",
    agent_type="langchain",        # or "langgraph"
    builder=lambda: agent,
)
```

**Step 2 — Register it (three one-liners):**

| File | Add |
| --- | --- |
| [`backend/core/registry.py`](backend/core/registry.py) | `"agents.movie_recommender"` to `_AGENT_MODULES` |
| [`backend/langgraph.json`](backend/langgraph.json) | `"movie_recommender": "./agents/movie_recommender.py:agent"` |
| [`frontend/src/components/agent-switcher.tsx`](frontend/src/components/agent-switcher.tsx) | `{ id: "movie_recommender", label: "Movie Recommender", emoji: "🎬" }` |

**Step 3 — Add tools if needed.** Define `@tool` functions in the agent file (or a
shared module) and pass them to `create_agent`. That's it — the registry, the
`/agents` endpoint, the Super Bot router, and the dropdown all pick it up. Restart
`langgraph dev` and the agent is live, both directly and via Super Bot routing.

**How streaming works:** the dropdown ([`agent-switcher.tsx`](frontend/src/components/agent-switcher.tsx))
sets the `assistantId` the chat streams against (and clears the thread). Every
agent — and the `superbot` supervisor — is a graph on the LangGraph server, so
streaming and the email agent's human-in-the-loop approval work natively in the UI.

## Adding an MCP server

MCP tools are loaded by a shared helper, so adding a server is a **config edit, not code**.

**Step 1 —** add the server to [`backend/tools/mcp_servers.json`](backend/tools/mcp_servers.json),
keyed by a short name. Values pass straight to `MultiServerMCPClient`, so any transport works:

```jsonc
{
  "travel": { "transport": "streamable_http", "url": "https://mcp.kiwi.com" },
  "github": { "transport": "stdio", "command": "npx",
              "args": ["-y", "@modelcontextprotocol/server-github"] }
}
```

**Step 2 —** pull its tools into any agent (one line — a flaky/down server degrades to
`[]` instead of breaking the agent; a retry interceptor is applied automatically):

```python
from tools.mcp import get_mcp_tools

agent = create_agent(model=get_chat_model(), tools=get_mcp_tools(["github"]), ...)
# get_mcp_tools() with no args loads every configured server.
```

The 💍 Wedding Planner uses this for its `travel` flight tools. Combine with the
"Adding an agent" recipe to register a whole MCP-backed agent on the platform.

## PDF uploads

Attach a PDF in the chat (paperclip or drag-drop) and ask about it — a `before_agent`
middleware in [`pdf_chatbot.py`](backend/agents/pdf_chatbot.py) chunks + embeds it into
the knowledge base **in the same process as the chat**, then strips the raw file from the
message before the model runs. The REST route `POST :8000/upload` is also available for
batch/API ingestion. (Both feed the same retrieval tool; the in-message path is what makes
the UI upload button "just work" for RAG.)

## Prerequisites

- Python 3.11–3.13
- Node.js 18+ and npm

## Setup (once)

```powershell
cd C:\Users\user\Music\MegaProject

# Python backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt

# Frontend
cd frontend
npm install
```

## Run

**Option A — one command (opens two windows):**

```powershell
cd C:\Users\user\Music\MegaProject
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

`start.ps1` launches three windows: the LangGraph dev server (all agents +
`superbot`, :2024), the FastAPI gateway (:8000), and the frontend (:3000).

**Option B — terminals:**

```powershell
# Terminal 1 — LangGraph server (all agents + superbot, streaming UI)
cd C:\Users\user\Music\MegaProject\backend
..\.venv\Scripts\langgraph.exe dev

# Terminal 2 — FastAPI gateway (/health, /agents, /chat, RAG /upload /ask)
cd C:\Users\user\Music\MegaProject\backend
..\.venv\Scripts\python.exe main.py

# Terminal 3 — frontend
cd C:\Users\user\Music\MegaProject\frontend
npm run dev
```

Then open **http://localhost:3000**. The dropdown defaults to **Super Bot (Auto)** —
just type and it routes. Or pick a specific agent. Try:

- Super Bot → *"What should I watch tonight? I loved Inception."* (routes to Movie Recommender)
- Personal Chef → *"I have eggs, spinach, and feta. What can I make?"*
- Email Agent → *"Authenticate me: julie@example.com / password123, then check my inbox and reply."* (you'll be asked to approve the send)
- Wedding Planner → *"Plan a wedding: from NYC to Lisbon, 80 guests, jazz playlist."*

Prefer the API? `POST http://localhost:8000/chat` with `{"query": "..."}` (LLM-routed)
or add `"agent_id": "personal_chef"` to target an agent directly; `GET /agents` lists them.

Stop a server with **Ctrl+C** in its window. LangSmith tracing is on (via
`.env`), so every run shows up in your LangSmith project.

## Notes

- **Wedding Planner** loads travel tools from a remote MCP server at startup, so
  its first request is slower and the flights tool can occasionally be flaky; the
  venue and playlist tools work independently.
- Switching agents in the dropdown intentionally **starts a fresh conversation** —
  each agent has a different state shape.
- **Super Bot routing** picks one agent per message from `MANIFEST.description`. Routing
  the Email Agent through Super Bot returns its final reply; to exercise the send-approval
  human-in-the-loop, select the Email Agent directly.
- **LLM provider** is set once by `default_llm_provider` in [`core/settings.py`](backend/core/settings.py)
  (Azure by default). Switch to `openai`/`anthropic`/`gemini`/`ollama` there or per call via
  `get_chat_model(provider=...)`; install the matching optional package from `requirements.txt`.
- **Redis** (PDF Chatbot memory) is read from `REDIS_*` in [`.env`](.env); if unreachable the
  agent runs without persistent memory rather than blocking startup.
- To change keys, ports, or the default agent, edit [`.env`](.env) and
  [`frontend/.env.local`](frontend/.env.local).
