# 🤖 SuperBot — Plug-and-Play AI Agent Platform

A plug-and-play AI agent platform: one chat UI (branded **SuperBot**), a
dropdown, and a **Super Bot** planner that breaks each message into a task DAG
and runs the right agents — in parallel where it can. Pick a specific agent to
talk to it directly, or pick **🤖 Super Bot (Auto)** and let the planner decide. Sign in with email + password — every
user gets their own private chat history.

> **[ARCHITECTURE.md](ARCHITECTURE.md)** — the design walkthrough: why a planner
> instead of an agent-of-agents, how the parallel DAG is bounded, the memory
> model, the decision log, and the known gaps.

Each agent doubles as a worked example of an agent-engineering concept:

| Concept | Where to look |
| --- | --- |
| **Planner–executor orchestration** | `superbot` graph — request → validated task DAG → agents → merged answer |
| **Parallel fan-out / fan-in (map-reduce)** | `superbot/executor.py` — `Send` per ready task, results merged through a reducer |
| **Live execution graph** | `frontend/.../execution-graph.tsx` - renders the task DAG with per-task status |
| **Capability registry** | `core/registry.py` — agents self-declare `capabilities`; planner and lookups read them |
| **Long-term memory (cross-agent)** | `core/store.py` + `core/memory.py` — a Mongo `BaseStore`, injected into every agent by middleware |
| **Multi-agent coordinator (agents-as-tools)** | Wedding Planner — flights (MCP) + venues (search) + SQL playlist |
| **RAG (retrieval-augmented generation)** | PDF Chatbot — chunk → embed → Atlas Vector Search → grounded answers |
| **Single agent, multiple tools** | Personal Chef / Movie Recommender — `create_agent` + web search tools |
| **Human-in-the-loop interrupts** | Email Agent — approval before anything is sent or destroyed |
| **Declarative approval policy** | `core/approval.py` — tools mark *themselves* destructive; the gate is derived |
| **Provider abstraction** | `tools/email/` — one protocol, a working mock, a documented Gmail seam |
| **MCP tool loading** | `tools/mcp.py` + `mcp_servers.json` — remote/stdio tool servers |

| Agent | What it does |
| --- | --- |
| 🤖 **Super Bot (Auto)** | Planner/orchestrator — breaks your message into a task DAG, runs independent tasks across the agents below **in parallel**, and merges the results into one answer. |
| 🍳 **Personal Chef** | Suggests recipes from your leftover ingredients (web search via Tavily). |
| ✉️ **Email Agent** | Searches and reads mail, looks up contacts, drafts, replies, archives and trashes — with **human-in-the-loop approval** before anything is sent or destroyed. Runs on a built-in mock mailbox out of the box (no credentials); swap providers with `EMAIL_PROVIDER`. |
| 💍 **Wedding Planner** | Multi-agent coordinator: flights (remote MCP), venues (web search), and a playlist (SQL over `Chinook.db`). |
| 📄 **PDF Chatbot** | RAG over PDFs — **attach a PDF in the chat** (or `POST /upload`) and ask about it. MongoDB Atlas Vector Search-backed, with **per-user document isolation** (you only ever retrieve your own uploads). |
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
│  ├─ registry.py     # AgentRegistry — the source of truth for which agents exist
│  ├─ store.py        # MongoDB-backed LangGraph BaseStore (long-term memory)
│  ├─ memory.py       # shared cross-agent memory + MemoryMiddleware
│  ├─ approval.py     # @requires_approval — declarative HITL for risky tools
│  ├─ concurrency.py  # one run at a time per thread_id
│  ├─ telemetry.py    # per-run tokens, latency, cost
│  ├─ lg_auth.py      # LangGraph JWT auth — isolates chats & PDFs per user
│  └─ prompts.py      # loads agent prompts from Mongo (versioned; see Prompt management)
├─ llm/
│  └─ factory.py      # get_chat_model(provider) — Azure default; openai/anthropic/gemini/ollama
├─ agents/            # each file exposes `agent` (compiled graph) + MANIFEST
├─ tools/
│  ├─ mcp.py          # get_mcp_tools() — shared MCP loader + retry interceptor
│  ├─ mcp_servers.json# MCP servers, keyed by name (add a server here)
│  └─ email/          # EmailProvider protocol + mock mailbox + Gmail seam
├─ tests/             # `cd backend && pytest` — hermetic, no DB or network
├─ superbot/
│  ├─ state.py        # state schema + reducers (the contract between nodes)
│  ├─ planner.py      # request → validated task DAG
│  ├─ executor.py     # dispatch / parallel execute / synthesize nodes
│  ├─ router.py       # single-agent classifier — the planner's fallback
│  └─ graph.py        # planner/executor graph, served as the `superbot` assistant
├─ api/app.py         # FastAPI: /health, /agents, /chat, + RAG /upload /ask
└─ langgraph.json     # graph ids served by `langgraph dev`
```

**Routing:** picking a specific agent in the dropdown talks to it directly. Picking
**Super Bot (Auto)** (or calling `POST /chat` without `agent_id`) runs the planner,
which reads each agent's `MANIFEST.description` and `capabilities` and produces a
**task DAG**:

```
plan → dispatch ──Send × N──→ execute ─┐   (independent tasks run in parallel)
         ↑                             │
         └─────────────────────────────┘
         └──→ synthesize → END
```

So *"recommend a sci-fi film and a dinner I can make with chicken and rice"* runs the
Movie Recommender and the Personal Chef concurrently, then merges both into one reply.
A request needing only one agent produces a one-task plan and returns that agent's
answer verbatim — no extra synthesis call. Guardrails: at most `MAX_TASKS` tasks and
`MAX_LAYERS` dispatch rounds per run, two attempts per task, and a failing agent is
recorded as a failed task rather than aborting the others. If planning itself fails,
the run degrades to the single-agent classifier in `superbot/router.py`.

**Memory:** every agent shares one long-term memory, scoped to the signed-in
user. When the model learns something durable it calls `remember`; on every later
turn — in any thread, through **any** agent — those facts are injected into the
system prompt. Tell the 🍳 Personal Chef you're vegetarian, and the 🎬 Movie
Recommender knows tomorrow.

```python
# That's the whole integration, per agent:
agent = create_agent(model, tools=[...], middleware=[MemoryMiddleware()])
```

The namespace comes only from authenticated identity, never from model output —
so memory cannot leak across accounts, and with no signed-in user it is disabled
rather than pooled. `GET /memories` and `DELETE /memories/{key}` let a user see
and remove anything stored about them. Details and rationale in
[ARCHITECTURE.md §4](ARCHITECTURE.md#4-memory).

## Tests

```bash
cd backend && pytest        # 83 tests, ~13s, no database and no network
```

`mongomock` backs the store tests, so the MongoDB `BaseStore` runs against a real
pymongo API surface without Atlas or Docker.

## Adding an agent (the plug-and-play recipe)

Adding an agent is **one new file + three one-line registrations**. No existing
agent is touched. The 🎬 **Movie Recommender** is the worked example — copy its
shape ([`backend/agents/movie_recommender.py`](backend/agents/movie_recommender.py)).

**Step 1 — Create the agent file** `backend/agents/movie_recommender.py`. Build the
model from the factory (never hardcode a provider), compile a graph as `agent`,
and declare a `MANIFEST` — the `description` and `capabilities` are what the Super Bot
planner reads when it decides which task belongs to which agent:

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
    capabilities=["movies", "search"],   # coarse domain keys; indexed by the registry
)
```

**Step 2 — Register it (three one-liners):**

| File | Add |
| --- | --- |
| [`backend/core/registry.py`](backend/core/registry.py) | `"agents.movie_recommender"` to `_AGENT_MODULES` |
| [`backend/langgraph.json`](backend/langgraph.json) | `"movie_recommender": "./agents/movie_recommender.py:agent"` |

The frontend dropdown discovers agents from `GET :8000/agents` at runtime, so
**no frontend change is needed** (the fallback list in
[`agent-switcher.tsx`](frontend/src/components/agent-switcher.tsx) is only used
while the gateway is unreachable).

**Step 3 — Add tools if needed.** Define `@tool` functions in the agent file (or a
shared module) and pass them to `create_agent`. That's it — the registry, the
`/agents` endpoint, the Super Bot planner, and the dropdown all pick it up. Restart
`langgraph dev` and the agent is live, both directly and as a task the planner can
delegate to.

**How streaming works:** the dropdown ([`agent-switcher.tsx`](frontend/src/components/agent-switcher.tsx))
sets the `assistantId` the chat streams against (and clears the thread). Every
agent — and the `superbot` planner — is a graph on the LangGraph server, so
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

Attach one or more PDFs in the chat (paperclip or drag-drop) and ask about them.
Each upload goes to `POST :8000/upload`, where [`core/pdf_ingest.py`](backend/core/pdf_ingest.py)
converts the text with **MarkItDown** and describes embedded images with a vision
model, then the chunks are embedded into Atlas Vector Search. Uploads run in a
worker thread, so **attaching several PDFs at once processes them in parallel**
instead of serializing (and timing out).

**Per-user isolation:** every chunk is stamped with the uploader's id (`owner`),
and the `ask_pdf_knowledge_base` tool filters retrieval to the calling user — so
one user can never retrieve another user's documents. Isolation is enforced by a
mandatory post-retrieval owner filter (a leak can't happen even if the Atlas
pre-filter is momentarily unavailable while its index rebuilds).

## Authentication & MongoDB Atlas

Users register/sign in with **email + password** (phone is an optional profile
field). The FastAPI gateway issues a JWT (`/auth/register`, `/auth/login`,
`/auth/me`); the LangGraph server validates the same JWT on every request via
custom auth ([`backend/core/lg_auth.py`](backend/core/lg_auth.py)) and stamps +
filters threads by `metadata.owner`. The PDF Chatbot applies the same ownership
model to uploaded documents, so **users never see each other's chats or PDFs**.

MongoDB Atlas backs several things (all in one cluster/db, `MONGODB_DB`):

| Collection | Used for |
| --- | --- |
| `users` | accounts (email, bcrypt password hash, optional phone) |
| `pdf_chunks` | Atlas **Vector Search** for PDF RAG — chunks stamped with `owner`, retrieval filtered so each user only queries their own PDFs |
| `prompts` / `prompt_registry` / `prompt_audit` | versioned agent prompts (see [Prompt management](#prompt-management)) |
| `checkpoints*` | gateway (`/chat`, `/ask`) conversation memory per user |

**Atlas setup (one time):**

1. In [cloud.mongodb.com](https://cloud.mongodb.com) create a cluster — the free
   **M0** tier is enough (Vector Search included); your credits cover an M10
   upgrade later if you need it.
2. *Database Access* → create a DB user; *Network Access* → allow your IP
   (or `0.0.0.0/0` while developing).
3. *Connect → Drivers* → copy the `mongodb+srv://...` string into `MONGODB_URI`
   in [`.env`](.env).

Without `MONGODB_URI` the platform still boots: PDF RAG falls back to an
in-memory store and `/auth/*` returns 503 — but login (and therefore the chat
UI) needs it, so set it first.

## Prompt management

Agent prompts (the system prompt for every agent and sub-agent) are **not
hardcoded** — they're stored and versioned in MongoDB and loaded at build time
by [`backend/core/prompts.py`](backend/core/prompts.py). The first time an agent
runs, its prompt is auto-seeded as version 1 from the in-code default, so it
shows up ready to edit; if Mongo is unreachable the agent falls back to that
default and keeps working.

A standalone FastAPI service, [`prompt_service/`](prompt_service/), is the
management API over those prompts — create, list versions, publish a new
(immutable) version, and roll back — with Swagger UI at `/docs`. It reads and
writes the same `prompts` / `prompt_registry` collections the agents use, so
**the backend and `prompt_service` must point at the same `MONGODB_DB`**. See
[`prompt_service/README.md`](prompt_service/README.md) for the endpoints and
examples.

Editing a prompt takes effect after the agent process restarts (prompts are read
when each graph is built).

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

Then open **http://localhost:3000**, create an account (email + password), and
you land in the chat. The dropdown defaults to **Super Bot (Auto)** —
just type and it routes. Or pick a specific agent. Try:

- Super Bot → *"What should I watch tonight? I loved Inception."* (routes to Movie Recommender)
- Personal Chef → *"I have eggs, spinach, and feta. What can I make?"*
- Email Agent → *"Authenticate me: julie@example.com / password123, then check my inbox and reply."* (you'll be asked to approve the send)
- Wedding Planner → *"Plan a wedding: from NYC to Lisbon, 80 guests, jazz playlist."*

Prefer the API? `POST http://localhost:8000/chat` with `{"query": "..."}` (LLM-routed)
or add `"agent_id": "personal_chef"` to target an agent directly; `GET /agents` lists them.
`/chat`, `/ask`, and `/upload` need an `Authorization: Bearer <token>` header from
`POST /auth/login`; `/chat` remembers the conversation per `thread_id` (stored in Mongo).

Stop a server with **Ctrl+C** in its window. LangSmith tracing is on (via
`.env`), so every run shows up in your LangSmith project.

## Notes

- **Wedding Planner** loads travel tools from a remote MCP server at startup, so
  its first request is slower and the flights tool can occasionally be flaky; the
  venue and playlist tools work independently.
- Switching agents in the dropdown intentionally **starts a fresh conversation** —
  each agent has a different state shape.
- The chat UI is **responsive**: on mobile the chat-history sidebar collapses to a
  slide-in drawer, and the header and message composer adapt so nothing overflows.
- **Super Bot routing** picks one agent per message from `MANIFEST.description`. Routing
  the Email Agent through Super Bot returns its final reply; to exercise the send-approval
  human-in-the-loop, select the Email Agent directly.
- **LLM provider** is set once by `default_llm_provider` in [`core/settings.py`](backend/core/settings.py)
  (Azure by default). Switch to `openai`/`anthropic`/`gemini`/`ollama` there or per call via
  `get_chat_model(provider=...)`; install the matching optional package from `requirements.txt`.
- **MongoDB Atlas** is read from `MONGODB_URI` in [`.env`](.env); if unset the platform
  boots with in-memory fallbacks (no login, no persistent RAG) rather than blocking startup.
- To change keys, ports, or the default agent, edit [`.env`](.env) and
  [`frontend/.env.local`](frontend/.env.local).
