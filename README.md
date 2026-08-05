# 🤖 SuperBot — a multi-agent AI platform

One chat UI over a registry of specialist AI agents. Ask for something and a
**planner** breaks it into a task DAG, runs the right agents — in parallel where
it can — and merges the results into a single answer. Or pick one agent from the
dropdown and talk to it directly.

> *"Recommend a sci-fi film for tonight, and a dinner I can make with chicken and rice"*
> → the Movie Recommender and the Personal Chef run **concurrently**, then one merged reply.

Built on **LangGraph** + **LangChain 1.x**, Next.js, FastAPI and MongoDB Atlas.

| | |
| --- | --- |
| 📐 **[ARCHITECTURE.md](ARCHITECTURE.md)** | Why a planner instead of an agent-of-agents, how the DAG is bounded, the memory model, a decision log, and the known gaps |
| 🚀 **[Quick start](#quick-start)** | Running locally in about five minutes |
| 🧩 **[Add your own agent](#adding-an-agent)** | One file + two registration lines |

---

## What's interesting here

If you're skimming the code, these are the parts worth opening:

| Decision | Where | Why it's not the obvious choice |
| --- | --- | --- |
| **Planner over agent-of-agents** | [`superbot/planner.py`](backend/superbot/planner.py) | An agent loop is unbounded by construction. The planner emits a *fixed, inspectable* DAG up front — one LLM call, then deterministic scheduling. |
| **Cycles are unrepresentable** | [`planner._validate`](backend/superbot/planner.py) | Dependencies may only point *backwards*. That one rule kills forward refs and cycles, so the scheduler can't deadlock on a bad plan — no cycle detection needed. |
| **Parallel fan-out via `Send`** | [`superbot/executor.py`](backend/superbot/executor.py) | N workers write `results` in the same superstep, which forces a reducer. Get that wrong and LangGraph raises `InvalidUpdateError`. |
| **A hand-written Mongo `BaseStore`** | [`core/store.py`](backend/core/store.py) | LangGraph ships no Mongo store. Adding Postgres for memory alone would mean a second datastore to operate. |
| **Deliberately no embeddings** | [`core/memory.py`](backend/core/memory.py) | Under a few dozen memories per user, loading them all beats retrieval — no recall failure mode, no index to keep warm. |
| **Tools declare their own risk** | [`core/approval.py`](backend/core/approval.py) | A hand-maintained approval map fails *open* when you forget it. `@requires_approval` makes a new destructive tool gated by default. |
| **Run-concurrency policy** | [`core/concurrency.py`](backend/core/concurrency.py) | Two runs on one `thread_id` share a checkpoint and interleave into a conversation that never happened. |

---

## The agents

| Agent | What it does |
| --- | --- |
| 🤖 **Super Bot (Auto)** | The orchestrator — plans a task DAG, runs agents in parallel, merges the results |
| 🍳 **Personal Chef** | Recipes from the ingredients you have (web search via Tavily) |
| ✉️ **Email Agent** | Searches, reads, drafts, replies, archives and trashes — with **approval required** before anything is sent or destroyed. Runs on a built-in mock mailbox, so no credentials needed |
| 💍 **Wedding Planner** | Agents-as-tools coordinator: flights (remote MCP), venues (web search), playlist (SQL over `Chinook.db`) |
| 📄 **PDF Chatbot** | RAG over your uploads — attach a PDF and ask about it. Atlas Vector Search, with per-user document isolation |
| 🎬 **Movie Recommender** | Film suggestions from your taste. The reference example for adding an agent |

Each one doubles as a worked example of an agent-engineering pattern:

| Pattern | Where to look |
| --- | --- |
| Planner–executor orchestration | `superbot/` — request → validated task DAG → agents → merged answer |
| Parallel fan-out / fan-in (map-reduce) | `superbot/executor.py` — `Send` per ready task, merged through a reducer |
| Live execution graph | [`execution-graph.tsx`](frontend/src/components/thread/execution-graph.tsx) — the DAG with per-task status |
| Capability registry | `core/registry.py` — agents self-declare `capabilities` |
| Cross-agent long-term memory | `core/store.py` + `core/memory.py` — a Mongo `BaseStore` behind middleware |
| Human-in-the-loop interrupts | Email Agent — approval before send |
| Declarative approval policy | `core/approval.py` — tools mark *themselves* destructive |
| Provider abstraction | `tools/email/` — one protocol, a working mock, a documented Gmail seam |
| RAG | PDF Chatbot — chunk → embed → Atlas Vector Search → grounded answers |
| MCP tool loading | `tools/mcp.py` + `mcp_servers.json` |
| Cost & latency telemetry | `core/telemetry.py` — per-run tokens, p95 latency, cost |

---

## Quick start

**Prerequisites:** Python 3.11–3.13, Node.js 18+, and a MongoDB Atlas connection
string (the free **M0** tier is enough — Vector Search is included).

```bash
git clone https://github.com/miteshjadav98/SuperAIBot.git
cd SuperAIBot

cp .env.example .env        # then fill in the keys below
```

Minimum to boot: an LLM provider (Azure OpenAI by default), `MONGODB_URI`, and
`AUTH_SECRET`. Everything else has a working default.

```bash
python -m venv .venv
.venv/bin/pip install -r backend/requirements.txt      # Windows: .venv\Scripts\pip.exe
cd frontend && npm install && cd ..
```

**Run it** — Windows one-liner, which opens three windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

Or three terminals, from the repo root (`.venv/bin/` on macOS/Linux,
`.venv\Scripts\` on Windows):

```bash
cd backend && ../.venv/bin/langgraph dev --no-browser --allow-blocking   # :2024  agent graphs + streaming
cd backend && ../.venv/bin/python main.py                                # :8000  FastAPI gateway
cd frontend && npm run dev                                               # :3000  chat UI
```

`--allow-blocking` is required: the PDF Chatbot does legitimate blocking
parse/embed work that the dev server's blocking-call detector would otherwise
reject.

Open **http://localhost:3000**, create an account, and you're in.

### Try these

| Agent | Prompt |
| --- | --- |
| 🤖 Super Bot | *"Recommend a sci-fi film for tonight, and a dinner I can make with chicken, rice and peppers."* — watch the execution graph show two agents running in parallel |
| 🍳 Personal Chef | *"I have eggs, spinach and feta. What can I make?"* |
| ✉️ Email Agent | *"Reply to Jane's coffee email saying Tuesday at 9 works for me."* — you'll be asked to approve before it sends |
| 💍 Wedding Planner | *"Plan a wedding: NYC to Lisbon, 80 guests, jazz playlist."* |
| 🧠 Memory | Tell the Chef *"I'm vegetarian"*, then start a new chat with the Movie Recommender and ask for a film and a snack |

Select the Email Agent **directly** (not via Super Bot) to exercise the approval
flow — see [known gaps](ARCHITECTURE.md#9-known-gaps).

---

## How it works

```text
Browser (Next.js chat UI :3000)
  │
  ├─ streaming chat ──────────►  LangGraph server :2024   (every agent + `superbot`)
  └─ auth / upload / metrics ─►  FastAPI gateway   :8000   (api/app.py)
                                       │
                                       └──►  MongoDB Atlas
                                             memory · checkpoints · users · vectors · prompts
```

### Planning

Picking an agent from the dropdown talks to it directly. Picking **Super Bot
(Auto)** — or calling `POST /chat` without `agent_id` — runs the planner:

```text
plan → dispatch ──Send × N──→ execute ─┐   (independent tasks run in parallel)
         ↑                             │
         └─────────────────────────────┘
         └──→ synthesize → END
```

A request needing one agent produces a one-task plan and returns that agent's
answer verbatim — no extra synthesis call. Everything is bounded: `MAX_TASKS`
fan-out width, `MAX_LAYERS` dispatch rounds, two attempts per task. A failing
agent is recorded as a failed task rather than aborting the others, and if
planning itself fails the run degrades to a single-agent classifier.

### Memory

Every agent shares one long-term memory, scoped to the signed-in user:

```python
# The whole integration, per agent:
agent = create_agent(model, tools=[...], middleware=[MemoryMiddleware()])
```

Tell the 🍳 Personal Chef you're vegetarian and the 🎬 Movie Recommender knows
tomorrow, in a different thread. The namespace comes **only** from authenticated
identity, never from model output, so memory can't leak across accounts — with
no signed-in user it's disabled rather than pooled. `GET /memories` and
`DELETE /memories/{key}` let users see and remove what's stored.

### Streaming

The chat renders **no tool calls** — an agent chat that prints every invocation
reads like a debug log, and the full trace belongs in LangSmith. What you see
instead is the execution graph.

Token streaming has to answer a question a single-agent app never faces: *whose*
tokens? The planner emits structured JSON and is suppressed; sub-agents stream
only when the plan has one task; with several agents running, only the final
merge streams. Details in [ARCHITECTURE.md §5b](ARCHITECTURE.md).

---

## Repo map

```text
backend/
├─ core/
│  ├─ settings.py     # one pydantic-settings singleton (reads ../.env)
│  ├─ base_agent.py   # the agent contract (AgentManifest)
│  ├─ registry.py     # which agents exist + capability lookup
│  ├─ store.py        # MongoDB-backed LangGraph BaseStore
│  ├─ memory.py       # shared cross-agent memory + MemoryMiddleware
│  ├─ approval.py     # @requires_approval — declarative HITL
│  ├─ concurrency.py  # one run at a time per thread_id
│  ├─ telemetry.py    # per-run tokens, latency, cost
│  ├─ lg_auth.py      # LangGraph JWT auth — isolates chats & PDFs per user
│  └─ prompts.py      # prompts from Mongo, versioned, code fallback
├─ superbot/
│  ├─ state.py        # state schema, reducers, and every bound
│  ├─ planner.py      # request → validated task DAG
│  ├─ executor.py     # dispatch / parallel execute / synthesize
│  ├─ router.py       # single-agent classifier — the planner's fallback
│  └─ graph.py        # topology assembly only, no logic
├─ agents/            # one file each: exports `agent` + `MANIFEST`
├─ tools/
│  ├─ mcp.py          # shared MCP loader + retry interceptor
│  └─ email/          # EmailProvider protocol + mock mailbox + Gmail seam
├─ llm/factory.py     # get_chat_model() — Azure default, 5 providers
├─ api/app.py         # FastAPI gateway
├─ tests/             # pytest — hermetic, no DB or network
└─ langgraph.json     # graph ids served by `langgraph dev`

frontend/             # Next.js chat UI
prompt_service/       # standalone prompt versioning API (Swagger at /docs)
```

**The rule that keeps this navigable:** dependencies point *inward*, toward
`core/`. An agent may import `core`; `core` never imports an agent.

---

## API

All of `/chat`, `/ask`, `/upload`, `/memories` and `/metrics` need
`Authorization: Bearer <token>` from `POST /auth/login`.

| Endpoint | Purpose |
| --- | --- |
| `POST /auth/register` · `/auth/login` · `GET /auth/me` | Email + password auth, returns a JWT |
| `GET /agents` | The registry — agents and their capabilities |
| `POST /chat` | `{"query": "..."}` plans across agents; add `"agent_id"` to target one. Returns the answer, the plan, and token usage |
| `POST /upload` · `POST /ask` | PDF ingestion and RAG Q&A |
| `GET /memories` · `DELETE /memories/{key}` | See and delete what's remembered about you |
| `GET /metrics` | Tokens, median/p95 latency, cost over recent runs |
| `GET /health` | Liveness + registered agents |

---

## Tests

```bash
cd backend && pytest        # 83 tests, ~13s — no database, no network
```

`mongomock` backs the store tests, so the MongoDB `BaseStore` is exercised
through a real pymongo API surface without Atlas or Docker.

---

## Adding an agent

**One new file + two registration lines.** No existing agent is touched. The
🎬 Movie Recommender is the worked example — copy
[`backend/agents/movie_recommender.py`](backend/agents/movie_recommender.py).

**1. Create the agent.** Build the model from the factory (never hardcode a
provider), compile a graph as `agent`, and declare a `MANIFEST` — its
`description` and `capabilities` are what the planner reads:

```python
from langchain.agents import create_agent
from core.base_agent import AgentManifest
from core.memory import MemoryMiddleware
from llm.factory import get_chat_model

agent = create_agent(
    model=get_chat_model(),
    tools=[...],
    system_prompt="...",
    middleware=[MemoryMiddleware()],      # shared long-term memory, one line
)

MANIFEST = AgentManifest(
    id="movie_recommender",
    label="Movie Recommender",
    emoji="🎬",
    description="Recommends movies based on the user's taste, mood, or films "
                "they liked. Use for anything about films or what to watch.",
    agent_type="langchain",
    builder=lambda: agent,
    capabilities=["movies", "search"],    # coarse domain keys, indexed by the registry
)
```

**2. Register it:**

| File | Add |
| --- | --- |
| [`backend/core/registry.py`](backend/core/registry.py) | `"agents.movie_recommender"` to `_AGENT_MODULES` |
| [`backend/langgraph.json`](backend/langgraph.json) | `"movie_recommender": "./agents/movie_recommender.py:agent"` |

**No frontend change is needed** — the dropdown discovers agents from
`GET /agents` at runtime. Restart `langgraph dev` and the agent is live, both
directly and as a task the planner can delegate to.

**Gate anything destructive.** A tool that sends, deletes or spends should say so
itself:

```python
from core.approval import requires_approval

@requires_approval(describe=lambda args: f"Send an email to {args['to']}")
@tool
def send_email(to: str, subject: str, body: str) -> str:
    ...
```

Then pass `approval_middleware(TOOLS)` to `create_agent`. Nothing is gated by
default, so the cost of forgetting to *unmark* something is an extra prompt —
not an unwanted send.

### Adding an MCP server

A config edit, not code. Add the server to
[`backend/tools/mcp_servers.json`](backend/tools/mcp_servers.json):

```jsonc
{
  "travel": { "transport": "streamable_http", "url": "https://mcp.kiwi.com" },
  "github": { "transport": "stdio", "command": "npx",
              "args": ["-y", "@modelcontextprotocol/server-github"] }
}
```

Then pull its tools into any agent. A down server degrades to `[]` rather than
breaking the agent, and a retry interceptor is applied automatically:

```python
from tools.mcp import get_mcp_tools

agent = create_agent(model=get_chat_model(), tools=get_mcp_tools(["github"]), ...)
```

---

## Configuration

Everything is read once into a single settings object
([`core/settings.py`](backend/core/settings.py)) from `.env` — no module reaches
into `os.environ` directly. See [`.env.example`](.env.example) for the full list.

| Setting | Default | Notes |
| --- | --- | --- |
| `DEFAULT_LLM_PROVIDER` | `azure` | Also `openai`, `anthropic`, `gemini`, `ollama` — install the matching optional package |
| `MONGODB_URI` | — | Required for login. Without it the platform still boots with in-memory fallbacks |
| `AUTH_SECRET` | — | JWT signing key: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `EMAIL_PROVIDER` | `mock` | A working in-memory mailbox. `gmail` is a documented seam, not yet implemented |
| `RUN_CONCURRENCY_POLICY` | `reject` | Or `enqueue`. What happens on a second request to a busy thread |
| `MODEL_PRICING` | unset | USD per million tokens as JSON. Tokens are always recorded; cost only when set, since rates vary by contract |

### MongoDB Atlas

One cluster and database backs everything:

| Collection | Used for |
| --- | --- |
| `users` | Accounts (email, bcrypt hash) |
| `agent_memory` | Cross-agent long-term memory, namespaced per user |
| `pdf_chunks` | Atlas Vector Search for PDF RAG, stamped with `owner` |
| `checkpoints*` | Per-user conversation memory |
| `run_metrics` | Tokens, latency and cost per run |
| `prompts` / `prompt_registry` / `prompt_audit` | Versioned agent prompts |

Setup: create a cluster → *Database Access* add a user → *Network Access* allow
your IP → *Connect → Drivers* → copy the `mongodb+srv://...` string into
`MONGODB_URI`. Indexes are created by the application on first use.

### Prompt management

Agent prompts aren't hardcoded — they're versioned in MongoDB and loaded when
each graph is built ([`core/prompts.py`](backend/core/prompts.py)). A prompt is
auto-seeded as version 1 from its in-code default on first run, and if Mongo is
unreachable the agent falls back to that default and keeps working.

[`prompt_service/`](prompt_service/) is a standalone FastAPI service for
creating, versioning, publishing and rolling back those prompts, with Swagger UI
at `/docs`. It must point at the same `MONGODB_DB` as the backend. Edits take
effect on the next agent restart.

---

## Notes and caveats

- **Switching agents starts a fresh conversation** — each agent has a different
  state shape.
- **The Wedding Planner loads MCP tools at startup**, so its first request is
  slower and the flights tool can be flaky. Venues and playlist work independently.
- **The Gmail provider is a stub.** [`tools/email/gmail.py`](backend/tools/email/gmail.py)
  documents exactly what to build and fails loudly; the mock is the working
  default. Shipping untested Gmail API calls would be worse than an honest gap.
- **Interrupts through the planner** don't resume cleanly yet — select the Email
  Agent directly for the approval flow. Tracked as gap #1.
- Honest list of what's unfinished: [ARCHITECTURE.md §9](ARCHITECTURE.md#9-known-gaps).

## Deployment

[`deployment_script.sh`](deployment_script.sh) provisions the whole stack on a
single Ubuntu VM — four systemd services, an nginx reverse proxy with SSE
support for token streaming, and Let's Encrypt SSL. It's written to share a box
safely with another app: uniquely-named services, non-default ports, and it
never touches the default nginx site.
