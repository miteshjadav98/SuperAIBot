"""FastAPI gateway for the Super Bot platform.

The public control plane: health, agent discovery, a non-streaming ``/chat``
convenience endpoint (manual agent or LLM-routed), and the PDF RAG endpoints
that used to live in ``main.py``. Live token streaming for the chat UI stays on
the LangGraph dev server (:2024); this gateway is for control + simple calls.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import datetime, timezone

import jwt as pyjwt
from bson import ObjectId
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from pymongo.errors import DuplicateKeyError, PyMongoError

from core import db, memory, telemetry
from core.base_agent import BaseAgent
from core.concurrency import RunBusy, thread_run
from core.prompts import get_prompt
from core.registry import registry
from core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from core.settings import settings
from llm.factory import get_chat_model

app = FastAPI(title="SuperBot Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Health & discovery -----------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "agents": registry.ids()}


@app.get("/agents")
async def list_agents():
    return {"agents": registry.manifests(), "capabilities": registry.capabilities()}


# --- Auth --------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    phone: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


def _public_user(doc: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "email": doc["email"],
        "phone": doc.get("phone"),
    }


def _auth_response(doc: dict) -> dict:
    return {
        "token": create_access_token(str(doc["_id"]), doc["email"]),
        "user": _public_user(doc),
    }


def _require_mongo():
    if not db.mongo_configured():
        raise HTTPException(
            status_code=503,
            detail="MongoDB is not configured. Set MONGODB_URI in .env (see README).",
        )


@app.post("/auth/register")
async def register(request: RegisterRequest):
    _require_mongo()
    if len(request.password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters.")

    doc = {
        "email": request.email.lower(),
        "phone": request.phone,
        "password_hash": hash_password(request.password),
        "created_at": datetime.now(timezone.utc),
    }
    try:
        result = db.users_collection().insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")

    doc["_id"] = result.inserted_id
    return _auth_response(doc)


@app.post("/auth/login")
async def login(request: LoginRequest):
    _require_mongo()
    try:
        doc = db.users_collection().find_one({"email": request.email.lower()})
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")

    if not doc or not verify_password(request.password, doc.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return _auth_response(doc)


async def get_current_user(authorization: str | None = Header(default=None)) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    try:
        payload = decode_access_token(authorization.split(" ", 1)[1])
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")

    _require_mongo()
    doc = db.users_collection().find_one({"_id": ObjectId(payload["sub"])})
    if not doc:
        raise HTTPException(status_code=401, detail="User no longer exists.")
    return doc


@app.get("/auth/me")
async def me(user: dict = Depends(get_current_user)):
    return _public_user(user)


# --- Chat (manual agent or LLM fallback) ------------------------------------

class ChatRequest(BaseModel):
    query: str
    agent_id: str | None = None  # explicit pick wins; omit to let the router decide
    thread_id: str = "default_thread"


# Graph variants compiled with a Mongo checkpointer, so gateway conversations
# have durable memory. The dev server (:2024) keeps its own persistence and
# rejects graphs that ship a checkpointer, hence the separate compile here.
_persistent_graphs: dict[str, object] = {}


def _with_checkpointer(key: str, compiled):
    """Recompile ``compiled`` against the Mongo checkpointer, cached by ``key``.
    Degrades to the checkpointer-less graph when Mongo isn't configured — no
    memory, but still functional."""
    if not db.mongo_configured():
        return compiled
    if key in _persistent_graphs:
        return _persistent_graphs[key]

    builder = getattr(compiled, "builder", None)
    if builder is None:
        return compiled

    from langgraph.checkpoint.mongodb import MongoDBSaver

    saver = MongoDBSaver(db.get_client(), db_name=settings.mongodb_db)
    graph = builder.compile(checkpointer=saver)
    _persistent_graphs[key] = graph
    return graph


def _gateway_graph(agent: BaseAgent):
    return _with_checkpointer(agent.id, agent.graph)


def _superbot_graph():
    """The planner/executor graph, which may span several agents in one turn."""
    from superbot.graph import agent as superbot

    return _with_checkpointer("superbot", superbot)


@app.post("/chat")
async def chat(request: ChatRequest, user: dict = Depends(get_current_user)):
    # A manual pick runs that agent directly. Otherwise the Super Bot plans the
    # request, which may fan out across several agents.
    agent_id = request.agent_id if registry.get(request.agent_id or "") else None

    if agent_id is None:
        graph = _superbot_graph()
    else:
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=503, detail="No agents are registered.")
        graph = _gateway_graph(agent)

    from langchain_core.messages import HumanMessage

    # Namespace threads by user so memory is private per account, and pass the
    # owner so the PDF RAG tool only retrieves this user's documents.
    owner = str(user["_id"])
    thread_id = f"{owner}:{request.thread_id}"
    config = {"configurable": {"thread_id": thread_id, "owner": owner}}

    try:
        # One run at a time per thread: concurrent runs share a checkpoint and
        # would interleave their messages into a conversation that never
        # happened.
        async with thread_run(thread_id):
            with telemetry.measure("chat", owner=owner, thread_id=thread_id) as metrics:
                result = await graph.ainvoke(
                    {"messages": [HumanMessage(content=request.query)]}, config=config
                )
                metrics.agent_id = agent_id or result.get("routed_to")
    except RunBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))

    last = result["messages"][-1]
    return {
        "agent_id": agent_id or result.get("routed_to"),
        "routed": request.agent_id is None,
        "plan": result.get("plan"),
        "answer": getattr(last, "content", str(last)),
        "usage": {
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "latency_ms": metrics.latency_ms,
            "cost_usd": metrics.cost_usd,
        },
    }


# --- Conversation titles ------------------------------------------------------


class TitleMessage(BaseModel):
    role: str
    content: str


class TitleRequest(BaseModel):
    messages: list[TitleMessage]


_TITLE_DEFAULT = """Write a title for this conversation: 3-6 words, no quotes, \
no trailing punctuation, capitalised like a headline.

Name what the conversation is *about*, not how it opened. A chat that starts \
"good morning" and becomes a daily briefing is "Monday Morning Briefing", never \
"Good Morning". If there is genuinely no subject yet, answer "New Chat"."""

_TITLE_FALLBACK = "New Chat"
_TITLE_MAX_CHARS = 60


@app.post("/threads/title")
async def thread_title(request: TitleRequest, user: dict = Depends(get_current_user)):
    """A short, human title for a conversation.

    The chat list used to show the first message, which made every briefing
    thread read "hello". The caller stores the result on the thread, so this
    runs once per conversation rather than once per render.
    """
    transcript = "\n".join(
        f"{m.role}: {m.content[:500]}" for m in request.messages[:6] if m.content.strip()
    )
    if not transcript:
        return {"title": _TITLE_FALLBACK}

    system = get_prompt("chat_title_system", _TITLE_DEFAULT, name="Chat Title")
    try:
        model = get_chat_model(temperature=0, max_tokens=24)
        answer = await model.ainvoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": transcript},
            ]
        )
        title = str(answer.content).strip().strip('"').strip()
    except Exception as exc:  # noqa: BLE001 — a title is never worth a 500
        print(f"[title] generation failed: {exc}")
        return {"title": _TITLE_FALLBACK}

    return {"title": title[:_TITLE_MAX_CHARS] or _TITLE_FALLBACK}


# --- Telemetry ---------------------------------------------------------------


@app.get("/metrics")
async def metrics(user: dict = Depends(get_current_user), mine: bool = True):
    """Token, latency and cost summary over recent runs.

    Scoped to the caller by default — per-user cost is the question people
    actually ask, and it avoids exposing platform-wide volume to every account.
    """
    return telemetry.summary(owner=str(user["_id"]) if mine else None)


# --- Memory ------------------------------------------------------------------
# Long-term memory is user-visible and user-deletable by design: it is the
# cheapest and most trustworthy correction mechanism for a memory that is wrong,
# stale, or simply unwelcome — and increasingly a compliance expectation.


@app.get("/memories")
async def get_memories(user: dict = Depends(get_current_user)):
    """Everything the platform remembers about the signed-in user."""
    return {"memories": memory.list_memories(str(user["_id"]))}


@app.delete("/memories/{key}")
async def delete_memory(key: str, user: dict = Depends(get_current_user)):
    """Delete one memory. Removal is real, not a soft flag."""
    if not memory.forget(str(user["_id"]), key):
        raise HTTPException(status_code=503, detail="Memory store is unavailable.")
    return {"deleted": key}


# --- PDF RAG (moved from main.py) -------------------------------------------

class AskRequest(BaseModel):
    query: str
    thread_id: str = "default_thread"


def _ingest_pdf_sync(data: bytes, filename: str, owner: str) -> dict:
    """Parse + embed a PDF. This is deliberately synchronous and blocking
    (MarkItDown, per-image vision calls, embeddings, and a poll until Atlas
    Vector Search can see the new chunks). ``upload_pdf`` runs it in a worker
    thread so concurrent uploads run in parallel instead of serializing on the
    event loop — the frontend uploads multiple PDFs at once, and without this a
    slow upload blocks the others until they trip the proxy read timeout.

    Every chunk is stamped with ``owner`` (the uploading user's id) so retrieval
    can keep each user's documents private to them."""
    from agents.pdf_chatbot import add_documents_to_store
    from core.pdf_ingest import pdf_to_documents

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(data)
            tmp_path = tmp_file.name

        # Text chunks (Document Intelligence OCR / MarkItDown) + vision
        # descriptions of embedded images. stats reports what happened so the
        # frontend can tell the user (e.g. "described 25 of 40 images").
        docs, stats = pdf_to_documents(tmp_path, source=filename)
        for doc in docs:
            doc.metadata["owner"] = owner

        # Block until the chunks are actually queryable by Atlas Vector Search,
        # so the caller can rely on RAG retrieval as soon as this returns.
        searchable = add_documents_to_store(docs, wait_for_index=True)
        return {
            "message": f"Successfully uploaded and processed {filename}",
            "chunks": len(docs),
            "pages": stats["pages"],
            "text_engine": stats["text_engine"],
            "images_total": stats["images_total"],
            "images_described": stats["images_described"],
            "image_cap": stats["image_cap"],
            "searchable": searchable,
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...), user: dict = Depends(get_current_user)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    data = await file.read()
    try:
        return await asyncio.to_thread(
            _ingest_pdf_sync, data, file.filename, str(user["_id"])
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ask")
async def ask_question(request: AskRequest, user: dict = Depends(get_current_user)):
    from langchain_core.messages import HumanMessage

    pdf_agent = registry.get("pdf_chatbot")
    if pdf_agent is None:
        raise HTTPException(status_code=503, detail="PDF chatbot is not registered.")

    owner = str(user["_id"])
    thread_id = f"{owner}:{request.thread_id}"
    config = {"configurable": {"thread_id": thread_id, "owner": owner}}

    try:
        async with thread_run(thread_id):
            with telemetry.measure("ask", owner=owner, thread_id=thread_id) as metrics:
                metrics.agent_id = "pdf_chatbot"
                graph = _gateway_graph(pdf_agent)
                response = await graph.ainvoke(
                    {"messages": [HumanMessage(content=request.query)]}, config=config
                )
        return {"answer": response["messages"][-1].content}
    except RunBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
