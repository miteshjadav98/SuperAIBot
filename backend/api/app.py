"""FastAPI gateway for the Super Bot platform.

The public control plane: health, agent discovery, a non-streaming ``/chat``
convenience endpoint (manual agent or LLM-routed), and the PDF RAG endpoints
that used to live in ``main.py``. Live token streaming for the chat UI stays on
the LangGraph dev server (:2024); this gateway is for control + simple calls.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone

import jwt as pyjwt
from bson import ObjectId
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from pymongo.errors import DuplicateKeyError, PyMongoError

from core import db
from core.base_agent import BaseAgent
from core.registry import registry
from core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from core.settings import settings
from superbot.router import route

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
    return {"agents": registry.manifests()}


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


def _gateway_graph(agent: BaseAgent):
    if not db.mongo_configured():
        return agent.graph  # no memory, but still functional
    if agent.id in _persistent_graphs:
        return _persistent_graphs[agent.id]

    builder = getattr(agent.graph, "builder", None)
    if builder is None:
        return agent.graph

    from langgraph.checkpoint.mongodb import MongoDBSaver

    saver = MongoDBSaver(db.get_client(), db_name=settings.mongodb_db)
    graph = builder.compile(checkpointer=saver)
    _persistent_graphs[agent.id] = graph
    return graph


@app.post("/chat")
async def chat(request: ChatRequest, user: dict = Depends(get_current_user)):
    # Manual override wins; otherwise the LLM router classifies.
    agent_id = request.agent_id if registry.get(request.agent_id or "") else None
    if agent_id is None:
        agent_id = await route(request.query)

    agent = registry.get(agent_id) or registry.get(settings.default_agent_id)
    if agent is None:
        raise HTTPException(status_code=503, detail="No agents are registered.")

    from langchain_core.messages import HumanMessage

    # Namespace threads by user so memory is private per account.
    config = {"configurable": {"thread_id": f"{user['_id']}:{request.thread_id}"}}
    try:
        graph = _gateway_graph(agent)
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=request.query)]}, config=config
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))

    last = result["messages"][-1]
    return {
        "agent_id": agent_id,
        "routed": request.agent_id is None,
        "answer": getattr(last, "content", str(last)),
    }


# --- PDF RAG (moved from main.py) -------------------------------------------

class AskRequest(BaseModel):
    query: str
    thread_id: str = "default_thread"


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...), user: dict = Depends(get_current_user)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    from agents.pdf_chatbot import add_documents_to_store
    from core.pdf_ingest import pdf_to_documents

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(await file.read())
            tmp_path = tmp_file.name

        # MarkItDown text chunks + vision descriptions of embedded images.
        docs = pdf_to_documents(tmp_path, source=file.filename)
        image_docs = sum(
            1 for d in docs if d.metadata.get("kind") == "image_description"
        )

        # Block until the chunks are actually queryable by Atlas Vector Search,
        # so the caller can rely on RAG retrieval as soon as this returns.
        searchable = add_documents_to_store(docs, wait_for_index=True)
        return {
            "message": f"Successfully uploaded and processed {file.filename}",
            "chunks": len(docs),
            "images_described": image_docs,
            "searchable": searchable,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/ask")
async def ask_question(request: AskRequest, user: dict = Depends(get_current_user)):
    from langchain_core.messages import HumanMessage

    pdf_agent = registry.get("pdf_chatbot")
    if pdf_agent is None:
        raise HTTPException(status_code=503, detail="PDF chatbot is not registered.")

    try:
        config = {
            "configurable": {"thread_id": f"{user['_id']}:{request.thread_id}"}
        }
        graph = _gateway_graph(pdf_agent)
        response = await graph.ainvoke(
            {"messages": [HumanMessage(content=request.query)]}, config=config
        )
        return {"answer": response["messages"][-1].content}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
