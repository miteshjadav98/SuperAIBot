"""FastAPI gateway for the Super Bot platform.

The public control plane: health, agent discovery, a non-streaming ``/chat``
convenience endpoint (manual agent or LLM-routed), and the PDF RAG endpoints
that used to live in ``main.py``. Live token streaming for the chat UI stays on
the LangGraph dev server (:2024); this gateway is for control + simple calls.
"""

from __future__ import annotations

import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.registry import registry
from core.settings import settings
from superbot.router import route

app = FastAPI(title="Super Bot Platform API")


# --- Health & discovery -----------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "agents": registry.ids()}


@app.get("/agents")
async def list_agents():
    return {"agents": registry.manifests()}


# --- Chat (manual agent or LLM fallback) ------------------------------------

class ChatRequest(BaseModel):
    query: str
    agent_id: str | None = None  # explicit pick wins; omit to let the router decide
    thread_id: str = "default_thread"


@app.post("/chat")
async def chat(request: ChatRequest):
    # Manual override wins; otherwise the LLM router classifies.
    agent_id = request.agent_id if registry.get(request.agent_id or "") else None
    if agent_id is None:
        agent_id = await route(request.query)

    agent = registry.get(agent_id) or registry.get(settings.default_agent_id)
    if agent is None:
        raise HTTPException(status_code=503, detail="No agents are registered.")

    config = {"configurable": {"thread_id": request.thread_id}}
    try:
        result = await agent.invoke(request.query, config=config)
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
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    from agents.pdf_chatbot import add_documents_to_store

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(await file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        splits = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        ).split_documents(documents)

        add_documents_to_store(splits)
        return {
            "message": f"Successfully uploaded and processed {file.filename}",
            "chunks": len(splits),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/ask")
async def ask_question(request: AskRequest):
    from langchain_core.messages import HumanMessage
    from agents.pdf_chatbot import agent as pdf_agent

    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        response = pdf_agent.invoke(
            {"messages": [HumanMessage(content=request.query)]}, config=config
        )
        return {"answer": response["messages"][-1].content}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
