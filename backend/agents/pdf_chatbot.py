import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

AZURE_DEPLOYMENT  = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
AZURE_ENDPOINT    = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY     = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")

azure_model = AzureChatOpenAI(
    azure_deployment=AZURE_DEPLOYMENT,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small", # Usually text-embedding-3-small is used, as per bonus_rag.ipynb
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION
)

VECTOR_INDEX_NAME = "pdf_vector_index"
EMBEDDING_DIMS = 1536  # text-embedding-3-small

_index_ready = False


def _ensure_vector_index() -> None:
    """Create the Atlas Vector Search index if missing. Retried on every store
    use (not just import) because Atlas' search service can lag behind the
    database being reachable."""
    global _index_ready
    if _index_ready or isinstance(vector_store, InMemoryVectorStore):
        return
    from core import db

    try:
        collection = db.pdf_chunks_collection()
        existing = [idx["name"] for idx in collection.list_search_indexes()]
        if VECTOR_INDEX_NAME not in existing:
            vector_store.create_vector_search_index(dimensions=EMBEDDING_DIMS)
            print(f"[pdf_chatbot] created Atlas search index '{VECTOR_INDEX_NAME}'")
        _index_ready = True
    except Exception as exc:  # noqa: BLE001 — retry on next use
        print(f"[pdf_chatbot] vector index not ready yet: {exc}")


def _build_vector_store():
    """Atlas Vector Search when MONGODB_URI is set (PDFs survive restarts);
    otherwise fall back to the old in-memory store so the agent still works."""
    from core import db

    if not db.mongo_configured():
        print("[pdf_chatbot] MONGODB_URI not set — using in-memory vector store.")
        return InMemoryVectorStore(embeddings)

    from langchain_mongodb import MongoDBAtlasVectorSearch

    return MongoDBAtlasVectorSearch(
        collection=db.pdf_chunks_collection(),
        embedding=embeddings,
        index_name=VECTOR_INDEX_NAME,
        relevance_score_fn="cosine",
    )


vector_store = _build_vector_store()
_ensure_vector_index()

def _wait_until_searchable(docs: List[Document], timeout: float = 45.0) -> bool:
    """Block until the just-added docs are queryable by Atlas Vector Search.

    Atlas' vector index is eventually consistent: `add_documents` returns as soon
    as the rows are written, but similarity_search won't surface them for ~10-15s
    while the search node indexes the new embeddings. We poll until a probe query
    finds our content so callers can guarantee first-turn RAG retrieval.
    Returns True once searchable, False if it timed out (best-effort)."""
    import time

    if isinstance(vector_store, InMemoryVectorStore):
        return True  # in-memory store is searchable immediately

    probe = (docs[0].page_content or "").strip()[:400] or "document"
    wanted = docs[0].page_content
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            results = vector_store.similarity_search(probe, k=5)
            if any(r.page_content == wanted for r in results):
                return True
        except Exception:  # noqa: BLE001 — index may not be live yet; retry
            pass
        time.sleep(2)
    return False


def add_documents_to_store(docs: List[Document], wait_for_index: bool = False):
    """Helper for the FastAPI server to add documents to the store.

    When `wait_for_index` is True, block until the documents are actually
    queryable (see `_wait_until_searchable`) so the caller can rely on RAG
    retrieval immediately after this returns."""
    if docs:
        _ensure_vector_index()
        vector_store.add_documents(documents=docs)
        if wait_for_index:
            return _wait_until_searchable(docs)
    return True

@tool
def ask_pdf_knowledge_base(query: str) -> str:
    """Ask a question to the PDF knowledge base to retrieve answers from the uploaded documents."""
    try:
        # Search for relevant documents
        _ensure_vector_index()
        results = vector_store.similarity_search(query, k=3)
        if not results:
            return "No relevant information found in the uploaded documents."
        
        # Combine the content of the retrieved documents
        combined_content = "\n\n---\n\n".join([doc.page_content for doc in results])
        return combined_content
    except Exception as e:
        return f"Error querying knowledge base: {str(e)}"

system_prompt = """
You are a helpful PDF Chatbot agent.
You can answer user questions based on the uploaded PDF documents.
Always use the `ask_pdf_knowledge_base` tool to search for information before answering questions related to the documents.
If the tool does not provide the answer, you can let the user know that the information is not present in the provided documents.
"""

from langchain.agents import create_agent

# NOTE: no custom checkpointer here. When this graph is served by the LangGraph
# API server (via langgraph.json) the platform provides persistence itself, and
# it rejects any graph that ships its own checkpointer (GraphLoadError). Every
# other agent on the platform follows the same rule, so PDF Chatbot does too.
agent = create_agent(
    model=azure_model,
    tools=[ask_pdf_knowledge_base],
    system_prompt=system_prompt,
)

from core.base_agent import AgentManifest

MANIFEST = AgentManifest(
    id="pdf_chatbot",
    label="PDF Chatbot",
    emoji="📄",
    description=(
        "Answers questions grounded in the content of PDF documents the user "
        "has uploaded. Use for anything about an uploaded file, document, "
        "report, paper, or 'the PDF'."
    ),
    agent_type="langchain",
    builder=lambda: agent,
    tags=["pdf", "rag", "documents"],
)
