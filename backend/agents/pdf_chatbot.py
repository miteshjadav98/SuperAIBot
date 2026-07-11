import os
import threading
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
TEXT_INDEX_NAME = "pdf_text_index"
EMBEDDING_DIMS = 1536  # text-embedding-3-small

# Hybrid retrieval: pull a wide candidate set from both searchers, then let the
# reranker cut it down to what actually goes into the model's context.
CANDIDATE_K = 10
FINAL_K = 4

_index_ready = False
_text_index_ready = False


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


def _ensure_text_index() -> None:
    """Create the Atlas full-text ($search) index used by hybrid retrieval.
    Same retry-on-every-use pattern as `_ensure_vector_index`. The index covers
    the `text` field because that's MongoDBAtlasVectorSearch's default
    text_key for chunk content."""
    global _text_index_ready
    if _text_index_ready or isinstance(vector_store, InMemoryVectorStore):
        return
    from core import db

    try:
        collection = db.pdf_chunks_collection()
        existing = [idx["name"] for idx in collection.list_search_indexes()]
        if TEXT_INDEX_NAME not in existing:
            from langchain_mongodb.index import create_fulltext_search_index

            create_fulltext_search_index(
                collection=collection,
                index_name=TEXT_INDEX_NAME,
                field="text",
            )
            print(f"[pdf_chatbot] created Atlas full-text index '{TEXT_INDEX_NAME}'")
        _text_index_ready = True
    except Exception as exc:  # noqa: BLE001 — retry on next use
        print(f"[pdf_chatbot] full-text index not ready yet: {exc}")


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
_ensure_text_index()

# In-memory mode has no Atlas $search, so hybrid retrieval needs its own
# keyword side: a BM25 retriever rebuilt whenever documents are added.
_memory_docs: List[Document] = []
_bm25_retriever = None

# Serializes writes to the shared in-memory store + BM25 index. /upload now runs
# add_documents_to_store in a worker thread, so parallel uploads can call it at
# the same time (Atlas writes are already thread-safe via pymongo).
_store_lock = threading.Lock()


def _rebuild_bm25() -> None:
    global _bm25_retriever
    if not _memory_docs:
        return
    try:
        from langchain_community.retrievers import BM25Retriever

        _bm25_retriever = BM25Retriever.from_documents(_memory_docs, k=CANDIDATE_K)
    except Exception as exc:  # noqa: BLE001 — vector-only retrieval still works
        print(f"[pdf_chatbot] BM25 unavailable, keyword search disabled: {exc}")


def _rrf_fuse(result_lists: List[List[Document]], rrf_k: int = 60) -> List[Document]:
    """Reciprocal rank fusion — same scheme Atlas hybrid search uses server-side,
    applied client-side for the in-memory store."""
    scores: dict = {}
    docs_by_key: dict = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = doc.page_content
            docs_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [docs_by_key[key] for key in ordered]


def _hybrid_retrieve(query: str) -> List[Document]:
    """Vector + keyword candidates fused by RRF. Atlas does the fusion in one
    aggregation via MongoDBAtlasHybridSearchRetriever; in-memory fuses vector
    similarity with BM25 locally. Falls back to pure vector search whenever the
    keyword side isn't available yet."""
    if isinstance(vector_store, InMemoryVectorStore):
        vector_hits = vector_store.similarity_search(query, k=CANDIDATE_K)
        keyword_hits = _bm25_retriever.invoke(query) if _bm25_retriever else []
        if not keyword_hits:
            return vector_hits
        return _rrf_fuse([vector_hits, keyword_hits])

    _ensure_text_index()
    if _text_index_ready:
        try:
            from langchain_mongodb.retrievers import MongoDBAtlasHybridSearchRetriever

            retriever = MongoDBAtlasHybridSearchRetriever(
                vectorstore=vector_store,
                search_index_name=TEXT_INDEX_NAME,
                k=CANDIDATE_K,
            )
            return retriever.invoke(query)
        except Exception as exc:  # noqa: BLE001 — degrade to vector-only
            print(f"[pdf_chatbot] hybrid search failed, using vector-only: {exc}")
    return vector_store.similarity_search(query, k=CANDIDATE_K)


_reranker = None


def _rerank(query: str, docs: List[Document], top_n: int = FINAL_K) -> List[Document]:
    """Cross-encoder precision pass over the fused candidates (FlashRank runs a
    small local ONNX model; first use downloads it to the temp dir). If the
    model can't load, keep the fused RRF order instead of failing retrieval."""
    global _reranker
    if len(docs) <= top_n:
        return docs
    try:
        from flashrank import Ranker, RerankRequest

        if _reranker is None:
            import tempfile

            _reranker = Ranker(cache_dir=os.path.join(tempfile.gettempdir(), "flashrank"))
        passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
        ranked = _reranker.rerank(RerankRequest(query=query, passages=passages))
        return [docs[item["id"]] for item in ranked[:top_n]]
    except Exception as exc:  # noqa: BLE001 — fused order is already ranked
        print(f"[pdf_chatbot] reranker unavailable, keeping fused order: {exc}")
        return docs[:top_n]

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
        _ensure_text_index()
        if isinstance(vector_store, InMemoryVectorStore):
            # In-memory store and BM25 index are process-shared mutable state.
            with _store_lock:
                vector_store.add_documents(documents=docs)
                _memory_docs.extend(docs)
                _rebuild_bm25()
        else:
            vector_store.add_documents(documents=docs)
        if wait_for_index:
            return _wait_until_searchable(docs)
    return True

@tool
def ask_pdf_knowledge_base(query: str) -> str:
    """Ask a question to the PDF knowledge base to retrieve answers from the uploaded documents."""
    try:
        # Hybrid retrieval (vector + keyword, RRF-fused), then rerank the
        # candidates down to the chunks that actually enter the context.
        _ensure_vector_index()
        candidates = _hybrid_retrieve(query)
        if not candidates:
            return "No relevant information found in the uploaded documents."

        results = _rerank(query, candidates)
        combined_content = "\n\n---\n\n".join([doc.page_content for doc in results])
        return combined_content
    except Exception as e:
        return f"Error querying knowledge base: {str(e)}"

from core.prompts import get_prompt

_DEFAULT_SYSTEM_PROMPT = """
You are a helpful PDF Chatbot agent.
You can answer user questions based on the uploaded PDF documents.
Always use the `ask_pdf_knowledge_base` tool to search for information before answering questions related to the documents.
If the tool does not provide the answer, you can let the user know that the information is not present in the provided documents.
"""

system_prompt = get_prompt(
    "pdf_chatbot_system",
    _DEFAULT_SYSTEM_PROMPT,
    name="PDF Chatbot — System Prompt",
)

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
