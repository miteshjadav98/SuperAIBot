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

# Global in-memory vector store
vector_store = InMemoryVectorStore(embeddings)

def add_documents_to_store(docs: List[Document]):
    """Helper for the FastAPI server to add documents to the store."""
    if docs:
        vector_store.add_documents(documents=docs)

@tool
def ask_pdf_knowledge_base(query: str) -> str:
    """Ask a question to the PDF knowledge base to retrieve answers from the uploaded documents."""
    try:
        # Search for relevant documents
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
import sys

# Connect to Redis for persistent LangGraph state memory
# pyrefly: ignore [missing-import]
from redis import Redis
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from simple_redis_saver import SimpleRedisSaver

redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))
redis_password = os.getenv("REDIS_PASSWORD", None)

try:
    redis_conn = Redis(host=redis_host, port=redis_port, db=0, password=redis_password)
    # Just to verify connection early
    redis_conn.ping()
    memory = SimpleRedisSaver(redis_conn)
except Exception as e:
    print(f"Failed to connect to Redis: {e}")
    memory = None

agent = create_agent(
    model=azure_model,
    tools=[ask_pdf_knowledge_base],
    system_prompt=system_prompt,
    checkpointer=memory
)
