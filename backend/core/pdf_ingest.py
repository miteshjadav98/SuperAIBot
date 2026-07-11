"""PDF → LangChain Documents for the RAG pipeline.

Text: MarkItDown converts the PDF's text layer to Markdown, so chunks keep
headings/lists/table structure instead of raw extracted text. Falls back to
PyPDFLoader if MarkItDown can't handle a file, so uploads never break.

Images: MarkItDown ignores images embedded in PDFs, so PyMuPDF extracts each
one and the platform's vision-capable chat model describes it (transcribing
visible text, explaining charts/diagrams). Each description becomes its own
Document with page metadata — downstream retrieval (hybrid search + reranking
in agents/pdf_chatbot.py) treats them as ordinary chunks.
"""

from __future__ import annotations

import base64
from typing import List

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Skip tiny embedded images (icons, bullets, separators) — not worth a vision call.
MIN_IMAGE_BYTES = 3000
# Cost guard: at most this many vision calls per uploaded PDF.
MAX_IMAGES_PER_PDF = 25

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

_VISION_SYSTEM_PROMPT = (
    "You describe images extracted from documents. Be concise and factual. "
    "Transcribe any visible text, and describe charts, diagrams, or photos so "
    "a reader who cannot see the image understands its content."
)


def _extract_text_markdown(pdf_path: str) -> str:
    """PDF text layer as Markdown via MarkItDown; PyPDFLoader as fallback."""
    try:
        from markitdown import MarkItDown

        return MarkItDown().convert(pdf_path).text_content
    except Exception as exc:  # noqa: BLE001 — never let conversion kill an upload
        print(f"[pdf_ingest] MarkItDown failed ({exc}); falling back to PyPDFLoader")
        from langchain_community.document_loaders import PyPDFLoader

        pages = PyPDFLoader(pdf_path).load()
        return "\n\n".join(page.page_content for page in pages)


def _describe_image(vision_model, image_bytes: bytes, ext: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    response = vision_model.invoke(
        [
            SystemMessage(content=_VISION_SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {"type": "text", "text": "Describe this image from a PDF document."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{ext};base64,{b64}"},
                    },
                ]
            ),
        ]
    )
    return (response.content or "").strip()


def _image_documents(pdf_path: str, source: str) -> List[Document]:
    """One Document per meaningful embedded image, described by the vision model."""
    import fitz  # PyMuPDF

    from llm.factory import get_chat_model

    docs: List[Document] = []
    vision_model = None
    seen_xrefs: set[int] = set()

    with fitz.open(pdf_path) as pdf:
        for page_index in range(len(pdf)):
            for img in pdf[page_index].get_images(full=True):
                xref = img[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                extracted = pdf.extract_image(xref)
                image_bytes = extracted["image"]
                if len(image_bytes) < MIN_IMAGE_BYTES:
                    continue
                if len(docs) >= MAX_IMAGES_PER_PDF:
                    print(
                        f"[pdf_ingest] {source}: image cap ({MAX_IMAGES_PER_PDF}) "
                        "reached; remaining images skipped"
                    )
                    return docs

                page_no = page_index + 1
                try:
                    if vision_model is None:
                        vision_model = get_chat_model()
                    description = _describe_image(
                        vision_model, image_bytes, extracted["ext"]
                    )
                except Exception as exc:  # noqa: BLE001 — skip image, keep the upload
                    print(f"[pdf_ingest] vision failed on page {page_no} of {source}: {exc}")
                    continue
                if not description:
                    continue

                docs.append(
                    Document(
                        page_content=(
                            f"[Image on page {page_no} of {source}]\n{description}"
                        ),
                        metadata={
                            "source": source,
                            "kind": "image_description",
                            "page": page_no,
                        },
                    )
                )
    return docs


def pdf_to_documents(pdf_path: str, source: str) -> List[Document]:
    """Full ingestion: Markdown text chunks + image-description documents."""
    markdown = _extract_text_markdown(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    docs = splitter.create_documents(
        [markdown], metadatas=[{"source": source, "kind": "text"}]
    )

    try:
        docs.extend(_image_documents(pdf_path, source))
    except Exception as exc:  # noqa: BLE001 — text-only ingestion still succeeds
        print(f"[pdf_ingest] image extraction failed for {source}: {exc}")

    return docs
