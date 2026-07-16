"""PDF → LangChain Documents for the RAG pipeline.

Text: Azure Document Intelligence (prebuilt-layout) is the primary extractor —
it returns Markdown AND OCRs scanned / image-only pages that have no text layer.
When it's unconfigured or its free-tier budget is exhausted, ingestion falls
back to MarkItDown, then PyPDFLoader, so uploads never break.

Images: the text extractors describe embedded images poorly (or not at all), so
PyMuPDF extracts each one and the platform's vision-capable chat model describes
it (transcribing visible text, explaining charts/diagrams). Each description
becomes its own Document with page metadata — downstream retrieval (hybrid
search + reranking in agents/pdf_chatbot.py) treats them as ordinary chunks.
Vision calls are capped per PDF (cost); ``pdf_to_documents`` reports how many
images existed vs. how many were described so the caller can tell the user.
"""

from __future__ import annotations

import base64
from typing import List, Tuple

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


def _extract_text(pdf_path: str, page_count: int) -> Tuple[str, str]:
    """Return ``(markdown_text, engine)``. Tries Azure Document Intelligence
    first (handles scanned pages via OCR), then falls back to MarkItDown."""
    from core import doc_intelligence

    if doc_intelligence.configured():
        try:
            with open(pdf_path, "rb") as fh:
                data = fh.read()
            markdown = doc_intelligence.extract_markdown(data, page_count)
            if markdown and markdown.strip():
                return markdown, "document_intelligence"
        except Exception as exc:  # noqa: BLE001 — fall back to MarkItDown
            print(f"[pdf_ingest] Document Intelligence read failed ({exc}); using MarkItDown")

    return _extract_text_markdown(pdf_path), "markitdown"


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


def _image_documents(pdf_path: str, source: str) -> Tuple[List[Document], int]:
    """Describe embedded images with the vision model, capped at
    ``MAX_IMAGES_PER_PDF``. Returns ``(documents, total_meaningful_images)`` —
    the total counts every image worth describing (deduped, above the size
    floor), even those skipped by the cap, so the caller can report "described
    N of M images" to the user."""
    import fitz  # PyMuPDF

    from llm.factory import get_chat_model

    docs: List[Document] = []
    vision_model = None
    seen_xrefs: set[int] = set()
    total_meaningful = 0

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
                total_meaningful += 1
                if len(docs) >= MAX_IMAGES_PER_PDF:
                    # Over the cap: count it (so the user is told), don't describe.
                    continue

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
    if total_meaningful > len(docs):
        print(
            f"[pdf_ingest] {source}: described {len(docs)} of {total_meaningful} "
            f"images (cap {MAX_IMAGES_PER_PDF})"
        )
    return docs, total_meaningful


def pdf_to_documents(pdf_path: str, source: str) -> Tuple[List[Document], dict]:
    """Full ingestion: text chunks (Document Intelligence OCR / MarkItDown) plus
    image-description documents. Returns ``(documents, stats)`` where ``stats``
    reports the text engine used, page count, and image totals so the upload
    endpoint can surface them to the user."""
    import fitz  # PyMuPDF

    try:
        with fitz.open(pdf_path) as pdf:
            page_count = len(pdf)
    except Exception:  # noqa: BLE001 — page count is only used for the budget/stats
        page_count = 0

    text, text_engine = _extract_text(pdf_path, page_count)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    docs = splitter.create_documents(
        [text], metadatas=[{"source": source, "kind": "text"}]
    )

    images_total = 0
    images_described = 0
    try:
        image_docs, images_total = _image_documents(pdf_path, source)
        docs.extend(image_docs)
        images_described = len(image_docs)
    except Exception as exc:  # noqa: BLE001 — text-only ingestion still succeeds
        print(f"[pdf_ingest] image extraction failed for {source}: {exc}")

    stats = {
        "text_engine": text_engine,
        "pages": page_count,
        "images_total": images_total,
        "images_described": images_described,
        "image_cap": MAX_IMAGES_PER_PDF,
    }
    return docs, stats
