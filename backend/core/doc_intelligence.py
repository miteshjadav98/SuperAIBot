"""Azure AI Document Intelligence — OCR + layout text extraction for PDFs.

The primary text extractor for the PDF chatbot's ingestion pipeline. The
``prebuilt-layout`` model returns Markdown (headings, lists, tables) and, unlike
MarkItDown, reads scanned / image-only pages via OCR — so scanned PDFs are fully
indexed instead of silently losing their text.

The free (F0) tier allows 500 pages/month and 20 calls/minute, so this module:
  * tracks a monthly page budget (in Mongo when configured, else in-process) and
  * retries on 429 (rate limit),
and returns ``None`` whenever Document Intelligence is unconfigured, over budget,
or erroring. The caller (core/pdf_ingest.py) then falls back to MarkItDown, so an
upload never hard-fails because of Document Intelligence.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Optional

from core.settings import settings


def configured() -> bool:
    return bool(settings.azure_docintel_endpoint and settings.azure_docintel_key)


def _current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


# In-process fallback counter for when Mongo isn't configured (dev / in-memory
# mode). Persistent tracking uses the usage_counters collection instead.
_local_lock = threading.Lock()
_local_counter: dict[str, int] = {}


def _pages_used_this_month() -> int:
    month = _current_month()
    from core import db

    if db.mongo_configured():
        try:
            doc = db.usage_counters_collection().find_one({"_id": f"docintel:{month}"})
            return int((doc or {}).get("pages", 0))
        except Exception as exc:  # noqa: BLE001 — never block ingestion on the counter
            print(f"[doc_intelligence] could not read page counter: {exc}")
            return 0
    with _local_lock:
        return _local_counter.get(month, 0)


def _record_pages(n: int) -> None:
    month = _current_month()
    from core import db

    if db.mongo_configured():
        try:
            db.usage_counters_collection().update_one(
                {"_id": f"docintel:{month}"},
                {
                    "$inc": {"pages": n},
                    "$set": {"updated_at": datetime.now(timezone.utc)},
                },
                upsert=True,
            )
            return
        except Exception as exc:  # noqa: BLE001 — best-effort accounting
            print(f"[doc_intelligence] could not record page usage: {exc}")
            return
    with _local_lock:
        _local_counter[month] = _local_counter.get(month, 0) + n


def within_budget(page_count: int) -> bool:
    budget = settings.azure_docintel_monthly_page_budget
    if budget <= 0:
        return True  # 0 / negative disables the monthly cap
    return _pages_used_this_month() + page_count <= budget


def extract_markdown(data: bytes, page_count: int) -> Optional[str]:
    """Return the PDF's text as Markdown via Document Intelligence, or ``None``
    if it's unconfigured, would exceed the monthly page budget, or errors."""
    if not configured():
        return None
    if not within_budget(page_count):
        print(
            "[doc_intelligence] monthly page budget "
            f"({settings.azure_docintel_monthly_page_budget}) reached — "
            "falling back to MarkItDown for this upload."
        )
        return None

    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential

        client = DocumentIntelligenceClient(
            endpoint=settings.azure_docintel_endpoint,
            credential=AzureKeyCredential(settings.azure_docintel_key),
        )
        markdown = _analyze_with_retry(client, data)
        # Only count pages against the budget on success; a failed call that
        # falls back to MarkItDown shouldn't consume quota.
        _record_pages(page_count)
        return markdown
    except Exception as exc:  # noqa: BLE001 — fall back to MarkItDown
        print(f"[doc_intelligence] analyze failed ({exc}); falling back to MarkItDown.")
        return None


def _analyze_with_retry(client, data: bytes, attempts: int = 3) -> str:
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
    from azure.core.exceptions import HttpResponseError

    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            poller = client.begin_analyze_document(
                settings.azure_docintel_model,
                AnalyzeDocumentRequest(bytes_source=data),
                output_content_format="markdown",
            )
            result = poller.result()
            return result.content or ""
        except HttpResponseError as exc:
            # 429 = the free tier's 20 calls/minute rate limit. Back off and
            # retry; anything else propagates to the caller's fallback.
            if getattr(exc, "status_code", None) == 429 and attempt < attempts - 1:
                wait = 5 * (2**attempt)
                print(f"[doc_intelligence] rate limited (429); retrying in {wait}s")
                time.sleep(wait)
                last_exc = exc
                continue
            raise
    if last_exc:
        raise last_exc
    return ""
