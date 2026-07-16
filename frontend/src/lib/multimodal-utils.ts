import { ContentBlock } from "@langchain/core/messages";
import { toast } from "sonner";
import { GATEWAY_URL } from "@/providers/Auth";

// Returns a Promise of a typed multimodal block for images or PDFs
export async function fileToContentBlock(
  file: File,
): Promise<ContentBlock.Multimodal.Data> {
  const supportedImageTypes = [
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
  ];
  const supportedFileTypes = [...supportedImageTypes, "application/pdf"];

  if (!supportedFileTypes.includes(file.type)) {
    toast.error(
      `Unsupported file type: ${file.type}. Supported types are: ${supportedFileTypes.join(", ")}`,
    );
    return Promise.reject(new Error(`Unsupported file type: ${file.type}`));
  }

  const data = await fileToBase64(file);

  if (supportedImageTypes.includes(file.type)) {
    return {
      type: "image",
      // source_type + mime_type (snake_case) are what LangChain's Python
      // OpenAI block translator looks for to build the `image_url` payload;
      // mimeType (camelCase) is kept for our own preview rendering.
      source_type: "base64",
      mime_type: file.type,
      mimeType: file.type,
      data,
      metadata: { name: file.name },
    } as ContentBlock.Multimodal.Data;
  }

  // PDF
  return {
    type: "file",
    source_type: "base64",
    mime_type: "application/pdf",
    mimeType: "application/pdf",
    data,
    filename: file.name,
    metadata: { filename: file.name },
  } as ContentBlock.Multimodal.Data;
}

// Helper to convert File to base64 string
export async function fileToBase64(file: File): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      // Remove the data:...;base64, prefix
      resolve(result.split(",")[1]);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// True for a PDF file content block (checks both snake_case and camelCase mime).
export function isPdfBlock(block: ContentBlock.Multimodal.Data): boolean {
  if (block.type !== "file") return false;
  const b = block as { mimeType?: unknown; mime_type?: unknown };
  return (
    b.mimeType === "application/pdf" || b.mime_type === "application/pdf"
  );
}

// Decode a base64 string into a Blob (browser atob → byte array).
function base64ToBlob(base64: string, mimeType: string): Blob {
  const byteChars = atob(base64);
  const byteNumbers = new Array(byteChars.length);
  for (let i = 0; i < byteChars.length; i += 1) {
    byteNumbers[i] = byteChars.charCodeAt(i);
  }
  return new Blob([new Uint8Array(byteNumbers)], { type: mimeType });
}

/**
 * Send a chat-attached PDF to the FastAPI `/upload` endpoint so it is chunked,
 * embedded, and persisted in the Atlas vector store — making it retrievable by
 * the pdf_chatbot's `ask_pdf_knowledge_base` RAG tool across future turns. The
 * inline base64 block is still sent with the message so the model can also read
 * the file directly on the current turn.
 */
export interface PdfIngestResult {
  filename: string;
  chunks: number;
  pages: number;
  textEngine: string;
  imagesTotal: number;
  imagesDescribed: number;
  imageCap: number;
}

export async function ingestPdfBlock(
  block: ContentBlock.Multimodal.Data,
  token: string | null,
): Promise<PdfIngestResult> {
  const meta = (block.metadata ?? {}) as { filename?: string };
  const filename =
    meta.filename ||
    (block as { filename?: string }).filename ||
    "document.pdf";

  const blob = base64ToBlob(
    (block as { data: string }).data,
    "application/pdf",
  );
  const formData = new FormData();
  formData.append("file", blob, filename);

  const res = await fetch(`${GATEWAY_URL}/upload`, {
    method: "POST",
    headers: token ? { Authorization: `Bearer ${token}` } : undefined,
    body: formData,
  });
  const data = await res.json().catch(() => ({}) as Record<string, unknown>);
  if (!res.ok) {
    throw new Error(
      typeof (data as { detail?: unknown }).detail === "string"
        ? (data as { detail: string }).detail
        : "PDF ingestion failed",
    );
  }
  const d = data as Record<string, unknown>;
  return {
    filename,
    chunks: Number(d.chunks ?? 0),
    pages: Number(d.pages ?? 0),
    textEngine: String(d.text_engine ?? ""),
    imagesTotal: Number(d.images_total ?? 0),
    imagesDescribed: Number(d.images_described ?? 0),
    imageCap: Number(d.image_cap ?? 0),
  };
}

// Read a block's MIME type tolerating both camelCase (`mimeType`, what we set
// on send) and snake_case (`mime_type`, what often survives the round-trip
// through the LangGraph server). Missing one of these was why attachments
// reloaded from chat history rendered as empty/black boxes.
export function blockMimeType(block: unknown): string | undefined {
  if (typeof block !== "object" || block === null) return undefined;
  const b = block as { mimeType?: unknown; mime_type?: unknown };
  if (typeof b.mimeType === "string") return b.mimeType;
  if (typeof b.mime_type === "string") return b.mime_type;
  return undefined;
}

// Type guard for Base64ContentBlock
export function isBase64ContentBlock(
  block: unknown,
): block is ContentBlock.Multimodal.Data {
  if (typeof block !== "object" || block === null || !("type" in block))
    return false;
  const type = (block as { type: unknown }).type;
  const mime = blockMimeType(block);
  if (!mime) return false;
  // file type (legacy): images or PDFs
  if (type === "file" && (mime.startsWith("image/") || mime === "application/pdf")) {
    return true;
  }
  // image type (new)
  if (type === "image" && mime.startsWith("image/")) {
    return true;
  }
  return false;
}
